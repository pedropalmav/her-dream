import numpy as np
import torch

import tools
from buffers.her_buffer import HERBuffer


class OnlineTrainer:
    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs, reward_function=None):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.reward_function = reward_function
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.obs_step_prob_log = bool(config.obs_step_prob_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat
        self._goal_sample = config.goal_sample
        self._goal_type = config.goal_type
        self._wm_only = bool(config.wm_only)
        self._train_text_only = bool(getattr(config, "train_text_only", False))
        # The policy is irrelevant for both world-model pretraining and
        # text-encoder distillation, so data is collected with random actions.
        self._random_actions = self._wm_only or self._train_text_only

        self.her = True if isinstance(self.replay_buffer, HERBuffer) else False

    @torch.no_grad()
    def _text_goal(self, agent, envs, indices):
        """Sample one-hot goals from the text encoder for the given env indices.

        Uses the same OneHotDist (with unimix_ratio) as the RSSM posterior so
        the goal distribution matches how the world model samples its z.

        Args:
            indices: list[int] of env indices that need a fresh goal (is_first=True).

        Returns:
            float32 one-hot tensor on agent.device with shape matching the
            env's goal space, or None if `indices` is empty.
        """
        if not indices:
            return None
        promises = [envs.envs[i].encoded_random_mission() for i in indices]
        missions = np.stack([p() for p in promises])  # (N, L) int8 token ids
        mission_t = torch.as_tensor(missions, device=agent.device)
        logits = agent.text_encoder(mission_t.unsqueeze(1))[:, 0]  # (N, S, K)
        one_hot = agent.rssm.get_dist(logits).rsample()  # (N, S, K)
        goal_shape = envs.observation_space["goal"].shape
        if len(goal_shape) == 1:
            return one_hot[:, 0, :]
        return one_hot

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        Environment stepping is executed on CPU to avoid GPU<->CPU synchronizations
        in the worker processes. Observations are moved back to GPU asynchronously
        (H2D with non_blocking=True) right before policy inference.
        """
        print("Evaluating the policy...")
        envs = self.eval_envs
        agent.eval()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)

        if self._goal_sample in ("buffer", "text"):
            goal_shape = envs.observation_space["goal"].shape
            goals = torch.zeros((envs.env_num, *goal_shape), dtype=torch.float32, device=agent.device)

        log_metrics = {}
        # cache is only used for video logging / open-loop prediction.
        cache = []
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            # Step environments on CPU.
            # (B, A)
            act_cpu = act.detach().to("cpu")
            # (B,)
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
            # Move observations back to GPU asynchronously for the agent.
            # dict of (B, 1, *)
            trans = trans_cpu.to(agent.device, non_blocking=True)
            # (B,)
            done = done_cpu.to(agent.device)

            # On is_first, refresh stored goals; then relabel trans["goal"]
            # before act so the agent conditions on the same goal used later
            # by the reward and stored in the buffer.
            if self._goal_sample in ("buffer", "text"):
                is_first = trans["is_first"][:, 0].bool()
                if is_first.any():
                    self._sample_goals(agent, envs, is_first, goals)
                self._relabel_goal(envs, goals, trans)

            # Store transition.
            # We keep the observation and the action that produced it together.

            trans["action"] = act
            if len(cache) < self.batch_length:
                cache.append(trans.clone())
            # (B, A)
            act, agent_state, _ = agent.act(trans, agent_state, eval=True, random=self._random_actions)

            self._apply_reward(agent_state["stoch"], trans)
            returns += trans["reward"][:, 0] * ~once_done

            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                    log_metrics[key] += value[:, 0] * ~once_done
            once_done |= done
        # dict of (B, T, *)
        cache = torch.stack(cache, dim=1) if len(cache) else None
        self.logger.scalar("episode/eval_score", returns.mean())
        self.logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
        for key, value in log_metrics.items():
            if key == "log_success":
                value = torch.clip(value, max=1.0)  # make sure 1.0 for success episode
            self.logger.scalar(f"episode/eval_{key[4:]}", value.mean())
        if cache is not None and "image" in cache:
            self.logger.video("eval_video", tools.to_np(cache["image"][:1]))
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            self.logger.video(
                "eval_open_loop",
                tools.to_np(
                    agent.video_pred(
                        cache[:1],  # give only first batch
                        (initial["stoch"], initial["deter"]),
                    )
                ),
            )
        self.logger.write(train_step)
        agent.train()

    def begin(self, agent):
        """Main online training loop.

        The loop is designed to overlap CPU environment stepping and GPU model
        execution. Environments are stepped on CPU, observations are pinned,
        then transferred to GPU with non_blocking=True.
        """
        envs = self.train_envs
        video_cache = []
        step = self.replay_buffer.count() * self._action_repeat
        update_count = 0
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        envs_ids = torch.arange(
            envs.env_num, dtype=torch.int32, device=agent.device
        )
        episode_ids = envs_ids.clone()  # used for HER to identify episodes in the buffer

        if self._goal_sample in ("buffer", "text"):
            goal_shape = envs.observation_space["goal"].shape
            goals = torch.zeros((envs.env_num, *goal_shape), dtype=torch.float32, device=agent.device)

        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while step < self.steps:
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0:
                self.eval(agent, step)

            # Save metrics
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        if i == 0 and len(video_cache) > 0:
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video("train_video", tools.to_np(video[None]))
                            video_cache = []
                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        self.logger.write(step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0

            step += int((~done).sum()) * self._action_repeat  # step is based on env side
            lengths += ~done

            # Step environments on CPU to avoid GPU<->CPU sync in the worker processes.
            # (B, A)
            act_cpu = act.detach().to("cpu")
            # (B,)
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)

            # Move observations back to GPU asynchronously for the agent.
            # dict of (B, 1, *)
            trans = trans_cpu.to(agent.device, non_blocking=True)
            # (B,)
            done = done_cpu.to(agent.device)

            # On is_first, refresh stored goals; then relabel trans["goal"]
            # before act so the agent conditions on the same goal used later
            # by the reward and stored in the buffer.
            if self._goal_sample in ("buffer", "text"):
                is_first = trans["is_first"][:, 0].bool()
                if is_first.any():
                    self._sample_goals(agent, envs, is_first, goals)
                self._relabel_goal(envs, goals, trans)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # In wm_only / train_text_only mode the actor is bypassed and uniform one-hot actions are used.
            # (B, A)
            act, agent_state, act_metrics = agent.act(
                trans.clone(), agent_state, eval=False, random=self._random_actions
            )
            if self.obs_step_prob_log:
                self.logger.write_step(
                    "rssm/obs_step_sample_log_prob",
                    act_metrics["obs_step_sample_log_prob"].item(),
                    step,
                )
                self.logger.write_step("rssm/obs_step_episode", episode_ids[0].float().item(), step)

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["env"] = envs_ids
            trans["episode"] = episode_ids

            self._apply_reward(trans["stoch"], trans)

            if "image" in trans:
                video_cache.append(trans["image"][0])

            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]

            episode_ids[done] += envs.env_num
            
            # Update models after enough data has accumulated
            if self._should_update(step):
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)
                for _ in range(update_num):
                    _metrics = agent.update(self.replay_buffer)
                    train_metrics = _metrics
                update_count += update_num
                # Log training metrics
                if self._should_log(step):
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", update_count)
                    if self.video_pred_log:
                        data, _, initial = self.replay_buffer.sample()
                        self.logger.video("open_loop", tools.to_np(agent.video_pred(data, initial)))
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(name, tools.to_np(param))
                    self.logger.write(step, fps=True)

    def _apply_reward(self, stoch, trans):
        if self.reward_function:
            trans["reward"] = self.reward_function(stoch, trans["goal"])

    def _relabel_goal(self, envs, goals, trans):
        # Si es que hay algun valor diferente de 0 en el goals, entonces relabel.
        # Esto siempre será True para mode="text" y para mode="buffer" será True si es que 
        # ya hay experiencias guardadas.
        mask = (goals != 0).view(envs.env_num, -1).any(dim=1)
        trans["goal"][mask] = goals[mask].clone()

    def _sample_goals(self, agent, envs, mask, goals):
        """Populate goals[i] for envs where mask[i] is True (typically is_first).

        Source is selected by self._goal_sample:
            - "buffer": sample a past stoch uniformly from the replay buffer.
                        Skipped silently when the buffer is empty.
            - "text":   sample a fresh goal from the live text encoder applied
                        to a random mission produced by each env.
        """
        if self._goal_sample == "buffer":
            if self.replay_buffer.count() == 0:
                return
            data, _, _ = self.replay_buffer.sample()
            goal_sample = data["stoch"]
            if self._goal_type == "first_row":
                goal_sample = goal_sample[..., 0, :]
            goal_sample = goal_sample.reshape(-1, *goal_sample.shape[2:])
            for i in range(envs.env_num):
                if mask[i]:
                    goals[i] = goal_sample[torch.randint(goal_sample.shape[0], (1,))]
        elif self._goal_sample == "text":
            indices = mask.nonzero(as_tuple=True)[0].tolist()
            new_goals = self._text_goal(agent, envs, indices)
            if new_goals is not None:
                goals[indices] = new_goals

    def _should_update(self, step):
        envs_num = self.train_envs.env_num
        return step // (envs_num * self._action_repeat) > self.batch_length + 1
