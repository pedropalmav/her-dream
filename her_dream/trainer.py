import numbers

import numpy as np
import torch

import her_dream.tools as tools
from her_dream.buffers.her_buffer import HERBuffer
from her_dream.goals import goal_from_latent, make_goal_spec, reward_state
from her_dream.peg import SEARCHES, explore_every_ep, go_steps

# Goal sources sampled by the trainer at episode start (everything except
# "random", where the env-generated goal in the obs is used as-is).
SAMPLED_GOAL_SOURCES = (
    "buffer",
    "text",
    "imagination",
    "imagination_refresh",
    "explore_imagination",
    "explore_imagination_refresh",
    "image",
    "peg",
)
# Goal sources that also re-sample the goal mid-episode, every
# `goal_refresh_every` agent steps (on top of the is_first sample).
REFRESHED_GOAL_SOURCES = ("imagination_refresh", "explore_imagination_refresh")
# Goal sources imagined with the explorer (`Dreamer.explore_goal`); they need
# plan2explore=True or lexa=True.
EXPLORER_GOAL_SOURCES = ("explore_imagination", "explore_imagination_refresh")


def check_goal_refresh_every(value):
    """Validate `env.goal_refresh_every` (the refresh period K) and return it.

    Must be a whole number >= 2 (any integer type, e.g. numpy's, but not bool);
    rejected instead of coerced, so 7.5 does not silently run as 7 and null
    does not surface as an anonymous int() error. K=1 is rejected because
    every step would be a goal switch, so the replay value loss would mask
    nearly all of its targets (ActorCritic.replay_value_loss).
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 2:
        raise ValueError(
            f"env.goal_refresh_every must be an integer >= 2, got {value!r} "
            "(it defaults to model.goal_imag_horizon; set it explicitly if that is below 2)."
        )
    return int(value)


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
        self._goal_spec = make_goal_spec(config)
        # Period K (agent steps) of the mid-episode goal refresh; None when the
        # goal source picks one goal per episode.
        self._goal_refresh_every = None
        # Logger namespace of the goal-selection score statistics (PEG and the
        # explorer-rollout sources); eval logs under "eval_" + this.
        self._goal_log_prefix = "explore_goal" if self._goal_sample in EXPLORER_GOAL_SOURCES else "peg"
        if self._goal_sample in REFRESHED_GOAL_SOURCES:
            self._goal_refresh_every = check_goal_refresh_every(config.goal_refresh_every)
        self._wm_only = bool(config.wm_only)
        self._train_text_only = bool(getattr(config, "train_text_only", False))
        # The policy is irrelevant for both world-model pretraining and
        # text-encoder distillation, so data is collected with random actions.
        self._random_actions = self._wm_only or self._train_text_only
        # LEXA splits data collection: the explorer drives every Nth episode and
        # the achiever the rest, so the buffer holds both novel states and
        # goal-directed ones. 0 disables the split (achiever only).
        self._lexa = bool(getattr(config, "lexa", False))
        self._explore_every_ep = explore_every_ep(getattr(config, "explore_every_ep", None), self._goal_sample)
        # PEG runs Go-Explore episodes: the achiever chases the selected goal for
        # this many agent steps, then the explorer takes over.
        self._peg_go_steps = None
        if self._goal_sample == "peg":
            peg_cfg = config.peg
            self._peg_go_steps = go_steps(peg_cfg.go_ratio, peg_cfg.time_limit, peg_cfg.action_repeat)
            # How the goal candidates are found. Only "buffer" is drawn here (stoch
            # the agent actually visited); the agent handles random and MPPI/CEM.
            self._peg_search = str(peg_cfg.search)
            if self._peg_search not in SEARCHES:
                raise ValueError(f"Unknown peg.search {self._peg_search!r}; expected one of {SEARCHES}.")
            self._peg_candidates = int(peg_cfg.num_candidates)

        self.her = isinstance(self.replay_buffer, HERBuffer)

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
        # Same OneHotDist (unimix_ratio) as the RSSM posterior, so a sampled goal
        # matches how the world model draws its z. goal_from_latent then selects
        # sample / argmax one-hot / raw logit per the goal type.
        stoch = agent.rssm.get_dist(logits).rsample()  # (N, S, K)
        goal = goal_from_latent(self._goal_spec, stoch=stoch, logit=logits)
        goal_shape = envs.observation_space["goal"].shape
        if len(goal_shape) == 1:
            return goal[:, 0, :]
        return goal

    @torch.no_grad()
    def _image_goal(self, agent, envs, indices):
        """Sample one-hot goals by encoding env-rendered goal images with the WM.

        Each env renders (with an auxiliary FixedGoal generator, see
        envs.goal_image.GoalImageGenerator) the observation of a synthetic
        state inside the current episode: the green square at the episode's
        goal position and the agent at a random cell. The frozen world model
        maps it to a goal z with a single posterior step (Dreamer.encode_observation),
        so the goal's z is consistent with the current episode's goal position
        by construction.

        Args:
            indices: list[int] of env indices that need a fresh goal (is_first=True).

        Returns:
            float32 one-hot tensor on agent.device with shape matching the
            env's goal space, or None if `indices` is empty.
        """
        if not indices:
            return None
        promises = [envs.envs[i].goal_observation() for i in indices]
        obs_list = [p() for p in promises]
        obs = {k: torch.as_tensor(np.stack([o[k] for o in obs_list]), device=agent.device) for k in obs_list[0]}
        goal = agent.encode_observation(obs)  # (N, S, K)
        goal_shape = envs.observation_space["goal"].shape
        if len(goal_shape) == 1:
            return goal[:, 0, :]
        return goal

    def _explore_mask(self, episode_ids, lengths=None):
        """Per-env mask selecting the exploration policy for the current step.

        LEXA hands every `explore_every_ep`-th episode to the explorer. Under
        goal_sample="peg" the explorer also takes over every other episode once
        it has run `peg.go_ratio` of its steps (`lengths` = actions already taken
        this episode). None outside LEXA, so every other mode keeps its single
        policy.
        """
        if not self._lexa:
            return None
        mask = None
        if self._explore_every_ep > 0:
            mask = (episode_ids % self._explore_every_ep) == 0
        if self._peg_go_steps is not None and lengths is not None:
            explore_phase = lengths >= self._peg_go_steps
            mask = explore_phase if mask is None else mask | explore_phase
        return mask

    def _refresh_mask(self, is_first, steps, last):
        """Envs whose goal is refreshed mid-episode at this step: (B,) bool.

        Refreshed sources (`imagination_refresh`) give every goal
        `goal_refresh_every` (K) agent steps: a refresh takes effect one step
        after it is imagined (see `_update_goals`), so it is imagined at
        steps K-1, 2K-1, ..., `steps` being the index of the current
        observation within its episode (0 at is_first). The is_first sample is
        not a refresh, and neither is the episode's last step (`last`): a goal
        picked there could never be pursued. The episode's last goal can
        still be cut short by the episode ending (the trainer does not know
        the time limit).
        """
        if self._goal_refresh_every is None:
            return torch.zeros_like(is_first)
        return (((steps + 1) % self._goal_refresh_every) == 0) & ~is_first & ~last

    def _update_goals(self, agent, envs, goals, trans, steps, last, state, log_prefix=None):
        """Pick new goals for this step and relabel trans["goal"].

        Goals sampled at is_first take effect immediately. A mid-episode refresh
        is imagined now, from the current observation, but only takes effect
        on the next step: this step is still scored, acted on and stored
        against the goal the agent was chasing to get here, so reaching it at
        the refresh step is rewarded. The returned pending refresh goes to
        `_commit_goals` once the step's reward has been applied.

        Returns:
            (mask, new_goals) of the pending refresh, or None.
        """
        is_first = trans["is_first"][:, 0].bool()
        refresh = self._refresh_mask(is_first, steps, last)
        pending = None
        if refresh.any():
            # Sample is_first and refresh envs in one call, into a copy so the
            # refreshed ones keep their current goal for this step.
            new_goals = goals.clone()
            self._sample_goals(agent, envs, is_first | refresh, new_goals, trans, log_prefix=log_prefix, state=state)
            goals[is_first] = new_goals[is_first]
            pending = (refresh, new_goals)
        elif is_first.any():
            self._sample_goals(agent, envs, is_first, goals, trans, log_prefix=log_prefix, state=state)
        self._relabel_goal(envs, goals, trans)
        return pending

    @staticmethod
    def _commit_goals(goals, pending):
        """Activate a refresh returned by `_update_goals` (from the next step on)."""
        if pending is not None:
            mask, new_goals = pending
            goals[mask] = new_goals[mask]

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

        if self._goal_sample in SAMPLED_GOAL_SOURCES:
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
            pending_goals = None
            if self._goal_sample in SAMPLED_GOAL_SOURCES:
                # Envs past their first episode keep stepping until all are done;
                # only refresh the ones still being scored.
                pending_goals = self._update_goals(
                    agent,
                    envs,
                    goals,
                    trans,
                    steps,
                    done | once_done,
                    agent_state,
                    log_prefix="eval_" + self._goal_log_prefix,
                )

            # Store transition.
            # We keep the observation and the action that produced it together.

            trans["action"] = act
            if len(cache) < self.batch_length:
                cache.append(trans.clone())
            # (B, A)
            act, agent_state, _ = agent.act(trans, agent_state, eval=True, random=self._random_actions)

            trans["stoch"] = agent_state["stoch"]
            if "logit" in agent_state:
                trans["logit"] = agent_state["logit"]
            self._apply_reward(trans, agent.rssm)
            if pending_goals is not None:
                self._commit_goals(goals, pending_goals)
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
        envs_ids = torch.arange(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_ids = envs_ids.clone()  # used for HER to identify episodes in the buffer

        if self._goal_sample in SAMPLED_GOAL_SOURCES:
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
            pending_goals = None
            if self._goal_sample in SAMPLED_GOAL_SOURCES:
                # lengths is the index of this observation within its episode;
                # done marks it as the episode's last.
                pending_goals = self._update_goals(agent, envs, goals, trans, lengths, done, agent_state)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # In wm_only / train_text_only mode the actor is bypassed and uniform one-hot actions are used.
            # (B, A)
            act, agent_state, act_metrics = agent.act(
                trans.clone(),
                agent_state,
                eval=False,
                random=self._random_actions,
                explore=self._explore_mask(episode_ids, lengths),
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
            if "logit" in agent_state:
                trans["logit"] = agent_state["logit"]

            self._apply_reward(trans, agent.rssm)
            if self._goal_refresh_every is not None:
                # Marks the step after which the goal changes, so the replay
                # value loss does not run its return across the switch
                # (ActorCritic.replay_value_loss). Stored under refreshed
                # sources only, so other runs' buffers are unchanged.
                switch = torch.zeros_like(trans["is_first"], dtype=torch.bool)
                if pending_goals is not None:
                    switch[pending_goals[0]] = True
                trans["goal_switch"] = switch
            if pending_goals is not None:
                self._commit_goals(goals, pending_goals)

            if "image" in trans:
                video_cache.append(trans["image"][0])

            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]

            episode_ids[done] += envs.env_num

            # Update models after enough data has accumulated
            if self._should_update(step):
                update_num = self.pretrain if self._should_pretrain() else self._updates_needed(step)
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

    def _apply_reward(self, trans, rssm=None):
        if self.reward_function:
            state = reward_state(self._goal_spec, stoch=trans["stoch"], logit=trans.get("logit"), rssm=rssm)
            trans["reward"] = self.reward_function(state, trans["goal"])

    def _relabel_goal(self, envs, goals, trans):
        # Si es que hay algun valor diferente de 0 en el goals, entonces relabel.
        # Esto siempre será True para mode="text" y para mode="buffer" será True si es que
        # ya hay experiencias guardadas.
        mask = (goals != 0).view(envs.env_num, -1).any(dim=1)
        trans["goal"][mask] = goals[mask].clone()

    def _sample_goals(self, agent, envs, mask, goals, trans, log_prefix=None, state=None):
        """Populate goals[i] for envs where mask[i] is True (typically is_first).

        `state` is the agent state before observing `trans` (the one `agent.act`
        is about to consume); only the refreshed sources read it, to imagine
        from where the agent is mid-episode. `log_prefix` namespaces the score
        statistics of "peg" / "explore_imagination*" (None: the source's
        training prefix, "peg" or "explore_goal").

        Source is selected by self._goal_sample:
            - "buffer":      sample a past stoch uniformly from the replay buffer.
                             Skipped silently when the buffer is empty.
            - "text":        sample a fresh goal from the live text encoder applied
                             to a random mission produced by each env.
            - "imagination": imagine `goal_imag_horizon` steps with random actions in
                             the world model from the episode's first observation
                             (in `trans`) and take the z reached at the last step,
                             so the goal is reachable from the current episode
                             by construction.
            - "imagination_refresh": as "imagination", but also re-run every
                             `goal_refresh_every` agent steps, imagining from
                             the agent's current posterior (`state` + `trans`)
                             rather than the episode's first observation.
            - "explore_imagination": roll the explorer out `explore_goal.num_rollouts`
                             times for `goal_imag_horizon` steps from the
                             episode's first observation and keep the final z
                             of the rollout the explorer critic values most
                             (`Dreamer.explore_goal`). Scores are logged under
                             `log_prefix`.
            - "explore_imagination_refresh": as "explore_imagination", also re-run
                             every `goal_refresh_every` agent steps from the
                             agent's current posterior, like "imagination_refresh".
            - "image":       render a synthetic goal observation in each env
                             (green square at the episode's goal position,
                             agent at a random cell) and encode it into z with
                             the frozen world model. See `_image_goal`.
            - "peg":         draw random z candidates, imagine the achiever
                             pursuing each from the episode's first observation,
                             and keep the one whose end state the explorer values
                             most (`Dreamer.peg_goal`). With peg.search="buffer"
                             the candidates are visited stoch from the replay
                             buffer (`_buffer_candidates`) instead. Score
                             statistics are logged under `log_prefix`.
        """
        if log_prefix is None:
            log_prefix = self._goal_log_prefix
        if self._goal_sample == "buffer":
            if self.replay_buffer.count() == 0:
                return
            goal_sample = self._buffer_goal_pool()
            for i in range(envs.env_num):
                if mask[i]:
                    goals[i] = goal_sample[torch.randint(goal_sample.shape[0], (1,))]
        elif self._goal_sample == "text":
            indices = mask.nonzero(as_tuple=True)[0].tolist()
            new_goals = self._text_goal(agent, envs, indices)
            if new_goals is not None:
                goals[indices] = new_goals
        elif self._goal_sample in ("imagination", "imagination_refresh"):
            indices = mask.nonzero(as_tuple=True)[0]
            # trans[indices] copies, so imagine_goal's in-place preprocess
            # does not touch the trans consumed later by agent.act.
            if self._goal_sample == "imagination_refresh":
                # Mid-episode the start posterior depends on the history, so
                # filter the obs onto the running state (obs_step still resets
                # the is_first envs).
                new_goals = agent.imagine_goal(trans[indices], state=state[indices])  # (N, S, K)
            else:
                new_goals = agent.imagine_goal(trans[indices])  # (N, S, K)
            goal_shape = envs.observation_space["goal"].shape
            if len(goal_shape) == 1:
                new_goals = new_goals[:, 0, :]
            goals[indices] = new_goals
        elif self._goal_sample in EXPLORER_GOAL_SOURCES:
            indices = mask.nonzero(as_tuple=True)[0]
            # trans[indices] copies, as for "imagination". The refreshed source
            # imagines from the running state mid-episode, like "imagination_refresh".
            if self._goal_sample == "explore_imagination_refresh":
                new_goals, metrics = agent.explore_goal(trans[indices], state=state[indices])
            else:
                new_goals, metrics = agent.explore_goal(trans[indices])
            if self._goal_spec.scope == "first_row":
                new_goals = new_goals[:, 0]
            goals[indices] = new_goals
            for name, value in metrics.items():
                self.logger.scalar(f"{log_prefix}/{name}", value)
        elif self._goal_sample == "peg":
            indices = mask.nonzero(as_tuple=True)[0]
            # trans[indices] copies, as for "imagination".
            candidates = None
            if self._peg_search == "buffer":
                candidates = self._buffer_candidates(len(indices))
            new_goals, metrics = agent.peg_goal(trans[indices], candidates=candidates)
            goals[indices] = new_goals
            for name, value in metrics.items():
                self.logger.scalar(f"{log_prefix}/{name}", value)
        elif self._goal_sample == "image":
            indices = mask.nonzero(as_tuple=True)[0].tolist()
            new_goals = self._image_goal(agent, envs, indices)
            if new_goals is not None:
                goals[indices] = new_goals

    def _buffer_goal_pool(self):
        """Goals from one replay batch: every (B, T) state as a (B*T, *goal_shape) goal."""
        data, _, _ = self.replay_buffer.sample()
        goal = goal_from_latent(self._goal_spec, stoch=data.get("stoch"), logit=data.get("logit"))
        if self._goal_spec.scope == "first_row":
            goal = goal[..., 0, :]
        return goal.reshape(-1, *goal.shape[2:])

    def _buffer_candidates(self, n_envs):
        """(n_envs, peg.num_candidates, *goal_shape) PEG candidates from the replay buffer.

        Replay batches are drawn until they hold enough states, then the
        candidates are taken without replacement, so no env sees a duplicate
        unless the buffer itself repeats a state. None while the buffer is
        empty, which makes the agent fall back to random candidates.
        """
        if self.replay_buffer.count() == 0:
            return None
        needed = n_envs * self._peg_candidates
        pools, total = [], 0
        while total < needed:
            pools.append(self._buffer_goal_pool())
            total += pools[-1].shape[0]
        pool = torch.cat(pools)
        pick = torch.randperm(pool.shape[0], device=pool.device)[:needed]
        return pool[pick].reshape(n_envs, self._peg_candidates, *pool.shape[1:])

    def _should_update(self, step):
        envs_num = self.train_envs.env_num
        return step // (envs_num * self._action_repeat) > self.batch_length + 1
