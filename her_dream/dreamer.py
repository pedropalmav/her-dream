import copy
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import her_dream.distributions as dists
import her_dream.goals as goals
import her_dream.networks as networks
import her_dream.rssm as rssm
import her_dream.tools as tools
from her_dream.networks import Projector
from her_dream.optim import LaProp, clip_grad_agc_
from her_dream.tools import to_f32


class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space, reward_function):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.goal_imag_horizon = int(config.goal_imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        self.rep_loss = str(config.rep_loss)
        # Vanilla-Dreamer world-model heads (off by default). The reward head is a
        # plain function of the RSSM feature, so it predicts `data["reward"]`
        # marginalized over goals; `imag_reward_source="head"` is therefore meant
        # for non-goal-conditioned baselines, and "goal" stays the default.
        self.use_reward_head = bool(getattr(config, "use_reward_head", False))
        self.use_cont_head = bool(getattr(config, "use_cont_head", False))
        self.imag_reward_source = str(getattr(config, "imag_reward_source", "goal"))
        if self.imag_reward_source not in ("goal", "head"):
            raise ValueError(f"Unknown imag_reward_source {self.imag_reward_source!r}; expected 'goal' or 'head'.")
        if self.imag_reward_source == "head" and not self.use_reward_head:
            raise ValueError("imag_reward_source='head' requires use_reward_head=True.")
        self.goal_type = str(config.goal_type)
        self._goal_spec = goals.make_goal_spec(config)
        self.mission_text = config.mission_text
        self.wm_only = bool(config.wm_only)
        self.freeze_wm = bool(getattr(config, "freeze_wm", False))
        if self.wm_only and self.freeze_wm:
            raise ValueError("wm_only=True and freeze_wm=True are mutually exclusive.")
        # Text-encoder distillation mode: world model and actor-critic are kept
        # fixed, and only the text encoder is trained to predict the (frozen)
        # posterior from the mission text. See train.py (train_text_only=True).
        self.train_text_only = bool(getattr(config, "train_text_only", False))
        if self.train_text_only and (self.wm_only or self.freeze_wm):
            raise ValueError("train_text_only is exclusive with wm_only and freeze_wm.")
        if self.train_text_only and not self.mission_text:
            raise ValueError("train_text_only=True requires mission_text=True.")

        # Cache action-space info for uniform random sampling (wm_only mode).
        # The env wrappers expose Box action spaces with custom `discrete` /
        # `multi_discrete` attributes (set to True) instead of a Discrete `.n`.
        self._act_multi = hasattr(act_space, "multi_discrete")
        self._act_disc = hasattr(act_space, "discrete") and not self._act_multi
        self._act_nvec = tuple(int(d) for d in act_space.shape) if self._act_multi else None
        self._act_n = int(act_space.shape[0]) if self._act_disc else None

        # World model components
        excluded = ("is_first", "is_last", "is_terminal", "reward", "mission")
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items() if k not in excluded}
        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        self.rssm = rssm.RSSM(
            config.rssm,
            self.embed_size,
            self.act_dim,
        )

        self.reward_function = reward_function
        # World-model heads: predict the replayed reward / continuation from the
        # RSSM feature alone, exactly as vanilla Dreamer does.
        if self.use_reward_head:
            self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        if self.use_cont_head:
            self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
        else:
            config.actor.dist = config.actor.dist.cont

        # Actor-critic components
        goal_shape = goals.goal_size(self._goal_spec, self.rssm)
        threshold_bins = int(round(1.0 / config.prob_threshold_step)) + 1 if self._goal_spec.uses_threshold else 0
        self.threshold_bins = threshold_bins
        self.actor = networks.MLPHead(config.actor, self.rssm.feat_size + goal_shape + threshold_bins)
        self.value = networks.MLPHead(config.critic, self.rssm.feat_size + goal_shape + threshold_bins)
        if threshold_bins > 0:
            idx = int(round(config.prob_threshold / config.prob_threshold_step))
            t_oh = torch.zeros(threshold_bins)
            t_oh[idx] = 1.0
            self.register_buffer("threshold_onehot", t_oh)  # shape (threshold_bins,)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)

        modules = {}
        if not self.freeze_wm and not self.train_text_only:
            modules["rssm"] = self.rssm
            modules["encoder"] = self.encoder
            # World-model heads, so they train under wm_only=True alongside the RSSM.
            if self.use_reward_head:
                modules["reward"] = self.reward
            if self.use_cont_head:
                modules["cont"] = self.cont
        if not self.wm_only and not self.train_text_only:
            modules["actor"] = self.actor
            modules["value"] = self.value

        # TODO: Create a method for this block
        if self.rep_loss == "dreamer":
            self.decoder = networks.MultiDecoder(
                config.decoder,
                self.rssm._deter,
                self.rssm.flat_stoch,
                shapes,
            )
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})
            if not self.freeze_wm and not self.train_text_only:
                modules.update({"decoder": self.decoder})
        elif self.rep_loss == "r2dreamer" or self.rep_loss == "infonce":
            # add projector for latent to embedding
            self.prj = Projector(self.rssm.feat_size, self.embed_size)
            if not self.freeze_wm and not self.train_text_only:
                modules.update({"projector": self.prj})
            self.barlow_lambd = float(config.r2dreamer.lambd)
        elif self.rep_loss == "dreamerpro":
            dpc = config.dreamer_pro
            self.warm_up = int(dpc.warm_up)
            self.num_prototypes = int(dpc.num_prototypes)
            self.proto_dim = int(dpc.proto_dim)
            self.temperature = float(dpc.temperature)
            self.sinkhorn_eps = float(dpc.sinkhorn_eps)
            self.sinkhorn_iters = int(dpc.sinkhorn_iters)
            self.ema_update_every = int(dpc.ema_update_every)
            self.ema_update_fraction = float(dpc.ema_update_fraction)
            self.freeze_prototypes_iters = int(dpc.freeze_prototypes_iters)
            self.aug_max_delta = float(dpc.aug.max_delta)
            self.aug_same_across_time = bool(dpc.aug.same_across_time)
            self.aug_bilinear = bool(dpc.aug.bilinear)

            self._prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim))
            self.obs_proj = nn.Linear(self.embed_size, self.proto_dim)
            self.feat_proj = nn.Linear(self.rssm.feat_size, self.proto_dim)
            self._ema_encoder = copy.deepcopy(self.encoder)
            self._ema_obs_proj = copy.deepcopy(self.obs_proj)
            for param in self._ema_encoder.parameters():
                param.requires_grad = False
            for param in self._ema_obs_proj.parameters():
                param.requires_grad = False
            self._ema_updates = 0
            if not self.freeze_wm and not self.train_text_only:
                modules.update({
                    "prototypes": self._prototypes,
                    "obs_proj": self.obs_proj,
                    "feat_proj": self.feat_proj,
                    "ema_encoder": self._ema_encoder,
                    "ema_obs_proj": self._ema_obs_proj,
                })

        # === Text encoder (auxiliar, solo si hay mission en el obs) ===
        if self.mission_text:
            self.text_encoder = networks.TextEncoderGRU(
                config=config.text_encoder,
                stoch=self.rssm._stoch,
                discrete=self.rssm._discrete,
                act=config.rssm.act,
            )
            if self.train_text_only or (not self.wm_only and not self.freeze_wm):
                modules["text_encoder"] = self.text_encoder

        # count number of parameters in each module
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")
        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    self._named_params[f"{name}.{param_name}"] = param
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self._scaler = GradScaler()

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        self.train()
        self.clone_and_freeze()
        if self.freeze_wm:
            self._apply_freeze_wm()
        if self.train_text_only:
            self._apply_train_text_only()
        if config.compile:
            print("Compiling update function with torch.compile...")
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

    def _update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode=True):
        super().train(mode)
        # slow_value should be always eval mode
        self._slow_value.train(False)
        return self

    @staticmethod
    def _freeze_clone(module):
        """Deep-copy `module` into a no-grad clone sharing its parameter storage.

        NOTE: "requires_grad" affects whether a parameter is updated, not whether
        gradients flow through its operations. Sharing `.data` keeps the clone in
        lockstep with the live module without re-copying every step.
        """
        clone = copy.deepcopy(module)
        for (name_orig, param_orig), (name_new, param_new) in zip(module.named_parameters(), clone.named_parameters()):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)
        return clone

    def clone_and_freeze(self):
        self._frozen_encoder = self._freeze_clone(self.encoder)
        self._frozen_rssm = self._freeze_clone(self.rssm)
        self._frozen_actor = self._freeze_clone(self.actor)
        self._frozen_value = self._freeze_clone(self.value)
        self._frozen_slow_value = self._freeze_clone(self._slow_value)
        if self.use_reward_head:
            self._frozen_reward = self._freeze_clone(self.reward)
        if self.use_cont_head:
            self._frozen_cont = self._freeze_clone(self.cont)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        if self.freeze_wm:
            self._apply_freeze_wm()
        if self.train_text_only:
            self._apply_train_text_only()
        return self

    def _apply_freeze_wm(self):
        """Disable gradients on all world-model parameters.

        Targets the live encoder/RSSM, the reward/continue heads and any auxiliary
        modules tied to the representation loss (decoder, projector, dreamerpro
        buffers) plus the text encoder. The frozen `_frozen_*` clones already have
        requires_grad disabled by `clone_and_freeze`.
        """
        for module in (self.encoder, self.rssm):
            for param in module.parameters():
                param.requires_grad_(False)
        if self.use_reward_head:
            for param in self.reward.parameters():
                param.requires_grad_(False)
        if self.use_cont_head:
            for param in self.cont.parameters():
                param.requires_grad_(False)
        if self.rep_loss == "dreamer" and hasattr(self, "decoder"):
            for param in self.decoder.parameters():
                param.requires_grad_(False)
        if self.rep_loss in ("r2dreamer", "infonce") and hasattr(self, "prj"):
            for param in self.prj.parameters():
                param.requires_grad_(False)
        if self.rep_loss == "dreamerpro":
            self._prototypes.requires_grad_(False)
            for module in (self.obs_proj, self.feat_proj, self._ema_encoder, self._ema_obs_proj):
                for param in module.parameters():
                    param.requires_grad_(False)
        if self.mission_text and hasattr(self, "text_encoder"):
            for param in self.text_encoder.parameters():
                param.requires_grad_(False)

    def _apply_train_text_only(self):
        """Freeze everything except the text encoder.

        Distillation mode: the world model and actor-critic stay fixed and only
        the `TextEncoderGRU` is optimized. Mirrors `_apply_freeze_wm` but also
        freezes the actor/value and explicitly keeps the text encoder trainable.
        """
        self._apply_freeze_wm()  # freezes encoder/rssm/rep modules + text encoder
        for module in (self.actor, self.value):
            for param in module.parameters():
                param.requires_grad_(False)
        # Re-enable the text encoder, which _apply_freeze_wm just disabled.
        for param in self.text_encoder.parameters():
            param.requires_grad_(True)

    @torch.no_grad()
    def act(self, obs, state, eval=False, random=False):
        """Policy inference step.

        When `random=True`, the actor is bypassed and a uniform one-hot action
        is sampled. The WM forward pass (encoder + obs_step) still runs so the
        agent_state (stoch/deter) is updated consistently for the buffer.
        """
        # obs: dict of (B, *), state: (stoch: (B, S, K), deter: (B, D), prev_action: (B, A))
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        # (B, E)
        embed = self._frozen_encoder(p_obs)
        prev_stoch, prev_deter, prev_action = (
            state["stoch"],
            state["deter"],
            state["prev_action"],
        )
        # (B, S, K), (B, D)
        stoch, deter, logit = self._frozen_rssm.obs_step(prev_stoch, prev_deter, prev_action, embed, obs["is_first"])
        obs_step_sample_log_prob = self._frozen_rssm.get_dist(logit).log_prob(stoch)[0]
        # (B, F)
        feat = self._frozen_rssm.get_feat(stoch, deter)

        B = stoch.shape[0]
        if random:
            action = self._random_action(B)
        else:
            # (B, F)
            goal = obs["goal"].reshape(feat.shape[0], -1)
            policy_input = torch.cat([feat, goal], dim=-1)
            if self.threshold_bins > 0:
                t = self.threshold_onehot.expand(feat.shape[0], -1)
                policy_input = torch.cat([policy_input, t], dim=-1)
            action_dist = self._frozen_actor(policy_input)
            # (B, A)
            action = action_dist.mode if eval else action_dist.rsample()
        state_dict = {"stoch": stoch, "deter": deter, "prev_action": action}
        if goals.stashes_logit(self._goal_spec):
            state_dict["logit"] = logit
        return (
            action,
            TensorDict(
                state_dict,
                batch_size=state.batch_size,
            ),
            {"obs_step_sample_log_prob": obs_step_sample_log_prob},
        )

    @torch.no_grad()
    def imagine_goal(self, obs):
        """Sample goals by imagining `goal_imag_horizon` random-action steps from the given observations.

        Used by goal_sample="imagination": at episode start, the world model is
        rolled out from the first observation and the z sampled after
        `goal_imag_horizon` steps becomes the episode goal, so the goal is reachable
        from the current episode by construction. The rollout uses uniformly
        random actions, so the goal distribution does not depend on the actor.

        Args:
            obs: dict of (N, 1, *) first transitions (is_first=True) of the
                envs that need a fresh goal. Must be a copy: preprocess
                modifies it in place.

        Returns:
            (N, S, K) float32 one-hot goal on self.device.
        """
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        # (N, E)
        embed = self._frozen_encoder(p_obs)
        N = obs["is_first"].shape[0]
        # is_first=True makes obs_step zero out the initial state and action.
        stoch, deter = self._frozen_rssm.initial(N)
        prev_action = torch.zeros(N, self.act_dim, dtype=torch.float32, device=self.device)
        # (N, S, K), (N, D)
        stoch, deter, logit = self._frozen_rssm.obs_step(stoch, deter, prev_action, embed, obs["is_first"])

        for _ in range(self.goal_imag_horizon):
            # (N, A)
            action = self._random_action(N)
            stoch, deter, logit = self._frozen_rssm.img_step(stoch, deter, action)

        # goal_repr selects sample (stoch) / mode (argmax one-hot) / raw logit, so the
        # goal matches whatever the reward function compares against.
        return goals.goal_from_latent(self._goal_spec, stoch=stoch, logit=logit)

    @torch.no_grad()
    def encode_observation(self, obs):
        """Encode an observation into a z with the frozen world model.

        Used by goal_sample="image": the env renders the observation of the
        desired state (see envs.goal_image.GoalImageGenerator) and a single
        posterior step from the initial latent state maps it to z — the same
        first step as `imagine_goal`, without the policy rollout.

        Args:
            obs: dict with "image" (N, H, W, C) uint8 plus any mlp keys the
                encoder uses (e.g. "direction" (N, 4) one-hot).

        Returns:
            (N, S, K) float32 one-hot goal on self.device.
        """
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        # (N, E)
        embed = self._frozen_encoder(p_obs)
        N = embed.shape[0]
        stoch, deter = self._frozen_rssm.initial(N)
        prev_action = torch.zeros(N, self.act_dim, dtype=torch.float32, device=self.device)
        is_first = torch.ones(N, dtype=torch.bool, device=self.device)
        # (N, S, K), (N, D), (N, S, K)
        stoch, _, logit = self._frozen_rssm.obs_step(stoch, deter, prev_action, embed, is_first)
        # goal_repr selects sample (stoch) / mode (argmax one-hot) / raw logit.
        return goals.goal_from_latent(self._goal_spec, stoch=stoch, logit=logit)

    @torch.no_grad()
    def _random_action(self, B):
        """Sample uniform actions in the format the env expects.

        Discrete: one-hot of size n.
        Multi-discrete: per-group one-hot, concatenated.
        Continuous: uniform in [-1, 1] (env post-scales via ScaleBox).
        """
        if self._act_n is not None:
            idx = torch.randint(0, self._act_n, (B,), device=self.device)
            return F.one_hot(idx, self._act_n).to(torch.float32)
        if self._act_multi:
            parts = []
            for dim in self._act_nvec:
                idx = torch.randint(0, dim, (B,), device=self.device)
                parts.append(F.one_hot(idx, dim).to(torch.float32))
            return torch.cat(parts, dim=-1)
        return torch.empty(B, self.act_dim, device=self.device).uniform_(-1.0, 1.0)

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        return TensorDict({"stoch": stoch, "deter": deter, "prev_action": action}, batch_size=(B,))

    @torch.no_grad()
    def video_pred(self, data, initial):
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        return self._video_pred(p_data, initial)

    def _video_pred(self, data, initial):
        """Video prediction utility."""
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")

        B = min(data["action"].shape[0], 6)
        # (B, T, E)
        embed = self.encoder(data)

        post_stoch, post_deter, _ = self.rssm.observe(
            embed[:B, :5],
            data["action"][:B, :5],
            tuple(val[:B] for val in initial),
            data["is_first"][:B, :5],
        )
        recon = self.decoder(post_stoch, post_deter)["image"].mode[:B]
        init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
        prior_stoch, prior_deter = self.rssm.imagine_with_action(
            init_stoch,
            init_deter,
            data["action"][:B, 5:],
        )
        openl = self.decoder(prior_stoch, prior_deter)["image"].mode
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:B]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    def update(self, replay_buffer):
        """Sample a batch from replay and perform one optimization step."""
        data, index, initial = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        self._update_slow_target()
        if self.rep_loss == "dreamerpro":
            self.ema_update()
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16):
            (stoch, deter), mets = self._cal_grad(p_data, initial)
        self._scaler.unscale_(self._optimizer)  # unscale grads in params
        if (
            self.rep_loss == "dreamerpro"
            and self._ema_updates < self.freeze_prototypes_iters
            and self._prototypes.grad is not None
        ):
            self._prototypes.grad.zero_()
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]  # log grads before clipping
            grad_norm = tools.compute_global_norm(grads)
            grad_rms = tools.compute_rms(grads)
            mets["opt/grad_norm"] = grad_norm
            mets["opt/grad_rms"] = grad_rms
        self._agc(self._named_params.values())  # clipping
        self._scaler.step(self._optimizer)  # update params
        self._scaler.update()  # adjust scale
        self._scheduler.step()  # increment scheduler
        self._optimizer.zero_grad(set_to_none=True)  # reset grads
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/lr_last"] = self._scheduler.get_last_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms
        metrics.update(mets)
        # update latent vectors in replay buffer
        replay_buffer.update(index, stoch.detach(), deter.detach())
        return metrics

    def _cal_grad(self, data, initial):
        """Compute gradients for one batch.

        Notes
        -----
        This function computes:
        1) World model loss (dynamics + representation)
        2) Optional representation loss variants (Dreamer, R2-Dreamer, InfoNCE, DreamerPro)
        3) Imagination rollouts for actor-critic updates
        4) Replay-based value learning
        """
        # data: dict of (B, T, *), initial: (stoch: (B, S, K), deter: (B, D))
        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout and KL losses ===
        if self.freeze_wm or self.train_text_only:
            # No grads anywhere through the WM. All representation / KL losses
            # are skipped — only the actor-critic branch below runs (freeze_wm),
            # or only the text encoder is trained (train_text_only).
            with torch.no_grad():
                embed = self.encoder(data)
                post_stoch, post_deter, post_logit = self.rssm.observe(embed, data["action"], initial, data["is_first"])
            if self.train_text_only:
                # Distill the text encoder against the frozen WM posterior. The
                # posterior is a fixed (no-grad) target; gradients flow only into
                # the text encoder. Imagination and actor-critic are skipped.
                text_logit = self.text_encoder(data["mission"])  # (B, T, S, K)
                text_kl = dists.kl(post_logit, text_logit).sum(-1)  # (B, T, S)
                text_kl = torch.clip(text_kl, min=self.kl_free)
                losses["text_kl"] = torch.mean(text_kl)
                metrics["text_kl"] = losses["text_kl"].detach()
                total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
                self._scaler.scale(total_loss).backward()
                metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
                metrics.update({"opt/loss": total_loss})
                return (post_stoch, post_deter), metrics
        else:
            # (B, T, E)
            embed = self.encoder(data)
            # (B, T, S, K), (B, T, D), (B, T, S, K)
            post_stoch, post_deter, post_logit = self.rssm.observe(embed, data["action"], initial, data["is_first"])
            # (B, T, S, K)
            _, prior_logit = self.rssm.prior(post_deter)
            dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
            losses["dyn"] = torch.mean(dyn_loss)
            losses["rep"] = torch.mean(rep_loss)

            # === Text KL loss (auxiliar) ===
            # El text encoder aprende a predecir el z del agente desde el texto.
            # detach en post_logit: la loss no modifica el world model, solo entrena text_encoder.
            if self.mission_text and not self.wm_only:
                # data["mission"]: (B, T, L) int8 token ids — TextEncoderGRU
                # promotes to long and materialises the one-hot internally.
                text_logit = self.text_encoder(data["mission"])  # (B, T, S, K)
                text_kl = dists.kl(post_logit.detach(), text_logit).sum(-1)  # (B, T, S)
                text_kl = torch.clip(text_kl, min=self.kl_free)
                losses["text_kl"] = torch.mean(text_kl)
                metrics["text_kl"] = losses["text_kl"].detach()

            # === Representation / auxiliary losses ===
            # (B, T, F)
            feat = self.rssm.get_feat(post_stoch, post_deter)

            # === Reward / continue heads (vanilla Dreamer, optional) ===
            # Placed inside this branch so freeze_wm / train_text_only (which take
            # the no-grad path above) skip them, while wm_only still trains them.
            if self.use_reward_head:
                losses["rew"] = torch.mean(-self.reward(feat).log_prob(to_f32(data["reward"])))
            if self.use_cont_head:
                cont = 1.0 - to_f32(data["is_terminal"])
                losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))

            # log
            metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
            metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())

        # TODO: Create a method for this block
        if self.freeze_wm:
            pass
        elif self.rep_loss == "dreamer":
            recon_losses = {
                key: torch.mean(-dist.log_prob(data[key])) for key, dist in self.decoder(post_stoch, post_deter).items()
            }
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            # R2-Dreamer: Barlow Twins style redundancy reduction between latent features and encoder embeddings.
            # Flatten batch/time dims for a single cross-correlation matrix.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important

            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
        elif self.rep_loss == "infonce":
            # Contrastive (InfoNCE) objective between projected latent features and encoder embeddings.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important
            logits = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(norm_logits.shape[0]).long().to(self.device)
            losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
        elif self.rep_loss == "dreamerpro":
            # DreamerPro uses augmentation + EMA targets + Sinkhorn assignment.
            with torch.no_grad():
                data_aug = self.augment_data(data)
                initial_aug = (
                    # (B, ...) -> (2B, ...)
                    torch.cat([initial[0], initial[0]], dim=0),
                    torch.cat([initial[1], initial[1]], dim=0),
                )
                ema_proj = self.ema_proj(data_aug)

            embed_aug = self.encoder(data_aug)
            post_stoch_aug, post_deter_aug, _ = self.rssm.observe(
                embed_aug, data_aug["action"], initial_aug, data_aug["is_first"]
            )
            proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj)
            losses.update(proto_losses)
        else:
            raise NotImplementedError

        # === Imagination rollout for actor-critic ===
        # Skipped entirely in wm_only mode: actor and value are frozen, so
        # policy/value/repval losses would only waste compute.
        if self.wm_only:
            total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
            self._scaler.scale(total_loss).backward()
            metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
            metrics.update({"opt/loss": total_loss})
            return (post_stoch, post_deter), metrics

        # (B*T, S, K), (B*T, D)
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        # (B, T, ...) -> (B*T, ...)
        goal = data["goal"].reshape(-1, *data["goal"].shape[2:])
        imag_feat, imag_action, imag_logit = self._imagine(start, self.imag_horizon + 1, goal)
        imag_feat, imag_action, imag_logit = imag_feat.detach(), imag_action.detach(), imag_logit.detach()

        # (B*T, T_imag, 1)
        S, K = self.rssm._stoch, self.rssm._discrete
        get_stoch_from_feat = lambda x: x[..., : S * K].reshape(*x.shape[:-1], S, K)  # noqa: E731
        imag_stoch = get_stoch_from_feat(imag_feat)
        imag_reward = self._imagination_reward(imag_feat, imag_stoch, imag_logit, goal)

        # (B*T, T_imag, 1)  probability of continuation. Without the head every
        # imagined step is assumed to continue, which is what this agent did
        # before the head was reintroduced.
        if self.use_cont_head:
            imag_cont = self._frozen_cont(imag_feat).mean
        else:
            imag_cont = torch.ones(*imag_feat.shape[:2], 1, dtype=torch.float32, device=imag_feat.device)

        imag_reward = to_f32(imag_reward)
        imag_cont = to_f32(imag_cont)

        # (B*T, T_imag, ...)
        imag_goal = goal.unsqueeze(1).expand(
            -1,
            self.imag_horizon + 1,
            *([-1] * (goal.ndim - 1)),
        )
        imag_goal = imag_goal.reshape(imag_goal.shape[0], imag_goal.shape[1], -1)
        imag_input = torch.cat([imag_feat, imag_goal], dim=-1)
        if self.threshold_bins > 0:
            B2, Ti = imag_input.shape[:2]
            t = self.threshold_onehot.expand(B2, Ti, -1)
            imag_input = torch.cat([imag_input, t], dim=-1)
        imag_value = self._frozen_value(imag_input).mode
        imag_slow_value = self._frozen_slow_value(imag_input).mode
        disc = 1 - 1 / self.horizon

        # (B*T, T_imag, 1)
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(
            last, term, imag_reward, imag_value, imag_value, disc, self.lamb
        )  # (B*T, T_imag-1, 1)
        ret_offset, ret_scale = self.return_ema(ret)
        # (B*T, T_imag-1, 1)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_input)
        # (B*T, T_imag-1, 1)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        imag_value_dist = self.value(imag_input)
        # (B*T, T_imag, 1)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )
        # log
        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))

        # === Replay-based value learning (keep gradients through world model) ===
        last, term, reward = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        feat = self.rssm.get_feat(post_stoch, post_deter)  # (B, T, F)
        boot = ret[:, 0].reshape(B, T, 1)
        goal = data["goal"].reshape(feat.shape[0], feat.shape[1], -1)
        value_input = torch.cat([feat, goal], dim=-1)
        if self.threshold_bins > 0:
            B2, T2 = value_input.shape[:2]
            t = self.threshold_onehot.expand(B2, T2, -1)
            value_input = torch.cat([value_input, t], dim=-1)
        value = self._frozen_value(value_input).mode
        slow_value = self._frozen_slow_value(value_input).mode
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        # Keep this attached to the world model so gradients can flow through
        value_dist = self.value(value_input)
        losses["repval"] = torch.mean(
            weight[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(
                -1
            )
        )
        # log
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), metrics

    def _imagination_reward(self, imag_feat, imag_stoch, imag_logit, goal):
        """Reward for the imagination rollout, selected by `imag_reward_source`.

        This is the seam the exploration objectives plug into: "goal" is the
        analytic goal reward from `rewards.make_reward`, "head" is the learned
        reward predictor. Returns (B*T, T_imag, 1).
        """
        if self.imag_reward_source == "head":
            return self._frozen_reward(imag_feat).mode
        reward_input = goals.reward_state(self._goal_spec, stoch=imag_stoch, logit=imag_logit, rssm=self.rssm)
        return self.reward_function(reward_input, goal)

    @torch.no_grad()
    def _imagine(self, start, imag_horizon, goal):
        """Roll out the policy in latent space."""
        # (B, S, K), (B, D)
        feats = []
        actions = []
        logits = []
        stoch, deter = start
        for _ in range(imag_horizon):
            # (B, F)
            feat = self._frozen_rssm.get_feat(stoch, deter)
            # (B, A)
            goal = goal.reshape(feat.shape[0], -1)
            policy_input = torch.cat([feat, goal], dim=-1)
            if self.threshold_bins > 0:
                t = self.threshold_onehot.expand(feat.shape[0], -1)
                policy_input = torch.cat([policy_input, t], dim=-1)
            action = self._frozen_actor(policy_input).rsample()
            # Append feat and its corresponding sampled action at the same time step.
            feats.append(feat)
            actions.append(action)
            stoch, deter, logit = self._frozen_rssm.img_step(stoch, deter, action)
            logits.append(logit)

        # Stack along sequence dim T_imag.
        # (B, T_imag, F), (B, T_imag, A), (B, T_imag, S, K)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1), torch.stack(logits, dim=1)

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"]) / 255.0
        return data

    @torch.no_grad()
    def augment_data(self, data):
        data_aug = {k: torch.cat([v, v], axis=0) for k, v in data.items()}
        # (B, T, H, W, C) -> (B, T, C, H, W)
        image = data_aug["image"].permute(0, 1, 4, 2, 3)
        data_aug["image"] = self.random_translate(
            image,
            self.aug_max_delta,
            same_across_time=self.aug_same_across_time,
            bilinear=self.aug_bilinear,
        )
        # (B, T, C, H, W) -> (B, T, H, W, C)
        data_aug["image"] = data_aug["image"].permute(0, 1, 3, 4, 2)
        return data_aug

    @torch.no_grad()
    def ema_proj(self, data):
        with torch.no_grad():
            embed = self._ema_encoder(data)
            proj = self._ema_obs_proj(embed)
        return F.normalize(proj, p=2, dim=-1)

    @torch.no_grad()
    def ema_update(self):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        self._prototypes.data.copy_(prototypes)
        if self._ema_updates % self.ema_update_every == 0:
            mix = self.ema_update_fraction if self._ema_updates > 0 else 1.0
            for s, d in zip(self.encoder.parameters(), self._ema_encoder.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
            for s, d in zip(self.obs_proj.parameters(), self._ema_obs_proj.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self._ema_updates += 1

    def sinkhorn(self, scores):
        """Sinkhorn-Knopp normalization.

        Notes
        -----
        Given a score matrix, we iteratively normalize rows and columns in log
        space so that the resulting assignment matrix is approximately doubly
        stochastic.
        """
        shape = scores.shape
        K = shape[0]
        scores = scores.reshape(-1)
        log_Q = F.log_softmax(scores / self.sinkhorn_eps, dim=0)
        log_Q = log_Q.reshape(K, -1)
        N = log_Q.shape[1]
        for _ in range(self.sinkhorn_iters):
            log_row_sums = torch.logsumexp(log_Q, dim=1, keepdim=True)
            log_Q = log_Q - log_row_sums - math.log(K)
            log_col_sums = torch.logsumexp(log_Q, dim=0, keepdim=True)
            log_Q = log_Q - log_col_sums - math.log(N)
        log_Q = log_Q + math.log(N)
        Q = torch.exp(log_Q)
        return Q.reshape(shape)

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)

        obs_proj = self.obs_proj(embed)
        obs_norm = torch.norm(obs_proj, dim=-1)
        obs_proj = F.normalize(obs_proj, p=2, dim=-1)

        B, T = obs_proj.shape[:2]
        # (B, T, P) -> (B*T, P)
        obs_proj = obs_proj.reshape(B * T, -1)
        obs_scores = torch.matmul(obs_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)
        obs_scores = obs_scores[:, :, self.warm_up :]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)

        # (B, T, P) -> (B*T, P)
        ema_proj = ema_proj.reshape(B * T, -1)
        ema_scores = torch.matmul(ema_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)
        ema_scores = ema_scores[:, :, self.warm_up :]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)

        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        feat_proj = self.feat_proj(feat)
        feat_norm = torch.norm(feat_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)

        # (B, T, P) -> (B*T, P)
        feat_proj = feat_proj.reshape(B * T, -1)
        feat_scores = torch.matmul(feat_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)
        feat_scores = feat_scores[:, :, self.warm_up :]
        feat_logits = F.log_softmax(feat_scores / self.temperature, dim=0)

        swav_loss = -0.5 * torch.mean(torch.sum(ema_targets_2 * obs_logits_1, dim=0)) - 0.5 * torch.mean(
            torch.sum(ema_targets_1 * obs_logits_2, dim=0)
        )
        temp_loss = -torch.mean(torch.sum(ema_targets * feat_logits, dim=0))
        norm_loss = torch.mean(torch.square(obs_norm - 1)) + torch.mean(torch.square(feat_norm - 1))

        return {
            "swav": swav_loss,
            "temp": temp_loss,
            "norm": norm_loss,
        }

    @torch.no_grad()
    def random_translate(self, x, max_delta, same_across_time=False, bilinear=False):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        pad = int(max_delta)

        # Pad
        x_padded = F.pad(x_flat, (pad, pad, pad, pad), "replicate")
        h_padded, w_padded = H + 2 * pad, W + 2 * pad

        # Create base grid
        eps_h = 1.0 / h_padded
        eps_w = 1.0 / w_padded
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h_padded, device=x.device, dtype=x.dtype)[:H]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w_padded, device=x.device, dtype=x.dtype)[:W]
        arange_h = arange_h.unsqueeze(1).repeat(1, W).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B * T, 1, 1, 1)

        # Create shift
        if same_across_time:
            shift = torch.randint(0, 2 * pad + 1, size=(B, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift = shift.repeat(1, T, 1, 1, 1).reshape(B * T, 1, 1, 2)
        else:
            shift = torch.randint(0, 2 * pad + 1, size=(B * T, 1, 1, 2), device=x.device, dtype=x.dtype)

        shift = shift * 2.0 / torch.tensor([w_padded, h_padded], device=x.device, dtype=x.dtype)

        # Apply shift and sample
        grid = base_grid + shift
        mode = "bilinear" if bilinear else "nearest"
        x_translated = F.grid_sample(x_padded, grid, mode=mode, padding_mode="zeros", align_corners=False)

        return x_translated.reshape(B, T, C, H, W)
