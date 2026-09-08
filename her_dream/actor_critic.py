import copy

import torch
from torch import nn

import her_dream.networks as networks
import her_dream.tools as tools
from her_dream.tools import to_f32


class ActorCritic(nn.Module):
    """Goal-conditioned actor + critic, its slow target and its frozen clones.

    Owns everything between the imagination rollout and the losses: the policy
    input layout (`feat ++ goal [++ threshold]`), lambda returns, the return
    EMA, and the policy / value / repval objectives.

    It never calls `backward()` — `imagination_loss` and `replay_value_loss`
    return unscaled scalar losses, exactly like `RSSM.kl_loss` — so the owner
    keeps the single shared optimizer and the single backward pass.

    `name` prefixes every loss key, metric name and optimizer module name, so a
    second instance (an exploration policy, say) can coexist with the task one
    without colliding. The default empty name leaves every key untouched.
    """

    def __init__(self, config, feat_size, act_space, goal_size, *, name=""):
        super().__init__()
        self.name = str(name)
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)

        # The actor head is shaped by the action space, and its `dist` node is
        # collapsed from the cont/disc/multi_disc triple down to the selected
        # one. Both edits are in place on the shared config node, so the collapse
        # is guarded: a second instance built from the same node (an exploration
        # policy alongside the task one) would otherwise look for `.disc` on an
        # already-collapsed node and raise.
        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        self.act_discrete = hasattr(act_space, "multi_discrete") or hasattr(act_space, "discrete")
        if "cont" in config.actor.dist:
            if hasattr(act_space, "multi_discrete"):
                config.actor.dist = config.actor.dist.multi_disc
            elif hasattr(act_space, "discrete"):
                config.actor.dist = config.actor.dist.disc
            else:
                config.actor.dist = config.actor.dist.cont

        # `log_prob` is the only goal type conditioning the policy on the
        # acceptance threshold; it appends a one-hot of the threshold bin. The
        # threshold describes the *goal* acceptance criterion, so a goal-agnostic
        # policy (goal_size=0, e.g. an explorer) takes neither.
        self.goal_size = int(goal_size)
        uses_threshold = bool(getattr(config, "uses_threshold", False)) and self.goal_size > 0
        self.threshold_bins = int(round(1.0 / config.prob_threshold_step)) + 1 if uses_threshold else 0
        self.input_size = feat_size + self.goal_size + self.threshold_bins

        # Construction order is load-bearing: it fixes the order of RNG draws in
        # the owner's __init__ and the parameter order in the shared optimizer.
        self.actor = networks.MLPHead(config.actor, self.input_size)
        self.value = networks.MLPHead(config.critic, self.input_size)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0
        self.return_ema = networks.ReturnEMA(device=self.device)

        if self.threshold_bins > 0:
            idx = int(round(config.prob_threshold / config.prob_threshold_step))
            t_oh = torch.zeros(self.threshold_bins)
            t_oh[idx] = 1.0
            self.register_buffer("threshold_onehot", t_oh)  # shape (threshold_bins,)

        self.clone_and_freeze()

    def key(self, base):
        """Namespace a loss / metric / module key with this instance's name."""
        return f"{self.name}_{base}" if self.name else base

    def optim_modules(self):
        """The trainable modules, keyed as they should appear in the optimizer.

        The slow target is deliberately absent: it is updated by Polyak mixing,
        not by gradient descent.
        """
        return {self.key("actor"): self.actor, self.key("value"): self.value}

    def loss_scales(self, base_scales):
        """Extra `loss_scales` entries this instance needs, mirroring the base.

        Empty for an unnamed instance, whose keys already exist in the config.
        """
        if not self.name:
            return {}
        return {self.key(k): base_scales[k] for k in ("policy", "value", "repval")}

    def policy_input(self, feat, goal):
        """Concatenate `feat` with the flattened goal (and threshold one-hot).

        `feat` is (..., F) and `goal` carries the same leading dims in any
        layout — (..., K) or (..., S, K) — so it is flattened to (..., G).

        A goal-agnostic instance (`goal_size=0`) ignores `goal` entirely and is
        conditioned on the feature alone; pass `None` for it.
        """
        parts = [feat]
        if self.goal_size > 0:
            parts.append(goal.reshape(*feat.shape[:-1], -1))
        if self.threshold_bins > 0:
            parts.append(self.threshold_onehot.expand(*feat.shape[:-1], -1))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else feat

    def policy(self, feat, goal):
        """Action distribution from the live actor (gradients flow)."""
        return self.actor(self.policy_input(feat, goal))

    def frozen_policy(self, feat, goal):
        """Action distribution from the frozen inference clone."""
        return self._frozen_actor(self.policy_input(feat, goal))

    def imagination_loss(self, imag_feat, imag_action, imag_reward, imag_cont, goal):
        """Policy-gradient actor loss and TD critic loss over an imagined rollout.

        All inputs are detached: the rollout is produced by the frozen clones,
        so the gradient reaches only this instance's actor and critic.

        Args:
            imag_feat: (B, T_imag, F) RSSM features along the rollout.
            imag_action: (B, T_imag, A) actions taken by the frozen actor.
            imag_reward: (B, T_imag, 1) reward per imagined step.
            imag_cont: (B, T_imag, 1) probability the episode continues.
            goal: (B, ...) goal held fixed along the rollout, or None for a
                goal-agnostic instance.

        Returns:
            (losses, metrics, ret) — `ret` is the (B, T_imag-1, 1) lambda
            return, which the replay branch bootstraps from.
        """
        horizon = imag_feat.shape[1]
        # (B, T_imag, ...) — the goal is constant along the rollout.
        imag_goal = None
        if goal is not None:
            imag_goal = goal.unsqueeze(1).expand(-1, horizon, *([-1] * (goal.ndim - 1)))
        imag_input = self.policy_input(imag_feat, imag_goal)

        imag_value = self._frozen_value(imag_input).mode
        imag_slow_value = self._frozen_slow_value(imag_input).mode
        disc = 1 - 1 / self.horizon

        # (B, T_imag, 1)
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        # (B, T_imag-1, 1)
        ret = self.lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
        ret_offset, ret_scale = self.return_ema(ret)
        # (B, T_imag-1, 1)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_input)
        # (B, T_imag-1, 1)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        policy_loss = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        imag_value_dist = self.value(imag_input)
        # (B, T_imag, 1)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        value_loss = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )

        losses = {self.key("policy"): policy_loss, self.key("value"): value_loss}
        ret_normed = (ret - ret_offset) / ret_scale
        metrics = {
            self.key("ret"): torch.mean(ret_normed),
            self.key("ret_005"): self.return_ema.ema_vals[0],
            self.key("ret_095"): self.return_ema.ema_vals[1],
            self.key("adv"): torch.mean(adv),
            self.key("adv_std"): torch.std(adv),
            self.key("con"): torch.mean(imag_cont),
            self.key("rew"): torch.mean(imag_reward),
            self.key("val"): torch.mean(imag_value),
            self.key("tar"): torch.mean(ret),
            self.key("slowval"): torch.mean(imag_slow_value),
            self.key("weight"): torch.mean(weight),
            self.key("action_entropy"): torch.mean(entropy),
        }
        metrics.update(tools.tensorstats(imag_action, self.key("action")))
        return losses, metrics, ret

    def replay_value_loss(self, feat, goal, last, term, reward, boot):
        """Critic loss on real replayed transitions.

        `feat` is deliberately the *attached* RSSM feature: this loss is what
        lets the critic's gradient reach the world model.

        Args:
            feat: (B, T, F) posterior features, still on the WM graph.
            goal: (B, T, ...) goal per transition.
            last / term / reward: (B, T, 1) float32 episode flags and reward.
            boot: (B, T, 1) bootstrap value, from the imagined lambda return.

        Returns:
            (losses, metrics).
        """
        value_input = self.policy_input(feat, goal)
        value = self._frozen_value(value_input).mode
        slow_value = self._frozen_slow_value(value_input).mode
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self.lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        # Keep this attached to the world model so gradients can flow through
        value_dist = self.value(value_input)
        repval_loss = torch.mean(
            weight[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(
                -1
            )
        )

        losses = {self.key("repval"): repval_loss}
        metrics = {}
        metrics.update(tools.tensorstats(ret, self.key("ret_replay")))
        metrics.update(tools.tensorstats(value, self.key("value_replay")))
        metrics.update(tools.tensorstats(slow_value, self.key("slow_value_replay")))
        return losses, metrics

    @torch.no_grad()
    def lambda_return(self, last, term, reward, value, boot, disc, lamb):
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

    def clone_and_freeze(self):
        """Rebuild the frozen inference clones from the live modules."""
        self._frozen_actor = tools.freeze_clone(self.actor)
        self._frozen_value = tools.freeze_clone(self.value)
        self._frozen_slow_value = tools.freeze_clone(self._slow_value)

    def update_slow_target(self):
        """Polyak-mix the live critic into the slow value target, on schedule."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def freeze(self):
        """Disable gradients on the actor and critic (the live ones)."""
        for module in (self.actor, self.value):
            for param in module.parameters():
                param.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # slow_value should be always eval mode
        self._slow_value.train(False)
        return self
