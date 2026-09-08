import torch
from torch import nn

import her_dream.distributions as dists
import her_dream.networks as networks
from her_dream.tools import WelfordAccumulator, to_f32


class Disagreement(nn.Module):
    """Ensemble of one-step latent predictors; their variance is the reward.

    Each member predicts the next latent from the current RSSM feature (and the
    action that drives the transition). Where the members disagree the world
    model is uncertain, so a policy rewarded by that variance seeks out states
    the model has not yet pinned down — this is Plan2Explore's intrinsic reward.

    The ensemble is a *probe*, not part of the representation: its inputs and
    targets are both detached, so training it never shapes the world model.
    Disagreement is meaningful only because the members differ: each `MLPHead`
    runs `weight_init_` itself, so they start from independent draws.
    """

    def __init__(self, config, feat_size, act_dim, rssm):
        super().__init__()
        cfg = config.disag
        self.target = str(cfg.target)
        if self.target not in ("stoch", "deter", "feat"):
            raise ValueError(f"Unknown disag.target {self.target!r}; expected 'stoch', 'deter' or 'feat'.")
        self.action_cond = bool(cfg.action_cond)
        self.offset = int(cfg.offset)
        if self.offset < 1:
            raise ValueError(f"disag.offset must be >= 1, got {self.offset}.")
        self.log = bool(cfg.log)
        self.intr_scale = float(cfg.intr_scale)
        self.models = int(cfg.models)
        if self.models < 2:
            raise ValueError(f"disag.models must be >= 2 for a variance to exist, got {self.models}.")

        self.target_size = {"stoch": rssm.flat_stoch, "deter": rssm._deter, "feat": feat_size}[self.target]
        self.input_size = feat_size + (act_dim if self.action_cond else 0)

        cfg.head.shape = [self.target_size]
        self.heads = nn.ModuleList([networks.MLPHead(cfg.head, self.input_size) for _ in range(self.models)])

    def target_from(self, stoch, deter, feat):
        """The tensor the ensemble is trained to predict, per `disag.target`."""
        if self.target == "stoch":
            # (B, T, S, K) -> (B, T, S*K)
            return stoch.reshape(*stoch.shape[:2], -1)
        if self.target == "deter":
            return deter
        return feat

    def _inputs(self, feat, action):
        if not self.action_cond:
            return feat
        return torch.cat([feat, action.to(feat.dtype)], dim=-1)

    def loss(self, feat, action, target):
        """One-step prediction loss over a replayed sequence.

        `action[:, t]` is the action leading *into* state t (the replay buffer
        shifts it, see `buffers/buffer.py`), so the transition t -> t+offset is
        driven by `action[:, offset:]` and pairs with `feat[:, :-offset]`.

        Args:
            feat: (B, T, F) posterior features.
            action: (B, T, A) actions, buffer-aligned.
            target: (B, T, D) the tensor to predict, from `target_from`.

        Returns:
            (loss, metrics) — an unscaled scalar, as every other loss here.
        """
        # Detached on both sides: the ensemble never trains the world model.
        inputs = self._inputs(feat[:, : -self.offset], action[:, self.offset :]).detach()
        targets = to_f32(target[:, self.offset :]).detach()

        # Summed over members, exactly as DreamerV2 does, so each member gets a
        # full-strength gradient rather than one scaled by 1/models.
        loss = -sum(dists.mse(head(inputs)).log_prob(targets).mean() for head in self.heads)
        metrics = {"disag_pred_err": (loss.detach() / self.models)}
        return loss, metrics

    @torch.no_grad()
    def intrinsic_reward(self, feat, action):
        """Ensemble std over an imagined rollout: the exploration reward.

        Accumulates the variance with `tools.WelfordAccumulator` rather than
        stacking every member's prediction, so memory is flat in the ensemble
        size — this runs over (B*T, T_imag, target_size), the largest tensor in
        a training step.

        Returns:
            ((B, T, 1) reward, metrics).
        """
        inputs = self._inputs(feat, action)
        acc = WelfordAccumulator()
        for head in self.heads:
            acc.update(to_f32(head(inputs)))
        # std over members, then mean over the predicted dimensions.
        disag = acc.std().mean(-1, keepdim=True)

        # Logged raw: `intr_scale` has to be set against this magnitude, because
        # ReturnEMA clips its scale at min=1.0 and so cannot amplify a tiny reward.
        metrics = {"disag_raw": disag.mean()}
        if self.log:
            disag = torch.log(disag + 1e-8)
        return self.intr_scale * disag, metrics
