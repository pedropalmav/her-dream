import torch
from torch import nn

import her_dream.networks as networks
from her_dream.tools import to_f32


class TemporalDistance(nn.Module):
    """Predicts how many steps separate a latent state from a goal.

    This is LEXA's goal-reaching reward: rather than asking whether the agent's
    latent *equals* the goal — which for a 32-group discrete `z` is an
    all-or-nothing signal — it asks how far away the goal still is, which gives
    the achiever a gradient everywhere.

    Trained by regression on pairs drawn from the same imagined trajectory,
    labelled with the (normalised) number of steps between them, plus negative
    pairs drawn from *different* trajectories and labelled maximally distant so
    the predictor does not collapse to a constant.
    """

    def __init__(self, config, latent_size):
        super().__init__()
        cfg = config.temporal_distance
        self.num_positives = int(cfg.num_positives)
        self.neg_sampling_factor = float(cfg.neg_sampling_factor)
        if self.neg_sampling_factor < 0:
            raise ValueError(f"neg_sampling_factor must be >= 0, got {self.neg_sampling_factor}.")
        self.latent_size = int(latent_size)
        # State and goal live in the same space, so the head sees both flattened.
        self.head = networks.MLPHead(cfg.head, 2 * self.latent_size)

    def _flat(self, x):
        """(..., S, K) -> (..., S*K); an already-flat latent passes through."""
        return x if x.shape[-1] == self.latent_size else x.reshape(*x.shape[:-2], -1)

    def distance(self, stoch, goal):
        """Predicted normalised steps from each state in `stoch` to `goal`.

        Args:
            stoch: (B, T, S, K) latents along a rollout.
            goal: (B, S, K) goal held fixed along the rollout.

        Returns:
            (B, T, 1) distance, ~[0, 1] — 0 when the goal is already reached.
        """
        state = self._flat(stoch)
        target = self._flat(goal).unsqueeze(1).expand(-1, state.shape[1], -1)
        return self.head(torch.cat([state, target], dim=-1))

    def loss(self, stoch):
        """Regress the step count between pairs drawn from `stoch`.

        Args:
            stoch: (B, T, S, K) trajectories — the imagined rollout, so the
                predictor is trained on the distribution the reward is read on.

        Returns:
            (loss, metrics) — an unscaled scalar, like every other loss here.
        """
        # Detached: the reward predictor is a probe, not a representation loss.
        state = self._flat(stoch).detach()
        B, T = state.shape[:2]
        if B < 2:
            raise ValueError("Temporal-distance training needs at least 2 trajectories for negatives.")
        device = state.device

        def predict(rows_a, steps_a, rows_b, steps_b):
            pair = torch.cat([state[rows_a, steps_a], state[rows_b, steps_b]], dim=-1)
            return to_f32(self.head(pair)).squeeze(-1)

        # Positives: two steps of one trajectory, labelled by how far apart they
        # are. Ordering the two draws keeps the label non-negative.
        n = self.num_positives
        rows = torch.randint(0, B, (n,), device=device)
        first = torch.randint(0, T, (n,), device=device)
        second = torch.randint(0, T, (n,), device=device)
        near, far = torch.minimum(first, second), torch.maximum(first, second)
        label = (far - near).float() / max(T - 1, 1)
        pred = predict(rows, near, rows, far)
        loss = torch.mean((pred - label) ** 2)
        metrics = {"temporal_label": label.mean(), "temporal_pred": pred.mean().detach()}

        # Negatives: states from different trajectories are assumed maximally far
        # apart. Without them the predictor can drive every distance to zero.
        num_neg = int(self.neg_sampling_factor * n)
        if num_neg > 0:
            rows_a = torch.randint(0, B, (num_neg,), device=device)
            # +[1, B) modulo B: a different trajectory by construction.
            rows_b = (rows_a + torch.randint(1, B, (num_neg,), device=device)) % B
            steps_a = torch.randint(0, T, (num_neg,), device=device)
            steps_b = torch.randint(0, T, (num_neg,), device=device)
            far_label = torch.ones(num_neg, device=device)
            # Summed rather than pooled, as in LEXA: the negatives are a tenth of
            # the samples but carry equal weight.
            loss = loss + torch.mean((predict(rows_a, steps_a, rows_b, steps_b) - far_label) ** 2)

        return loss, metrics
