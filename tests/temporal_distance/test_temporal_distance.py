"""Tests for LEXA's learned steps-to-goal reward."""

import pytest
import torch

from tests.temporal_distance.conftest import B, K, L, S, T, make_td


class TestConstruction:
    def test_head_sees_the_state_and_the_goal(self, td):
        first = next(m for m in td.head.modules() if isinstance(m, torch.nn.Linear))
        assert first.in_features == 2 * L

    def test_predicts_a_single_value(self, td, stoch):
        assert td.distance(stoch, stoch[:, 0]).shape == (B, T, 1)

    def test_negative_factor_must_be_non_negative(self):
        with pytest.raises(ValueError, match="neg_sampling_factor"):
            make_td(model__temporal_distance__neg_sampling_factor=-1.0)


class TestDistance:
    def test_accepts_a_flat_goal(self, td, stoch):
        # The goal arrives as (B, S, K) from the buffer but flat from some callers.
        grouped = td.distance(stoch, stoch[:, 0])
        flat = td.distance(stoch, stoch[:, 0].reshape(B, L))
        assert torch.equal(grouped, flat)

    def test_the_goal_is_held_fixed_along_the_rollout(self, td, stoch):
        # Same state at two steps with one goal must score identically.
        repeated = stoch[:, :1].expand(-1, T, -1, -1).contiguous()
        out = td.distance(repeated, stoch[:, 0])
        assert torch.allclose(out, out[:, :1].expand_as(out), atol=1e-5)

    def test_depends_on_the_goal(self, td, stoch):
        near = td.distance(stoch, stoch[:, 0])
        far = td.distance(stoch, stoch[:, -1])
        assert not torch.allclose(near, far)


class TestLoss:
    def test_returns_a_scalar(self, td, stoch):
        loss, _ = td.loss(stoch)
        assert loss.ndim == 0

    def test_reports_the_label_and_prediction(self, td, stoch):
        _, metrics = td.loss(stoch)
        assert {"temporal_label", "temporal_pred"} <= set(metrics)
        # Labels are normalised step counts, so they live in [0, 1].
        assert 0.0 <= float(metrics["temporal_label"]) <= 1.0

    def test_no_gradient_reaches_the_world_model(self, td):
        # The predictor is a probe on detached latents, like the ensemble.
        stoch = torch.randn(B, T, S, K, requires_grad=True)
        loss, _ = td.loss(stoch)
        loss.backward()
        assert stoch.grad is None

    def test_gradient_reaches_the_head(self, td, stoch):
        loss, _ = td.loss(stoch)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in td.head.parameters())

    def test_a_single_trajectory_cannot_be_negatively_sampled(self, td):
        with pytest.raises(ValueError, match="at least 2 trajectories"):
            td.loss(torch.randn(1, T, S, K))

    def test_zero_negatives_still_trains(self, stoch):
        td = make_td(model__temporal_distance__neg_sampling_factor=0.0)
        loss, _ = td.loss(stoch)
        assert torch.isfinite(loss)

    def test_training_reduces_the_loss(self, td, stoch):
        opt = torch.optim.Adam(td.parameters(), lr=1e-2)
        torch.manual_seed(0)
        first, last = None, None
        for _ in range(40):
            loss, _ = td.loss(stoch)
            first = loss.item() if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
        assert last < first

    def test_learns_that_nearby_states_are_close(self):
        # The point of the objective: after fitting one fixed trajectory batch,
        # a state should be predicted nearer to a step just ahead of it than to
        # the far end of the rollout.
        torch.manual_seed(0)
        td = make_td(model__temporal_distance__num_positives=512)
        stoch = torch.randn(B, T, S, K)
        opt = torch.optim.Adam(td.parameters(), lr=3e-3)
        for _ in range(300):
            loss, _ = td.loss(stoch)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            near = td.distance(stoch[:, :1], stoch[:, 1]).mean()
            far = td.distance(stoch[:, :1], stoch[:, -1]).mean()
        assert near < far
