"""Tests for the Plan2Explore disagreement ensemble."""

import pytest
import torch

from her_dream.plan2explore import Disagreement
from tests.dreamer.conftest import build_model_config
from tests.plan2explore.conftest import A, B, D, F, K, S, T, _StubRSSM, make_disag


class TestConstruction:
    def test_builds_the_configured_number_of_members(self, disag):
        assert len(disag.heads) == 8

    def test_members_are_independently_initialised(self, disag):
        # Identical members would make the disagreement identically zero.
        first = next(disag.heads[0].parameters())
        assert not any(torch.equal(first, next(h.parameters())) for h in disag.heads[1:])

    @pytest.mark.parametrize("target, size", [("stoch", S * K), ("deter", D), ("feat", F)])
    def test_target_selects_the_prediction_width(self, target, size):
        assert make_disag(model__disag__target=target).target_size == size

    def test_action_conditioning_widens_the_input(self):
        assert make_disag(model__disag__action_cond=True).input_size == F + A
        assert make_disag(model__disag__action_cond=False).input_size == F

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown disag.target"):
            make_disag(model__disag__target="embed")

    def test_a_single_member_raises(self):
        # A variance needs at least two predictions.
        with pytest.raises(ValueError, match="must be >= 2"):
            make_disag(model__disag__models=1)

    def test_zero_offset_raises(self):
        with pytest.raises(ValueError, match="offset must be >= 1"):
            make_disag(model__disag__offset=0)


class TestTargetFrom:
    def test_stoch_target_is_flattened(self, disag, batch):
        out = disag.target_from(batch.stoch, batch.deter, batch.feat)
        assert out.shape == (B, T, S * K)
        assert torch.equal(out, batch.stoch.reshape(B, T, S * K))

    def test_deter_target_passes_through(self, batch):
        d = make_disag(model__disag__target="deter")
        assert torch.equal(d.target_from(batch.stoch, batch.deter, batch.feat), batch.deter)

    def test_feat_target_passes_through(self, batch):
        d = make_disag(model__disag__target="feat")
        assert torch.equal(d.target_from(batch.stoch, batch.deter, batch.feat), batch.feat)


class TestLoss:
    def test_returns_a_scalar(self, disag, batch):
        target = disag.target_from(batch.stoch, batch.deter, batch.feat)
        loss, _ = disag.loss(batch.feat, batch.action, target)
        assert loss.ndim == 0

    def test_reports_the_per_member_error(self, disag, batch):
        target = disag.target_from(batch.stoch, batch.deter, batch.feat)
        _, metrics = disag.loss(batch.feat, batch.action, target)
        assert "disag_pred_err" in metrics

    def test_pairs_each_state_with_the_action_that_leaves_it(self, disag, batch):
        """`action[:, t]` leads *into* state t (the buffer shifts it), so the
        transition t -> t+offset is driven by `action[:, offset:]`."""
        seen = {}

        def spy(x):
            seen["inputs"] = x
            return torch.zeros(*x.shape[:-1], disag.target_size)

        disag.heads = torch.nn.ModuleList([_Lambda(spy)])
        disag.models = 1
        target = disag.target_from(batch.stoch, batch.deter, batch.feat)
        disag.loss(batch.feat, batch.action, target)

        inputs = seen["inputs"]
        assert inputs.shape[1] == T - disag.offset
        assert torch.equal(inputs[..., :F], batch.feat[:, : -disag.offset])
        assert torch.equal(inputs[..., F:], batch.action[:, disag.offset :])

    def test_target_is_shifted_forward(self, disag, batch):
        # The loss must compare against the *next* latent, not the current one.
        target = disag.target_from(batch.stoch, batch.deter, batch.feat)
        shifted = target[:, disag.offset :]
        assert shifted.shape[1] == T - disag.offset
        assert torch.equal(shifted[:, 0], target[:, disag.offset])

    def test_no_gradient_reaches_the_world_model(self, disag, batch):
        # Both sides are detached: the ensemble is a probe, never a shaping loss.
        feat = batch.feat.clone().requires_grad_(True)
        stoch = batch.stoch.clone().requires_grad_(True)
        target = disag.target_from(stoch, batch.deter, feat)
        loss, _ = disag.loss(feat, batch.action, target)
        loss.backward()
        assert feat.grad is None
        assert stoch.grad is None

    def test_gradient_reaches_every_member(self, disag, batch):
        target = disag.target_from(batch.stoch, batch.deter, batch.feat)
        loss, _ = disag.loss(batch.feat, batch.action, target)
        loss.backward()
        assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in h.parameters()) for h in disag.heads)

    def test_training_reduces_the_loss(self, disag, batch):
        target = disag.target_from(batch.stoch, batch.deter, batch.feat)
        opt = torch.optim.Adam(disag.parameters(), lr=1e-2)
        first = None
        for _ in range(20):
            loss, _ = disag.loss(batch.feat, batch.action, target)
            first = loss.item() if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() < first


class _Lambda(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) + self.w


class TestIntrinsicReward:
    def test_shape_is_one_reward_per_step(self, disag, batch):
        reward, _ = disag.intrinsic_reward(batch.feat, batch.action)
        assert reward.shape == (B, T, 1)

    def test_identical_members_disagree_about_nothing(self, disag, batch):
        for head in disag.heads[1:]:
            head.load_state_dict(disag.heads[0].state_dict())
        reward, _ = disag.intrinsic_reward(batch.feat, batch.action)
        assert torch.allclose(reward, torch.zeros_like(reward), atol=1e-5)

    def test_differing_members_produce_positive_reward(self, disag, batch):
        reward, _ = disag.intrinsic_reward(batch.feat, batch.action)
        assert (reward > 0).all()

    def test_matches_an_explicit_std_over_members(self, disag, batch):
        # The streaming two-moment accumulator must equal the naive stack.
        inputs = disag._inputs(batch.feat, batch.action)
        with torch.no_grad():
            preds = torch.stack([h(inputs) for h in disag.heads])
        expected = preds.std(0).mean(-1, keepdim=True)
        reward, _ = disag.intrinsic_reward(batch.feat, batch.action)
        assert torch.allclose(reward, expected, atol=1e-4)

    def test_reports_the_raw_magnitude(self, disag, batch):
        _, metrics = disag.intrinsic_reward(batch.feat, batch.action)
        assert torch.isfinite(metrics["disag_raw"])

    def test_intr_scale_scales_the_reward(self, batch):
        base, _ = make_disag().intrinsic_reward(batch.feat, batch.action)
        scaled, _ = make_disag(model__disag__intr_scale=3.0).intrinsic_reward(batch.feat, batch.action)
        # Different inits, so compare magnitudes rather than exact values.
        assert scaled.mean() > base.mean()

    def test_log_compresses_the_reward(self, disag, batch):
        cfg = build_model_config(model__disag__log=True)
        logged = Disagreement(cfg.model, F, A, _StubRSSM())
        logged.load_state_dict(disag.state_dict())
        raw, _ = disag.intrinsic_reward(batch.feat, batch.action)
        out, _ = logged.intrinsic_reward(batch.feat, batch.action)
        assert torch.allclose(out, torch.log(raw + 1e-8), atol=1e-5)

    def test_is_computed_without_gradients(self, disag, batch):
        feat = batch.feat.clone().requires_grad_(True)
        reward, _ = disag.intrinsic_reward(feat, batch.action)
        assert not reward.requires_grad
