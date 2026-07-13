"""Tests for goals.py — the GoalSpec descriptors built from the goal_type config.

Covers the factory/validation, each helper's branch per descriptor value, and a
config drift-guard: every ``configs/goal_type/*.yaml`` must build a valid spec whose
goal_type is supported by ``rewards.make_reward``.
"""

import glob
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

import goals
from rewards import make_reward

CONFIG_DIR = os.path.join(os.path.dirname(goals.__file__), "configs", "goal_type")


def _cfg(**kw):
    base = dict(goal_type="full", state_repr="stoch", goal_repr="sample", scope="full", uses_threshold=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _spec(**kw):
    return goals.make_goal_spec(_cfg(**kw))


class TestMakeGoalSpec:
    def test_reads_all_fields(self):
        spec = goals.make_goal_spec(_cfg(goal_type="argmax_full", state_repr="logit", goal_repr="argmax", scope="full"))
        assert (spec.goal_type, spec.state_repr, spec.goal_repr, spec.scope) == (
            "argmax_full",
            "logit",
            "argmax",
            "full",
        )

    def test_uses_threshold_defaults_false_when_key_absent(self):
        cfg = SimpleNamespace(goal_type="full", state_repr="stoch", goal_repr="sample", scope="full")
        assert goals.make_goal_spec(cfg).uses_threshold is False

    def test_uses_threshold_true(self):
        assert _spec(uses_threshold=True).uses_threshold is True

    @pytest.mark.parametrize(
        "field, bad",
        [("state_repr", "bad"), ("goal_repr", "bad"), ("scope", "bad")],
    )
    def test_invalid_descriptor_raises(self, field, bad):
        with pytest.raises(ValueError):
            goals.make_goal_spec(_cfg(**{field: bad}))


class TestStashesLogit:
    @pytest.mark.parametrize("state_repr, expected", [("stoch", False), ("dist", True), ("logit", True)])
    def test_stash_iff_not_stoch(self, state_repr, expected):
        assert goals.stashes_logit(_spec(state_repr=state_repr)) is expected


class TestRewardState:
    def test_stoch_returns_stoch(self):
        stoch, logit = torch.randn(2, 4, 4), torch.randn(2, 4, 4)
        out = goals.reward_state(_spec(state_repr="stoch"), stoch=stoch, logit=logit, rssm=MagicMock())
        assert out is stoch

    def test_logit_returns_logit(self):
        stoch, logit = torch.randn(2, 4, 4), torch.randn(2, 4, 4)
        out = goals.reward_state(_spec(state_repr="logit"), stoch=stoch, logit=logit, rssm=MagicMock())
        assert out is logit

    def test_dist_returns_rssm_get_dist(self):
        stoch, logit = torch.randn(2, 4, 4), torch.randn(2, 4, 4)
        rssm = MagicMock()
        out = goals.reward_state(_spec(state_repr="dist"), stoch=stoch, logit=logit, rssm=rssm)
        rssm.get_dist.assert_called_once_with(logit)
        assert out is rssm.get_dist.return_value


class TestGoalFromLatent:
    def test_sample_returns_stoch(self):
        stoch, logit = torch.randn(2, 4, 4), torch.randn(2, 4, 4)
        out = goals.goal_from_latent(_spec(goal_repr="sample"), stoch=stoch, logit=logit)
        assert out is stoch

    def test_logit_returns_logit(self):
        stoch, logit = torch.randn(2, 4, 4), torch.randn(2, 4, 4)
        out = goals.goal_from_latent(_spec(goal_repr="logit"), stoch=stoch, logit=logit)
        assert out is logit

    def test_argmax_returns_one_hot_of_argmax(self):
        logit = torch.tensor([[[0.0, 5.0, 0.0], [3.0, 0.0, 0.0]]])  # argmax -> col 1, col 0
        out = goals.goal_from_latent(_spec(goal_repr="argmax"), stoch=torch.zeros_like(logit), logit=logit)
        expected = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
        assert torch.equal(out, expected)


class TestGoalSize:
    def test_first_row_uses_discrete(self):
        rssm = SimpleNamespace(_discrete=16, flat_stoch=512)
        assert goals.goal_size(_spec(scope="first_row"), rssm) == 16

    def test_full_uses_flat_stoch(self):
        rssm = SimpleNamespace(_discrete=16, flat_stoch=512)
        assert goals.goal_size(_spec(scope="full"), rssm) == 512


class TestConfigDriftGuard:
    """Every goal_type config file must build a valid spec and a valid reward fn."""

    @pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml"))))
    def test_config_builds_valid_spec_and_reward(self, path):
        cfg = OmegaConf.load(path)
        spec = goals.make_goal_spec(cfg)  # validates descriptor values
        assert spec.goal_type == os.path.splitext(os.path.basename(path))[0]
        # make_reward must support this goal_type (raises ValueError otherwise).
        make_reward(cfg)

    def test_default_descriptors_matches_config(self):
        for path in glob.glob(os.path.join(CONFIG_DIR, "*.yaml")):
            cfg = OmegaConf.load(path)
            d = goals.default_descriptors(cfg.goal_type)
            assert d["state_repr"] == cfg.state_repr
            assert d["goal_repr"] == cfg.goal_repr
            assert d["scope"] == cfg.scope
