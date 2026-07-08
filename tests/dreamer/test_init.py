"""Tests for `Dreamer.__init__` — the construction/wiring paths.

Covers: representation-loss branch selection (which auxiliary module gets built),
action-space handling (discrete / multi-discrete / continuous / gym-`.n`), the
`goal_type`-dependent input sizing (`first_row` goal shape, `log_prob` threshold
bins), the optional text encoder, the `compile` wrapping, and the three mutually
exclusive-mode `ValueError` guards. These build a real (tiny) `Dreamer` but never
run an optimization step.
"""

import pytest
import torch

from dreamer import Dreamer
from tests.dreamer.conftest import (
    ACT,
    act_multi,
    build_model_config,
    make_obs_space,
    make_real_dreamer,
)


class TestRepLossBranch:
    def test_r2dreamer_builds_projector(self):
        agent, _ = make_real_dreamer(model__rep_loss="r2dreamer")
        assert hasattr(agent, "prj")
        assert not hasattr(agent, "decoder")

    def test_infonce_builds_projector(self):
        agent, _ = make_real_dreamer(model__rep_loss="infonce")
        assert hasattr(agent, "prj")

    def test_dreamer_builds_decoder(self):
        agent, _ = make_real_dreamer(model__rep_loss="dreamer")
        assert hasattr(agent, "decoder")

    def test_dreamer_reexpands_recon_loss_scale(self):
        # The "recon" scale is popped and re-keyed onto each decoder output.
        agent, _ = make_real_dreamer(model__rep_loss="dreamer")
        assert "recon" not in agent._loss_scales
        for key in agent.decoder.all_keys:
            assert key in agent._loss_scales

    def test_dreamerpro_builds_prototypes_and_projections(self):
        agent, _ = make_real_dreamer(model__rep_loss="dreamerpro")
        assert isinstance(agent._prototypes, torch.nn.Parameter)
        assert hasattr(agent, "obs_proj")
        assert hasattr(agent, "feat_proj")
        assert hasattr(agent, "_ema_encoder")


class TestActionSpace:
    def test_discrete_sets_act_dim_and_flags(self):
        agent, _ = make_real_dreamer(act="discrete")
        assert agent.act_dim == ACT
        assert agent.act_discrete is True
        assert agent._act_n == ACT
        assert agent._act_multi is False
        assert agent._act_nvec is None

    def test_multi_discrete_sets_nvec(self):
        agent, _ = make_real_dreamer(act="multi")
        assert agent._act_multi is True
        assert agent._act_nvec == (2, 3)
        assert agent.act_dim == 5  # sum of the nvec
        assert agent.act_discrete is True

    def test_continuous_is_not_discrete(self):
        agent, _ = make_real_dreamer(act="cont")
        assert agent.act_discrete is False
        assert agent._act_n is None
        assert agent._act_multi is False

    def test_gym_discrete_n_attribute(self):
        # act_space with `.n` -> act_dim = n and actor.shape = (n,).
        agent, _ = make_real_dreamer(act="n")
        assert agent.act_dim == ACT


class TestGoalTypeSizing:
    def test_first_row_uses_discrete_goal_shape(self):
        # first_row goals are (K,), so the actor input adds only K, not S*K.
        agent, _ = make_real_dreamer(goal_type="first_row")
        expected = agent.rssm.feat_size + agent.rssm._discrete
        # MLPHead stores its input dim on the first linear layer's in_features.
        assert _mlphead_in_features(agent.actor) == expected

    def test_full_uses_flat_stoch_goal_shape(self):
        agent, _ = make_real_dreamer(goal_type="full")
        expected = agent.rssm.feat_size + agent.rssm.flat_stoch
        assert _mlphead_in_features(agent.actor) == expected

    def test_log_prob_builds_threshold_bins(self):
        agent, _ = make_real_dreamer(goal_type="log_prob")
        assert agent.threshold_bins > 0
        assert hasattr(agent, "threshold_onehot")
        assert agent.threshold_onehot.sum().item() == pytest.approx(1.0)

    def test_non_log_prob_has_no_threshold_bins(self):
        agent, _ = make_real_dreamer(goal_type="full")
        assert agent.threshold_bins == 0
        assert not hasattr(agent, "threshold_onehot")


class TestTextEncoder:
    def test_mission_text_builds_text_encoder(self):
        agent, _ = make_real_dreamer(mission_text=True)
        assert hasattr(agent, "text_encoder")

    def test_no_mission_text_no_text_encoder(self):
        agent, _ = make_real_dreamer(mission_text=False)
        assert not hasattr(agent, "text_encoder")


class TestCompile:
    def test_compile_wraps_cal_grad(self):
        # compile=True replaces _cal_grad with a torch.compile'd callable; we
        # only assert construction succeeds and the attribute is still callable.
        agent, _ = make_real_dreamer(model__compile=True)
        assert callable(agent._cal_grad)


class TestModeGuards:
    def _build_raw(self, **model_overrides):
        cfg = build_model_config(**model_overrides)
        from rewards import make_reward

        return Dreamer(
            cfg.model,
            make_obs_space(),
            act_multi(),
            reward_function=make_reward(cfg.model),
        )

    def test_wm_only_and_freeze_wm_conflict(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._build_raw(wm_only=True, freeze_wm=True)

    def test_train_text_only_with_wm_only_conflict(self):
        with pytest.raises(ValueError, match="exclusive"):
            self._build_raw(train_text_only=True, wm_only=True, mission_text=True)

    def test_train_text_only_requires_mission_text(self):
        with pytest.raises(ValueError, match="mission_text"):
            self._build_raw(train_text_only=True, mission_text=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mlphead_in_features(head):
    """Return the input dim of an MLPHead by inspecting its first Linear layer."""
    for module in head.modules():
        if isinstance(module, torch.nn.Linear):
            return module.in_features
    raise AssertionError("no Linear layer found in MLPHead")
