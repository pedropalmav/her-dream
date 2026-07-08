"""Tests for the `goal_imag_horizon` config wiring (goal-imagination branch).

`goal_imag_horizon` lives in `configs/model/_base_.yaml` and defaults to
`${model.imag_horizon}` so existing runs (which never set it) keep imagining
goals for `imag_horizon` steps. These tests pin that contract: the default
follows `imag_horizon`, and an explicit override decouples the two.
"""

from hydra import compose, initialize


def _compose(*overrides):
    # config_path is relative to this test file's directory.
    with initialize(version_base=None, config_path="../../configs"):
        return compose(config_name="configs", overrides=list(overrides))


class TestDefault:
    def test_defaults_to_imag_horizon(self):
        cfg = _compose()
        assert cfg.model.goal_imag_horizon == cfg.model.imag_horizon

    def test_default_is_fifteen(self):
        assert _compose().model.goal_imag_horizon == 15


class TestOverride:
    def test_explicit_override_takes_effect(self):
        cfg = _compose("model.goal_imag_horizon=30")
        assert cfg.model.goal_imag_horizon == 30

    def test_override_does_not_change_imag_horizon(self):
        cfg = _compose("model.goal_imag_horizon=30")
        assert cfg.model.imag_horizon == 15

    def test_tracks_imag_horizon_when_only_imag_horizon_overridden(self):
        # Interpolation means changing imag_horizon also moves the goal horizon...
        cfg = _compose("model.imag_horizon=7")
        assert cfg.model.goal_imag_horizon == 7

    def test_explicit_goal_horizon_wins_over_imag_horizon(self):
        # ...unless goal_imag_horizon is set explicitly, which decouples them.
        cfg = _compose("model.imag_horizon=7", "model.goal_imag_horizon=20")
        assert cfg.model.imag_horizon == 7
        assert cfg.model.goal_imag_horizon == 20


class TestDreamerReadsConfig:
    def test_model_cfg_exposes_goal_imag_horizon(self):
        # Dreamer.__init__ reads `int(config.goal_imag_horizon)` off the model cfg.
        cfg = _compose("model.goal_imag_horizon=12")
        assert "goal_imag_horizon" in cfg.model
        assert int(cfg.model.goal_imag_horizon) == 12
