"""Tests for the `goal_imag_horizon` config wiring (goal-imagination branch).

`goal_imag_horizon` lives in `configs/model/_base_.yaml` and defaults to
`${model.imag_horizon}` so existing runs (which never set it) keep imagining
goals for `imag_horizon` steps. These tests pin that contract: the default
follows `imag_horizon`, and an explicit override decouples the two.
"""

import pathlib

from hydra import compose, initialize_config_dir

import her_dream.goals as goals

# Hydra configs now ship inside the her_dream package (next to goals.py).
_CONFIG_DIR = str(pathlib.Path(goals.__file__).parent / "configs")


def _compose(*overrides):
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
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


class TestExploreGoalConfig:
    """`model.explore_goal` (goal_sample=explore_imagination[_refresh])."""

    def test_default_rollouts(self):
        assert _compose().model.explore_goal.num_rollouts == 16

    def test_composes_with_the_explorer_source(self):
        cfg = _compose("env.goal_sample=explore_imagination_refresh", "lexa=True", "model.explore_goal.num_rollouts=4")
        assert cfg.trainer.goal_sample == "explore_imagination_refresh"
        assert cfg.model.explore_goal.num_rollouts == 4


class TestGoalRefreshEvery:
    """`env.goal_refresh_every` (imagination_refresh) follows the goal horizon by default."""

    def test_defaults_to_goal_imag_horizon(self):
        cfg = _compose()
        assert cfg.env.goal_refresh_every == cfg.model.goal_imag_horizon == 15
        assert cfg.trainer.goal_refresh_every == 15

    def test_tracks_the_horizon(self):
        cfg = _compose("model.imag_horizon=30")
        assert cfg.env.goal_refresh_every == cfg.trainer.goal_refresh_every == 30

    def test_explicit_value_decouples_it(self):
        cfg = _compose("model.goal_imag_horizon=30", "env.goal_refresh_every=10")
        assert cfg.env.goal_refresh_every == cfg.trainer.goal_refresh_every == 10
