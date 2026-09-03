"""Tests for `backfill_config` — keys added after archived runs were trained.

`zflow/serve.py` and `viz/common.py` rebuild a `Dreamer` from an old run's saved
`.hydra/config.yaml`, so every key the constructor reads must be filled in for
configs that predate it.
"""

from omegaconf import OmegaConf

from her_dream.tools.config_compat import backfill_config


def _old_config(**model_extra):
    """A goal-conditioned run's config as saved before the head keys existed."""
    return OmegaConf.create({
        "goal_type": "full",
        "model": {"goal_type": "full", "imag_horizon": 15, "goal_imag_horizon": 15, **model_extra},
    })


class TestOptionalHeadDefaults:
    def test_head_keys_default_off(self):
        config, wm_only = backfill_config(_old_config())
        assert wm_only is False
        assert config.model.use_reward_head is False
        assert config.model.use_cont_head is False
        assert config.model.imag_reward_source == "goal"

    def test_existing_values_are_preserved(self):
        config, _ = backfill_config(_old_config(use_reward_head=True, imag_reward_source="head"))
        assert config.model.use_reward_head is True
        assert config.model.imag_reward_source == "head"

    def test_backfilled_for_pre_goal_conditioning_configs(self):
        # No top-level goal_type -> the wm_only branch, which merges its own
        # model defaults on top; the head keys must survive that deep merge.
        config, wm_only = backfill_config(OmegaConf.create({"model": {"imag_horizon": 15}}))
        assert wm_only is True
        assert config.model.use_cont_head is False
        assert config.model.goal_imag_horizon == 15
