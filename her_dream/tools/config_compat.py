"""Backfill checkpoint configs saved before newer keys existed.

Older runs' ``.hydra/config.yaml`` files (loaded by ``zflow/serve.py`` and viz)
predate goal-conditioning and/or the goal-type descriptor keys. `backfill_config`
fills in just enough for the current ``Dreamer`` constructor and ``make_goal_spec``
to run.
"""

from omegaconf import OmegaConf

from her_dream import goals

# Harmless dummy goal-conditioning defaults for pre-goal-conditioning (WM-only)
# checkpoints — they only size the actor/value heads, which are neither loaded
# nor run in WM-only mode; make_reward is skipped entirely.
_WM_ONLY_GOAL_DEFAULTS = {
    "goal_type": "full",
    "mission_text": False,
    "wm_only": False,
    "train_text_only": False,
    "freeze_wm": False,
    "prob_threshold": 0.5,
    "prob_threshold_step": 0.05,  # canonical (was 0.1 in the old zflow copy)
}


def backfill_config(config, *, force_wm_only=False):
    """Fill config keys added after older experiments were trained.

    Returns ``(config, wm_only)``. ``wm_only`` is True for pre-goal-conditioning
    checkpoints (absence of the top-level ``goal_type`` key); only the world model
    is used in that mode.
    """
    updates = {}
    if "goal_imag_horizon" not in config.model:
        updates["model"] = {"goal_imag_horizon": int(config.model.imag_horizon)}

    wm_only = force_wm_only or ("goal_type" not in config)
    if wm_only:
        # Dummy goal-conditioning keys so main's Dreamer can be *constructed*.
        # These only size the actor/value heads, which are neither loaded nor
        # run in WM-only mode; make_reward is skipped entirely.
        top = {k: v for k, v in _WM_ONLY_GOAL_DEFAULTS.items() if k not in config}
        model_missing = {k: v for k, v in _WM_ONLY_GOAL_DEFAULTS.items() if k not in config.model}
        # Deep-merge: preserves the model.goal_imag_horizon backfill above.
        updates = OmegaConf.merge(updates, {**top, "model": model_missing})

    config = OmegaConf.merge(config, updates) if updates else config

    if "goal_type" in config.model and "state_repr" not in config.model:
        # Goal-type descriptors were added later; load them from the config group.
        goals.with_default_descriptors(config.model)
    return config, wm_only
