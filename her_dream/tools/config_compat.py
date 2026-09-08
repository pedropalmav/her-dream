"""Backfill checkpoint configs saved before newer keys existed.

Older runs' ``.hydra/config.yaml`` files (loaded by ``zflow/serve.py`` and viz)
predate goal-conditioning and/or the goal-type descriptor keys. `backfill_config`
fills in just enough for the current ``Dreamer`` constructor and ``make_goal_spec``
to run.
"""

from omegaconf import OmegaConf

from her_dream import goals

# Optional-module keys added after the archived runs were trained. All off, which
# matches those checkpoints' state dicts: `Dreamer` builds no reward/continue head
# and no Plan2Explore explorer or ensemble.
_OPTIONAL_HEAD_DEFAULTS = {
    "use_reward_head": False,
    "use_cont_head": False,
    "imag_reward_source": "goal",
    "plan2explore": False,
}

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
    model_defaults = {}
    if "goal_imag_horizon" not in config.model:
        model_defaults["goal_imag_horizon"] = int(config.model.imag_horizon)
    # The optional reward/continue heads and the imagination reward source
    # postdate every archived run, so default them off for any config that
    # predates them (unlike the goal keys below, this applies to *all* old
    # configs, not just the pre-goal-conditioning ones).
    for key, default in _OPTIONAL_HEAD_DEFAULTS.items():
        if key not in config.model:
            model_defaults[key] = default
    if model_defaults:
        updates["model"] = model_defaults

    wm_only = force_wm_only or ("goal_type" not in config)
    if wm_only:
        # Dummy goal-conditioning keys so main's Dreamer can be *constructed*.
        # These only size the actor/value heads, which are neither loaded nor
        # run in WM-only mode; make_reward is skipped entirely.
        top = {k: v for k, v in _WM_ONLY_GOAL_DEFAULTS.items() if k not in config}
        model_missing = {k: v for k, v in _WM_ONLY_GOAL_DEFAULTS.items() if k not in config.model}
        # Deep-merge: preserves the model-level backfills above.
        updates = OmegaConf.merge(updates, {**top, "model": model_missing})

    config = OmegaConf.merge(config, updates) if updates else config

    if "goal_type" in config.model and "state_repr" not in config.model:
        # Goal-type descriptors were added later; load them from the config group
        # and fan them into every section that carries them (mirrors configs.yaml).
        # Notably `env` needs them too: the GoalConditioned wrapper reads the spec
        # from config.env, so backfilling only `model` left `env.state_repr` missing
        # and old goal_type checkpoints failed to serve in zflow.
        d = goals.default_descriptors(config.model.goal_type)
        core = {"state_repr": d["state_repr"], "goal_repr": d["goal_repr"], "scope": d["scope"]}
        section_patch = {}
        for section in ("env", "model", "buffer", "trainer"):
            if section in config and "state_repr" not in config[section]:
                # Only `model` carries uses_threshold in configs.yaml.
                section_patch[section] = {**core, "uses_threshold": d["uses_threshold"]} if section == "model" else core
        config = OmegaConf.merge(config, section_patch)
    return config, wm_only
