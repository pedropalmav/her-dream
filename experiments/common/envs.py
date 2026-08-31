"""Helpers to reach the base grid env behind the wrapper chain and read its
ground-truth state (agent pose, goal position), plus build a controllable
fixed-goal env config."""

import numpy as np
from omegaconf import OmegaConf


def fixed_goal_cfg(env_cfg, *, goal_pos=None, agent_start_pos=None, agent_start_dir=None):
    """A copy of `env_cfg` forced to the controllable fixed-goal env, with any of
    the green-square / agent-spawn positions overridden.

    Lets an experiment dictate the layout instead of reading it off a random reset:
    FixedGoal honours `goal_pos` / `agent_start_pos` on reset, so building an env
    from this config places the square and agent exactly where asked. Positions
    left as None keep whatever the run's config already had (so a fixed-goal run
    keeps its own `goal_pos`). Accepts an OmegaConf config or a plain dict; the
    original is never mutated.
    """
    raw = env_cfg if isinstance(env_cfg, dict) else OmegaConf.to_container(env_cfg, resolve=True)
    cfg = OmegaConf.create(dict(raw))
    OmegaConf.set_struct(cfg, False)
    cfg.task = "fixed-goal_"
    if goal_pos is not None:
        cfg.goal_pos_x, cfg.goal_pos_y = int(goal_pos[0]), int(goal_pos[1])
    if agent_start_pos is not None:
        cfg.agent_start_pos_x, cfg.agent_start_pos_y = int(agent_start_pos[0]), int(agent_start_pos[1])
    if agent_start_dir is not None:
        cfg.agent_start_dir = int(agent_start_dir)
    return cfg


def unwrap_env(env):
    """Peel the wrapper chain down to the base env (the one exposing agent_pos)."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def goal_pos(base_env) -> np.ndarray:
    """Green-square position as an int array.

    RandomGoal exposes `_goal_pos` (set in place_obj); FixedGoal uses `goal_pos`.
    """
    raw = getattr(base_env, "_goal_pos", None)
    if raw is None:
        raw = getattr(base_env, "goal_pos", None)
    if raw is None:
        raise AttributeError(f"No goal_pos / _goal_pos on {type(base_env).__name__}")
    return np.array(raw, dtype=np.int32)


def agent_pose(base_env) -> tuple:
    """`((ax, ay), dir)` of the agent in the base env."""
    ax, ay = base_env.agent_pos
    return (int(ax), int(ay)), int(base_env.agent_dir)
