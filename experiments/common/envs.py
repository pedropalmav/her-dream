"""Helpers to reach the base grid env behind the wrapper chain and read its
ground-truth state (agent pose, goal position)."""

import numpy as np


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
