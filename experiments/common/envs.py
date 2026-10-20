"""Helpers to reach the base grid env behind the wrapper chain and read its
ground-truth state (agent pose, goal position), plus build a config whose layout
the experiment controls instead of the env's sampler."""

import numpy as np
from omegaconf import OmegaConf

# Suites whose layout includes hazards, so pinning the green square and the agent
# is not enough to reproduce a state.
_LAVA_SUITES = ("lava-grid",)


def pinned_layout_cfg(env_cfg, *, goal_pos=None, agent_start_pos=None, agent_start_dir=None, lava_pos=None):
    """A copy of `env_cfg` with the layout pinned: green square, agent spawn, lava.

    Lets an experiment dictate the layout instead of reading it off a random reset.
    Building an env from the result places everything exactly where asked. Positions
    left as None keep whatever the run's config already had (so a fixed-goal run
    keeps its own `goal_pos`). Accepts an OmegaConf config or a plain dict; the
    original is never mutated.

    **Which env it builds depends on the run.** For the goal-grid suites
    (`random-goal` / `fixed-goal`) the task is forced to `fixed-goal_`: those two
    render identically and differ only in where the square goes, so the pinnable one
    is the controllable stand-in for both. A `lava-grid` run keeps its own suite —
    its world model was trained on observations that contain lava, and measuring it
    on a lava-free env would be measuring it off-distribution. `lava_pos` is what
    makes that env controllable in turn, so it is required rather than optional
    there: without it the lava resamples every reset and nothing is pinned at all.
    """
    raw = env_cfg if isinstance(env_cfg, dict) else OmegaConf.to_container(env_cfg, resolve=True)
    cfg = OmegaConf.create(dict(raw))
    OmegaConf.set_struct(cfg, False)

    if suite_of(raw) in _LAVA_SUITES:
        if lava_pos is None:
            raise ValueError(
                "pinned_layout_cfg needs lava_pos on a lava run: the lava resamples on every reset, "
                "so pinning only the square and the agent does not pin the layout. Pass the cells to "
                "use (they must avoid the square and the agent spawn)."
            )
        cfg.lava_pos = [[int(x), int(y)] for x, y in lava_pos]
    else:
        cfg.task = "fixed-goal_"

    if goal_pos is not None:
        cfg.goal_pos_x, cfg.goal_pos_y = int(goal_pos[0]), int(goal_pos[1])
    if agent_start_pos is not None:
        cfg.agent_start_pos_x, cfg.agent_start_pos_y = int(agent_start_pos[0]), int(agent_start_pos[1])
        # A run trained with agent_start_random=True would otherwise ignore the pinned
        # spawn (envs/__init__.py resamples it), and the agent starts somewhere else.
        cfg.agent_start_random = False
    if agent_start_dir is not None:
        cfg.agent_start_dir = int(agent_start_dir)
    return cfg


def suite_of(env_cfg) -> str:
    """The suite half of `env.task` (the part before the first underscore)."""
    raw = env_cfg if isinstance(env_cfg, dict) else OmegaConf.to_container(env_cfg, resolve=True)
    return str(raw.get("task", "")).split("_", 1)[0]


def has_lava(env_cfg) -> bool:
    """Whether this run's env carries lava, and so needs its layout pinned in full."""
    return suite_of(env_cfg) in _LAVA_SUITES


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


def lava_pos(base_env) -> list:
    """This episode's lava cells as a list of `(x, y)`, empty on an env without lava."""
    return [(int(x), int(y)) for x, y in getattr(base_env, "lava_positions", [])]
