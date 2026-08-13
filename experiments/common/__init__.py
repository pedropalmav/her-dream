"""Shared utilities for the interpretability experiment scripts.

Import the pieces you need, e.g.::

    from experiments.common import load_agent, posterior_rollout, random_policy
    from experiments.common import save_fig, experiment_outdir, dump_json
"""

from .cli import add_common_args
from .envs import agent_pose, fixed_goal_cfg, goal_pos, unwrap_env
from .grid import FORWARD, TURN_LEFT, TURN_RIGHT, bfs_plan, sample_cell, successors, walkable
from .io import dump_json, experiment_outdir, save_fig
from .loading import env_factory, load_agent
from .metrics import (
    entropy,
    groups_matched,
    pairwise_hamming,
    posterior_probs,
    reward_of,
    sym_kl,
    upper_triangular_mean,
)
from .obs import encode_goal_image, onehot_action, preprocess_obs, stack_obs
from .oracle import oracle_z_check
from .pairs import (
    FIXED_SEED,
    MIN_PLAN_LEN,
    N_PAIRS,
    PAIRS_SEED,
    generate_fixed_goal_pairs,
    generate_pairs,
    make_suite,
    suite_meta,
    sweep_pairs,
)
from .rollout import Step, actor_policy, posterior_rollout, random_policy, scripted_policy

__all__ = [
    "add_common_args",
    "agent_pose",
    "fixed_goal_cfg",
    "goal_pos",
    "unwrap_env",
    "FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "bfs_plan",
    "sample_cell",
    "successors",
    "walkable",
    "dump_json",
    "experiment_outdir",
    "save_fig",
    "env_factory",
    "load_agent",
    "entropy",
    "groups_matched",
    "pairwise_hamming",
    "posterior_probs",
    "reward_of",
    "sym_kl",
    "upper_triangular_mean",
    "encode_goal_image",
    "onehot_action",
    "preprocess_obs",
    "stack_obs",
    "oracle_z_check",
    "FIXED_SEED",
    "MIN_PLAN_LEN",
    "N_PAIRS",
    "PAIRS_SEED",
    "generate_fixed_goal_pairs",
    "generate_pairs",
    "make_suite",
    "suite_meta",
    "sweep_pairs",
    "Step",
    "actor_policy",
    "posterior_rollout",
    "random_policy",
    "scripted_policy",
]
