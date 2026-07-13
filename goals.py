"""Goal-type behavior descriptors.

Each `goal_type` drives three orthogonal axes of behavior, previously branched on the
`goal_type` string at ~13 sites across the codebase:

- ``state_repr`` — what the reward function's state argument is:
  ``"stoch"`` (a sampled one-hot), ``"dist"`` (``rssm.get_dist(logit)``) or ``"logit"``.
- ``goal_repr`` — what the goal tensor is (it must match ``state_repr``'s comparison):
  ``"sample"`` (a stoch one-hot), ``"argmax"`` (``one_hot(argmax(logit))``) or ``"logit"``.
- ``scope`` — the actor/critic goal input: ``"first_row"`` (a single group, size ``K``)
  or ``"full"`` (all groups, size ``S*K``).

The descriptor values live in each ``configs/goal_type/<type>.yaml`` and are read into a
``GoalSpec`` with :func:`make_goal_spec`. Every consumer builds its spec once from its own
sub-config and then routes through the helpers below, so adding a goal type is a matter of
adding a reward function (``rewards.py``) and a config file — no new branches.
"""

import os
from dataclasses import dataclass

import torch.nn.functional as F
from omegaconf import OmegaConf

_GOAL_TYPE_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "goal_type")

_STATE_REPRS = {"stoch", "dist", "logit"}
_GOAL_REPRS = {"sample", "argmax", "logit"}
_SCOPES = {"first_row", "full"}


@dataclass(frozen=True)
class GoalSpec:
    goal_type: str
    state_repr: str  # "stoch" | "dist" | "logit"  — reward fn's state arg
    goal_repr: str  # "sample" | "argmax" | "logit" — the goal tensor
    scope: str  # "first_row" | "full"          — actor/critic goal input
    uses_threshold: bool = False  # log_prob only — appends a threshold one-hot


def make_goal_spec(config) -> GoalSpec:
    """Build a `GoalSpec` from a (sub-)config carrying the descriptor keys.

    `config` is any config section that received the fanned-out descriptors
    (`model`, `env`, `buffer`, `trainer`). `uses_threshold` defaults to False for
    sections that don't carry it (only `model` needs it).
    """
    spec = GoalSpec(
        goal_type=config.goal_type,
        state_repr=config.state_repr,
        goal_repr=config.goal_repr,
        scope=config.scope,
        uses_threshold=bool(getattr(config, "uses_threshold", False)),
    )
    if spec.state_repr not in _STATE_REPRS:
        raise ValueError(f"Invalid state_repr {spec.state_repr!r} for goal_type {spec.goal_type!r}")
    if spec.goal_repr not in _GOAL_REPRS:
        raise ValueError(f"Invalid goal_repr {spec.goal_repr!r} for goal_type {spec.goal_type!r}")
    if spec.scope not in _SCOPES:
        raise ValueError(f"Invalid scope {spec.scope!r} for goal_type {spec.goal_type!r}")
    return spec


def stashes_logit(spec: GoalSpec) -> bool:
    """Whether `act()` must stash the RSSM logit for later reward/goal use.

    Needed iff the reward compares against something other than the sampled stoch.
    """
    return spec.state_repr != "stoch"


def reward_state(spec: GoalSpec, *, stoch, logit, rssm):
    """Return the reward function's state argument for this goal type (Axis A)."""
    if spec.state_repr == "dist":
        return rssm.get_dist(logit)
    if spec.state_repr == "logit":
        return logit
    return stoch


def goal_from_latent(spec: GoalSpec, *, stoch, logit):
    """Turn a latent (stoch sample + logit) into the goal tensor for this type (Axis B)."""
    if spec.goal_repr == "argmax":
        K = logit.shape[-1]
        return F.one_hot(logit.argmax(dim=-1), num_classes=K).float()
    if spec.goal_repr == "logit":
        return logit
    return stoch


def goal_size(spec: GoalSpec, rssm) -> int:
    """Flattened goal size fed to the actor/critic alongside the RSSM feature (Axis C)."""
    return rssm._discrete if spec.scope == "first_row" else rssm.flat_stoch


def default_descriptors(goal_type: str) -> dict:
    """Load a goal type's descriptor keys from its ``configs/goal_type/<type>.yaml``.

    Used to backfill configs saved before the descriptors existed (e.g. the zflow /
    viz loaders reading an older run's ``.hydra/config.yaml``), keeping the config
    files the single source of truth rather than a hardcoded map.
    """
    cfg = OmegaConf.load(os.path.join(_GOAL_TYPE_CONFIG_DIR, f"{goal_type}.yaml"))
    return {
        "state_repr": cfg.state_repr,
        "goal_repr": cfg.goal_repr,
        "scope": cfg.scope,
        "uses_threshold": bool(cfg.get("uses_threshold", False)),
    }
