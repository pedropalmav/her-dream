"""Fixtures for exercising `ActorCritic` on its own, without a `Dreamer`.

The sub-module only needs a config node, a feature size, an action space and a
goal size, so the tests build it directly from the real Hydra tree — no encoder,
RSSM, optimizer or replay buffer involved.
"""

from types import SimpleNamespace

import pytest
import torch

from her_dream.actor_critic import ActorCritic
from tests.dreamer.conftest import act_discrete, build_model_config

F = 24  # feature size
G = 16  # goal size (flattened)
B, T_IMAG = 3, 4  # batch, imagination horizon
A = 5  # discrete action dim


def make_ac(goal_type="full", act=None, *, name="", **cfg_overrides):
    """Build a real `ActorCritic` from the composed Hydra config."""
    cfg = build_model_config(goal_type=goal_type, **cfg_overrides)
    return ActorCritic(cfg.model, F, act or act_discrete(A), G, name=name)


@pytest.fixture
def ac():
    return make_ac()


@pytest.fixture
def imag_batch():
    """A detached imagination rollout, as `Dreamer._cal_grad` hands it over."""
    return SimpleNamespace(
        feat=torch.randn(B, T_IMAG, F),
        action=torch.nn.functional.one_hot(torch.zeros(B, T_IMAG, dtype=torch.long), A).float(),
        reward=torch.full((B, T_IMAG, 1), -1.0),
        cont=torch.ones(B, T_IMAG, 1),
        goal=torch.randn(B, G),
    )


@pytest.fixture
def replay_batch():
    """An attached replay batch, as the replay-value branch hands it over."""
    return SimpleNamespace(
        feat=torch.randn(B, T_IMAG, F, requires_grad=True),
        goal=torch.randn(B, T_IMAG, G),
        last=torch.zeros(B, T_IMAG, 1),
        term=torch.zeros(B, T_IMAG, 1),
        reward=torch.full((B, T_IMAG, 1), -1.0),
        boot=torch.zeros(B, T_IMAG, 1),
    )
