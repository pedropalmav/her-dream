"""Fixtures for exercising `TemporalDistance` on its own."""

import pytest
import torch

from her_dream.temporal_distance import TemporalDistance
from tests.dreamer.conftest import build_model_config

B, T = 6, 8  # trajectories, steps per trajectory
S, K = 8, 8  # stoch groups / categories
L = S * K  # flattened latent size


def make_td(**cfg_overrides):
    """Build a real `TemporalDistance` from the composed Hydra config."""
    cfg = build_model_config(**cfg_overrides)
    return TemporalDistance(cfg.model, L)


@pytest.fixture
def td():
    return make_td()


@pytest.fixture
def stoch():
    """A batch of imagined trajectories, as `_cal_grad` hands them over."""
    return torch.randn(B, T, S, K)
