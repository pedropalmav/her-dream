"""Fixtures for exercising `Disagreement` on its own.

The ensemble only needs a config node, a feature size, an action dim and an RSSM
for its target sizes, so the tests build it directly from the real Hydra tree —
no `Dreamer`, encoder or replay buffer involved.
"""

from types import SimpleNamespace

import pytest
import torch

from her_dream.plan2explore import Disagreement
from tests.dreamer.conftest import build_model_config

B, T = 2, 6  # batch, sequence length
A = 5  # action dim
S, K, D = 8, 8, 32  # rssm stoch groups / categories / deter
F = S * K + D  # feature size


class _StubRSSM:
    """Only the three size attributes `Disagreement` reads."""

    flat_stoch = S * K
    _deter = D


def make_disag(**cfg_overrides):
    """Build a real `Disagreement` from the composed Hydra config."""
    cfg = build_model_config(**cfg_overrides)
    return Disagreement(cfg.model, F, A, _StubRSSM())


@pytest.fixture
def disag():
    return make_disag()


@pytest.fixture
def batch():
    """A replayed sequence, shaped as `_cal_grad` hands it over."""
    return SimpleNamespace(
        feat=torch.randn(B, T, F),
        action=torch.nn.functional.one_hot(torch.randint(0, A, (B, T)), A).float(),
        stoch=torch.randn(B, T, S, K),
        deter=torch.randn(B, T, D),
    )
