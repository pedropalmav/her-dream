import pytest
import torch

from deter import Deter

DETER = 32
STOCH = 32
ACT_DIM = 4
HIDDEN = 16
BLOCKS = 4
DYNLAYERS = 1
S, K = 8, 4
B = 2


def make_deter(**overrides):
    defaults = dict(
        deter=DETER,
        stoch=STOCH,
        act_dim=ACT_DIM,
        hidden=HIDDEN,
        blocks=BLOCKS,
        dynlayers=DYNLAYERS,
    )
    defaults.update(overrides)
    return Deter(**defaults)


@pytest.fixture
def model():
    return make_deter()


@pytest.fixture
def stoch_input():
    return torch.randn(B, S, K)


@pytest.fixture
def deter_input():
    return torch.randn(B, DETER)


@pytest.fixture
def action():
    return torch.randn(B, ACT_DIM)
