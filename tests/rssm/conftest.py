from types import SimpleNamespace

import pytest
import torch

from her_dream.rssm import RSSM

B, T = 2, 3
S, K = 8, 8  # stoch groups, categories per group → flat_stoch = 64
D = 32  # deter dim (must be divisible by blocks=4)
A = 4  # action dim
E = 16  # embed size


def make_rssm_config(**overrides):
    defaults = dict(
        stoch=S,
        discrete=K,
        deter=D,
        hidden=16,
        obs_layers=1,
        img_layers=1,
        dyn_layers=1,
        blocks=4,
        act="SiLU",
        unimix_ratio=0.01,
        initial="learned",
        device="cpu",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def rssm():
    return RSSM(make_rssm_config(), embed_size=E, act_dim=A)


@pytest.fixture
def stoch():
    return torch.randn(B, S, K)


@pytest.fixture
def deter():
    return torch.randn(B, D)


@pytest.fixture
def action():
    return torch.randn(B, A)


@pytest.fixture
def embed():
    return torch.randn(B, E)


@pytest.fixture
def reset():
    return torch.zeros(B, dtype=torch.bool)


@pytest.fixture
def embed_seq():
    return torch.randn(B, T, E)


@pytest.fixture
def action_seq():
    return torch.randn(B, T, A)


@pytest.fixture
def reset_seq():
    return torch.zeros(B, T, dtype=torch.bool)


@pytest.fixture
def logit():
    return torch.randn(B, S, K)


@pytest.fixture
def post_logit():
    return torch.randn(B, T, S, K)


@pytest.fixture
def prior_logit():
    return torch.randn(B, T, S, K)


@pytest.fixture
def initial(rssm):
    return rssm.initial(B)
