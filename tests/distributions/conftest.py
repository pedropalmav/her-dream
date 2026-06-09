import pytest
import torch

B, T, K, S = 4, 3, 8, 8


@pytest.fixture
def logits_1d():
    return torch.randn(B, K)


@pytest.fixture
def logits_2d():
    return torch.randn(B, T, K)


@pytest.fixture
def mode_tensor():
    return torch.randn(B, T, S)
