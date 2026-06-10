import pytest
import torch
import torch.nn as nn

from networks.rmsnorm_2d import RMSNorm2D

from .conftest import B, C, H, W


class TestRMSNorm2D:
    @pytest.fixture
    def norm(self):
        return RMSNorm2D(C)

    def test_output_shape(self, norm):
        assert norm(torch.randn(B, C, H, W)).shape == (B, C, H, W)

    def test_is_rms_norm_subclass(self, norm):
        assert isinstance(norm, nn.RMSNorm)

    def test_weight_shape(self, norm):
        assert norm.weight.shape == (C,)

    def test_gradient_flows(self, norm):
        norm(torch.randn(B, C, H, W)).sum().backward()
        assert norm.weight.grad is not None

    def test_permute_applied(self, norm):
        out = norm(torch.randn(B, C, H, W))
        assert out.shape[1] == C
