import pytest
import torch

from networks.projector import Projector

from .conftest import B, T


class TestProjector:
    @pytest.fixture
    def projector(self):
        return Projector(32, 16)

    def test_output_shape(self, projector):
        assert projector(torch.randn(B, T, 32)).shape == (B, T, 16)

    def test_no_bias(self, projector):
        assert projector.w.bias is None

    def test_weight_shape(self, projector):
        assert projector.w.weight.shape == (16, 32)

    def test_weight_init_applied(self, projector):
        assert not torch.all(projector.w.weight == 0)

    def test_gradient_flows(self, projector):
        projector(torch.randn(B, T, 32)).sum().backward()
        assert projector.w.weight.grad is not None
