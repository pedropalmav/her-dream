import pytest
import torch

from her_dream.networks.block_linear import BlockLinear


class TestBlockLinear:
    @pytest.fixture
    def layer(self):
        return BlockLinear(32, 32, 4)

    def test_weight_shape(self, layer):
        assert layer.weight.shape == (32 // 4, 32 // 4, 4)

    def test_bias_shape(self, layer):
        assert layer.bias.shape == (32,)

    def test_forward_2d(self, layer):
        assert layer(torch.randn(8, 32)).shape == (8, 32)

    def test_forward_3d(self, layer):
        assert layer(torch.randn(2, 3, 32)).shape == (2, 3, 32)

    def test_bias_added(self, layer):
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.fill_(1.0)
        assert torch.allclose(layer(torch.randn(4, 32)), torch.ones(4, 32))

    def test_gradient_flows(self, layer):
        layer(torch.randn(4, 32)).sum().backward()
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None

    def test_outscale_stored(self):
        assert BlockLinear(32, 32, 4, outscale=2.0).outscale == 2.0

    def test_different_in_out(self):
        layer = BlockLinear(32, 64, 4)
        assert layer(torch.randn(3, 32)).shape == (3, 64)
