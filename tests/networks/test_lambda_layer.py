import torch
import torch.nn as nn

from her_dream.networks.lambda_layer import LambdaLayer


class TestLambdaLayer:
    def test_forward_doubles(self):
        layer = LambdaLayer(lambda x: x * 2)
        x = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(layer(x), x * 2)

    def test_forward_identity(self):
        layer = LambdaLayer(lambda x: x)
        x = torch.randn(4, 8)
        assert torch.equal(layer(x), x)

    def test_lambd_stored(self):
        def fn(x):
            return x + 1

        layer = LambdaLayer(fn)
        assert layer.lambd is fn

    def test_is_nn_module(self):
        layer = LambdaLayer(lambda x: x)
        assert isinstance(layer, nn.Module)

    def test_gradient_propagates(self):
        layer = LambdaLayer(lambda x: x * 2)
        x = torch.randn(4, requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
