import torch

import her_dream.distributions as dists
from her_dream.networks.mlp import MLP

from .conftest import B, T, make_mlp_config


class TestMLP:
    def test_out_dim(self):
        cfg = make_mlp_config(units=32)
        mlp = MLP(cfg, inp_dim=16)
        assert mlp.out_dim == 32

    def test_output_shape(self):
        cfg = make_mlp_config(layers=2, units=32)
        mlp = MLP(cfg, inp_dim=16)
        x = torch.randn(B, T, 16)
        assert mlp(x).shape == (B, T, 32)

    def test_symlog_false_branch(self):
        cfg = make_mlp_config(symlog_inputs=False, layers=1)
        mlp = MLP(cfg, inp_dim=8)
        mlp.eval()
        x = torch.randn(B, T, 8)
        with torch.no_grad():
            out = mlp(x)
            expected = mlp.layers(x)
        assert torch.allclose(out, expected)

    def test_symlog_true_branch(self):
        cfg = make_mlp_config(symlog_inputs=True, layers=1)
        mlp = MLP(cfg, inp_dim=8)
        mlp.eval()
        x = torch.randn(B, T, 8) * 100
        with torch.no_grad():
            out = mlp(x)
            expected = mlp.layers(dists.symlog(x))
        assert torch.allclose(out, expected)

    def test_layer_count(self):
        cfg = make_mlp_config(layers=3, units=32)
        mlp = MLP(cfg, inp_dim=16)
        # 3 layers × 3 modules (Linear + RMSNorm + act) = 9
        assert len(list(mlp.layers.children())) == 9

    def test_gradient_flows(self):
        cfg = make_mlp_config(layers=1, units=16)
        mlp = MLP(cfg, inp_dim=8)
        x = torch.randn(B, T, 8)
        mlp(x).sum().backward()
        first_linear = list(mlp.layers.children())[0]
        assert first_linear.weight.grad is not None
