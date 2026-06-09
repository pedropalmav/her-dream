from types import SimpleNamespace

import pytest
import torch
import torch.distributions as torchd

from distributions.distributions import MultiOneHotDist, OneHotDist, TwoHot
from networks.mlp_head import MLPHead

from .conftest import B, T, make_mlphead_config

INP = 32


def _fwd(head):
    return head(torch.randn(B, T, INP))


def _bounded_normal_cfg(outscale=1.0):
    return SimpleNamespace(
        act="SiLU",
        symlog_inputs=False,
        device="cpu",
        layers=1,
        units=32,
        name="test",
        shape=[4],
        outscale=outscale,
        dist=SimpleNamespace(name="bounded_normal", min_std=0.1, max_std=1.0),
    )


class TestMLPHeadBoundedNormal:
    @pytest.fixture
    def head(self):
        return MLPHead(make_mlphead_config("bounded_normal"), inp_dim=INP)

    def test_last_layer_size(self, head):
        assert head.last.out_features == 4 * 2

    def test_forward_returns_independent_normal(self, head):
        assert isinstance(_fwd(head), torchd.independent.Independent)

    def test_forward_sample_shape(self, head):
        assert _fwd(head).rsample().shape == (B, T, 4)


class TestMLPHeadOnehot:
    @pytest.fixture
    def head(self):
        return MLPHead(make_mlphead_config("onehot"), inp_dim=INP)

    def test_last_layer_size(self, head):
        assert head.last.out_features == 8

    def test_forward_returns_onehot_dist(self, head):
        assert isinstance(_fwd(head), OneHotDist)


class TestMLPHeadMultiOnehot:
    @pytest.fixture
    def head(self):
        return MLPHead(make_mlphead_config("multi_onehot"), inp_dim=INP)

    def test_last_layer_size(self, head):
        assert head.last.out_features == 4 + 4

    def test_forward_returns_multi_onehot_dist(self, head):
        assert isinstance(_fwd(head), MultiOneHotDist)


class TestMLPHeadSymexpTwohot:
    @pytest.fixture
    def head(self):
        return MLPHead(make_mlphead_config("symexp_twohot"), inp_dim=INP)

    def test_last_layer_size(self, head):
        assert head.last.out_features == 5

    def test_forward_returns_twohot_dist(self, head):
        assert isinstance(_fwd(head), TwoHot)


class TestMLPHeadBinary:
    @pytest.fixture
    def head(self):
        return MLPHead(make_mlphead_config("binary"), inp_dim=INP)

    def test_last_layer_size(self, head):
        assert head.last.out_features == 4

    def test_forward_returns_independent_bernoulli(self, head):
        dist = _fwd(head)
        assert isinstance(dist, torchd.independent.Independent)
        assert isinstance(dist.base_dist, torchd.Bernoulli)


class TestMLPHeadIdentity:
    @pytest.fixture
    def head(self):
        return MLPHead(make_mlphead_config("identity"), inp_dim=INP)

    def test_last_layer_size(self, head):
        assert head.last.out_features == 4

    def test_forward_returns_tensor(self, head):
        out = _fwd(head)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (B, T, 4)


class TestMLPHeadEdgeCases:
    def test_unknown_dist_raises(self):
        cfg = SimpleNamespace(
            act="SiLU",
            symlog_inputs=False,
            device="cpu",
            layers=1,
            units=32,
            name="test",
            shape=[4],
            outscale=1.0,
            dist=SimpleNamespace(name="mse"),
        )
        with pytest.raises(NotImplementedError):
            MLPHead(cfg, inp_dim=INP)

    def test_outscale_not_one_scales_weight(self):
        torch.manual_seed(42)
        head_one = MLPHead(_bounded_normal_cfg(outscale=1.0), inp_dim=INP)
        torch.manual_seed(42)
        head_half = MLPHead(_bounded_normal_cfg(outscale=0.5), inp_dim=INP)
        assert torch.allclose(head_half.last.weight, head_one.last.weight * 0.5)

    def test_outscale_one_skips_scaling(self):
        head = MLPHead(_bounded_normal_cfg(outscale=1.0), inp_dim=INP)
        assert head._outscale == 1.0
        assert not torch.all(head.last.weight == 0)
