import pytest
import torch
import torch.distributions as torchd

from her_dream.distributions.constructors import (
    binary,
    bounded_normal,
    identity,
    kl,
    mse,
    multi_onehot,
    normal_std_fixed,
    onehot,
    symexp_twohot,
    symlog_mse,
)
from her_dream.distributions.distributions import Bound, MSEDist, MultiOneHotDist, OneHotDist, SymlogDist, TwoHot

from .conftest import B, K, T


class TestBoundedNormal:
    def test_returns_independent_normal(self):
        x = torch.randn(B, K * 2)
        dist = bounded_normal(x, min_std=0.1, max_std=1.0)
        assert isinstance(dist, torchd.independent.Independent)

    def test_output_sample_shape(self):
        x = torch.randn(B, K * 2)
        dist = bounded_normal(x, min_std=0.1, max_std=1.0)
        sample = dist.rsample()
        assert sample.shape == (B, K)

    def test_std_in_range(self):
        x = torch.randn(B, K * 2)
        min_std, max_std = 0.1, 1.0
        dist = bounded_normal(x, min_std=min_std, max_std=max_std)
        std = dist.base_dist.scale
        assert torch.all(std >= min_std - 1e-6)
        assert torch.all(std <= max_std + 1e-6)


class TestNormalStdFixed:
    def test_returns_bound(self):
        mean = torch.randn(B, K)
        std = torch.ones(B, K) * 0.5
        dist = normal_std_fixed(mean, std)
        assert isinstance(dist, Bound)

    def test_mode_in_unit_interval(self):
        mean = torch.randn(B, K) * 10
        std = torch.ones(B, K)
        dist = normal_std_fixed(mean, std)
        mode = dist.mode
        assert torch.all(mode >= -1.0 - 1e-6)
        assert torch.all(mode <= 1.0 + 1e-6)


class TestOnehot:
    def test_returns_onehot_dist(self):
        logits = torch.randn(B, K)
        dist = onehot(logits, unimix_ratio=0.0)
        assert isinstance(dist, OneHotDist)


class TestMultiOnehot:
    def test_returns_multi_onehot_dist(self):
        shape = (K, K)
        logits = torch.randn(B, sum(shape))
        dist = multi_onehot(logits, unimix_ratio=0.0, shape=shape)
        assert isinstance(dist, MultiOneHotDist)


class TestBinary:
    def test_returns_independent_bernoulli(self):
        logits = torch.randn(B, K)
        dist = binary(logits)
        assert isinstance(dist, torchd.independent.Independent)
        assert isinstance(dist.base_dist, torchd.Bernoulli)


class TestSymexpTwohot:
    def test_odd_bin_count(self):
        logits = torch.randn(B, 5)
        dist = symexp_twohot(logits, bin_num=5)
        assert isinstance(dist, TwoHot)
        assert len(dist.bins) == 5

    def test_even_bin_count(self):
        logits = torch.randn(B, 4)
        dist = symexp_twohot(logits, bin_num=4)
        assert isinstance(dist, TwoHot)
        assert len(dist.bins) == 4

    def test_bins_are_symmetric(self):
        for bin_num in [5, 6, 255]:
            logits = torch.randn(B, bin_num)
            dist = symexp_twohot(logits, bin_num=bin_num)
            bins = dist.bins
            for i in range(len(bins) // 2):
                assert bins[i].item() == pytest.approx(-bins[-(i + 1)].item(), abs=1e-5)

    def test_bins_contain_zero_for_odd(self):
        bin_num = 5
        logits = torch.randn(B, bin_num)
        dist = symexp_twohot(logits, bin_num=bin_num)
        center = dist.bins[(bin_num - 1) // 2]
        assert center.item() == pytest.approx(0.0, abs=1e-6)

    def test_default_255_bins(self):
        logits = torch.randn(B, 255)
        dist = symexp_twohot(logits, bin_num=255)
        assert len(dist.bins) == 255


class TestSymlogMse:
    def test_returns_symlog_dist(self):
        logits = torch.randn(B, K)
        dist = symlog_mse(logits)
        assert isinstance(dist, SymlogDist)


class TestMse:
    def test_returns_mse_dist(self):
        logits = torch.randn(B, K)
        dist = mse(logits)
        assert isinstance(dist, MSEDist)


class TestIdentity:
    def test_returns_input_unchanged(self):
        logits = torch.randn(B, K)
        out = identity(logits)
        assert out is logits


class TestKl:
    def test_kl_same_dist_is_zero(self):
        logits = torch.randn(B, T, K)
        result = kl(logits, logits)
        assert torch.allclose(result, torch.zeros(B, T), atol=1e-5)

    def test_kl_non_negative(self):
        left = torch.randn(B, T, K)
        right = torch.randn(B, T, K)
        result = kl(left, right)
        assert torch.all(result >= -1e-6)

    def test_kl_output_shape(self):
        left = torch.randn(B, T, K)
        right = torch.randn(B, T, K)
        result = kl(left, right)
        assert result.shape == (B, T)

    def test_kl_matches_torch(self):
        left = torch.randn(B, K)
        right = torch.randn(B, K)
        expected = torchd.kl_divergence(
            torchd.Categorical(logits=left),
            torchd.Categorical(logits=right),
        )
        actual = kl(left, right)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_kl_asymmetric(self):
        left = torch.randn(B, K)
        right = torch.randn(B, K)
        # KL(p||q) != KL(q||p) in general
        kl_pq = kl(left, right)
        kl_qp = kl(right, left)
        assert not torch.allclose(kl_pq, kl_qp)
