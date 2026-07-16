import pytest
import torch
import torch.distributions as torchd

from her_dream.distributions.distributions import Bound, MSEDist, MultiOneHotDist, OneHotDist, SymlogDist, TwoHot
from her_dream.distributions.functional import symexp, symlog

from .conftest import B, K, T


class TestOneHotDist:
    def test_mode_is_one_hot(self, logits_1d):
        dist = OneHotDist(logits_1d)
        mode = dist.mode
        assert mode.shape == logits_1d.shape
        assert torch.all(mode.sum(dim=-1) == 1)
        assert set(mode.unique().tolist()).issubset({0.0, 1.0})

    def test_mode_argmax_matches_logits(self, logits_1d):
        dist = OneHotDist(logits_1d)
        mode = dist.mode
        expected = torch.argmax(logits_1d, dim=-1)
        actual = torch.argmax(mode.detach(), dim=-1)
        assert torch.equal(expected, actual)

    def test_mode_has_gradient(self, logits_1d):
        logits = logits_1d.requires_grad_(True)
        dist = OneHotDist(logits)
        loss = (dist.mode * logits).sum()
        loss.backward()
        assert logits.grad is not None

    def test_unimix_blends_uniform(self, logits_1d):
        unimix = 0.5
        dist = OneHotDist(logits_1d, unimix_ratio=unimix)
        probs = dist.probs
        assert torch.all(probs >= unimix / K - 1e-6)

    def test_rsample_is_hard_one_hot(self, logits_1d):
        dist = OneHotDist(logits_1d)
        sample = dist.rsample()
        assert sample.shape == logits_1d.shape
        assert torch.allclose(sample.sum(dim=-1), torch.ones(B))
        assert torch.all((sample == 0) | (sample == 1))

    def test_rsample_shape(self, logits_1d):
        dist = OneHotDist(logits_1d)
        assert dist.rsample().shape == (B, K)

    def test_rsample_differentiable(self, logits_1d):
        logits = logits_1d.requires_grad_(True)
        dist = OneHotDist(logits)
        loss = dist.rsample().sum()
        loss.backward()
        assert logits.grad is not None

    def test_rsample_temperature_accepted(self, logits_1d):
        dist = OneHotDist(logits_1d)
        out = dist.rsample(temperature=0.5)
        assert out.shape == logits_1d.shape

    def test_sample_raises(self, logits_1d):
        dist = OneHotDist(logits_1d)
        with pytest.raises(NotImplementedError):
            dist.sample()


class TestMultiOneHotDist:
    def setup_method(self):
        self.shape = (K, K)  # two equal-sized groups
        self.total = sum(self.shape)

    def make_dist(self):
        logits = torch.randn(B, self.total)
        return MultiOneHotDist(logits, shape=self.shape)

    def test_mode_shape(self):
        dist = self.make_dist()
        assert dist.mode.shape == (B, self.total)

    def test_rsample_shape(self):
        dist = self.make_dist()
        assert dist.rsample().shape == (B, self.total)

    def test_log_prob_shape(self):
        dist = self.make_dist()
        value = dist.rsample()
        assert dist.log_prob(value).shape == (B,)

    def test_entropy_shape(self):
        dist = self.make_dist()
        assert dist.entropy().shape == (B,)

    def test_sample_raises(self):
        dist = self.make_dist()
        with pytest.raises(NotImplementedError):
            dist.sample()


class TestTwoHot:
    def make_dist(self, n_bins=5):
        logits = torch.randn(B, n_bins)
        bins = torch.linspace(-2.0, 2.0, n_bins)
        return TwoHot(logits, bins), bins

    def test_init_mismatch_raises(self):
        logits = torch.randn(B, 5)
        bins = torch.linspace(-1, 1, 4)
        with pytest.raises(AssertionError):
            TwoHot(logits, bins)

    def test_mode_odd_bins_shape(self):
        dist, _ = self.make_dist(n_bins=5)
        assert dist.mode.shape == (B, 1)

    def test_mode_even_bins_shape(self):
        dist, _ = self.make_dist(n_bins=4)
        assert dist.mode.shape == (B, 1)

    def test_log_prob_shape(self):
        dist, _ = self.make_dist(n_bins=5)
        target = torch.zeros(B, 1)
        assert dist.log_prob(target).shape == (B,)

    def test_log_prob_target_outside_range(self):
        dist, bins = self.make_dist(n_bins=5)
        far_above = torch.full((B, 1), bins[-1].item() + 100.0)
        far_below = torch.full((B, 1), bins[0].item() - 100.0)
        # should not crash and return finite values
        assert torch.all(torch.isfinite(dist.log_prob(far_above)))
        assert torch.all(torch.isfinite(dist.log_prob(far_below)))

    def test_log_prob_dtype_assertion(self):
        dist, _ = self.make_dist(n_bins=5)
        target = torch.zeros(B, 1, dtype=torch.float64)
        with pytest.raises(AssertionError):
            dist.log_prob(target)

    def test_log_prob_target_on_bin_equal_weight(self):
        n_bins = 5
        bins = torch.linspace(-2.0, 2.0, n_bins)
        logits = torch.zeros(B, n_bins)
        dist = TwoHot(logits, bins)
        # target on bins[2] (the center bin for odd n_bins=5) → below == above → equal weight
        target = bins[2].expand(B, 1).clone()
        log_p = dist.log_prob(target)
        assert torch.all(torch.isfinite(log_p))

    def test_mode_with_squash_unsquash(self):
        n_bins = 5
        bins = torch.linspace(-2.0, 2.0, n_bins)
        logits = torch.zeros(B, n_bins)
        dist = TwoHot(logits, bins, squash=symlog, unsquash=symexp)
        out = dist.mode
        assert out.shape == (B, 1)
        assert torch.all(torch.isfinite(out))


class TestMSEDist:
    def test_mode_property(self, mode_tensor):
        dist = MSEDist(mode_tensor)
        assert torch.equal(dist.mode, mode_tensor.float())

    def test_mean_property(self, mode_tensor):
        dist = MSEDist(mode_tensor)
        assert torch.equal(dist.mean, mode_tensor.float())

    def test_log_prob_perfect_prediction_is_zero(self, mode_tensor):
        dist = MSEDist(mode_tensor)
        log_p = dist.log_prob(mode_tensor.float())
        assert torch.allclose(log_p, torch.zeros(B, T))

    def test_log_prob_sum_aggregation(self):
        mode = torch.ones(B, T, 4)
        value = torch.zeros(B, T, 4)
        dist = MSEDist(mode, agg="sum")
        # distance per element = 1, sum over 4 dims = 4, negative → -4
        assert torch.allclose(dist.log_prob(value), torch.full((B, T), -4.0))

    def test_log_prob_mean_aggregation(self):
        mode = torch.ones(B, T, 4)
        value = torch.zeros(B, T, 4)
        dist = MSEDist(mode, agg="mean")
        # mean over 4 dims = 1.0, negative → -1
        assert torch.allclose(dist.log_prob(value), torch.full((B, T), -1.0))

    def test_log_prob_sum_greater_than_mean(self):
        mode = torch.ones(B, T, 4)
        value = torch.zeros(B, T, 4)
        sum_lp = MSEDist(mode, agg="sum").log_prob(value)
        mean_lp = MSEDist(mode, agg="mean").log_prob(value)
        # both are negative; sum is more negative (larger magnitude)
        assert torch.all(sum_lp < mean_lp)

    def test_log_prob_shape_mismatch_raises(self, mode_tensor):
        dist = MSEDist(mode_tensor)
        wrong = torch.randn(B, T, 5)
        with pytest.raises(AssertionError):
            dist.log_prob(wrong)

    def test_log_prob_dtype_mismatch_raises(self, mode_tensor):
        dist = MSEDist(mode_tensor)
        wrong_dtype = mode_tensor.double()
        with pytest.raises(AssertionError):
            dist.log_prob(wrong_dtype)

    def test_log_prob_invalid_agg_raises(self, mode_tensor):
        dist = MSEDist(mode_tensor, agg="invalid")
        with pytest.raises(NotImplementedError):
            dist.log_prob(mode_tensor.float())


class TestSymlogDist:
    def test_mode_returns_symexp(self, mode_tensor):
        dist = SymlogDist(mode_tensor)
        assert torch.allclose(dist.mode, symexp(mode_tensor.float()))

    def test_mean_equals_mode(self, mode_tensor):
        dist = SymlogDist(mode_tensor)
        assert torch.equal(dist.mode, dist.mean)

    def test_log_prob_mse_perfect_is_zero(self, mode_tensor):
        stored = mode_tensor.float()
        dist = SymlogDist(stored, dist="mse")
        # target = symexp(stored) → symlog(target) == stored → distance == 0
        target = symexp(stored)
        log_p = dist.log_prob(target)
        assert torch.allclose(log_p, torch.zeros(B, T))

    def test_log_prob_abs_perfect_is_zero(self, mode_tensor):
        stored = mode_tensor.float()
        dist = SymlogDist(stored, dist="abs")
        target = symexp(stored)
        log_p = dist.log_prob(target)
        # symlog(symexp(x)) has float32 round-trip error ~3e-8; abs distance doesn't
        # square it down, so use atol that covers float32 precision
        assert torch.allclose(log_p, torch.zeros(B, T), atol=1e-6)

    def test_log_prob_tolerance_clamp(self):
        mode = torch.zeros(B, T, 4)
        # target = symexp(0) = 0 → symlog(0) = 0 → distance = tol/2 < tol → clamped to 0
        tiny_offset = torch.full((B, T, 4), 1e-9)
        dist = SymlogDist(mode + tiny_offset, dist="mse", tol=1e-8)
        target = symexp(mode + tiny_offset)
        log_p = dist.log_prob(target)
        assert torch.allclose(log_p, torch.zeros(B, T), atol=1e-5)

    def test_log_prob_invalid_dist_raises(self, mode_tensor):
        dist = SymlogDist(mode_tensor, dist="invalid")
        with pytest.raises(NotImplementedError):
            dist.log_prob(symexp(mode_tensor.float()))

    def test_log_prob_invalid_agg_raises(self, mode_tensor):
        dist = SymlogDist(mode_tensor, agg="invalid")
        with pytest.raises(NotImplementedError):
            dist.log_prob(symexp(mode_tensor.float()))

    def test_log_prob_shape_mismatch_raises(self, mode_tensor):
        dist = SymlogDist(mode_tensor)
        with pytest.raises(AssertionError):
            dist.log_prob(torch.randn(B, T, 5))


class TestBound:
    def _make_bound(self, mean_val):
        mean = torch.full((B, 4), mean_val)
        std = torch.ones(B, 4)
        inner = torchd.independent.Independent(torchd.Normal(mean, std), 1)
        return Bound(inner)

    def test_mode_clips_large_positive(self):
        dist = self._make_bound(10.0)
        mode = dist.mode
        assert torch.all(mode <= 1.0 + 1e-6)

    def test_mode_clips_large_negative(self):
        dist = self._make_bound(-10.0)
        mode = dist.mode
        assert torch.all(mode >= -1.0 - 1e-6)

    def test_mode_passthrough_small(self):
        val = 0.5
        dist = self._make_bound(val)
        # |mean| < 1 → clip denominator is 1 → output == mean
        assert torch.allclose(dist.mode, torch.full((B, 4), val), atol=1e-6)

    def test_sample_clips_large_values(self):
        dist = self._make_bound(100.0)
        sample = dist.sample()
        assert torch.all(sample <= 1.0 + 1e-6)

    def test_entropy_delegates(self):
        dist = self._make_bound(0.0)
        entropy = dist.entropy()
        assert entropy.shape == (B,)

    def test_log_prob_delegates(self):
        dist = self._make_bound(0.0)
        x = torch.zeros(B, 4)
        log_p = dist.log_prob(x)
        assert log_p.shape == (B,)

    def test_getattr_proxies_to_inner(self):
        mean_val = 0.3
        mean = torch.full((B, 4), mean_val)
        std = torch.ones(B, 4)
        inner = torchd.independent.Independent(torchd.Normal(mean, std), 1)
        bound = Bound(inner)
        # .mean should proxy to inner Independent → returns the Normal mean
        assert torch.allclose(bound.mean, mean)
