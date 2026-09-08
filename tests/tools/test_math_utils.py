import pytest
import torch

from her_dream.tools.math_utils import WelfordAccumulator, compute_global_norm, compute_rms


class TestComputeRms:
    def test_known_value(self):
        # RMS([3, 4]) = norm([3,4]) / sqrt(2) = 5 / sqrt(2)
        t = torch.tensor([3.0, 4.0])
        result = compute_rms([t])
        assert result.item() == pytest.approx(5.0 / 2**0.5, rel=1e-5)

    def test_none_tensors_filtered(self):
        t = torch.tensor([1.0, 1.0])
        result = compute_rms([None, t, None])
        assert torch.isfinite(result)
        assert result.item() == pytest.approx(1.0, rel=1e-5)

    def test_empty_tensor_returns_zero(self):
        result = compute_rms([torch.empty(0)])
        assert result.item() == 0.0

    def test_multiple_tensors_concatenated(self):
        result = compute_rms([torch.ones(4), torch.ones(4)])
        assert result.item() == pytest.approx(1.0, rel=1e-5)

    def test_single_element(self):
        t = torch.tensor([3.0])
        result = compute_rms([t])
        assert result.item() == pytest.approx(3.0, rel=1e-5)


class TestComputeGlobalNorm:
    def test_known_value(self):
        t = torch.tensor([3.0, 4.0])
        result = compute_global_norm([t])
        assert result.item() == pytest.approx(5.0, rel=1e-5)

    def test_none_tensors_filtered(self):
        t = torch.tensor([3.0, 4.0])
        result = compute_global_norm([None, t, None])
        assert result.item() == pytest.approx(5.0, rel=1e-5)

    def test_empty_tensor_returns_zero(self):
        result = compute_global_norm([torch.empty(0)])
        assert result.item() == 0.0

    def test_multiple_tensors(self):
        # norm([3,4,0,0]) = 5
        result = compute_global_norm([torch.tensor([3.0, 4.0]), torch.tensor([0.0, 0.0])])
        assert result.item() == pytest.approx(5.0, rel=1e-5)


class TestWelfordAccumulator:
    def test_matches_torch_var_std_unbiased(self):
        samples = torch.randn(6, 4, 3)
        acc = WelfordAccumulator()
        for x in samples:
            acc.update(x)

        assert torch.allclose(acc.mean, samples.mean(0), atol=1e-5)
        assert torch.allclose(acc.variance(), samples.var(0, unbiased=True), atol=1e-5)
        assert torch.allclose(acc.std(), samples.std(0, unbiased=True), atol=1e-5)

    def test_matches_torch_var_std_biased(self):
        samples = torch.randn(6, 4, 3)
        acc = WelfordAccumulator()
        for x in samples:
            acc.update(x)

        assert torch.allclose(acc.variance(unbiased=False), samples.var(0, unbiased=False), atol=1e-5)
        assert torch.allclose(acc.std(unbiased=False), samples.std(0, unbiased=False), atol=1e-5)

    def test_two_known_samples(self):
        acc = WelfordAccumulator()
        acc.update(torch.tensor([1.0, 5.0]))
        acc.update(torch.tensor([3.0, 7.0]))

        assert torch.allclose(acc.mean, torch.tensor([2.0, 6.0]))
        # var([1,3]) = var([5,7]) = 2 with Bessel's correction
        assert torch.allclose(acc.variance(), torch.tensor([2.0, 2.0]))
        assert torch.allclose(acc.std(), torch.tensor([2.0, 2.0]).sqrt())

    def test_count_tracks_number_of_updates(self):
        acc = WelfordAccumulator()
        assert acc.count == 0
        for _ in range(4):
            acc.update(torch.zeros(2))
        assert acc.count == 4

    def test_update_returns_self_for_chaining(self):
        acc = WelfordAccumulator()
        result = acc.update(torch.ones(2))
        assert result is acc

    def test_identical_samples_give_zero_variance(self):
        acc = WelfordAccumulator()
        x = torch.tensor([1.0, 2.0, 3.0])
        for _ in range(5):
            acc.update(x)

        assert torch.allclose(acc.variance(), torch.zeros_like(x))
        assert torch.allclose(acc.std(), torch.zeros_like(x))

    def test_variance_requires_at_least_two_samples(self):
        acc = WelfordAccumulator()
        with pytest.raises(ValueError):
            acc.variance()

        acc.update(torch.ones(3))
        with pytest.raises(ValueError):
            acc.variance()

        acc.update(torch.ones(3))
        acc.variance()  # no longer raises
