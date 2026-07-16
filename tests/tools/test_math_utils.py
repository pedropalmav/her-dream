import pytest
import torch

from her_dream.tools.math_utils import compute_global_norm, compute_rms


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
