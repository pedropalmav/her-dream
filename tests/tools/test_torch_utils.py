import numpy as np
import torch

from her_dream.tools.torch_utils import rpad, to_f32, to_i32, to_np


class TestToNp:
    def test_returns_numpy_array(self):
        x = torch.tensor([1.0, 2.0])
        assert isinstance(to_np(x), np.ndarray)

    def test_values_preserved(self):
        x = torch.tensor([3.0, 4.0])
        np.testing.assert_array_equal(to_np(x), [3.0, 4.0])

    def test_detaches_gradient(self):
        x = torch.tensor([1.0], requires_grad=True)
        result = to_np(x)
        assert isinstance(result, np.ndarray)

    def test_moves_to_cpu(self):
        x = torch.tensor([1.0, 2.0])
        result = to_np(x)
        assert result.dtype == np.float32


class TestToF32:
    def test_converts_float64_to_float32(self):
        x = torch.tensor([1.0], dtype=torch.float64)
        assert to_f32(x).dtype == torch.float32

    def test_float32_stays_float32(self):
        x = torch.tensor([1.0], dtype=torch.float32)
        assert to_f32(x).dtype == torch.float32

    def test_int_converted_to_float32(self):
        x = torch.tensor([1], dtype=torch.int32)
        assert to_f32(x).dtype == torch.float32


class TestToI32:
    def test_converts_int64_to_int32(self):
        x = torch.tensor([1], dtype=torch.int64)
        assert to_i32(x).dtype == torch.int32

    def test_int32_stays_int32(self):
        x = torch.tensor([1], dtype=torch.int32)
        assert to_i32(x).dtype == torch.int32


class TestRpad:
    def test_pad_zero_no_change(self):
        x = torch.tensor([1.0, 2.0])
        assert rpad(x, 0).shape == torch.Size([2])

    def test_pad_one_adds_singleton_dim(self):
        x = torch.tensor([1.0, 2.0])
        assert rpad(x, 1).shape == torch.Size([2, 1])

    def test_pad_two_adds_two_singleton_dims(self):
        x = torch.tensor([1.0, 2.0])
        assert rpad(x, 2).shape == torch.Size([2, 1, 1])

    def test_pad_three_adds_three_singleton_dims(self):
        x = torch.tensor([1.0, 2.0])
        assert rpad(x, 3).shape == torch.Size([2, 1, 1, 1])

    def test_values_preserved(self):
        x = torch.tensor([5.0, 6.0])
        result = rpad(x, 2)
        assert result[0, 0, 0].item() == 5.0
        assert result[1, 0, 0].item() == 6.0
