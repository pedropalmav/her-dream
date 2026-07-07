import numpy as np
import pytest
import torch
import torch.nn as nn

from tools.nn_utils import convert, tensorstats, weight_init_


class _EmptyWeightModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0))


class TestWeightInit:
    def test_rmsnorm_weight_filled_with_ones(self):
        m = nn.RMSNorm(4)
        with torch.no_grad():
            m.weight.fill_(0.0)
        weight_init_(m)
        assert torch.all(m.weight == 1.0)

    def test_rmsnorm_returns_early_without_fan_calculation(self):
        m = nn.RMSNorm(4)
        weight_init_(m)  # must not raise

    def test_no_weight_attr_returns_early(self):
        m = nn.ReLU()
        weight_init_(m)  # must not raise

    def test_empty_weight_returns_early(self):
        m = _EmptyWeightModule()
        weight_init_(m)  # must not raise

    def test_linear_fan_in_fills_weight(self):
        m = nn.Linear(8, 4)
        weight_init_(m, fan_type="in")
        assert not torch.all(m.weight == 0)

    def test_linear_fan_out_fills_weight(self):
        m = nn.Linear(8, 4)
        weight_init_(m, fan_type="out")
        assert not torch.all(m.weight == 0)

    def test_linear_fan_avg_fills_weight(self):
        m = nn.Linear(8, 4)
        weight_init_(m, fan_type="avg")
        assert not torch.all(m.weight == 0)

    def test_bias_set_to_zero(self):
        m = nn.Linear(4, 4)
        m.bias.data.fill_(99.0)
        weight_init_(m)
        assert torch.all(m.bias == 0.0)

    def test_no_bias_does_not_crash(self):
        m = nn.Linear(4, 4, bias=False)
        weight_init_(m)  # must not raise

    def test_invalid_fan_type_raises_key_error(self):
        m = nn.Linear(4, 4)
        with pytest.raises(KeyError):
            weight_init_(m, fan_type="invalid")


class TestConvert:
    def test_float32(self):
        arr = convert(np.array([1.0], dtype=np.float64), precision=32)
        assert arr.dtype == np.float32

    def test_float64(self):
        arr = convert(np.array([1.0], dtype=np.float32), precision=64)
        assert arr.dtype == np.float64

    def test_float16(self):
        arr = convert(np.array([1.0], dtype=np.float32), precision=16)
        assert arr.dtype == np.float16

    def test_int32(self):
        arr = convert(np.array([1], dtype=np.int64), precision=32)
        assert arr.dtype == np.int32

    def test_int64(self):
        arr = convert(np.array([1], dtype=np.int32), precision=64)
        assert arr.dtype == np.int64

    def test_int16(self):
        arr = convert(np.array([1], dtype=np.int32), precision=16)
        assert arr.dtype == np.int16

    def test_uint8_stays_uint8(self):
        arr = convert(np.array([1], dtype=np.uint8), precision=32)
        assert arr.dtype == np.uint8

    def test_bool_stays_bool(self):
        arr = convert(np.array([True, False]))
        assert arr.dtype == bool

    def test_unsupported_dtype_raises(self):
        arr = np.array([1 + 2j], dtype=np.complex64)
        with pytest.raises(NotImplementedError):
            convert(arr)

    def test_dict_recurses(self):
        result = convert({"k": np.array([1.0], dtype=np.float64)}, precision=32)
        assert isinstance(result, dict)
        assert result["k"].dtype == np.float32

    def test_dict_drops_precision_bug(self):
        # BUG: dict branch calls convert(val) without forwarding precision,
        # so inner values always use the default precision=32.
        result = convert({"k": np.array([1.0], dtype=np.float64)}, precision=64)
        assert result["k"].dtype == np.float32  # not float64


class TestTensorstats:
    def test_returns_four_keys(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        stats = tensorstats(t, "loss")
        assert set(stats.keys()) == {"loss_mean", "loss_std", "loss_min", "loss_max"}

    def test_prefix_applied_to_all_keys(self):
        stats = tensorstats(torch.ones(4), "train")
        assert all(k.startswith("train_") for k in stats)

    def test_mean_value(self):
        t = torch.tensor([2.0, 4.0])
        assert tensorstats(t, "x")["x_mean"].item() == pytest.approx(3.0)

    def test_min_value(self):
        t = torch.tensor([2.0, 4.0])
        assert tensorstats(t, "x")["x_min"].item() == pytest.approx(2.0)

    def test_max_value(self):
        t = torch.tensor([2.0, 4.0])
        assert tensorstats(t, "x")["x_max"].item() == pytest.approx(4.0)
