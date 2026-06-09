from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from .conftest import A, D, K, S, make_buffer_config, make_her_config, make_transition


class TestBufferInit:
    def test_device_attribute(self, buffer):
        assert buffer.device == torch.device("cpu")

    def test_batch_size_attribute(self, buffer):
        assert buffer.batch_size == 2

    def test_batch_length_attribute(self, buffer):
        assert buffer.batch_length == 4

    def test_num_eps_starts_at_zero(self, buffer):
        assert buffer.num_eps == 0


class TestBufferCount:
    def test_count_returns_zero_when_empty(self, buffer):
        assert buffer.count() == 0

    def test_count_increases_after_add(self, buffer):
        data = make_transition(n_envs=2)
        buffer.add_transition(data)
        assert buffer.count() > 0

    def test_count_accumulates_with_multiple_adds(self, buffer):
        data = make_transition(n_envs=2)
        buffer.add_transition(data)
        count_after_one = buffer.count()
        buffer.add_transition(data)
        assert buffer.count() > count_after_one


class TestMoveToDevice:
    def test_same_device_stays_on_cpu(self, buffer):
        td = TensorDict({"x": torch.zeros(4, 8)}, batch_size=[4, 8], device="cpu")
        result = buffer._move_sample_to_device(td)
        assert result.device.type == "cpu"

    def test_different_device_calls_to(self, buffer):
        mock_td = MagicMock()
        mock_td.device = torch.device("cuda")  # differs from "cpu", not a cpu→cuda pair
        mock_td.to.return_value = mock_td
        buffer._move_sample_to_device(mock_td)
        mock_td.to.assert_called_once_with(buffer.device, non_blocking=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_to_cuda_moves_data(self, buf_config):
        from buffers.buffer import Buffer

        config = make_buffer_config(device="cuda", storage_device="cpu")
        buf = Buffer(config)
        td = TensorDict({"x": torch.zeros(4, 8)}, batch_size=[4, 8], device="cpu")
        result = buf._move_sample_to_device(td)
        assert result.device.type == "cuda"


class TestMakeBuffer:
    def _make_config(self, buffer_type, **her_kwargs):
        base = make_buffer_config()
        if buffer_type == "her":
            buffer_ns = SimpleNamespace(**vars(make_her_config(**her_kwargs)), type=buffer_type)
        else:
            buffer_ns = SimpleNamespace(**vars(base), type=buffer_type)
        return SimpleNamespace(buffer=buffer_ns)

    def test_normal_type_returns_buffer(self):
        from buffers import make_buffer
        from buffers.buffer import Buffer

        config = self._make_config("normal")
        buf = make_buffer(config, reward_function=None)
        assert isinstance(buf, Buffer)

    def test_her_type_returns_her_buffer(self):
        from buffers import make_buffer
        from buffers.her_buffer import HERBuffer

        config = self._make_config("her")
        buf = make_buffer(config, reward_function=lambda a, d: torch.zeros(a.shape[0]))
        assert isinstance(buf, HERBuffer)

    def test_invalid_type_raises_value_error(self):
        from buffers import make_buffer

        config = self._make_config("unknown")
        with pytest.raises(ValueError):
            make_buffer(config, reward_function=None)


class TestBufferSample:
    def test_returns_three_tuple(self, filled_buffer):
        assert len(filled_buffer.sample()) == 3

    def test_data_shapes(self, filled_buffer):
        data, _, _ = filled_buffer.sample()
        B, T = filled_buffer.batch_size, filled_buffer.batch_length
        assert data["stoch"].shape == (B, T, S, K)
        assert data["deter"].shape == (B, T, D)
        assert data["action"].shape == (B, T, A)

    def test_initial_shapes(self, filled_buffer):
        _, _, (init_stoch, init_deter) = filled_buffer.sample()
        B = filled_buffer.batch_size
        assert init_stoch.shape == (B, S, K)
        assert init_deter.shape == (B, D)

    def test_index_shape(self, filled_buffer):
        _, index, _ = filled_buffer.sample()
        B, T = filled_buffer.batch_size, filled_buffer.batch_length
        assert len(index) == 2
        assert index[0].shape == (B, T)
        assert index[1].shape == (B, T)

    def test_data_on_correct_device(self, filled_buffer):
        data, _, _ = filled_buffer.sample()
        assert data.device.type == filled_buffer.device.type


class TestBufferUpdate:
    def test_update_does_not_raise(self, filled_buffer):
        _, index, _ = filled_buffer.sample()
        B, T = filled_buffer.batch_size, filled_buffer.batch_length
        filled_buffer.update(index, torch.zeros(B, T, S, K), torch.zeros(B, T, D))

    def test_update_preserves_buffer_count(self, filled_buffer):
        _, index, _ = filled_buffer.sample()
        count_before = filled_buffer.count()
        B, T = filled_buffer.batch_size, filled_buffer.batch_length
        filled_buffer.update(index, torch.ones(B, T, S, K), torch.ones(B, T, D))
        assert filled_buffer.count() == count_before

    def test_update_buffer_remains_sampleable(self, filled_buffer):
        _, index, _ = filled_buffer.sample()
        B, T = filled_buffer.batch_size, filled_buffer.batch_length
        filled_buffer.update(index, torch.ones(B, T, S, K), torch.ones(B, T, D))
        data, _, _ = filled_buffer.sample()
        assert data["stoch"].shape == (B, T, S, K)
