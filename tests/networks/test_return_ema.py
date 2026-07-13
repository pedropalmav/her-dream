import pytest
import torch

from her_dream.networks.return_ema import ReturnEMA

from .conftest import B, T


class TestReturnEMA:
    @pytest.fixture
    def ema(self):
        return ReturnEMA(torch.device("cpu"), alpha=1e-2)

    def test_initial_ema_zeros(self, ema):
        assert torch.allclose(ema.ema_vals, torch.zeros(2))

    def test_returns_tuple(self, ema):
        result = ema(torch.randn(B, T))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_scale_clipped_to_one_narrow_input(self, ema):
        _, scale = ema(torch.zeros(100))
        assert scale.item() == 1.0

    def test_scale_exceeds_one_after_wide_updates(self):
        ema = ReturnEMA(torch.device("cpu"), alpha=0.1)
        x = torch.linspace(-100.0, 100.0, 1000)
        for _ in range(50):
            _, scale = ema(x)
        assert scale.item() > 1.0

    def test_ema_vals_update_after_call(self, ema):
        ema(torch.randn(100))
        assert not torch.allclose(ema.ema_vals, torch.zeros(2))

    def test_outputs_detached(self, ema):
        offset, scale = ema(torch.randn(100))
        assert not offset.requires_grad
        assert not scale.requires_grad

    def test_ema_vals_registered_buffer(self, ema):
        assert "ema_vals" in dict(ema.named_buffers())

    def test_offset_equals_lower_ema(self):
        ema = ReturnEMA(torch.device("cpu"), alpha=0.5)
        x = torch.linspace(-10.0, 10.0, 100)
        offset, _ = ema(x)
        assert torch.allclose(offset, ema.ema_vals[0].detach())

    def test_scale_equals_clipped_spread(self):
        ema = ReturnEMA(torch.device("cpu"), alpha=0.5)
        x = torch.linspace(-10.0, 10.0, 100)
        _, scale = ema(x)
        expected = torch.clip(ema.ema_vals[1] - ema.ema_vals[0], min=1.0).detach()
        assert torch.allclose(scale, expected)
