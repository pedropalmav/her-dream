import pytest
import torch

from her_dream.networks.conv_encoder import ConvEncoder
from her_dream.networks.rmsnorm_2d import RMSNorm2D

from .conftest import B, C, H, T, W, make_cnn_enc_config


def _expected_out_dim(cfg, h, w):
    depths = tuple(int(cfg.depth) * int(m) for m in cfg.mults)
    for _ in depths:
        h, w = h // 2, w // 2
    return depths[-1] * h * w


class TestConvEncoderNoNorm:
    @pytest.fixture
    def encoder(self):
        return ConvEncoder(make_cnn_enc_config(norm=False), (H, W, C))

    def test_out_dim(self, encoder):
        assert encoder.out_dim == _expected_out_dim(make_cnn_enc_config(norm=False), H, W)

    def test_forward_shape(self, encoder):
        assert encoder(torch.randn(B, T, H, W, C)).shape == (B, T, encoder.out_dim)

    def test_no_rmsnorm2d_in_layers(self, encoder):
        norms = [m for m in encoder.layers.modules() if isinstance(m, RMSNorm2D)]
        assert len(norms) == 0

    def test_obs_shift(self, encoder):
        encoder.eval()
        x = torch.randn(B, T, H, W, C)
        with torch.no_grad():
            out = encoder(x)
            shifted = x - 0.5
            flat = shifted.reshape(-1, H, W, C).permute(0, 3, 1, 2)
            processed = encoder.layers(flat).reshape(B * T, -1)
            expected = processed.reshape(B, T, -1)
        assert torch.allclose(out, expected)

    def test_gradient_flows(self, encoder):
        encoder(torch.randn(B, T, H, W, C)).sum().backward()
        first_conv = list(encoder.layers.children())[0]
        assert first_conv.weight.grad is not None


class TestConvEncoderWithNorm:
    @pytest.fixture
    def encoder(self):
        return ConvEncoder(make_cnn_enc_config(norm=True), (H, W, C))

    def test_rmsnorm2d_present(self, encoder):
        norms = [m for m in encoder.layers.modules() if isinstance(m, RMSNorm2D)]
        assert len(norms) > 0

    def test_forward_shape(self, encoder):
        assert encoder(torch.randn(B, T, H, W, C)).shape == (B, T, encoder.out_dim)

    def test_out_dim_matches_no_norm(self, encoder):
        enc_no = ConvEncoder(make_cnn_enc_config(norm=False), (H, W, C))
        assert encoder.out_dim == enc_no.out_dim
