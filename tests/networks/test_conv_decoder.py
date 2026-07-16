import pytest
import torch

from her_dream.networks.block_linear import BlockLinear
from her_dream.networks.conv_decoder import ConvDecoder

from .conftest import FLAT_STOCH, B, D, K, S, T, make_cnn_dec_config

SHAPE = (3, 16, 16)  # (C, H, W)


class TestConvDecoder:
    @pytest.fixture
    def decoder(self):
        return ConvDecoder(make_cnn_dec_config(), deter=D, flat_stoch=FLAT_STOCH, shape=SHAPE)

    def test_output_shape(self, decoder, stoch, deter):
        assert decoder(stoch, deter).shape == (B, T, SHAPE[1], SHAPE[2], SHAPE[0])

    def test_output_in_unit_interval(self, decoder, stoch, deter):
        out = decoder(stoch, deter)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_min_shape_attr(self, decoder):
        cfg = make_cnn_dec_config()
        factor = 2 ** len(tuple(cfg.depth * m for m in cfg.mults))
        h, w = SHAPE[1] // factor, SHAPE[2] // factor
        depths = tuple(int(cfg.depth) * int(m) for m in cfg.mults)
        assert decoder.min_shape == (h, w, depths[-1])

    def test_sp0_is_block_linear(self, decoder):
        assert isinstance(decoder.sp0, BlockLinear)

    def test_gradient_flows_sp0(self, decoder, stoch, deter):
        decoder(stoch, deter).sum().backward()
        assert decoder.sp0.weight.grad is not None

    def test_gradient_flows_sp1(self, decoder, stoch, deter):
        decoder(stoch, deter).sum().backward()
        sp1_linear = list(decoder.sp1.children())[0]
        assert sp1_linear.weight.grad is not None

    def test_different_batch_sizes(self, decoder):
        for b in (1, 4):
            out = decoder(torch.randn(b, T, S, K), torch.randn(b, T, D))
            assert out.shape == (b, T, SHAPE[1], SHAPE[2], SHAPE[0])
