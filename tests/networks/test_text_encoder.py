import pytest
import torch

from networks.text_encoder import TextEncoderGRU

from .conftest import B, K, S, T, make_text_encoder_config

VOCAB = 46
L = 20


class TestTextEncoderGRU:
    @pytest.fixture
    def encoder(self):
        return TextEncoderGRU(make_text_encoder_config(vocab_size=VOCAB), stoch=S, discrete=K, act="SiLU")

    def test_output_shape_3d_leading(self, encoder):
        assert encoder(torch.randint(0, VOCAB, (B, T, L))).shape == (B, T, S, K)

    def test_output_shape_2d_leading(self, encoder):
        assert encoder(torch.randint(0, VOCAB, (B, L))).shape == (B, S, K)

    def test_integer_tokens_accepted(self, encoder):
        assert torch.all(torch.isfinite(encoder(torch.randint(0, VOCAB, (B, T, L)))))

    def test_max_token_boundary(self, encoder):
        tokens = torch.full((B, T, L), VOCAB - 1, dtype=torch.long)
        assert encoder(tokens).shape == (B, T, S, K)

    def test_stoch_attr(self, encoder):
        assert encoder.stoch == S

    def test_discrete_attr(self, encoder):
        assert encoder.discrete == K

    def test_vocab_size_attr(self, encoder):
        assert encoder.vocab_size == VOCAB

    def test_gradient_flows(self, encoder):
        encoder(torch.randint(0, VOCAB, (B, T, L))).sum().backward()
        assert encoder.char_proj.weight.grad is not None
