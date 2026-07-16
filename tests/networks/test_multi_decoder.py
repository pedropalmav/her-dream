import pytest

from her_dream.distributions.distributions import MSEDist, SymlogDist
from her_dream.networks.multi_decoder import MultiDecoder

from .conftest import FLAT_STOCH, B, C, D, H, T, W, make_multi_decoder_config

CNN_SHAPE = (H, W, C)
MLP_SHAPE = (8,)


class TestMultiDecoderCNNAndMLP:
    @pytest.fixture
    def decoder(self):
        cfg = make_multi_decoder_config(cnn_keys="^image$", mlp_keys="^goal$")
        return MultiDecoder(cfg, D, FLAT_STOCH, {"image": CNN_SHAPE, "goal": MLP_SHAPE})

    @pytest.fixture
    def out(self, decoder, stoch, deter):
        return decoder(stoch, deter)

    def test_both_keys_in_output(self, out):
        assert "image" in out
        assert "goal" in out

    def test_image_dist_type(self, out):
        assert isinstance(out["image"], MSEDist)

    def test_goal_dist_type(self, out):
        assert isinstance(out["goal"], SymlogDist)

    def test_image_mode_shape(self, out):
        assert out["image"].mode.shape == (B, T, H, W, C)

    def test_goal_mode_shape(self, out):
        assert out["goal"].mode.shape == (B, T, 8)

    def test_all_keys_contains_both(self, decoder):
        assert "image" in decoder.all_keys
        assert "goal" in decoder.all_keys


class TestMultiDecoderCNNOnly:
    @pytest.fixture
    def decoder(self):
        cfg = make_multi_decoder_config(cnn_keys="^image$", mlp_keys="^goal$")
        return MultiDecoder(cfg, D, FLAT_STOCH, {"image": CNN_SHAPE})

    @pytest.fixture
    def out(self, decoder, stoch, deter):
        return decoder(stoch, deter)

    def test_cnn_block_present(self, decoder):
        assert hasattr(decoder, "_cnn")

    def test_mlp_block_absent(self, decoder):
        assert not hasattr(decoder, "_mlp")

    def test_forward_keys(self, out):
        assert list(out.keys()) == ["image"]

    def test_image_shape(self, out):
        assert out["image"].mode.shape == (B, T, H, W, C)


class TestMultiDecoderMLPOnly:
    @pytest.fixture
    def decoder(self):
        cfg = make_multi_decoder_config(cnn_keys="^image$", mlp_keys="^goal$")
        return MultiDecoder(cfg, D, FLAT_STOCH, {"goal": MLP_SHAPE})

    @pytest.fixture
    def out(self, decoder, stoch, deter):
        return decoder(stoch, deter)

    def test_mlp_block_present(self, decoder):
        assert hasattr(decoder, "_mlp")

    def test_cnn_block_absent(self, decoder):
        assert not hasattr(decoder, "_cnn")

    def test_forward_keys(self, out):
        assert list(out.keys()) == ["goal"]

    def test_goal_shape(self, out):
        assert out["goal"].mode.shape == (B, T, 8)


class TestMultiDecoderExcludedKeys:
    def test_excluded_not_in_all_keys(self):
        cfg = make_multi_decoder_config(cnn_keys="^image$", mlp_keys="^goal$")
        shapes = {
            "image": CNN_SHAPE,
            "is_first": (1,),
            "is_last": (1,),
            "is_terminal": (1,),
            "mission": (1,),
        }
        dec = MultiDecoder(cfg, D, FLAT_STOCH, shapes)
        for key in ("is_first", "is_last", "is_terminal", "mission"):
            assert key not in dec.all_keys
