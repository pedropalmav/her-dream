import pytest
import torch

from her_dream.networks.multi_encoder import MultiEncoder

from .conftest import B, C, H, T, W, make_multi_encoder_config

CNN_SHAPE = (H, W, C)
MLP_SHAPE = (8,)


class TestMultiEncoderCNNAndMLP:
    @pytest.fixture
    def encoder(self):
        cfg = make_multi_encoder_config(cnn_keys="^image$", mlp_keys="^feat$")
        return MultiEncoder(cfg, {"image": CNN_SHAPE, "feat": MLP_SHAPE})

    def test_both_shapes_populated(self, encoder):
        assert encoder.cnn_shapes
        assert encoder.mlp_shapes

    def test_out_dim_is_sum(self, encoder):
        assert encoder.out_dim == encoder.encoders[0].out_dim + encoder.encoders[1].out_dim

    def test_two_encoders_registered(self, encoder):
        assert len(encoder.encoders) == 2

    def test_fuser_concatenates(self, encoder):
        obs = {"image": torch.randn(B, T, H, W, C), "feat": torch.randn(B, T, 8)}
        assert encoder(obs).shape[-1] == encoder.out_dim

    def test_forward_output_shape(self, encoder):
        obs = {"image": torch.randn(B, T, H, W, C), "feat": torch.randn(B, T, 8)}
        assert encoder(obs).shape == (B, T, encoder.out_dim)

    def test_excluded_keys_filtered(self):
        cfg = make_multi_encoder_config(cnn_keys="^image$", mlp_keys="^feat$")
        shapes = {
            "image": CNN_SHAPE,
            "feat": MLP_SHAPE,
            "is_first": (1,),
            "reward": (1,),
            "log_reward": (1,),
            "mission": (1,),
        }
        enc = MultiEncoder(cfg, shapes)
        all_shape_keys = set(enc.cnn_shapes.keys()) | set(enc.mlp_shapes.keys())
        for excluded in ("is_first", "reward", "log_reward", "mission"):
            assert excluded not in all_shape_keys


class TestMultiEncoderCNNOnly:
    @pytest.fixture
    def encoder(self):
        cfg = make_multi_encoder_config(cnn_keys="^image$", mlp_keys="^feat$")
        return MultiEncoder(cfg, {"image": CNN_SHAPE})

    def test_one_encoder_registered(self, encoder):
        assert len(encoder.encoders) == 1

    def test_fuser_returns_single(self, encoder):
        obs = {"image": torch.randn(B, T, H, W, C)}
        assert encoder(obs).shape == (B, T, encoder.out_dim)

    def test_forward_shape(self, encoder):
        obs = {"image": torch.randn(B, T, H, W, C)}
        assert encoder(obs).shape == (B, T, encoder.out_dim)

    def test_mlp_shapes_empty(self, encoder):
        assert not encoder.mlp_shapes


class TestMultiEncoderMLPOnly:
    @pytest.fixture
    def encoder(self):
        cfg = make_multi_encoder_config(cnn_keys="^image$", mlp_keys="^feat$")
        return MultiEncoder(cfg, {"feat": MLP_SHAPE})

    def test_one_encoder_registered(self, encoder):
        assert len(encoder.encoders) == 1

    def test_forward_shape(self, encoder):
        assert encoder({"feat": torch.randn(B, T, 8)}).shape == (B, T, encoder.out_dim)

    def test_cnn_shapes_empty(self, encoder):
        assert not encoder.cnn_shapes


class TestMultiEncoderNoValidShapes:
    def test_raises_all_excluded(self):
        cfg = make_multi_encoder_config()
        with pytest.raises(NotImplementedError):
            MultiEncoder(cfg, {"is_first": (1,), "is_last": (1,)})

    def test_raises_no_regex_match(self):
        cfg = make_multi_encoder_config(cnn_keys="^image$", mlp_keys="^feat$")
        with pytest.raises(NotImplementedError):
            MultiEncoder(cfg, {"obs": CNN_SHAPE})
