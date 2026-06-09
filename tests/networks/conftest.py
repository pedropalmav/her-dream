from types import SimpleNamespace

import pytest
import torch

B, T = 2, 3
H, W, C = 16, 16, 3
S, K = 8, 4
D = 64
FLAT_STOCH = S * K  # 32


class NS:
    """Namespace with mapping protocol so ``**NS(name="mse")`` works."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def keys(self):
        return vars(self).keys()

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(vars(self))


def make_mlp_config(**overrides):
    defaults = dict(act="SiLU", symlog_inputs=False, device="cpu", layers=2, units=32, name="test")
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_cnn_enc_config(**overrides):
    defaults = dict(act="SiLU", norm=False, kernel_size=3, depth=8, mults=[1, 2])
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_cnn_dec_config(**overrides):
    defaults = dict(act="SiLU", depth=8, mults=[2, 1], bspace=4, kernel_size=3, units=64)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


_DIST_KWARGS = {
    "bounded_normal": dict(name="bounded_normal", min_std=0.1, max_std=1.0),
    "onehot": dict(name="onehot", unimix_ratio=0.0),
    "multi_onehot": dict(name="multi_onehot", unimix_ratio=0.0),
    "symexp_twohot": dict(name="symexp_twohot", bin_num=5),
    "binary": dict(name="binary"),
    "identity": dict(name="identity"),
    "mse": dict(name="mse"),
}

_DIST_SHAPE = {
    "bounded_normal": [4],
    "onehot": [8],
    "multi_onehot": [4, 4],
    "symexp_twohot": [5],
    "binary": [4],
    "identity": [4],
    "mse": [4],
}


def make_mlphead_config(dist_name, outscale=1.0):
    return SimpleNamespace(
        act="SiLU",
        symlog_inputs=False,
        device="cpu",
        layers=1,
        units=32,
        name="test",
        shape=_DIST_SHAPE[dist_name],
        outscale=outscale,
        dist=SimpleNamespace(**_DIST_KWARGS[dist_name]),
    )


def make_text_encoder_config(**overrides):
    defaults = dict(vocab_size=46, embed_dim=32, hidden=64, num_layers=1)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_multi_encoder_config(cnn_keys="^image$", mlp_keys="^feat$"):
    return SimpleNamespace(
        cnn_keys=cnn_keys,
        mlp_keys=mlp_keys,
        cnn=make_cnn_enc_config(),
        mlp=make_mlp_config(),
    )


def make_multi_decoder_config(cnn_keys="^image$", mlp_keys="^goal$"):
    mlp_inner = SimpleNamespace(
        act="SiLU",
        symlog_inputs=False,
        device="cpu",
        layers=1,
        units=32,
        name="mlp_dec",
        shape=None,
        outscale=1.0,
        dist=SimpleNamespace(name="identity"),
    )
    return SimpleNamespace(
        cnn_keys=cnn_keys,
        mlp_keys=mlp_keys,
        cnn=make_cnn_dec_config(),
        mlp=mlp_inner,
        cnn_dist=NS(name="mse"),
        mlp_dist=NS(name="symlog_mse"),
    )


@pytest.fixture
def stoch():
    return torch.randn(B, T, S, K)


@pytest.fixture
def deter():
    return torch.randn(B, T, D)


@pytest.fixture
def obs_image():
    return torch.randn(B, T, H, W, C)
