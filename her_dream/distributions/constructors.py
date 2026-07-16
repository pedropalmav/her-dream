import torch
import torch.distributions as torchd

from her_dream.tools import to_f32

from .distributions import (
    Bound,
    MSEDist,
    MultiOneHotDist,
    OneHotDist,
    SymlogDist,
    TwoHot,
)
from .functional import symexp


def bounded_normal(x: torch.Tensor, min_std: float, max_std: float, **kwargs):
    mean, std = torch.chunk(x, 2, dim=-1)
    std = (max_std - min_std) * torch.sigmoid(std + 2.0) + min_std
    # NOTE: Bound can be added
    dist = torchd.normal.Normal(torch.tanh(to_f32(mean)), to_f32(std))
    return torchd.independent.Independent(dist, 1)


def normal_std_fixed(mean: torch.Tensor, std: torch.Tensor, **kwargs):
    dist = torchd.normal.Normal(to_f32(mean), to_f32(std))
    return Bound(torchd.independent.Independent(dist, 1))


def onehot(mean: torch.Tensor, unimix_ratio: float, **kwargs):
    return OneHotDist(to_f32(mean), unimix_ratio=unimix_ratio)


def multi_onehot(mean: torch.Tensor, unimix_ratio: float, shape: tuple, **kwargs):
    return MultiOneHotDist(to_f32(mean), shape, unimix_ratio=unimix_ratio)


def binary(logits: torch.Tensor, **kwargs):
    return torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=to_f32(logits)), 1)


def symexp_twohot(logits: torch.Tensor, bin_num: int, **kwargs):
    if bin_num % 2 == 1:
        half = torch.linspace(-20, 0, (bin_num - 1) // 2 + 1, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half[:-1].flip(dims=(0,))], 0)
    else:
        half = torch.linspace(-20, 0, bin_num // 2, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half.flip(dims=(0,))], 0)
    return TwoHot(to_f32(logits), bins)


def symlog_mse(logits: torch.Tensor, **kwargs):
    return SymlogDist(to_f32(logits))


def mse(logits: torch.Tensor, **kwargs):
    return MSEDist(to_f32(logits))


def identity(logits: torch.Tensor, **kwargs):
    return logits


def kl(logits_left: torch.Tensor, logits_right: torch.Tensor):
    # (..., K), (..., K)
    logprob_left = torch.log_softmax(logits_left, -1)
    logprob_right = torch.log_softmax(logits_right, -1)
    prob = torch.softmax(logits_left, -1)
    return (prob * (logprob_left - logprob_right)).sum(-1)  # (...)
