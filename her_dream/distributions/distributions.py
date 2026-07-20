from collections.abc import Callable

import torch
import torch.distributions as torchd
from torch.nn import functional as F
from torch.types import _size

from her_dream.tools import to_f32, to_i32

from .functional import symexp, symlog


class OneHotDist(torchd.OneHotCategorical):
    def __init__(self, logits: torch.Tensor, unimix_ratio: float = 0.0):
        # (..., K)
        probs = F.softmax(to_f32(logits), dim=-1)
        uniform = unimix_ratio / probs.shape[-1]
        probs = probs * (1.0 - unimix_ratio) + torch.ones_like(probs, dtype=torch.float32) * uniform
        logits = torch.log(probs)
        super().__init__(logits=logits)

    @property
    def mode(self) -> torch.Tensor:
        # (..., K)
        _mode = F.one_hot(torch.argmax(self.logits, axis=-1), self.logits.shape[-1])
        return _mode.detach() + self.logits - self.logits.detach()

    def rsample(self, sample_shape: _size = (), temperature: float = 1.0) -> torch.Tensor:
        # (..., K)
        return F.gumbel_softmax(self.logits, tau=temperature, hard=True, dim=-1)

    def sample(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class MultiOneHotDist(torchd.Distribution):
    def __init__(self, logits: torch.Tensor, shape: tuple, unimix_ratio: float = 0.0):
        self.shape = shape
        splits = torch.split(logits, shape, dim=-1)
        self.onehots = [OneHotDist(s, unimix_ratio=unimix_ratio) for s in splits]

    @property
    def mode(self) -> torch.Tensor:
        _modes = [dist.mode for dist in self.onehots]
        return torch.cat(_modes, dim=-1)

    def rsample(self, sample_shape: _size = ()) -> torch.Tensor:
        _rsamples = [dist.rsample() for dist in self.onehots]
        return torch.cat(_rsamples, dim=-1)

    def sample(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        splits = torch.split(value, self.shape, dim=-1)
        _log_probs = [dist.log_prob(s) for dist, s in zip(self.onehots, splits)]
        return sum(_log_probs)

    def entropy(self) -> torch.Tensor:
        _entropies = [dist.entropy() for dist in self.onehots]
        return sum(_entropies)


class TwoHot(torchd.Distribution):
    def __init__(
        self,
        logits: torch.Tensor,
        bins: torch.Tensor,
        squash: Callable = None,
        unsquash: Callable = None,
    ):
        # (..., N_bins), (N_bins,)
        self.logits = to_f32(logits)
        assert self.logits.shape[-1] == len(bins), (self.logits.shape, len(bins))

        self.bins = bins
        self.probs = F.softmax(self.logits, dim=-1)  # (..., N_bins)
        self.squash = squash if squash is not None else (lambda x: x)
        self.unsquash = unsquash if unsquash is not None else (lambda x: x)

    @property
    def mode(self) -> torch.Tensor:
        # (..., N_bins), (N_bins,) -> (..., 1)
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m : m + 1]
            p3 = self.probs[..., m + 1 :]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m : m + 1]
            b3 = self.bins[..., m + 1 :]
            wavg = (p2 * b2).sum(dim=-1, keepdim=True) + ((p1 * b1).flip(dims=(-1,)) + (p3 * b3)).sum(
                dim=-1, keepdim=True
            )
            return self.unsquash(wavg)
        p1 = self.probs[..., : n // 2]
        p2 = self.probs[..., n // 2 :]
        b1 = self.bins[..., : n // 2]
        b2 = self.bins[..., n // 2 :]
        wavg = ((p1 * b1).flip(dims=(-1,)) + (p2 * b2)).sum(dim=-1, keepdim=True)
        return self.unsquash(wavg)

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        # (..., 1)
        assert target.dtype == self.probs.dtype
        target = target.squeeze(-1)  # (...,)
        target_squashed = self.squash(target).detach()  # (...,)
        # below/above: (...,)
        below = to_i32(self.bins <= target_squashed.unsqueeze(-1)).sum(dim=-1) - 1
        above = len(self.bins) - to_i32(self.bins > target_squashed.unsqueeze(-1)).sum(dim=-1)
        below = torch.clamp(below, 0, len(self.bins) - 1)
        above = torch.clamp(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[below] - target_squashed).abs(),
        )
        dist_to_above = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[above] - target_squashed).abs(),
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        oh_below = to_f32(F.one_hot(below, num_classes=len(self.bins)))
        oh_above = to_f32(F.one_hot(above, num_classes=len(self.bins)))
        # (..., N_bins)
        mixed_target = oh_below * weight_below.unsqueeze(-1) + oh_above * weight_above.unsqueeze(-1)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)  # (..., N_bins)
        return (mixed_target * log_pred).sum(dim=-1)  # (...)


class MSEDist(torchd.Distribution):
    def __init__(self, mode: torch.Tensor, agg: str = "sum"):
        # (..., D)
        self._mode = to_f32(mode)
        self._agg = agg

    @property
    def mode(self) -> torch.Tensor:
        return self._mode

    @property
    def mean(self) -> torch.Tensor:
        return self._mode

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # (..., D)
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        assert self._mode.dtype == value.dtype, (self._mode.dtype, value.dtype)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss  # (...)


class SymlogDist(torchd.Distribution):
    def __init__(self, mode: torch.Tensor, dist: str = "mse", agg: str = "sum", tol: float = 1e-8):
        # (..., D)
        self._mode = to_f32(mode)
        self._dist = dist
        self._agg = agg
        self._tol = tol

    @property
    def mode(self) -> torch.Tensor:
        return symexp(self._mode)

    @property
    def mean(self) -> torch.Tensor:
        return symexp(self._mode)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # (..., D)
        assert self._mode.shape == value.shape
        assert self._mode.dtype == value.dtype
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss  # (...)


class Bound:
    def __init__(self, dist: torchd.Distribution):
        super().__init__()
        self._dist = dist

    def __getattr__(self, name: str):
        return getattr(self._dist, name)

    @property
    def mode(self) -> torch.Tensor:
        out = self._dist.mean
        return out / torch.clip(torch.abs(out), min=1.0).detach()

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()

    def sample(self, sample_shape: _size = ()) -> torch.Tensor:
        out = self._dist.rsample(sample_shape)
        return out / torch.clip(torch.abs(out), min=1.0).detach()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(x)
