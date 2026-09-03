import torch


class WelfordAccumulator:
    """Streaming mean/variance over a sequence of same-shaped tensors.

    Numerically stable alternative to a naive sum-of-squares variance, which
    catastrophically cancels once the samples agree and floors at a spurious
    ~1e-4 of pure float error instead of zero. Memory is flat in the number of
    `update()` calls: nothing is stacked or kept beyond the running mean/M2.
    """

    def __init__(self):
        self._mean = None
        self._m2 = None
        self._count = 0

    def update(self, x):
        """Fold one more sample `x` into the running mean/variance."""
        self._count += 1
        if self._mean is None:
            self._mean = torch.zeros_like(x)
            self._m2 = torch.zeros_like(x)
        delta = x - self._mean
        self._mean = self._mean + delta / self._count
        self._m2 = self._m2 + delta * (x - self._mean)
        return self

    @property
    def count(self):
        return self._count

    @property
    def mean(self):
        return self._mean

    def variance(self, unbiased=True):
        """Sample variance (Bessel's correction) unless `unbiased=False`."""
        if self._count < 2:
            raise ValueError(f"variance needs >= 2 samples, got {self._count}.")
        denom = self._count - 1 if unbiased else self._count
        return (self._m2 / denom).clamp_min(0.0)

    def std(self, unbiased=True):
        return self.variance(unbiased=unbiased).sqrt()


def compute_rms(tensors):
    """Compute the root mean square (RMS) of a list of tensors."""
    flattened = torch.cat([t.view(-1) for t in tensors if t is not None])
    if len(flattened) == 0:
        return torch.tensor(0.0)
    return torch.linalg.norm(flattened, ord=2) / (flattened.numel() ** 0.5)


def compute_global_norm(tensors):
    """Compute the global norm (L2 norm) across a list of tensors."""
    flattened = torch.cat([t.view(-1) for t in tensors if t is not None])
    if len(flattened) == 0:
        return torch.tensor(0.0)
    return torch.linalg.norm(flattened, ord=2)
