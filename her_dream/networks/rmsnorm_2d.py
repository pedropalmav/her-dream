import torch
from torch import nn


class RMSNorm2D(nn.RMSNorm):
    """RMSNorm over channel-last format applied to 4D tensors."""

    def __init__(self, ch: int, eps: float = 1e-3, dtype=None):
        super().__init__(ch, eps=eps, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply RMSNorm over the channel dimension.
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
