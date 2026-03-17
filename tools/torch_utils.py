import torch
import numpy as np


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def to_f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.float32)


def to_i32(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.int32)


# TODO: Rename to right_pad or something more descriptive
def rpad(x: torch.Tensor, pad: int) -> torch.Tensor:
    for _ in range(pad):
        x = x.unsqueeze(-1)
    return x
