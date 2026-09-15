import numpy as np
import torch


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def to_f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.float32)


def to_i32(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.int32)


def pin_for(td, device):
    """Pin a CPU TensorDict when `device` is CUDA, leaving it on CPU either way.

    Pinned memory exists for one reason: it lets the host-to-device copy run
    asynchronously under `non_blocking=True`, which only CUDA can do. Pinning ahead
    of a CPU or MPS target buys nothing.

    On macOS it costs rather than buys. `pin_memory()` pins through whatever
    accelerator torch finds, so on a box with MPS it hands back storage on `mps:0`
    and the follow-up copy dies with "Attempted to set the storage of a tensor on
    device cpu to a storage on different device mps:0". That is why the collection
    loop could not run locally at all, and why two scripts under experiments/ used to
    monkey-patch `TensorDictBase.pin_memory` into a no-op to get around it.

    Separate from `to_device` because the producer and the consumer are not always
    the same place: `envs/parallel.py` builds the batch and hands it back on CPU, so
    it has to pin for a device it is not the one copying to.
    """
    return td.pin_memory() if torch.device(device).type == "cuda" else td


def to_device(td, device):
    """Move a TensorDict to `device`, pinning the source first only when that helps.

    The pinning rule is `pin_for`'s; this is the common case where one place both
    pins and copies. It already lived inline in `buffers/buffer.py` — this is that
    rule, in one place, for every caller.
    """
    device = torch.device(device)
    if td.device is not None and td.device == device:
        return td
    if td.device is None or td.device.type == "cpu":
        return pin_for(td, device).to(device, non_blocking=True)
    return td.to(device, non_blocking=True)


# TODO: Rename to right_pad or something more descriptive
def rpad(x: torch.Tensor, pad: int) -> torch.Tensor:
    for _ in range(pad):
        x = x.unsqueeze(-1)
    return x
