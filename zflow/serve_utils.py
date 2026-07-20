"""Shared helpers for serve.py and clustering.py.

Split out so clustering.py can reuse these without serve.py importing back
from clustering.py (which would create a circular import).
"""

import base64
import io

import numpy as np
import torch
from PIL import Image
from tensordict import TensorDict


def encode_image(img_np: np.ndarray) -> str:
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_np, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_trans(obs: dict, action: torch.Tensor, device) -> TensorDict:
    """Build a TensorDict from the raw obs dict + action, matching evaluate.py exactly."""
    trans = TensorDict(
        {k: torch.as_tensor(v, device="cpu").unsqueeze(0) for k, v in obs.items()},
        batch_size=(1,),
        device="cpu",
    ).pin_memory()
    trans = trans.to(device, non_blocking=True)
    trans["action"] = action
    return trans
