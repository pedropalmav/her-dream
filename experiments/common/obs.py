"""Observation / action preprocessing shared across the experiment scripts."""

import numpy as np
import torch


def preprocess_obs(obs: dict, device: str) -> dict:
    """obs dict (numpy) -> (1, ...) float tensors on device. uint8 images /255.

    Only ndarray/bool values are kept (the keys the encoder expects); any other
    type (e.g. a raw string) is silently dropped.
    """
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            t = torch.as_tensor(v, dtype=torch.float32, device=device)
            if v.dtype == np.uint8:
                t = t / 255.0
            out[k] = t.unsqueeze(0)  # (1, ...)
        elif isinstance(v, (bool, np.bool_)):
            out[k] = torch.tensor([[float(v)]], dtype=torch.float32, device=device)
    return out


def stack_obs(per_pos_seqs: list, t: int, repeats: int, device: str) -> dict:
    """Stack step `t`'s obs across positions and repeat R times -> (P*R, ...).

    Batched sibling of `preprocess_obs`: same dtype/uint8/bool rules, but the
    batch dim is built by stacking P conditions then `repeat_interleave(R)`.
    """
    keys = per_pos_seqs[0][t].keys()
    out = {}
    for k in keys:
        vals = [seq[t][k] for seq in per_pos_seqs]
        if isinstance(vals[0], np.ndarray):
            arr = np.stack(vals, axis=0)
            tens = torch.as_tensor(arr, dtype=torch.float32, device=device)
            if arr.dtype == np.uint8:
                tens = tens / 255.0
            out[k] = tens.repeat_interleave(repeats, dim=0)
        elif isinstance(vals[0], (bool, np.bool_)):
            tens = torch.tensor([[float(v)] for v in vals], dtype=torch.float32, device=device)
            out[k] = tens.repeat_interleave(repeats, dim=0)
    return out


def onehot_action(idx: int, n_actions: int) -> np.ndarray:
    """One-hot float32 action vector with `idx` set."""
    act = np.zeros(n_actions, dtype=np.float32)
    act[idx] = 1.0
    return act
