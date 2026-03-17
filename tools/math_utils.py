import torch


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
