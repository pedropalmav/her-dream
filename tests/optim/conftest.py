import torch


def make_param(shape, grad_val=None):
    """Create a Parameter with an optional constant gradient."""
    p = torch.nn.Parameter(torch.randn(*shape))
    if grad_val is not None:
        p.grad = torch.full(shape, float(grad_val))
    return p
