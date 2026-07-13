"""Small distribution / divergence helpers shared across experiments.

These operate on RSSM categorical posteriors with shape (..., S, K).
"""

import torch

import her_dream.distributions as dists


def posterior_probs(agent, logit: torch.Tensor) -> torch.Tensor:
    """Categorical probabilities of the RSSM posterior (with unimix), (..., S, K)."""
    return agent.rssm.get_dist(logit).base_dist.probs


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """Per-group Shannon entropy H = -sum_k p log p over the last dim, (..., S)."""
    return -(probs * (probs + 1e-8).log()).sum(-1)


def sym_kl(logit_a: torch.Tensor, logit_b: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between two posteriors given their logits, summed over K.

    Uses the canonical `distributions.kl` (logit-space), matching the RSSM's own
    KL. Returns shape (..., S).
    """
    return (dists.kl(logit_a, logit_b) + dists.kl(logit_b, logit_a)).sum(-1)


def pairwise_hamming(modes: torch.Tensor) -> torch.Tensor:
    """Normalized pairwise Hamming distance over group-argmax modes.

    `modes` is (N, S) of argmax categories; returns the (N, N) matrix where
    entry (i, j) is the fraction of the S groups whose mode differs.
    """
    diff = (modes.unsqueeze(0) != modes.unsqueeze(1)).float()  # (N, N, S)
    return diff.mean(-1)  # (N, N)


def upper_triangular_mean(matrix: torch.Tensor) -> float:
    """Mean of the strict upper triangle of a square matrix (i < j)."""
    n = matrix.shape[0]
    mask = torch.triu(torch.ones(n, n, device=matrix.device), diagonal=1).bool()
    return float(matrix[mask].mean().item())
