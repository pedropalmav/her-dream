from unittest.mock import MagicMock

import torch

B, T, S, K = 2, 3, 4, 8


def make_dist_mock(logits: torch.Tensor, log_prob_value: torch.Tensor) -> MagicMock:
    dist = MagicMock()
    dist.base_dist.logits = logits
    dist.log_prob.return_value = log_prob_value
    return dist
