import math

import torch
import torch.nn.functional as F

# The exact-match rewards (first_row / row_by_row / full / argmax_full) used to
# compare one-hot tensors with `state == goal`. That float equality is fragile:
# under fp16 autocast + torch.compile on GPU a straight-through one-hot whose
# "1" entry is e.g. 0.9995 never equals the exact-1.0 goal, so the reward
# collapses to ~-1 even when the states actually match (the CPU/fp32 collection
# path is fine, which is why episode/score stayed dense while train/rew pinned
# at -1). We compare argmax (category) indices instead — semantically identical
# for one-hots and immune to any precision/graph perturbation.
#
# ONEHOT_ATOL: how far the mode (argmax) entry may sit from 1.0 (and the row sum
# from 1.0) before we consider the input not a valid one-hot.
ONEHOT_ATOL = 1e-3


def make_reward(config):
    match config.goal_type:
        case "first_row":
            return first_row_reward
        case "row_by_row":
            return row_by_row_reward
        case "full":
            return full_goal_reward
        case "argmax_full":
            return argmax_full_reward
        case "log_prob":
            log_threshold = math.log(config.prob_threshold)
            return lambda dist, goal: log_prob_reward(dist, goal, log_threshold)
        case "prob":
            return prob_reward
        case "max_cosine":
            return max_cosine_reward
        case _:
            raise ValueError(f"Tipo de objetivo no soportado: {config.goal_type}")


def _mode_idx(x: torch.Tensor, atol: float = ONEHOT_ATOL) -> torch.Tensor:
    """Argmax (category index) over the last dim of a (near-)one-hot tensor.

    Asserts the input really is one-hot to within `atol`: the winning entry must
    be within `atol` of 1.0 and each row must sum to ~1.0. This guards against
    silently taking the argmax of a degenerate / non-one-hot tensor.
    """
    peak_err = (x.amax(dim=-1) - 1.0).abs().amax()
    sum_err = (x.sum(dim=-1) - 1.0).abs().amax()
    assert bool(peak_err < atol) and bool(sum_err < atol), (
        f"reward input is not (close to) one-hot: |max-1|={peak_err.item():.2e}, "
        f"|sum-1|={sum_err.item():.2e} (atol={atol:g})"
    )
    return x.argmax(dim=-1)


def first_row_reward(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """
    Compute reward for a given state and goal.

    Args:
        state (torch.Tensor): The current state of the environment.
                              Shape (B, S, K) or (B, T, S, K).
        goal (torch.Tensor): The desired goal state.
                             Shape (K,) or (B, K).

    Returns:
        torch.Tensor: Reward tensor of shape (B, 1) or (B, T, 1).
    """
    if state.dim() == 3:
        # Caso (B, S, K) con goal (K,)
        # (B,) == () -> (B,)
        matches = (_mode_idx(state[:, 0, :]) == _mode_idx(goal)).unsqueeze(-1)

    elif state.dim() == 4:
        # Caso (B, T, S, K) con goal (B, K)
        # (B, T) == (B, 1) -> (B, T)
        matches = (_mode_idx(state[:, :, 0, :]) == _mode_idx(goal).unsqueeze(1)).unsqueeze(-1)

    else:
        raise ValueError(f"Estado con número de dimensiones no soportado: {state.dim()}")

    return torch.where(matches, torch.tensor(0), torch.tensor(-1))


def row_by_row_reward(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """
    Compute reward based on how many rows of the state match the goal.

    Args:
        state (torch.Tensor): The current state of the environment.
                              Shape (B, S, K) or (B, T, S, K).
        goal (torch.Tensor): The desired goal state.
                             Shape (S, K) or (B, S, K).

    Returns:
        torch.Tensor: Reward tensor of shape (B, 1) or (B, T, 1).
    """
    if state.dim() == 3:
        # Caso (B, S, K) con goal (S, K)
        _, S, _ = state.shape
        # (B, S) == (S,) -> (B, S)
        matches = _mode_idx(state) == _mode_idx(goal)
        num_matching_rows = matches.sum(dim=1, keepdim=True)

    elif state.dim() == 4:
        # Caso (B, T, S, K) con goal (B, S, K)
        _, _, S, _ = state.shape
        # (B, T, S) == (B, 1, S) -> (B, T, S)
        matches = _mode_idx(state) == _mode_idx(goal).unsqueeze(1)
        num_matching_rows = matches.sum(dim=2, keepdim=True)

    else:
        raise ValueError(f"Estado con número de dimensiones no soportado: {state.dim()}")

    return num_matching_rows / S - 1


def full_goal_reward(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """
    Compute reward based on whether the entire state matches the goal.

    Args:
        state (torch.Tensor): The current state of the environment.
                              Shape (B, S, K) or (B, T, S, K).
        goal (torch.Tensor): The desired goal state.
                             Shape (S, K) or (B, S, K).

    Returns:
        torch.Tensor: Reward tensor of shape (B, 1) or (B, T, 1).
    """
    if state.dim() == 3:
        # Caso (B, S, K) con goal (S, K): all S groups must share the argmax category
        matches = torch.all(_mode_idx(state) == _mode_idx(goal), dim=1).unsqueeze(-1)

    elif state.dim() == 4:
        # Caso (B, T, S, K) con goal (B, S, K)
        matches = torch.all(_mode_idx(state) == _mode_idx(goal).unsqueeze(1), dim=2).unsqueeze(-1)

    else:
        raise ValueError(f"Estado con número de dimensiones no soportado: {state.dim()}")

    return torch.where(matches, torch.tensor(0), torch.tensor(-1))


def log_prob_reward(dist, goal: torch.Tensor, log_threshold: float) -> torch.Tensor:
    """
    Args:
        dist: Independent(OneHotCategorical(...), 1) built from RSSM logits
        goal: (S, K) or (B, S, K) — one-hot goal
        log_threshold: log of the probability threshold
    Returns:
        Reward tensor (B, 1) or (B, T, 1): 0 if log_prob >= log_threshold, else -1.
    """
    if dist.base_dist.logits.dim() == 4:  # (B, T, S, K) case
        goal = goal.unsqueeze(1).expand_as(dist.base_dist.logits)
    total_log_prob = dist.log_prob(goal).unsqueeze(-1)
    return torch.where(total_log_prob >= log_threshold, torch.tensor(0), torch.tensor(-1))


def prob_reward(dist, goal: torch.Tensor) -> torch.Tensor:
    """
    Args:
        dist: Independent(OneHotCategorical(...), 1) built from RSSM logits
        goal: (S, K) or (B, S, K) — one-hot goal
    Returns:
        Reward tensor (B, 1) or (B, T, 1): probability in [0, 1].
    """
    if dist.base_dist.logits.dim() == 4:  # (B, T, S, K) case
        goal = goal.unsqueeze(1).expand_as(dist.base_dist.logits)
    return dist.log_prob(goal).exp().unsqueeze(-1)


def argmax_full_reward(logit: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """
    Compute reward by comparing the goal against the argmax one-hot of the logits.

    Args:
        logit (torch.Tensor): Prior logits. Shape (B, S, K) or (B, T, S, K).
        goal (torch.Tensor): The desired goal state. Shape (S, K) or (B, S, K).

    Returns:
        torch.Tensor: Reward tensor of shape (B, 1) or (B, T, 1).
    """
    K = logit.shape[-1]
    hard_stoch = F.one_hot(torch.argmax(logit, dim=-1), num_classes=K).float()
    return full_goal_reward(hard_stoch, goal)


def max_cosine_reward(state: torch.Tensor, goal: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Max-normalized dot product between the state and goal logit vectors.

    Each (S, K) latent is flattened into a single S*K vector; the reward is
    r = (g/m)^T (s/m) = (g . s) / m^2, with m = max(||g||, ||s||). It peaks at
    1 when g == s and stays roughly in [-1, 1]. Both `state` and `goal` are the
    raw logits returned by RSSM's obs_step/img_step (not one-hot samples).

    Args:
        state (torch.Tensor): Current-state logits. Shape (B, S, K) or (B, T, S, K).
        goal (torch.Tensor): Goal logits. Shape (S, K) or (B, S, K).
        eps (float): Guards the all-zero degenerate case.

    Returns:
        torch.Tensor: Reward tensor of shape (B, 1) or (B, T, 1).
    """
    if state.dim() == 3:
        # Caso (B, S, K) con goal (S, K) o (B, S, K)
        s = state.flatten(start_dim=1)  # (B, S*K)
        g = goal.reshape(*goal.shape[:-2], -1)  # (S*K,) o (B, S*K)

    elif state.dim() == 4:
        # Caso (B, T, S, K) con goal (B, S, K)
        s = state.flatten(start_dim=2)  # (B, T, S*K)
        g = goal.reshape(goal.shape[0], -1).unsqueeze(1)  # (B, 1, S*K)

    else:
        raise ValueError(f"Estado con número de dimensiones no soportado: {state.dim()}")

    dot = (s * g).sum(dim=-1, keepdim=True)
    s_norm = torch.linalg.vector_norm(s, dim=-1, keepdim=True)
    g_norm = torch.linalg.vector_norm(g, dim=-1, keepdim=True)
    m = torch.maximum(s_norm, g_norm)
    return dot / m.pow(2).clamp_min(eps)
