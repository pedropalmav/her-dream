import torch


def make_reward(config):
    match config.goal_type:
        case "first_row":
            return first_row_reward
        case "row_by_row":
            return row_by_row_reward
        case "dont_care":
            return dont_care_reward
        case "full":
            return full_goal_reward
        case _:
            raise ValueError(f"Tipo de objetivo no soportado: {config.goal_type}")


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
        first_rows = state[:, 0, :]
        matches = torch.all(first_rows == goal, dim=1, keepdim=True)

    elif state.dim() == 4:
        # Caso (B, T, S, K) con goal (B, K)
        first_rows = state[:, :, 0, :]
        goal_expanded = goal.unsqueeze(1).expand_as(first_rows)
        matches = torch.all(first_rows == goal_expanded, dim=-1, keepdim=True)

    else:
        raise ValueError(
            f"Estado con número de dimensiones no soportado: {state.dim()}"
        )

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
        matches = torch.all(state == goal, dim=-1)
        num_matching_rows = matches.sum(dim=1, keepdim=True)

    elif state.dim() == 4:
        # Caso (B, T, S, K) con goal (B, S, K)
        _, _, S, _ = state.shape
        goal_expanded = goal.unsqueeze(1).expand_as(state)
        matches = torch.all(state == goal_expanded, dim=-1)
        num_matching_rows = matches.sum(dim=2, keepdim=True)

    else:
        raise ValueError(
            f"Estado con número de dimensiones no soportado: {state.dim()}"
        )

    return num_matching_rows / S - 1


def dont_care_reward(
    state: torch.Tensor, goal: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute reward based on how many rows of the state match the goal, ignoring masked positions.

    Args:
        state (torch.Tensor): The current state of the environment.
                              Shape (B, S, K) or (B, T, S, K).
        goal (torch.Tensor): The desired goal state.
                             Shape (S, K) or (B, S, K).
        mask (torch.Tensor): A binary tensor indicating which positions to ignore.
                             Shape (S, K) or (B, S, K).

    Returns:
        torch.Tensor: Reward tensor of shape (B, 1) or (B, T, 1).
    """
    masked_state = torch.where(mask == 1, state, 0)
    return full_goal_reward(masked_state, goal)


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
        # Caso (B, S, K) con goal (S, K)
        matches = torch.all(state == goal, dim=(1, 2)).unsqueeze(-1)

    elif state.dim() == 4:
        # Caso (B, T, S, K) con goal (B, S, K)
        goal_expanded = goal.unsqueeze(1).expand_as(state)
        matches = torch.all(state == goal_expanded, dim=(2, 3)).unsqueeze(-1)

    else:
        raise ValueError(
            f"Estado con número de dimensiones no soportado: {state.dim()}"
        )

    return torch.where(matches, torch.tensor(0), torch.tensor(-1))
