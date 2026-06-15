from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .buffer import Buffer


class HERStrategy(Enum):
    FUTURE = 0
    FINAL = 1
    EPISODE = 2


class HERBuffer(Buffer):
    def __init__(self, config, reward_function):
        super().__init__(config)

        self.reward_function = reward_function
        self.goal_type = config.goal_type
        self.rssm = None
        self.her_ratio = float(config.her_ratio)
        self.her_strategy = HERStrategy[config.her_strategy.upper()]

        self.n_envs = int(config.n_envs)
        self.max_columns = int(config.max_size / self.n_envs)
        self.pos = 0

        self.ep_start = np.zeros((self.n_envs, self.max_columns), dtype=np.int64)
        self.ep_length = np.zeros((self.n_envs, self.max_columns), dtype=np.int64)
        self._current_ep_start = np.zeros(self.n_envs, dtype=np.int64)

    def add_transition(self, data):
        # TODO: Invalidate overwritten episodes
        # When the buffer is full, we rewrite on old episodes. When we start to
        # rewrite on an old episodes, we want the whole old episode to be deleted
        # (and not only the transition on which we rewrite). To do this, we set
        # the length of the old episode to 0, so it can't be sampled anymore.
        for env_id in range(self.n_envs):
            episode_start = self.ep_start[env_id, self.pos]
            episode_length = self.ep_length[env_id, self.pos]
            if episode_length > 0:
                episode_end = episode_start + episode_length
                episode_indices = np.arange(self.pos, episode_end) % self.max_columns
                self.ep_length[env_id, episode_indices] = 0

        self.ep_start[:, self.pos] = self._current_ep_start.copy()
        super().add_transition(data)
        self.pos = (self.pos + 1) % self.max_columns

        for env_id in range(self.n_envs):
            self._compute_episode_length(env_id, data["is_last"][env_id])

    def _compute_episode_length(self, env_id: int, is_last: bool) -> None:
        episode_start = self._current_ep_start[env_id]
        episode_end = self.pos
        if episode_end < episode_start:
            episode_end += self.max_columns
        episode_indices = np.arange(episode_start, episode_end) % self.max_columns
        self.ep_length[env_id, episode_indices] = episode_end - episode_start
        if is_last:
            self._current_ep_start[env_id] = self.pos

    def sample(self):
        # Info contains the indices of the sampled transitions in format (step, episode)
        sample_td, info = self._get_samples()
        envs, steps = self._parse_indices(info)
        valid_batches = self._get_valid_batches(envs, steps)
        valid_indices = np.flatnonzero(valid_batches)

        n_her = min(int(self.her_ratio * self.batch_size), len(valid_indices))
        her_batch_indices = np.random.choice(valid_indices, size=n_her, replace=False)
        sample_td = self._apply_her(sample_td, envs, steps, her_batch_indices)

        return self._parse_sample(sample_td, info)

    def _parse_indices(self, info):
        # (B*(T+1), ...) -> (B, T+1, ...)
        indices = info["index"]
        envs = indices[1].view(-1, self.batch_length + 1)
        steps = indices[0].view(-1, self.batch_length + 1)
        return envs, steps

    def _get_valid_batches(self, envs, steps):
        is_valid = self.ep_length > 0
        envs, steps = envs.numpy(), steps.numpy()
        return is_valid[envs, steps].all(axis=1)

    def _apply_her(
        self,
        sample_td: TensorDict,
        envs,
        steps,
        her_batch_indices: np.ndarray,
    ):
        for batch_idx in her_batch_indices:
            new_goal = self._sample_goal(envs[batch_idx], steps[batch_idx])
            sample_td["goal"][batch_idx] = new_goal

            if self.goal_type == "log_prob" and self.rssm is not None:
                achieved = self.rssm.get_dist(sample_td["logit"][batch_idx])
            else:
                achieved = sample_td["stoch"][batch_idx]
            desired = new_goal.to(self.device)
            new_reward = self.reward_function(achieved, desired)
            sample_td["reward"][batch_idx] = new_reward
        return sample_td

    def _sample_goal(self, env_indices, step_indices):
        ep_start = self.ep_start[env_indices, step_indices]
        ep_length = self.ep_length[env_indices, step_indices]
        match self.her_strategy:
            case HERStrategy.FINAL:
                transition_indices_in_episode = ep_length - 1
            case HERStrategy.FUTURE:
                current_indices_in_episode = (step_indices - ep_start) % self.max_columns
                transition_indices_in_episode = np.random.randint(current_indices_in_episode, ep_length)
            case HERStrategy.EPISODE:
                transition_indices_in_episode = np.random.randint(0, ep_length)
            case _:
                raise ValueError(f"Invalid HER strategy: {self.her_strategy}")

        transition_indices = (ep_start + transition_indices_in_episode) % self.max_columns
        new_goal = self._buffer[env_indices, transition_indices]
        if self.goal_type == "argmax_full":
            new_goal = new_goal["logit"]
            K = new_goal.shape[-1]
            new_goal = torch.argmax(new_goal, dim=-1)
            new_goal = F.one_hot(new_goal, num_classes=K).float()
        else:
            new_goal = new_goal["stoch"]
        return new_goal[:, 0] if self.goal_type == "first_row" else new_goal
