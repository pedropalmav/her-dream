"""Shared fixtures for the envs tests.

Covers two independent needs:
- the goal-image rendering used by goal_sample="image" (`generator`), and
- mock gymnasium envs (`discrete_env` / `box_env`) for the generic env tests.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

import her_dream.goals as goals
from her_dream.envs.fixed_goal import make_fixed_goal_env
from her_dream.envs.goal_image import GoalImageGenerator

SIZE = 10  # grid size; interior cells are 1..SIZE-2

N_CLASSES = 4
N_ROWS = 2


@pytest.fixture
def generator():
    return GoalImageGenerator(lambda: make_fixed_goal_env(size=SIZE))


class DictNamespace(SimpleNamespace):
    """SimpleNamespace that also supports .get() like a dict."""

    def get(self, key, default=None):
        return getattr(self, key, default)


def make_goal_config(**overrides):
    defaults = dict(
        stochastic_classes=N_CLASSES,
        stochastic_rows=N_ROWS,
        goal_type="first_row",
    )
    merged = goals.with_default_descriptors({**defaults, **overrides})
    return DictNamespace(**merged)


def make_discrete_env(n_actions=4, obs_keys=None):
    """Returns a MagicMock that behaves like a gymnasium Discrete env with Dict obs."""
    if obs_keys is None:
        obs_keys = ["image"]

    spaces = {k: gym.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8) for k in obs_keys}
    obs_space = gym.spaces.Dict(spaces)

    env = MagicMock(spec=gym.Env)
    env.action_space = gym.spaces.Discrete(n_actions)
    env.observation_space = obs_space
    env.get_wrapper_attr = None  # mark as non-gymnasium-wrapper for some tests

    def _reset():
        return {k: np.zeros((8, 8, 3), dtype=np.uint8) for k in obs_keys}

    def _step(action):
        obs = {k: np.zeros((8, 8, 3), dtype=np.uint8) for k in obs_keys}
        return obs, 0.0, False, {}

    env.reset.side_effect = _reset
    env.step.side_effect = _step
    return env


def make_box_env(low=-1.0, high=1.0, n_actions=3, obs_keys=None):
    """Returns a MagicMock that behaves like a gymnasium Box-action env with Dict obs."""
    if obs_keys is None:
        obs_keys = ["image"]

    spaces = {k: gym.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8) for k in obs_keys}
    obs_space = gym.spaces.Dict(spaces)

    env = MagicMock(spec=gym.Env)
    env.action_space = gym.spaces.Box(
        low=np.full(n_actions, low, dtype=np.float32),
        high=np.full(n_actions, high, dtype=np.float32),
        dtype=np.float32,
    )
    env.observation_space = obs_space

    def _reset():
        return {k: np.zeros((8, 8, 3), dtype=np.uint8) for k in obs_keys}

    def _step(action):
        obs = {k: np.zeros((8, 8, 3), dtype=np.uint8) for k in obs_keys}
        return obs, 0.0, False, {}

    env.reset.side_effect = _reset
    env.step.side_effect = _step
    return env


@pytest.fixture
def discrete_env():
    return make_discrete_env()


@pytest.fixture
def box_env():
    return make_box_env()
