import sys
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest


def _make_time_step(size=(64, 64), reward=0.5, last=False, discount=1.0):
    time_step = MagicMock()
    time_step.observation = np.zeros((*size, 3), dtype=np.uint8)
    time_step.reward = reward
    time_step.discount = discount
    time_step.last.return_value = last
    return time_step


def _make_dm_env_mock(size=(64, 64), n_actions=5):
    dm_env = MagicMock()
    dm_env.action_spec.return_value.num_values = n_actions
    dm_env.reset.return_value = _make_time_step(size, reward=None)
    dm_env.step.return_value = _make_time_step(size)
    dm_env.some_attr = "proxy_value"
    return dm_env


@pytest.fixture
def mock_task(monkeypatch):
    """Stub the whole `memory_maze` package.

    Importing the real one defaults MUJOCO_GL to egl and initialises a dm_control
    renderer, which is unavailable in CI and on macOS. `MemoryMaze` imports it
    lazily, so putting a mock in `sys.modules` first is enough.
    """
    dm_env = _make_dm_env_mock()
    fake_memory_maze = MagicMock()
    fake_memory_maze.tasks.memory_maze_9x9 = lambda **kwargs: dm_env
    monkeypatch.setitem(sys.modules, "memory_maze", fake_memory_maze)
    monkeypatch.setitem(sys.modules, "memory_maze.tasks", fake_memory_maze.tasks)
    return dm_env


class TestMemoryMazeInit:
    def test_builds_dm_env_task(self, mock_task):
        from her_dream.envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9", size=(64, 64), seed=0)
        assert mm._size == (64, 64)
        assert mm._env is mock_task

    def test_unknown_task_raises(self, mock_task):
        from her_dream.envs.memorymaze import MemoryMaze

        with pytest.raises(ValueError):
            MemoryMaze("7x7")


class TestMemoryMazeObsAndActionSpace:
    def test_observation_space_structure(self, mock_task):
        from her_dream.envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9", size=(64, 64))
        spaces = mm.observation_space.spaces
        assert "image" in spaces
        assert "is_first" in spaces
        assert "is_last" in spaces
        assert "is_terminal" in spaces
        assert spaces["image"].shape == (64, 64, 3)

    def test_action_space_is_discrete(self, mock_task):
        from her_dream.envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        assert isinstance(mm.action_space, gym.spaces.Discrete)
        assert mm.action_space.n == 5


class TestMemoryMazeStep:
    def test_step_is_first_false(self, mock_task):
        from her_dream.envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        obs, rew, done, info = mm.step(0)
        assert obs["is_first"] is False
        assert rew == 0.5

    def test_step_is_last_matches_done(self, mock_task):
        mock_task.step.return_value = _make_time_step(last=True, discount=1.0)
        from her_dream.envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        obs, _, done, _ = mm.step(0)
        assert obs["is_last"] is True
        assert done is True

    def test_step_is_terminal_when_discount_zero(self, mock_task):
        mock_task.step.return_value = _make_time_step(last=True, discount=0.0)
        from her_dream.envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        obs, _, _, info = mm.step(0)
        assert obs["is_terminal"] is True
        assert info["is_terminal"] is True

    def test_truncation_is_not_terminal(self, mock_task):
        """dm_env time-limit truncation ends the episode but keeps discount 1."""
        mock_task.step.return_value = _make_time_step(last=True, discount=1.0)
        from her_dream.envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        obs, _, done, _ = mm.step(0)
        assert done is True
        assert obs["is_terminal"] is False

    def test_none_reward_becomes_zero(self, mock_task):
        mock_task.step.return_value = _make_time_step(reward=None)
        from her_dream.envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        _, rew, _, _ = mm.step(0)
        assert rew == 0.0


class TestMemoryMazeReset:
    def test_reset_is_first(self, mock_task):
        from her_dream.envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        obs = mm.reset()
        assert obs["is_first"] is True
        assert obs["is_last"] is False
        assert obs["is_terminal"] is False

    def test_reset_has_image(self, mock_task):
        from her_dream.envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        obs = mm.reset()
        assert "image" in obs


class TestMemoryMazeGetAttr:
    def test_proxies_attribute_to_inner_env(self, mock_task):
        from her_dream.envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        assert mm.some_attr == "proxy_value"

    def test_dunder_attr_raises_attribute_error(self, mock_task):
        from her_dream.envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        with pytest.raises(AttributeError):
            _ = mm.__nonexistent__

    def test_missing_attr_raises_value_error(self, mock_task):
        del mock_task.nonexistent_attr
        from her_dream.envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        with pytest.raises(ValueError):
            _ = mm.nonexistent_attr
