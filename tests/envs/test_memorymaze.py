from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest


def _make_memory_maze_mock(task="9x9", size=(64, 64)):
    mock_old_env = MagicMock()
    mock_old_env.observation_space = MagicMock()
    mock_old_env.observation_space.spaces = {}
    mock_old_env.action_space.n = 5
    mock_old_env.reset.return_value = np.zeros((*size, 3), dtype=np.uint8)
    mock_old_env.step.return_value = (
        np.zeros((*size, 3), dtype=np.uint8),
        0.5,
        False,
        {"is_terminal": False},
    )
    mock_old_env.some_attr = "proxy_value"
    return mock_old_env


@pytest.fixture
def mock_gym_make(monkeypatch):
    mock_old_env = _make_memory_maze_mock()
    import gym as old_gym

    monkeypatch.setattr(old_gym, "make", lambda *a, **kw: mock_old_env)
    return mock_old_env


class TestMemoryMazeInit:
    def test_creates_old_gym_env(self, mock_gym_make):
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9", size=(64, 64), seed=0)
        assert mm._size == (64, 64)

    def test_obs_is_dict_set(self, mock_gym_make):
        mock_gym_make.observation_space.spaces = {"image": None}
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        assert mm._obs_is_dict is True

    def test_obs_is_dict_false_when_no_spaces(self, mock_gym_make):
        del mock_gym_make.observation_space.spaces
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        assert mm._obs_is_dict is False


class TestMemoryMazeObsAndActionSpace:
    def test_observation_space_structure(self, mock_gym_make):
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9", size=(64, 64))
        spaces = mm.observation_space.spaces
        assert "image" in spaces
        assert "is_first" in spaces
        assert "is_last" in spaces
        assert "is_terminal" in spaces
        assert spaces["image"].shape == (64, 64, 3)

    def test_action_space_is_discrete(self, mock_gym_make):
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        assert isinstance(mm.action_space, gym.spaces.Discrete)
        assert mm.action_space.n == 5


class TestMemoryMazeStep:
    def test_step_is_first_false(self, mock_gym_make):
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        obs, rew, done, info = mm.step(0)
        assert obs["is_first"] is False

    def test_step_is_last_matches_done(self, mock_gym_make):
        mock_gym_make.step.return_value = (np.zeros((64, 64, 3), dtype=np.uint8), 0.0, True, {"is_terminal": False})
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        obs, _, done, _ = mm.step(0)
        assert obs["is_last"] is True
        assert done is True

    def test_step_is_terminal_from_info(self, mock_gym_make):
        mock_gym_make.step.return_value = (np.zeros((64, 64, 3), dtype=np.uint8), 0.0, True, {"is_terminal": True})
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        obs, _, _, _ = mm.step(0)
        assert obs["is_terminal"] is True

    def test_step_is_terminal_defaults_to_false(self, mock_gym_make):
        mock_gym_make.step.return_value = (np.zeros((64, 64, 3), dtype=np.uint8), 0.0, False, {})
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        obs, _, _, _ = mm.step(0)
        assert obs["is_terminal"] is False


class TestMemoryMazeReset:
    def test_reset_is_first(self, mock_gym_make):
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        obs = mm.reset()
        assert obs["is_first"] is True
        assert obs["is_last"] is False
        assert obs["is_terminal"] is False

    def test_reset_has_image(self, mock_gym_make):
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        obs = mm.reset()
        assert "image" in obs


class TestMemoryMazeGetAttr:
    def test_proxies_attribute_to_inner_env(self, mock_gym_make):
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        assert mm.some_attr == "proxy_value"

    def test_dunder_attr_raises_attribute_error(self, mock_gym_make):
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        with pytest.raises(AttributeError):
            _ = mm.__nonexistent__

    def test_missing_attr_raises_value_error(self, mock_gym_make):
        mock_gym_make.nonexistent_attr = MagicMock(side_effect=AttributeError)
        del mock_gym_make.nonexistent_attr
        from envs.memorymaze import MemoryMaze

        mm = MemoryMaze("9x9")
        with pytest.raises(ValueError):
            _ = mm.nonexistent_attr
