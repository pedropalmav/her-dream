import sys
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest


def _make_crafter_mocks():
    """Return (mock_crafter_module, mock_env_instance)."""
    mock_env = MagicMock()
    mock_env.observation_space.shape = (64, 64, 3)
    mock_env.action_space.n = 17
    mock_env.reset.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
    mock_env.step.return_value = (
        np.zeros((64, 64, 3), dtype=np.uint8),
        1.0,
        False,
        {"achievements": {"collect_wood": 1, "make_table": 0}, "discount": 1},
    )
    mock_env.render.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

    achievements = {"collect_wood": 0, "make_table": 0}

    mock_crafter = MagicMock()
    mock_crafter.Env.return_value = mock_env
    mock_crafter.constants.achievements = achievements

    return mock_crafter, mock_env


@pytest.fixture(autouse=True)
def patch_crafter():
    mock_crafter, mock_env = _make_crafter_mocks()
    with patch.dict(sys.modules, {"crafter": mock_crafter}):
        yield mock_crafter, mock_env


class TestCrafterInit:
    def test_task_reward(self, patch_crafter):
        mock_crafter, _ = patch_crafter
        from envs.crafter import Crafter

        Crafter("reward", size=(64, 64), seed=0)
        mock_crafter.Env.assert_called_with(size=(64, 64), reward=True, seed=0)

    def test_task_noreward(self, patch_crafter):
        mock_crafter, _ = patch_crafter
        from envs.crafter import Crafter

        Crafter("noreward", size=(64, 64), seed=1)
        mock_crafter.Env.assert_called_with(size=(64, 64), reward=False, seed=1)

    def test_invalid_task_raises(self, patch_crafter):
        from envs.crafter import Crafter

        with pytest.raises(AssertionError):
            Crafter("invalid_task")


class TestCrafterObsAndActionSpace:
    def test_observation_space_has_image_key(self, patch_crafter):
        from envs.crafter import Crafter

        c = Crafter("reward")
        assert "image" in c.observation_space.spaces

    def test_observation_space_has_log_keys(self, patch_crafter):
        from envs.crafter import Crafter

        c = Crafter("reward")
        spaces = c.observation_space.spaces
        assert "log_collect_wood" in spaces
        assert "log_make_table" in spaces

    def test_action_space_is_discrete(self, patch_crafter):
        from envs.crafter import Crafter

        c = Crafter("reward")
        assert isinstance(c.action_space, gym.spaces.Discrete)
        assert c.action_space.n == 17


class TestCrafterStep:
    def test_step_includes_log_keys(self, patch_crafter):
        from envs.crafter import Crafter

        c = Crafter("reward")
        obs, reward, done, info = c.step(0)
        assert "log_collect_wood" in obs
        assert "log_make_table" in obs

    def test_step_reward_converted_to_float32(self, patch_crafter):
        from envs.crafter import Crafter

        c = Crafter("reward")
        _, reward, _, _ = c.step(0)
        assert isinstance(reward, np.float32)

    def test_step_is_terminal_when_discount_zero(self, patch_crafter):
        mock_crafter, mock_env = patch_crafter
        mock_env.step.return_value = (
            np.zeros((64, 64, 3), dtype=np.uint8),
            0.0,
            True,
            {"achievements": {"collect_wood": 0, "make_table": 0}, "discount": 0},
        )
        from envs.crafter import Crafter

        c = Crafter("reward")
        obs, _, _, _ = c.step(0)
        assert obs["is_terminal"] is True

    def test_step_not_terminal_when_discount_nonzero(self, patch_crafter):
        from envs.crafter import Crafter

        c = Crafter("reward")
        obs, _, _, _ = c.step(0)
        assert obs["is_terminal"] is False

    def test_step_is_last_matches_done(self, patch_crafter):
        mock_crafter, mock_env = patch_crafter
        mock_env.step.return_value = (
            np.zeros((64, 64, 3), dtype=np.uint8),
            0.0,
            True,
            {"achievements": {"collect_wood": 0, "make_table": 0}, "discount": 1},
        )
        from envs.crafter import Crafter

        c = Crafter("reward")
        obs, _, done, _ = c.step(0)
        assert obs["is_last"] is True
        assert done is True

    def test_step_is_first_always_false(self, patch_crafter):
        from envs.crafter import Crafter

        c = Crafter("reward")
        obs, _, _, _ = c.step(0)
        assert obs["is_first"] is False


class TestCrafterReset:
    def test_reset_is_first(self, patch_crafter):
        from envs.crafter import Crafter

        c = Crafter("reward")
        obs = c.reset()
        assert obs["is_first"] is True
        assert obs["is_last"] is False
        assert obs["is_terminal"] is False

    def test_reset_log_keys_zeroed(self, patch_crafter):
        from envs.crafter import Crafter

        c = Crafter("reward")
        obs = c.reset()
        assert obs["log_collect_wood"] == 0.0
        assert obs["log_make_table"] == 0.0

    def test_reset_has_image(self, patch_crafter):
        from envs.crafter import Crafter

        c = Crafter("reward")
        obs = c.reset()
        assert "image" in obs


class TestCrafterRender:
    def test_render_delegates_to_inner(self, patch_crafter):
        mock_crafter, mock_env = patch_crafter
        from envs.crafter import Crafter

        c = Crafter("reward")
        c.render()
        mock_env.render.assert_called_once()
