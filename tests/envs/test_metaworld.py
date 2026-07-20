import sys
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest


def _make_metaworld_env_mock(size=(64, 64)):
    """Return a mock metaworld inner env."""
    mock_env = MagicMock()
    mock_env.observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)
    mock_env.action_space.low = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    mock_env.action_space.high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    mock_env.render.return_value = np.zeros((*size, 3), dtype=np.uint8)
    mock_env.step.return_value = (
        np.zeros(9, dtype=np.float32),
        1.0,
        False,
        False,
        {"success": 0.0},
    )
    mock_env.reset.return_value = (np.zeros(9, dtype=np.float32), {})
    mock_env.model.cam_pos = [[0, 0, 0], [0, 0, 0], [0.5, 0.05, 0.6]]
    mock_env.mujoco_renderer = MagicMock()
    mock_env._freeze_rand_vec = True
    return mock_env


def _make_mt1_mock(env_mock):
    mock_mt1 = MagicMock()
    mock_mt1.train_classes = {"reach-v3": MagicMock(return_value=env_mock)}
    mock_mt1.train_tasks = [MagicMock()]
    return mock_mt1


def _make_metaworld(name="reach", camera=None, size=(64, 64), action_repeat=1, seed=0):
    env_mock = _make_metaworld_env_mock(size=size)
    mt1_mock = _make_mt1_mock(env_mock)

    mock_metaworld = MagicMock()
    mock_metaworld.MT1.return_value = mt1_mock

    with patch.dict(sys.modules, {"metaworld": mock_metaworld}):
        from her_dream.envs.metaworld import MetaWorld

        env = MetaWorld(name, action_repeat=action_repeat, size=size, camera=camera, seed=seed)
    return env, env_mock


class TestMetaWorldInit:
    def test_default_camera_none(self):
        env, mock_env = _make_metaworld(camera=None)
        assert env._camera is None

    def test_corner2_camera_adjusts_cam_pos(self):
        env_mock = _make_metaworld_env_mock()
        mt1_mock = _make_mt1_mock(env_mock)
        mock_metaworld = MagicMock()
        mock_metaworld.MT1.return_value = mt1_mock
        with patch.dict(sys.modules, {"metaworld": mock_metaworld}):
            from her_dream.envs.metaworld import MetaWorld

            MetaWorld("reach", camera="corner2")
        assert env_mock.model.cam_pos[2] == [0.75, 0.075, 0.7]

    def test_renderer_size_set(self):
        env, mock_env = _make_metaworld(size=(84, 84))
        assert mock_env.mujoco_renderer.width == 84
        assert mock_env.mujoco_renderer.height == 84

    def test_freeze_rand_vec_disabled(self):
        env, mock_env = _make_metaworld()
        assert mock_env._freeze_rand_vec is False


class TestMetaWorldObsAndActionSpace:
    def test_observation_space_has_image(self):
        env, _ = _make_metaworld()
        assert "image" in env.observation_space.spaces

    def test_observation_space_has_state(self):
        env, _ = _make_metaworld()
        assert "state" in env.observation_space.spaces

    def test_observation_space_has_log_success(self):
        env, _ = _make_metaworld()
        assert "log_success" in env.observation_space.spaces

    def test_action_space_is_box(self):
        env, _ = _make_metaworld()
        assert isinstance(env.action_space, gym.spaces.Box)


class TestMetaWorldStep:
    def test_step_reward_accumulates_over_repeats(self):
        env, mock_env = _make_metaworld(action_repeat=3)
        mock_env.step.return_value = (np.zeros(9), 2.0, False, False, {"success": 0.0})
        obs, reward, _, _ = env.step(np.zeros(4))
        assert reward == 6.0

    def test_step_early_exit_on_terminated(self):
        env, mock_env = _make_metaworld(action_repeat=4)
        call_count = [0]

        def step_side(action):
            call_count[0] += 1
            return np.zeros(9), 1.0, True, False, {"success": 0.0}

        mock_env.step.side_effect = step_side
        env.step(np.zeros(4))
        assert call_count[0] == 1

    def test_step_early_exit_on_truncated(self):
        env, mock_env = _make_metaworld(action_repeat=4)
        call_count = [0]

        def step_side(action):
            call_count[0] += 1
            return np.zeros(9), 1.0, False, True, {"success": 0.0}

        mock_env.step.side_effect = step_side
        env.step(np.zeros(4))
        assert call_count[0] == 1

    def test_step_success_clamped_to_1(self):
        env, mock_env = _make_metaworld(action_repeat=3)
        mock_env.step.return_value = (np.zeros(9), 0.0, False, False, {"success": 1.0})
        obs, _, _, _ = env.step(np.zeros(4))
        assert obs["log_success"] is True

    def test_step_is_last_on_terminated(self):
        env, mock_env = _make_metaworld()
        mock_env.step.return_value = (np.zeros(9), 0.0, True, False, {"success": 0.0})
        obs, _, is_last, _ = env.step(np.zeros(4))
        assert is_last is True

    def test_step_is_last_on_truncated(self):
        env, mock_env = _make_metaworld()
        mock_env.step.return_value = (np.zeros(9), 0.0, False, True, {"success": 0.0})
        obs, _, is_last, _ = env.step(np.zeros(4))
        assert is_last is True

    def test_step_is_terminal_only_on_terminated(self):
        env, mock_env = _make_metaworld()
        mock_env.step.return_value = (np.zeros(9), 0.0, False, True, {"success": 0.0})
        obs, _, _, _ = env.step(np.zeros(4))
        assert obs["is_terminal"] is False

    def test_step_finite_action_required(self):
        env, _ = _make_metaworld()
        with pytest.raises(AssertionError):
            env.step(np.array([np.inf, 0.0, 0.0, 0.0]))

    def test_step_is_first_false(self):
        env, _ = _make_metaworld()
        obs, _, _, _ = env.step(np.zeros(4))
        assert obs["is_first"] is False


class TestMetaWorldReset:
    def test_reset_is_first(self):
        env, _ = _make_metaworld()
        obs = env.reset()
        assert obs["is_first"] is True
        assert obs["is_last"] is False
        assert obs["is_terminal"] is False

    def test_reset_log_success_false(self):
        env, _ = _make_metaworld()
        obs = env.reset()
        assert obs["log_success"] is False


class TestMetaWorldRender:
    def test_render_wrong_mode_raises(self):
        env, _ = _make_metaworld()
        with pytest.raises(ValueError, match="rgb_array"):
            env.render(mode="human")

    def test_render_corner2_flips_axis(self):
        env, mock_env = _make_metaworld(camera="corner2")
        img = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        mock_env.render.return_value = img
        result = env.render()
        np.testing.assert_array_equal(result, np.flip(img, axis=0))

    def test_render_non_corner2_returns_direct(self):
        env, mock_env = _make_metaworld(camera=None)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_env.render.return_value = img
        result = env.render()
        assert result is img
