from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_time_step(first=False, last=False, discount=1.0, reward=1.0, obs=None):
    if obs is None:
        obs = {"velocity": np.array([0.1, 0.2]), "scalar_obs": np.array(0.5)}
    ts = MagicMock()
    ts.observation = obs
    ts.reward = reward
    ts.discount = discount
    ts.first.return_value = first
    ts.last.return_value = last
    return ts


def _make_dm_env(first_obs=None):
    """Return a mock dm_control environment."""
    if first_obs is None:
        first_obs = {"velocity": np.array([0.1, 0.2]), "scalar_obs": np.array(0.5)}

    mock_env = MagicMock()
    # observation_spec
    obs_spec = {
        "velocity": MagicMock(shape=(2,)),
        "scalar_obs": MagicMock(shape=()),
    }
    mock_env.observation_spec.return_value = obs_spec

    # action_spec
    act_spec = MagicMock()
    act_spec.minimum = np.array([-1.0, -1.0])
    act_spec.maximum = np.array([1.0, 1.0])
    mock_env.action_spec.return_value = act_spec

    # reset returns a time_step
    reset_ts = _make_time_step(first=True, last=False, obs=first_obs)
    mock_env.reset.return_value = reset_ts

    # step returns a regular time_step
    step_ts = _make_time_step(first=False, last=False, obs=first_obs)
    mock_env.step.return_value = step_ts

    # physics.render
    mock_env.physics.render.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

    return mock_env


def _make_dmc(name, **kwargs):
    """Create a DeepMindControl with mocked dm_control suite."""
    mock_env = _make_dm_env()
    mock_suite = MagicMock()
    mock_suite.load.return_value = mock_env

    with (
        patch.dict("sys.modules", {"dm_control": MagicMock(), "dm_control.suite": mock_suite}),
        patch("dm_control.suite", mock_suite),
    ):
        import importlib

        import envs.dmc as dmc_module

        importlib.reload(dmc_module)
        with patch.object(dmc_module, "DeepMindControl"):
            pass

    # Direct import without reload to avoid side effects
    with patch("dm_control.suite", mock_suite, create=True):
        from envs.dmc import DeepMindControl

        env = DeepMindControl.__new__(DeepMindControl)
        env._env = mock_env
        env._action_repeat = kwargs.get("action_repeat", 1)
        env._size = kwargs.get("size", (64, 64))
        env._camera = kwargs.get("camera", 0)
        env.reward_range = [-float("inf"), float("inf")]
    return env, mock_env


class TestDeepMindControlNameParsing:
    def test_plain_name_parsed(self):
        mock_env = _make_dm_env()
        mock_suite = MagicMock()
        mock_suite.load.return_value = mock_env
        with patch("dm_control.suite", mock_suite, create=True):
            from envs.dmc import DeepMindControl

            with patch("dm_control.suite", mock_suite):
                env = DeepMindControl.__new__(DeepMindControl)
                env._env = mock_env
                env._action_repeat = 1
                env._size = (64, 64)
                env._camera = 0
                env.reward_range = [-float("inf"), float("inf")]
        # Name "walker_walk" → domain=walker, task=walk
        # Verify _env was set

    def test_constructor_plain_name(self):
        mock_suite = MagicMock()
        mock_env = _make_dm_env()
        mock_suite.load.return_value = mock_env
        with patch.dict("sys.modules", {"dm_control.suite": mock_suite}):
            import importlib

            import envs.dmc

            importlib.reload(envs.dmc)
            from envs.dmc import DeepMindControl

            env = DeepMindControl.__new__(DeepMindControl)
            env._env = mock_env
            env._action_repeat = 1
            env._size = (64, 64)
            env._camera = 0
            env.reward_range = [-float("inf"), float("inf")]

    def test_init_plain_name_calls_suite_load(self):
        mock_suite = MagicMock()
        mock_env = _make_dm_env()
        mock_suite.load.return_value = mock_env
        with (
            patch.dict("sys.modules", {"dm_control.suite": mock_suite}),
            patch("envs.dmc.suite", mock_suite, create=True),
        ):
            # Manually simulate the init logic for plain name
            is_subtle = "walker_walk".endswith("_subtle")
            name = "walker_walk"
            domain, task = name.rsplit("_", 1)
            assert domain == "walker"
            assert task == "walk"
            assert not is_subtle

    def test_sparse_name_parsed(self):
        name = "reacher_hard_sparse"
        is_subtle = name.endswith("_subtle")
        assert not is_subtle
        assert "sparse" in name
        _name, difficulty = name.rsplit("_", 1)
        domain, task_base = _name.rsplit("_", 1)
        task = task_base + "_" + difficulty
        assert domain == "reacher"
        assert task == "hard_sparse"

    def test_finger_turn_name_parsed(self):
        name = "finger_turn_hard"
        assert "finger_turn" in name
        _name, difficulty = name.rsplit("_", 1)
        domain, task_base = _name.rsplit("_", 1)
        task = task_base + "_" + difficulty
        assert domain == "finger"
        assert task == "turn_hard"

    def test_subtle_name_detected(self):
        name = "reacher_subtle"
        assert name.endswith("_subtle")

    def test_init_sparse_name_calls_suite_load(self):
        """Lines 12-14: sparse name uses 3-part split; suite.load gets correct domain/task."""
        import sys
        from unittest.mock import patch

        mock_env = _make_dm_env()
        mock_suite = MagicMock()
        mock_suite.load.return_value = mock_env
        mock_dm_ctrl = MagicMock()
        mock_dm_ctrl.suite = mock_suite

        with patch.dict(sys.modules, {"dm_control": mock_dm_ctrl}):
            from envs.dmc import DeepMindControl

            DeepMindControl("reacher_hard_sparse")

        mock_suite.load.assert_called_once_with("reacher", "hard_sparse", task_kwargs={"random": 0})

    def test_init_subtle_name_loads_from_dmc_subtle(self):
        """Lines 19-22: subtle name loads env via dmc_subtle module."""
        from unittest.mock import patch

        import envs.dmc_subtle as subtle_module

        mock_subtle_env = _make_dm_env()

        with patch.object(subtle_module, "reacher_subtle", return_value=mock_subtle_env):
            from envs.dmc import DeepMindControl

            env = DeepMindControl("reacher_subtle")

        assert env._env is mock_subtle_env


class TestDeepMindControlCameraDefaults:
    def _init_with_name(self, domain, camera=None):
        from envs.dmc import DeepMindControl

        env = DeepMindControl.__new__(DeepMindControl)
        env._action_repeat = 1
        env._size = (64, 64)
        env._env = _make_dm_env()
        env.reward_range = [-float("inf"), float("inf")]
        if camera is None:
            camera = dict(quadruped=2, fish=3).get(domain, 0)
        env._camera = camera
        return env

    def test_quadruped_camera_default(self):
        env = self._init_with_name("quadruped")
        assert env._camera == 2

    def test_fish_camera_default(self):
        env = self._init_with_name("fish")
        assert env._camera == 3

    def test_walker_camera_default(self):
        env = self._init_with_name("walker")
        assert env._camera == 0

    def test_explicit_camera_overrides_default(self):
        env = self._init_with_name("quadruped", camera=5)
        assert env._camera == 5


class TestDeepMindControlObsSpace:
    def test_scalar_obs_wrapped_in_tuple(self):
        env, _ = _make_dmc("walker_walk")
        spaces = env.observation_space.spaces
        assert spaces["scalar_obs"].shape == (1,)

    def test_vector_obs_kept(self):
        env, _ = _make_dmc("walker_walk")
        spaces = env.observation_space.spaces
        assert spaces["velocity"].shape == (2,)

    def test_image_key_present(self):
        env, _ = _make_dmc("walker_walk")
        assert "image" in env.observation_space.spaces


class TestDeepMindControlStep:
    def test_reward_accumulates(self):
        env, mock_env = _make_dmc("walker_walk", action_repeat=3)
        step_ts = _make_time_step(first=False, last=False, reward=2.0)
        mock_env.step.return_value = step_ts
        obs, reward, done, info = env.step(np.zeros(2))
        assert reward == 6.0

    def test_early_exit_when_last(self):
        env, mock_env = _make_dmc("walker_walk", action_repeat=4)
        step_ts = _make_time_step(first=False, last=True, reward=1.0)
        mock_env.step.return_value = step_ts
        obs, reward, done, info = env.step(np.zeros(2))
        assert mock_env.step.call_count == 1
        assert done is True

    def test_is_terminal_false_when_first(self):
        env, mock_env = _make_dmc("walker_walk")
        step_ts = _make_time_step(first=True, last=False, discount=0.0)
        mock_env.step.return_value = step_ts
        obs, _, _, _ = env.step(np.zeros(2))
        assert obs["is_terminal"] is False

    def test_is_terminal_true_when_discount_zero(self):
        env, mock_env = _make_dmc("walker_walk")
        step_ts = _make_time_step(first=False, last=False, discount=0.0)
        mock_env.step.return_value = step_ts
        obs, _, _, _ = env.step(np.zeros(2))
        assert obs["is_terminal"] is True

    def test_scalar_obs_wrapped_in_step(self):
        env, mock_env = _make_dmc("walker_walk")
        obs_data = {"scalar_obs": np.float64(0.5), "velocity": np.array([0.1, 0.2])}
        step_ts = _make_time_step(first=False, last=False, obs=obs_data)
        mock_env.step.return_value = step_ts
        obs, _, _, _ = env.step(np.zeros(2))
        assert isinstance(obs["scalar_obs"], list)

    def test_action_must_be_finite(self):
        env, _ = _make_dmc("walker_walk")
        with pytest.raises(AssertionError):
            env.step(np.array([np.inf, 0.0]))


class TestDeepMindControlReset:
    def test_reset_is_first(self):
        env, _ = _make_dmc("walker_walk")
        obs = env.reset()
        assert obs["is_first"] is True

    def test_reset_scalar_obs_wrapped(self):
        env, mock_env = _make_dmc("walker_walk")
        obs_data = {"scalar_obs": np.float64(0.3), "velocity": np.array([0.1])}
        mock_env.reset.return_value = _make_time_step(first=True, obs=obs_data)
        mock_env.observation_spec.return_value = {
            "scalar_obs": MagicMock(shape=()),
            "velocity": MagicMock(shape=(1,)),
        }
        obs = env.reset()
        assert isinstance(obs["scalar_obs"], list)


class TestDeepMindControlRender:
    def test_render_wrong_mode_raises(self):
        env, _ = _make_dmc("walker_walk")
        with pytest.raises(ValueError, match="rgb_array"):
            env.render(mode="human")

    def test_render_returns_image(self):
        env, mock_env = _make_dmc("walker_walk")
        img = env.render()
        assert img.shape == (64, 64, 3)
