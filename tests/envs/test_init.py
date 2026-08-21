import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest

import her_dream.envs
import her_dream.goals as goals
from tests.envs.conftest import DictNamespace


@contextmanager
def patch_env_module(module_name, dummy):
    """Stub out a lazily-imported `her_dream.envs.<x>` submodule.

    Patching `sys.modules` alone is not enough. `make_env` does
    `import her_dream.envs.x as x`, which binds from the *parent package
    attribute* whenever that attribute already exists — and it does as soon as
    any earlier test in the session imports the real module (e.g. test_atari.py).
    The mock is then silently bypassed and the real env is constructed, so the
    failure depends on test ordering. Patching both keeps it deterministic.
    """
    attr = module_name.rsplit(".", 1)[1]
    with (
        patch.dict(sys.modules, {module_name: dummy}),
        patch.object(her_dream.envs, attr, dummy, create=True),
    ):
        yield


# ---------------------------------------------------------------------------
# Shared config factory
# ---------------------------------------------------------------------------


def make_config(task="random-goal_default", **overrides):
    """Create a minimal config for make_env."""
    defaults = dict(
        task=task,
        action_repeat=1,
        size=(64, 64),
        gray=True,
        noops=0,
        lives="unused",
        sticky=False,
        actions="all",
        time_limit=100,
        pooling=2,
        aggregate="max",
        resize="pillow",
        autostart=False,
        clip_reward=False,
        seed=0,
        camera=None,
        env_size=10,
        agent_start_pos_x=1,
        agent_start_pos_y=1,
        agent_start_dir=0,
        goal_pos_x=8,
        goal_pos_y=1,
        render_mode="rgb_array",
        mission_text=False,
        stochastic_classes=4,
        stochastic_rows=1,
        goal_type="first_row",
        env_num=2,
        eval_episode_num=1,
        device="cpu",
    )
    merged = goals.with_default_descriptors({**defaults, **overrides})
    return DictNamespace(**merged)


_SPACES = {
    "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
    "is_first": gym.spaces.Box(0, 1, (), bool),
    "is_last": gym.spaces.Box(0, 1, (), bool),
    "is_terminal": gym.spaces.Box(0, 1, (), bool),
}
_OBS = {k: np.zeros(v.shape, dtype=v.dtype) for k, v in _SPACES.items()}


class _DummyDiscreteEnv(gym.Env):
    """Real gymnasium env with Discrete action space for wrapper testing."""

    def __init__(self, n_actions=4):
        self.action_space = gym.spaces.Discrete(n_actions)
        self.observation_space = gym.spaces.Dict(_SPACES)

    def reset(self, **kwargs):
        return dict(_OBS)

    def step(self, action):
        return dict(_OBS), 0.0, False, {}


class _DummyBoxEnv(gym.Env):
    """Real gymnasium env with Box action space for wrapper testing (e.g. DMC)."""

    def __init__(self):
        self.action_space = gym.spaces.Box(np.float32(-1.0), np.float32(1.0), (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(_SPACES)

    def reset(self, **kwargs):
        return dict(_OBS)

    def step(self, action):
        return dict(_OBS), 0.0, False, {}


def _make_dummy_gym_env(n_actions=4):
    return _DummyDiscreteEnv(n_actions=n_actions)


# ---------------------------------------------------------------------------
# make_env — individual suite branches
# ---------------------------------------------------------------------------


class TestMakeEnvSuites:
    def test_invalid_suite_raises_not_implemented(self):
        from her_dream.envs import make_env

        config = make_config(task="unknown_task")
        with pytest.raises(NotImplementedError):
            make_env(config, 0)

    def test_dmc_suite_creates_env_with_normalize_actions(self):
        dummy = MagicMock()
        dummy_env = _DummyBoxEnv()
        dummy.DeepMindControl.return_value = dummy_env

        with patch_env_module("her_dream.envs.dmc", dummy):
            from her_dream.envs import make_env

            config = make_config(task="dmc_walker_walk")
            env = make_env(config, 0)
        from her_dream.envs.wrappers import Dtype

        assert isinstance(env, Dtype)

    def test_atari_suite_wraps_with_one_hot_action(self):

        dummy_atari = MagicMock()
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(18)
        dummy_atari.Atari.return_value = dummy_env

        with patch_env_module("her_dream.envs.atari", dummy_atari):
            from her_dream.envs import make_env

            config = make_config(task="atari_pong")
            env = make_env(config, 0)
        from her_dream.envs.wrappers import Dtype

        assert isinstance(env, Dtype)

    def test_memorymaze_suite_wraps_with_one_hot_action(self):
        dummy_mem = MagicMock()
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(5)
        dummy_mem.MemoryMaze.return_value = dummy_env

        with patch_env_module("her_dream.envs.memorymaze", dummy_mem):
            from her_dream.envs import make_env

            config = make_config(task="memorymaze_9x9")
            env = make_env(config, 0)
        from her_dream.envs.wrappers import Dtype

        assert isinstance(env, Dtype)

    def test_crafter_suite_wraps_with_one_hot_action(self):
        dummy_crafter = MagicMock()
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(17)
        dummy_crafter.Crafter.return_value = dummy_env

        with patch_env_module("her_dream.envs.crafter", dummy_crafter):
            from her_dream.envs import make_env

            config = make_config(task="crafter_reward")
            env = make_env(config, 0)
        from her_dream.envs.wrappers import Dtype

        assert isinstance(env, Dtype)

    def test_metaworld_suite(self):
        dummy_mw = MagicMock()
        dummy_env = _DummyBoxEnv()
        dummy_mw.MetaWorld.return_value = dummy_env

        with patch_env_module("her_dream.envs.metaworld", dummy_mw):
            from her_dream.envs import make_env

            config = make_config(task="metaworld_reach")
            env = make_env(config, 0)
        from her_dream.envs.wrappers import Dtype

        assert isinstance(env, Dtype)

    # `random-goal` and `fixed-goal` are the same suite now (cookie_env.GoalGrid);
    # both task strings stay supported because archived run configs carry them.
    # What separates the variants is goal_pos: a tuple pins the square, None
    # resamples it. `make_config` always injects goal_pos_x/y, so the random case
    # passes them as None explicitly.

    def _patch_goal_grid(self, dummy_env):
        return patch("cookie_env.envs.goal_grid.make_goal_grid_env", return_value=dummy_env)

    @pytest.mark.parametrize("task", ["random-goal_default", "fixed-goal_default", "goal-grid_default"])
    def test_goal_grid_suites_build_without_mission_text(self, task):
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(7)

        with self._patch_goal_grid(dummy_env):
            from her_dream.envs import make_env

            env = make_env(make_config(task=task, mission_text=False), 0)
        from her_dream.envs.wrappers import Dtype

        assert isinstance(env, Dtype)

    def test_configured_goal_pos_pins_the_square(self):
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(7)

        with self._patch_goal_grid(dummy_env) as mock_make:
            from her_dream.envs import make_env

            make_env(make_config(task="fixed-goal_default", goal_pos_x=8, goal_pos_y=1, mission_text=False), 0)

        assert mock_make.call_args.kwargs["goal_pos"] == (8, 1)

    def test_absent_goal_pos_leaves_the_square_random(self):
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(7)

        with self._patch_goal_grid(dummy_env) as mock_make:
            from her_dream.envs import make_env

            config = make_config(task="random-goal_default", goal_pos_x=None, goal_pos_y=None, mission_text=False)
            make_env(config, 0)

        assert mock_make.call_args.kwargs["goal_pos"] is None

    def test_random_start_passes_none_pos(self):
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(7)

        with self._patch_goal_grid(dummy_env) as mock_make:
            from her_dream.envs import make_env

            config = make_config(task="random-goal_default", mission_text=False, agent_start_random=True)
            make_env(config, 0)

        assert mock_make.call_args.kwargs["agent_start_pos"] is None

    def test_fixed_start_passes_configured_pos(self):
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(7)

        with self._patch_goal_grid(dummy_env) as mock_make:
            from her_dream.envs import make_env

            # agent_start_random absent -> getattr default False -> use configured pos
            make_env(make_config(task="random-goal_default", mission_text=False), 0)

        assert mock_make.call_args.kwargs["agent_start_pos"] == (1, 1)

    @pytest.mark.parametrize("task", ["random-goal_default", "fixed-goal_default"])
    def test_goal_grid_with_mission_text_uses_mission_wrapper(self, task):
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(7)

        with self._patch_goal_grid(dummy_env):
            with patch("her_dream.envs.wrappers.MissionGridWrapper") as mock_mgw:
                mock_mgw.return_value = dummy_env
                from her_dream.envs import make_env

                make_env(make_config(task=task, mission_text=True), 0)
            mock_mgw.assert_called_once_with(dummy_env)


class TestMakeEnvWrapperStack:
    def test_all_paths_end_with_dtype_wrapper(self):
        from her_dream.envs.wrappers import Dtype

        dummy_crafter = MagicMock()
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(17)
        dummy_crafter.Crafter.return_value = dummy_env

        with patch_env_module("her_dream.envs.crafter", dummy_crafter):
            from her_dream.envs import make_env

            config = make_config(task="crafter_reward")
            env = make_env(config, 0)
        assert isinstance(env, Dtype)

    def test_all_paths_include_time_limit(self):
        from her_dream.envs.wrappers import TimeLimit

        dummy_crafter = MagicMock()
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(17)
        dummy_crafter.Crafter.return_value = dummy_env

        with patch_env_module("her_dream.envs.crafter", dummy_crafter):
            from her_dream.envs import make_env

            config = make_config(task="crafter_reward", time_limit=50, action_repeat=1)
            env = make_env(config, 0)
        from her_dream.envs.wrappers import get_wrapper

        assert get_wrapper(env, TimeLimit) is not None


# ---------------------------------------------------------------------------
# make_envs
# ---------------------------------------------------------------------------


class TestMakeEnvs:
    def test_returns_four_tuple(self):
        dummy_crafter = MagicMock()
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(17)
        dummy_crafter.Crafter.return_value = dummy_env

        with (
            patch_env_module("her_dream.envs.crafter", dummy_crafter),
            patch("her_dream.envs.parallel.ParallelEnv") as mock_penv,
        ):
            penv_instance = MagicMock()
            penv_instance.observation_space = dummy_env.observation_space
            penv_instance.action_space = dummy_env.action_space
            mock_penv.return_value = penv_instance

            from her_dream.envs import make_envs

            config = make_config(task="crafter_reward", env_num=2, eval_episode_num=1)
            train_envs, eval_envs, obs_space, act_space = make_envs(config)
        assert train_envs is penv_instance
        assert eval_envs is penv_instance
        assert obs_space is penv_instance.observation_space
        assert act_space is penv_instance.action_space

    def test_env_constructor_returns_callable(self):
        """Line 6: env_constructor(idx) returns a lambda; capture and call it."""
        dummy_crafter = MagicMock()
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(17)
        dummy_crafter.Crafter.return_value = dummy_env

        captured = {}

        def fake_penv(constructor, env_num, device):
            captured["fn"] = constructor
            instance = MagicMock()
            instance.observation_space = dummy_env.observation_space
            instance.action_space = dummy_env.action_space
            return instance

        with (
            patch_env_module("her_dream.envs.crafter", dummy_crafter),
            patch("her_dream.envs.parallel.ParallelEnv", side_effect=fake_penv),
        ):
            from her_dream.envs import make_envs

            config = make_config(task="crafter_reward", env_num=1, eval_episode_num=1)
            make_envs(config)

        # Calling env_constructor(0) executes line 6: return lambda: make_env(config, idx)
        fn = captured["fn"](0)
        assert callable(fn)

    def test_creates_train_and_eval_parallel_envs(self):
        dummy_crafter = MagicMock()
        dummy_env = _make_dummy_gym_env()
        dummy_env.action_space = gym.spaces.Discrete(17)
        dummy_crafter.Crafter.return_value = dummy_env

        with (
            patch_env_module("her_dream.envs.crafter", dummy_crafter),
            patch("her_dream.envs.parallel.ParallelEnv") as mock_penv,
        ):
            penv_instance = MagicMock()
            penv_instance.observation_space = dummy_env.observation_space
            penv_instance.action_space = dummy_env.action_space
            mock_penv.return_value = penv_instance

            from her_dream.envs import make_envs

            config = make_config(task="crafter_reward", env_num=4, eval_episode_num=2)
            make_envs(config)

        calls = mock_penv.call_args_list
        assert len(calls) == 2
        # First call = train (env_num=4)
        assert calls[0][0][1] == 4
        # Second call = eval (eval_episode_num=2)
        assert calls[1][0][1] == 2
