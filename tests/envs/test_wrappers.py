import gymnasium as gym
import numpy as np
import pytest
import torch

from envs.wrappers import (
    Dtype,
    GoalConditioned,
    MiniGridWrapper,
    MissionGridWrapper,
    MultiOneHotAction,
    NoMission,
    NormalizeActions,
    OneHotAction,
    OneHotDirection,
    RewardObs,
    TimeLimit,
    encode_mission,
    get_wrapper,
)

from .conftest import N_CLASSES, N_ROWS, make_box_env, make_discrete_env, make_goal_config

# ---------------------------------------------------------------------------
# get_wrapper
# ---------------------------------------------------------------------------


class TestGetWrapper:
    def test_returns_wrapper_when_found(self):
        inner = make_discrete_env()
        env = TimeLimit(inner, duration=10)
        inner._step = 0
        result = get_wrapper(env, TimeLimit)
        assert isinstance(result, TimeLimit)

    def test_returns_none_when_not_found(self):
        env = make_discrete_env()
        result = get_wrapper(env, TimeLimit)
        assert result is None


# ---------------------------------------------------------------------------
# TimeLimit
# ---------------------------------------------------------------------------


class TestTimeLimit:
    def _make_env(self, duration=3):
        inner = make_discrete_env()
        return TimeLimit(inner, duration=duration), inner

    def test_reset_sets_step_to_zero(self):
        env, _ = self._make_env()
        env.reset()
        assert env._step == 0

    def test_step_increments_counter(self):
        env, _ = self._make_env()
        env.reset()
        env.step(0)
        assert env._step == 1

    def test_step_at_duration_sets_is_last(self):
        env, inner = self._make_env(duration=2)
        inner.step.side_effect = lambda a: ({"is_last": False}, 0.0, False, {})
        env.reset()
        env.step(0)
        obs, _, done, info = env.step(0)
        assert obs["is_last"] is True
        assert done is True

    def test_step_at_duration_adds_discount_when_absent(self):
        env, inner = self._make_env(duration=1)
        inner.step.side_effect = lambda a: ({"is_last": False}, 0.0, False, {})
        env.reset()
        _, _, _, info = env.step(0)
        assert "discount" in info
        assert info["discount"] == np.float32(1.0)

    def test_step_at_duration_keeps_existing_discount(self):
        env, inner = self._make_env(duration=1)
        inner.step.side_effect = lambda a: ({"is_last": False}, 0.0, False, {"discount": np.float32(0.0)})
        env.reset()
        _, _, _, info = env.step(0)
        assert info["discount"] == np.float32(0.0)

    def test_step_resets_step_counter_at_end(self):
        env, inner = self._make_env(duration=1)
        inner.step.side_effect = lambda a: ({"is_last": False}, 0.0, False, {})
        env.reset()
        env.step(0)
        assert env._step is None

    def test_step_asserts_when_not_reset(self):
        env, _ = self._make_env()
        with pytest.raises(AssertionError, match="Must reset"):
            env.step(0)


# ---------------------------------------------------------------------------
# NormalizeActions
# ---------------------------------------------------------------------------


class TestNormalizeActions:
    def test_finite_bounds_scaled(self):
        env = make_box_env(low=0.0, high=2.0, n_actions=2)
        wrapped = NormalizeActions(env)
        # action=1 in [-1,1] should map to high=2.0
        wrapped.step(np.ones(2, dtype=np.float32))
        called_action = env.step.call_args[0][0]
        np.testing.assert_allclose(called_action, np.array([2.0, 2.0]), atol=1e-5)

    def test_infinite_bounds_passthrough(self):
        inner = make_box_env(n_actions=1)
        inner.action_space = gym.spaces.Box(
            low=np.array([-np.inf], dtype=np.float32),
            high=np.array([np.inf], dtype=np.float32),
        )
        wrapped = NormalizeActions(inner)
        action = np.array([3.14], dtype=np.float32)
        wrapped.step(action)
        called_action = inner.step.call_args[0][0]
        np.testing.assert_allclose(called_action, action, atol=1e-5)

    def test_mixed_bounds(self):
        inner = make_box_env(n_actions=2)
        inner.action_space = gym.spaces.Box(
            low=np.array([0.0, -np.inf], dtype=np.float32),
            high=np.array([4.0, np.inf], dtype=np.float32),
        )
        wrapped = NormalizeActions(inner)
        action = np.array([0.0, 7.0], dtype=np.float32)
        wrapped.step(action)
        called = inner.step.call_args[0][0]
        # first dim: (0+1)/2 * (4-0) + 0 = 2.0
        assert abs(called[0] - 2.0) < 1e-5
        # second dim: inf bounds, passthrough
        assert called[1] == 7.0


# ---------------------------------------------------------------------------
# OneHotAction
# ---------------------------------------------------------------------------


class TestOneHotAction:
    def _wrapped(self, n=4):
        env = make_discrete_env(n_actions=n)
        return OneHotAction(env), env

    def test_valid_one_hot_converted_to_index(self):
        wrapped, inner = self._wrapped(n=4)
        action = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        wrapped.step(action)
        inner.step.assert_called_with(2)

    def test_invalid_one_hot_raises_value_error(self):
        wrapped, _ = self._wrapped(n=4)
        action = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Invalid one-hot"):
            wrapped.step(action)

    def test_reset_passthrough(self):
        wrapped, inner = self._wrapped()
        wrapped.reset()
        inner.reset.assert_called_once()

    def test_sample_action_is_valid_one_hot(self):
        wrapped, _ = self._wrapped(n=6)
        action = wrapped._sample_action()
        assert action.sum() == 1.0
        assert action.shape == (6,)

    def test_action_space_is_box_with_discrete_flag(self):
        wrapped, _ = self._wrapped(n=5)
        assert isinstance(wrapped.action_space, gym.spaces.Box)
        assert wrapped.action_space.discrete is True


# ---------------------------------------------------------------------------
# MultiOneHotAction
# ---------------------------------------------------------------------------


class _MultiDiscreteEnv(gym.Env):
    """Minimal real gym.Env with MultiDiscrete action space for testing.

    Adds a `.low` attribute to the action space to match the wrappers.py expectation
    (older gymnasium/gym compat shim).
    """

    def __init__(self, nvec):
        md = gym.spaces.MultiDiscrete(nvec=np.array(nvec))
        # Patch `.low` since gymnasium >= 1.0 removed it
        md.low = np.zeros(len(nvec), dtype=np.int64)
        self.action_space = md
        self.observation_space = gym.spaces.Dict({"obs": gym.spaces.Box(0, 1, (1,))})
        self._step_result = None

    def reset(self, **kwargs):
        return {"obs": np.zeros((1,))}, {}

    def step(self, *args):
        self._step_result = args
        return {"obs": np.zeros((1,))}, 0.0, False, False, {}


class TestMultiOneHotAction:
    def test_convert_single_dimension(self):
        inner = _MultiDiscreteEnv(nvec=[3])
        wrapped = MultiOneHotAction(inner, device="cpu")
        action = torch.tensor([[0.0, 1.0, 0.0]])
        result = wrapped.convert(action)
        assert result[0, 0].item() == 1

    def test_convert_multiple_dimensions(self):
        inner = _MultiDiscreteEnv(nvec=[3, 2])
        wrapped = MultiOneHotAction(inner, device="cpu")
        # action: [1,0,0, 0,1] → indices [0, 1]
        action = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]])
        result = wrapped.convert(action)
        assert result[0, 0].item() == 0
        assert result[0, 1].item() == 1

    def test_step_delegates(self):
        inner = _MultiDiscreteEnv(nvec=[2])
        wrapped = MultiOneHotAction(inner, device="cpu")
        a1 = torch.tensor([[1.0, 0.0]])
        a2 = torch.tensor([[0.0, 1.0]])
        wrapped.step(a1, a2, False)
        # Should have called inner.step
        assert inner._step_result is not None


# ---------------------------------------------------------------------------
# RewardObs
# ---------------------------------------------------------------------------


class TestRewardObs:
    def test_init_adds_obs_reward_when_absent(self):
        env = make_discrete_env()
        wrapped = RewardObs(env)
        assert "obs_reward" in wrapped.observation_space.spaces

    def test_init_does_not_duplicate_obs_reward(self):
        env = make_discrete_env(obs_keys=["image", "obs_reward"])
        initial_count = len(env.observation_space.spaces)
        wrapped = RewardObs(env)
        assert len(wrapped.observation_space.spaces) == initial_count

    def test_step_adds_obs_reward_when_absent(self):
        env = make_discrete_env()
        wrapped = RewardObs(env)
        env.step.side_effect = lambda a: ({"image": np.zeros((8, 8, 3))}, 5.0, False, {})
        obs, reward, _, _ = wrapped.step(0)
        assert "obs_reward" in obs
        np.testing.assert_allclose(obs["obs_reward"], [5.0])

    def test_step_keeps_existing_obs_reward(self):
        env = make_discrete_env()
        wrapped = RewardObs(env)
        env.step.side_effect = lambda a: (
            {"image": np.zeros((8, 8, 3)), "obs_reward": np.array([99.0])},
            5.0,
            False,
            {},
        )
        obs, _, _, _ = wrapped.step(0)
        np.testing.assert_allclose(obs["obs_reward"], [99.0])

    def test_reset_adds_obs_reward_when_absent(self):
        env = make_discrete_env()
        wrapped = RewardObs(env)
        env.reset.side_effect = lambda: {"image": np.zeros((8, 8, 3))}
        obs = wrapped.reset()
        assert "obs_reward" in obs
        np.testing.assert_allclose(obs["obs_reward"], [0.0])

    def test_reset_keeps_existing_obs_reward(self):
        env = make_discrete_env()
        wrapped = RewardObs(env)
        env.reset.side_effect = lambda: {"image": np.zeros((8, 8, 3)), "obs_reward": np.array([7.0])}
        obs = wrapped.reset()
        np.testing.assert_allclose(obs["obs_reward"], [7.0])


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------


class TestDtype:
    def test_step_converts_reward_to_float32(self):
        env = make_discrete_env(obs_keys=["image"])
        wrapped = Dtype(env)
        env.step.side_effect = lambda a: ({"image": np.zeros((8, 8, 3), dtype=np.uint8)}, 1.0, False, {})
        obs, rew, done, info = wrapped.step(0)
        assert isinstance(rew, np.float32)

    def test_step_converts_float_obs(self):
        env = make_discrete_env(obs_keys=["value"])
        env.step.side_effect = lambda a: ({"value": np.array([1.0], dtype=np.float64)}, 1.0, False, {})
        wrapped = Dtype(env)
        obs, _, _, _ = wrapped.step(0)
        assert obs["value"].dtype == np.float32

    def test_reset_converts_float_obs(self):
        env = make_discrete_env(obs_keys=["value"])
        env.reset.side_effect = lambda: {"value": np.array([1.0], dtype=np.float64)}
        wrapped = Dtype(env)
        obs = wrapped.reset()
        assert obs["value"].dtype == np.float32


# ---------------------------------------------------------------------------
# GoalConditioned
# ---------------------------------------------------------------------------


class TestGoalConditioned:
    def _make(self, **cfg_overrides):
        env = make_discrete_env()
        env.step.side_effect = lambda a: ({"image": np.zeros((8, 8, 3), dtype=np.uint8)}, 0.0, False, {})
        env.reset.side_effect = lambda: {"image": np.zeros((8, 8, 3), dtype=np.uint8)}
        cfg = make_goal_config(**cfg_overrides)
        return GoalConditioned(env, cfg), env

    def test_first_row_with_goal_index_deterministic(self):
        wrapped, _ = self._make(goal_type="first_row", goal_index=2)
        obs = wrapped.reset()
        assert obs["goal"].shape == (N_CLASSES,)
        assert obs["goal"][2] == 1.0
        assert obs["goal"].sum() == 1.0

    def test_first_row_without_goal_index_is_random_one_hot(self):
        wrapped, _ = self._make(goal_type="first_row")
        obs = wrapped.reset()
        assert obs["goal"].shape == (N_CLASSES,)
        assert obs["goal"].sum() == 1.0

    def test_multi_row_goal_shape(self):
        wrapped, _ = self._make(goal_type="multi_row")
        obs = wrapped.reset()
        assert obs["goal"].shape == (N_ROWS, N_CLASSES)
        for row in obs["goal"]:
            assert row.sum() == 1.0

    def test_step_attaches_goal(self):
        wrapped, _ = self._make(goal_type="first_row", goal_index=0)
        wrapped.reset()
        obs, _, _, _ = wrapped.step(0)
        assert "goal" in obs
        assert obs["goal"][0] == 1.0

    def test_goal_consistent_within_episode(self):
        wrapped, _ = self._make(goal_type="first_row")
        obs_reset = wrapped.reset()
        goal_reset = obs_reset["goal"].copy()
        obs_step, _, _, _ = wrapped.step(0)
        np.testing.assert_array_equal(obs_step["goal"], goal_reset)

    def test_goal_type_first_row_adds_1d_space(self):
        wrapped, _ = self._make(goal_type="first_row")
        assert wrapped.observation_space.spaces["goal"].shape == (N_CLASSES,)

    def test_goal_type_multi_row_adds_2d_space(self):
        wrapped, _ = self._make(goal_type="multi_row")
        assert wrapped.observation_space.spaces["goal"].shape == (N_ROWS, N_CLASSES)


# ---------------------------------------------------------------------------
# NoMission
# ---------------------------------------------------------------------------


class TestNoMission:
    def test_init_removes_mission_from_space(self):
        env = make_discrete_env(obs_keys=["image", "mission"])
        wrapped = NoMission(env)
        assert "mission" not in wrapped.observation_space.spaces

    def test_observation_removes_mission(self):
        env = make_discrete_env(obs_keys=["image", "mission"])
        wrapped = NoMission(env)
        obs = {"image": np.zeros((8, 8, 3)), "mission": "go right"}
        result = wrapped.observation(obs)
        assert "mission" not in result

    def test_observation_without_mission_key_no_error(self):
        env = make_discrete_env(obs_keys=["image"])
        wrapped = NoMission(env)
        obs = {"image": np.zeros((8, 8, 3))}
        result = wrapped.observation(obs)
        assert "image" in result


# ---------------------------------------------------------------------------
# OneHotDirection
# ---------------------------------------------------------------------------


class TestOneHotDirection:
    @pytest.mark.parametrize("direction,expected_idx", [(0, 0), (1, 1), (2, 2), (3, 3)])
    def test_direction_to_one_hot(self, direction, expected_idx):
        env = make_discrete_env(obs_keys=["image", "direction"])
        env.observation_space.spaces["direction"] = gym.spaces.Discrete(4)
        wrapped = OneHotDirection(env)
        obs = {"image": np.zeros((8, 8, 3)), "direction": direction}
        result = wrapped.observation(obs)
        assert result["direction"].shape == (4,)
        assert result["direction"][expected_idx] == 1.0
        assert result["direction"].sum() == 1.0


# ---------------------------------------------------------------------------
# MiniGridWrapper
# ---------------------------------------------------------------------------


class _FakeMiniGridEnv(gym.Env):
    """Minimal MiniGrid-compatible env for testing MiniGridWrapper."""

    def __init__(self, terminated=False, truncated=False):
        self._terminated = terminated
        self._truncated = truncated
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8),
            "direction": gym.spaces.Discrete(4),
            "mission": gym.spaces.Text(max_length=100),
        })
        self.action_space = gym.spaces.Discrete(7)

    def reset(self, **kwargs):
        obs = {"image": np.zeros((8, 8, 3), dtype=np.uint8), "direction": 0, "mission": "go"}
        return obs, {}

    def step(self, action):
        obs = {"image": np.zeros((8, 8, 3), dtype=np.uint8), "direction": 1, "mission": "go"}
        return obs, 0.0, self._terminated, self._truncated, {}


class TestMiniGridWrapper:
    def test_reset_sets_is_first(self):
        wrapped = MiniGridWrapper(_FakeMiniGridEnv())
        obs = wrapped.reset()
        assert obs["is_first"] is True
        assert obs["is_last"] is False
        assert obs["is_terminal"] is False

    def test_step_normal_clears_is_first(self):
        wrapped = MiniGridWrapper(_FakeMiniGridEnv())
        wrapped.reset()
        obs, _, _, _ = wrapped.step(0)
        assert obs["is_first"] is False
        assert obs["is_last"] is False
        assert obs["is_terminal"] is False

    def test_step_terminated_sets_is_last_and_is_terminal(self):
        wrapped = MiniGridWrapper(_FakeMiniGridEnv(terminated=True))
        wrapped.reset()
        obs, _, done, _ = wrapped.step(0)
        assert obs["is_last"] is True
        assert obs["is_terminal"] is True
        assert done is True

    def test_step_truncated_sets_is_last_but_not_is_terminal(self):
        wrapped = MiniGridWrapper(_FakeMiniGridEnv(truncated=True))
        wrapped.reset()
        obs, _, done, _ = wrapped.step(0)
        assert obs["is_last"] is True
        assert obs["is_terminal"] is False
        assert done is True


# ---------------------------------------------------------------------------
# encode_mission
# ---------------------------------------------------------------------------


class TestEncodeMission:
    def test_known_characters_mapped_correctly(self):
        ids = encode_mission("ab")
        assert ids[0] == 2  # 'a' = index 1 in abc... = 2 (1 + 1)
        assert ids[1] == 3  # 'b' = 3

    def test_padding_fills_remainder_with_zero(self):
        ids = encode_mission("a", max_len=5)
        assert ids[0] != 0
        assert all(ids[i] == 0 for i in range(1, 5))

    def test_truncation_when_text_exceeds_max_len(self):
        ids = encode_mission("abcde", max_len=3)
        assert len(ids) == 3

    def test_lowercasing_applied(self):
        ids_lower = encode_mission("abc")
        ids_upper = encode_mission("ABC")
        np.testing.assert_array_equal(ids_lower, ids_upper)

    def test_returns_int8_array(self):
        ids = encode_mission("hello")
        assert ids.dtype == np.int8


# ---------------------------------------------------------------------------
# MissionGridWrapper
# ---------------------------------------------------------------------------


class _FakeMissionEnv(_FakeMiniGridEnv):
    """Extends _FakeMiniGridEnv with random_mission and _build_mission."""

    def random_mission(self, rng=None):
        return "random mission"

    def _build_mission(self):
        return "current mission"


class TestMissionGridWrapper:
    def _make(self):
        base_env = _FakeMissionEnv()
        return MissionGridWrapper(base_env)

    def test_reset_encodes_mission_in_obs(self):
        wrapped = self._make()
        obs = wrapped.reset()
        assert "mission" in obs
        assert isinstance(obs["mission"], np.ndarray)
        assert obs["mission"].dtype == np.int8

    def test_step_with_mission_in_obs_re_encodes(self):
        wrapped = self._make()
        wrapped.reset()
        obs, _, _, _ = wrapped.step(0)
        assert "mission" in obs

    def test_step_without_mission_in_obs_not_added(self):
        class NomissionFakeEnv(_FakeMissionEnv):
            def step(self, action):
                obs = {"image": np.zeros((8, 8, 3), dtype=np.uint8), "direction": 1}
                return obs, 0.0, False, False, {}

        wrapped = MissionGridWrapper(NomissionFakeEnv())
        wrapped.reset()
        obs, _, _, _ = wrapped.step(0)
        assert "mission" not in obs

    def test_encoded_random_mission_returns_int8_array(self):
        wrapped = self._make()
        result = wrapped.encoded_random_mission()
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int8

    def test_encoded_current_mission_returns_int8_array(self):
        wrapped = self._make()
        wrapped.reset()
        result = wrapped.encoded_current_mission()
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int8

    def test_reset_sets_is_first(self):
        wrapped = self._make()
        obs = wrapped.reset()
        assert obs["is_first"] is True

    def test_step_terminated_sets_flags(self):
        wrapped = MissionGridWrapper(_FakeMissionEnv(terminated=True))
        wrapped.reset()
        obs, _, done, _ = wrapped.step(0)
        assert obs["is_last"] is True
        assert obs["is_terminal"] is True

    def test_step_truncated_sets_is_last_not_terminal(self):
        wrapped = MissionGridWrapper(_FakeMissionEnv(truncated=True))
        wrapped.reset()
        obs, _, done, _ = wrapped.step(0)
        assert obs["is_last"] is True
        assert obs["is_terminal"] is False
