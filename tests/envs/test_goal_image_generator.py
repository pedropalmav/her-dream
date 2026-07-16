"""Tests for `GoalImageGenerator` (goal_sample="image").

Renders, via an auxiliary `FixedGoal` env, the observation of a synthetic goal
state — green square at the episode's goal position, agent at a random interior
cell — that the frozen WM later encodes into the goal z
(`Dreamer.encode_observation`). Covers the observation contract (keys, shapes,
dtypes), one-hot direction, reproducibility under a seeded RNG, sensitivity to
the goal position, agent placement, and lazy reuse of the auxiliary env.
"""

import numpy as np
import pytest

from her_dream.envs.fixed_goal import FixedGoal, make_fixed_goal_env
from her_dream.envs.wrappers import GoalImageObservation
from tests.envs.conftest import SIZE


class TestObservationContract:
    def test_keys(self, generator):
        obs = generator.observation((1, 1))
        assert set(obs) == {"image", "direction"}

    def test_image_is_uint8_hwc(self, generator):
        img = generator.observation((1, 1))["image"]
        assert img.dtype == np.uint8
        assert img.ndim == 3 and img.shape[-1] == 3

    def test_direction_is_one_hot_float32_len4(self, generator):
        d = generator.observation((1, 1))["direction"]
        assert d.dtype == np.float32 and d.shape == (4,)
        assert d.sum() == 1.0 and set(np.unique(d)).issubset({0.0, 1.0})


class TestReproducibility:
    def test_same_seed_same_observation(self, generator):
        a = generator.observation((2, 3), np.random.RandomState(0))
        b = generator.observation((2, 3), np.random.RandomState(0))
        assert np.array_equal(a["image"], b["image"])
        assert np.array_equal(a["direction"], b["direction"])


class TestGoalPositionAffectsImage:
    def test_different_goal_positions_differ(self, generator):
        # Same RNG seed fixes the agent placement, so any pixel difference is
        # due to the green square moving.
        a = generator.observation((1, 1), np.random.RandomState(0))
        b = generator.observation((SIZE - 2, SIZE - 2), np.random.RandomState(0))
        assert not np.array_equal(a["image"], b["image"])

    def test_goal_pos_set_on_core_env(self, generator):
        generator.observation((3, 4), np.random.RandomState(0))
        assert generator._env.unwrapped.goal_pos == (3, 4)


class TestAgentPlacement:
    @pytest.mark.parametrize("seed", range(8))
    def test_agent_in_interior(self, generator, seed):
        generator.observation((1, 1), np.random.RandomState(seed))
        ax, ay = generator._env.unwrapped.agent_pos
        assert 1 <= ax <= SIZE - 2 and 1 <= ay <= SIZE - 2

    def test_agent_dir_in_range(self, generator):
        generator.observation((1, 1), np.random.RandomState(0))
        assert 0 <= generator._env.unwrapped.agent_dir <= 3


class TestLazyEnvReuse:
    def test_env_created_lazily(self, generator):
        assert generator._env is None
        generator.observation((1, 1))
        assert generator._env is not None

    def test_env_reused_across_calls(self, generator):
        generator.observation((1, 1))
        first = generator._env
        generator.observation((2, 2))
        assert generator._env is first


class TestWrapperIntegration:
    def test_goal_observation_renders_at_env_goal_position(self):
        # GoalImageObservation renders the goal at the wrapped env's goal_position.
        env = FixedGoal(size=SIZE, goal_pos=(8, 1))
        wrapped = GoalImageObservation(env, lambda: make_fixed_goal_env(size=SIZE))
        obs = wrapped.goal_observation()
        assert set(obs) == {"image", "direction"}
        assert wrapped._generator._env.unwrapped.goal_pos == (8, 1)
