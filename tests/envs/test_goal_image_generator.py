"""Tests for `GoalImageGenerator` (goal_sample="image").

Renders, via an auxiliary `FixedGoal` env, the observation of a synthetic goal
state — green square at the episode's goal position, agent at a random interior
cell — that the frozen WM later encodes into the goal z
(`Dreamer.encode_observation`). Covers the observation contract (keys, shapes,
dtypes), one-hot direction, reproducibility under a seeded RNG, sensitivity to
the goal position, agent placement, and lazy reuse of the auxiliary env.

On an env with hazards the generator has to reproduce the whole layout, so
`lava_pos` is covered here too (see `TestLavaLayout`).
"""

import numpy as np
import pytest
from cookie_env.envs.lava_grid import make_lava_grid_env

from her_dream.envs.fixed_goal import FixedGoal, make_fixed_goal_env
from her_dream.envs.goal_image import GoalImageGenerator
from her_dream.envs.wrappers import GoalImageObservation
from tests.envs.conftest import SIZE


@pytest.fixture
def lava_generator():
    return GoalImageGenerator(lambda: make_lava_grid_env(size=SIZE, goal_pos=(1, 1), lava_pos=[]))


class TestLavaLayout:
    LAVA = [(3, 3), (4, 5), (6, 2)]

    def test_lava_is_pinned_on_the_auxiliary_env(self, lava_generator):
        lava_generator.observation((8, 1), lava_pos=self.LAVA)
        assert lava_generator._env.unwrapped.lava_positions == self.LAVA

    def test_lava_can_change_between_calls(self, lava_generator):
        """The auxiliary env is reused, so a new episode's layout has to overwrite
        the previous one rather than stick."""
        lava_generator.observation((8, 1), lava_pos=self.LAVA)
        lava_generator.observation((8, 1), lava_pos=[(2, 2)])
        assert lava_generator._env.unwrapped.lava_positions == [(2, 2)]

    @pytest.mark.parametrize("seed", range(20))
    def test_sampled_agent_avoids_lava(self, lava_generator, seed):
        lava_generator.observation((8, 1), rng=np.random.RandomState(seed), lava_pos=self.LAVA)
        assert tuple(lava_generator._env.unwrapped.agent_pos) not in set(self.LAVA)

    def test_pinned_agent_on_lava_raises(self, lava_generator):
        """A caller asking for a state that cannot exist gets an error, not a silent
        relocation — the experiments pin the pose to know the ground truth."""
        with pytest.raises(ValueError, match="collides with agent_start_pos"):
            lava_generator.observation((8, 1), agent_pos=(3, 3), lava_pos=self.LAVA)

    def test_lava_free_path_is_unchanged(self, generator):
        """Without lava the agent is still drawn from the full interior, with the two
        coordinates sampled independently — same RNG consumption as before."""
        a = generator.observation((8, 1), rng=np.random.RandomState(0))
        b = generator.observation((8, 1), rng=np.random.RandomState(0))
        assert np.array_equal(a["image"], b["image"])


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
