"""The `lava-grid` suite: her_dream driving cookie-env's `LavaGrid`.

These go through the real cookie_env package rather than a mock. The point of the
suite is that the two repos compose — the import resolves, the wrapper chain
accepts a cookie-env env, and `terminated` survives to `is_terminal` — and a mock
would assert none of it.
"""

import numpy as np
import pytest

import her_dream.envs.wrappers as wrappers
import her_dream.goals as goals
from her_dream.envs import make_env
from her_dream.envs.wrappers import Dtype, GoalImageObservation, TimeLimit, get_wrapper
from tests.envs.conftest import DictNamespace

SIZE = 10
FORWARD = 2  # MiniGrid's "move forward" action index


def make_config(**overrides):
    defaults = dict(
        task="lava-grid_",
        action_repeat=1,
        time_limit=40,
        seed=0,
        env_size=SIZE,
        agent_start_dir=0,
        agent_start_pos_x=1,
        agent_start_pos_y=1,
        agent_start_random=False,
        goal_pos_x=None,
        goal_pos_y=None,
        n_lava=1,
        render_mode="rgb_array",
        mission_text=False,
        stochastic_classes=4,
        stochastic_rows=2,
        goal_type="full",
    )
    return DictNamespace(**goals.with_default_descriptors({**defaults, **overrides}))


def forward_action(env):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[FORWARD] = 1.0
    return action


@pytest.fixture
def env():
    return make_env(make_config(), 0)


class TestWrapperStack:
    def test_ends_with_dtype(self, env):
        assert isinstance(env, Dtype)

    def test_includes_time_limit(self, env):
        assert get_wrapper(env, TimeLimit) is not None

    def test_goal_image_observation_attached(self, env):
        assert get_wrapper(env, GoalImageObservation) is not None

    def test_observation_keys(self, env):
        obs = env.reset()
        assert {"image", "direction", "goal", "is_first", "is_last", "is_terminal"} <= set(obs)

    def test_goal_shape_follows_config(self, env):
        obs = env.reset()
        assert obs["goal"].shape == (2, 4)

    def test_mission_text_encodes_the_mission(self):
        env = make_env(make_config(mission_text=True), 0)
        obs = env.reset()
        # MissionGridWrapper replaces the string with token ids (widened to int32
        # by the outer Dtype wrapper), so the mission reaches the agent as text.
        assert obs["mission"].shape == (wrappers.MAX_LEN,)
        assert np.issubdtype(obs["mission"].dtype, np.integer)
        assert obs["mission"].any()


class TestGoalImage:
    """goal_sample="image" has to reproduce the WHOLE layout, lava included.

    A goal image rendered with the lava somewhere else is a state the live episode
    cannot reach — the same unreachable-goal failure buffer-sampled goals have on
    random_goal, which is the thing this wrapper exists to avoid.
    """

    def generator_env(self, env):
        return get_wrapper(env, GoalImageObservation)._generator._env.unwrapped

    def test_rendered_goal_uses_the_live_lava(self, env):
        env.reset()
        env.get_wrapper_attr("goal_observation")()
        live = sorted(tuple(c) for c in env.get_wrapper_attr("lava_positions"))
        rendered = sorted(tuple(c) for c in self.generator_env(env).lava_positions)
        assert rendered == live

    def test_rendered_goal_uses_the_live_square(self, env):
        env.reset()
        env.get_wrapper_attr("goal_observation")()
        assert tuple(self.generator_env(env).goal_position) == tuple(env.get_wrapper_attr("goal_position"))

    def test_layout_follows_across_episodes(self):
        env = make_env(make_config(n_lava=4), 0)
        goal_observation = env.get_wrapper_attr("goal_observation")
        for _ in range(30):
            env.reset()
            goal_observation()
            live = sorted(tuple(c) for c in env.get_wrapper_attr("lava_positions"))
            rendered = sorted(tuple(c) for c in self.generator_env(env).lava_positions)
            assert rendered == live

    def test_sampled_agent_never_stands_on_lava(self):
        """Unlike the green square, the agent coinciding with lava is a death state."""
        env = make_env(make_config(n_lava=6), 0)
        goal_observation = env.get_wrapper_attr("goal_observation")
        for _ in range(50):
            env.reset()
            goal_observation()
            core = self.generator_env(env)
            assert tuple(core.agent_pos) not in {tuple(c) for c in core.lava_positions}

    def test_returns_an_observation(self, env):
        env.reset()
        obs = env.get_wrapper_attr("goal_observation")()
        assert obs["image"].shape == (80, 80, 3)
        assert obs["direction"].shape == (4,)


class TestLavaTermination:
    def test_lava_step_is_terminal(self, env):
        env.reset()
        core = env.get_wrapper_attr("unwrapped")
        lava_x, lava_y = env.get_wrapper_attr("lava_positions")[0]
        # Stand west of a lava cell facing east, so one forward step walks into it.
        core.agent_pos = (lava_x - 1, lava_y)
        core.agent_dir = 0

        obs, _, done, info = env.step(forward_action(env))

        assert done
        assert obs["is_terminal"]
        assert obs["is_last"]
        assert info["lava"]

    def test_ordinary_step_does_not_terminate(self, env):
        env.reset()
        obs, _, done, _ = env.step(forward_action(env))
        assert not done
        assert not obs["is_terminal"]

    def test_reaching_the_goal_does_not_terminate(self, env):
        """Inherited from GoalGrid: lava is the only terminal, the green square is not."""
        env.reset()
        core = env.get_wrapper_attr("unwrapped")
        goal_x, goal_y = env.get_wrapper_attr("goal_position")
        core.agent_pos = (goal_x - 1, goal_y)
        core.agent_dir = 0

        obs, _, done, _ = env.step(forward_action(env))

        assert tuple(core.agent_pos) == (goal_x, goal_y)
        assert not done
        assert not obs["is_terminal"]


class TestConfigPassthrough:
    def test_n_lava_is_honoured(self):
        env = make_env(make_config(n_lava=4), 0)
        env.reset()
        assert len(env.get_wrapper_attr("lava_positions")) == 4

    def test_null_goal_pos_resamples_the_square(self):
        env = make_env(make_config(agent_start_random=True), 0)
        seen = set()
        for _ in range(20):
            env.reset()
            seen.add(tuple(env.get_wrapper_attr("goal_position")))
        assert len(seen) > 1

    def test_configured_goal_pos_pins_the_square(self):
        env = make_env(make_config(goal_pos_x=8, goal_pos_y=1), 0)
        for _ in range(5):
            env.reset()
            assert tuple(env.get_wrapper_attr("goal_position")) == (8, 1)

    def test_agent_start_random_moves_the_agent(self):
        env = make_env(make_config(agent_start_random=True), 0)
        core = env.get_wrapper_attr("unwrapped")
        seen = set()
        for _ in range(20):
            env.reset()
            seen.add(tuple(core.agent_pos))
        assert len(seen) > 1

    def test_fixed_start_uses_configured_pos(self, env):
        core = env.get_wrapper_attr("unwrapped")
        env.reset()
        assert tuple(core.agent_pos) == (1, 1)
