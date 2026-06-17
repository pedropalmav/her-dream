import numpy as np
from minigrid.wrappers import RGBImgObsWrapper

from envs.random_goal import DIRECTIONS, RandomGoal, make_random_goal_env


class TestRandomGoalInit:
    def test_default_size(self):
        env = RandomGoal(size=5)
        assert env.size == 5

    def test_agent_start_pos_stored(self):
        env = RandomGoal(size=5, agent_start_pos=(1, 1))
        assert env.agent_start_pos == (1, 1)


class TestGenMission:
    def test_returns_reach_goal(self):
        assert RandomGoal._gen_mission() == "reach goal"


class TestPutAgent:
    def test_with_explicit_start_pos(self):
        env = RandomGoal(size=5, agent_start_pos=(1, 1), agent_start_dir=2)
        env.reset()
        assert env.agent_pos == (1, 1)
        assert env.agent_dir == 2

    def test_with_none_start_pos_places_randomly(self):
        env = RandomGoal(size=5, agent_start_pos=None)
        obs, _ = env.reset()
        # Agent should be placed somewhere in the interior
        x, y = env.agent_pos
        assert 0 <= x < 5
        assert 0 <= y < 5


class TestBuildMission:
    def test_mission_string_format(self):
        env = RandomGoal(size=5, agent_start_pos=(1, 1), agent_start_dir=0)
        env.reset()
        mission = env._build_mission()
        ax, ay = env.agent_pos
        direction = DIRECTIONS[env.agent_dir]
        gx, gy = env._goal_pos
        expected = f"agent at ({ax},{ay}) facing {direction}. goal at ({gx},{gy})"
        assert mission == expected


class TestRandomMission:
    def test_with_explicit_rng(self):
        env = RandomGoal(size=7)
        env.reset()
        rng = np.random.RandomState(42)
        mission = env.random_mission(rng=rng)
        assert "agent at" in mission
        assert "goal at" in mission
        assert "facing" in mission

    def test_without_rng_creates_own(self):
        env = RandomGoal(size=7)
        env.reset()
        mission = env.random_mission()
        assert "agent at" in mission

    def test_positions_within_interior(self):
        env = RandomGoal(size=7)
        env.reset()
        rng = np.random.RandomState(0)
        for _ in range(20):
            mission = env.random_mission(rng=rng)
            # Parse positions from mission string
            import re

            nums = re.findall(r"\d+", mission)
            positions = [int(n) for n in nums]
            for pos in positions:
                assert 1 <= pos <= 5  # interior of size-7 grid: 1 to 5


class TestResetAndStep:
    def test_reset_adds_mission_to_obs(self):
        env = RandomGoal(size=5)
        obs, _ = env.reset()
        assert "mission" in obs

    def test_step_terminated_always_false(self):
        env = RandomGoal(size=5)
        env.reset()
        _, _, terminated, _, _ = env.step(0)
        assert terminated is False

    def test_step_reward_not_on_goal(self):
        # Ensure agent is not at goal
        env = RandomGoal(size=5, agent_start_pos=(1, 1))
        env.reset()
        # Move in a direction that doesn't place us on the goal
        reward = env._reward()
        agent_cell = env.grid.get(*env.agent_pos)
        if agent_cell is None or agent_cell.type != "goal":
            assert reward == -1

    def test_step_reward_on_goal(self):
        env = RandomGoal(size=5)
        env.reset()
        # Manually move agent to goal
        env.agent_pos = env._goal_pos
        assert env._reward() == 0

    def test_step_includes_mission(self):
        env = RandomGoal(size=5)
        env.reset()
        obs, _, _, _, _ = env.step(0)
        assert "mission" in obs


class TestMakeRandomGoalEnv:
    def test_wraps_in_rgb_obs_wrapper(self):
        env = make_random_goal_env(size=5)
        assert isinstance(env, RGBImgObsWrapper)

    def test_kwargs_passed_through(self):
        env = make_random_goal_env(size=5, agent_start_dir=3)
        env.reset()
        assert env.unwrapped.agent_start_dir == 3
