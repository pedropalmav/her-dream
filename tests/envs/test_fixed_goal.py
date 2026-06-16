import numpy as np
from minigrid.wrappers import RGBImgObsWrapper

from envs.fixed_goal import DIRECTIONS, FixedGoal, make_fixed_goal_env


class TestFixedGoalInit:
    def test_goal_pos_stored(self):
        env = FixedGoal(size=5, goal_pos=(3, 1))
        assert env.goal_pos == (3, 1)

    def test_agent_start_pos_stored(self):
        env = FixedGoal(size=5, agent_start_pos=(1, 3))
        assert env.agent_start_pos == (1, 3)


class TestGenMission:
    def test_returns_reach_goal(self):
        assert FixedGoal._gen_mission() == "reach goal"


class TestPutAgent:
    def test_with_explicit_start_pos(self):
        env = FixedGoal(size=5, agent_start_pos=(1, 3), agent_start_dir=1, goal_pos=(3, 1))
        env.reset()
        assert env.agent_pos == (1, 3)
        assert env.agent_dir == 1

    def test_with_none_start_pos_places_randomly(self):
        env = FixedGoal(size=5, agent_start_pos=None, goal_pos=(3, 1))
        env.reset()
        x, y = env.agent_pos
        assert 0 <= x < 5
        assert 0 <= y < 5


class TestBuildMission:
    def test_mission_string_format(self):
        env = FixedGoal(size=5, agent_start_pos=(1, 3), agent_start_dir=0, goal_pos=(3, 1))
        env.reset()
        mission = env._build_mission()
        ax, ay = env.agent_pos
        direction = DIRECTIONS[env.agent_dir]
        gx, gy = env.goal_pos
        expected = f"agent at ({ax},{ay}) facing {direction}. goal at ({gx},{gy})"
        assert mission == expected


class TestRandomMission:
    def test_goal_pos_fixed_in_mission(self):
        env = FixedGoal(size=10, agent_start_pos=(1, 8), goal_pos=(3, 2))
        env.reset()
        rng = np.random.RandomState(42)
        mission = env.random_mission(rng=rng)
        # Goal position must always be (3,2)
        assert "goal at (3,2)" in mission

    def test_without_rng_creates_own(self):
        env = FixedGoal(size=10, agent_start_pos=(1, 5), goal_pos=(3, 2))
        env.reset()
        mission = env.random_mission()
        assert "goal at (3,2)" in mission

    def test_agent_position_randomized(self):
        env = FixedGoal(size=10, agent_start_pos=(1, 5), goal_pos=(3, 2))
        env.reset()
        positions = set()
        rng = np.random.RandomState(0)
        for _ in range(30):
            m = env.random_mission(rng=rng)
            positions.add(m.split("agent at ")[1].split(")")[0])
        # Should see more than one agent position across 30 samples
        assert len(positions) > 1


class TestResetAndStep:
    def test_reset_adds_mission(self):
        env = FixedGoal(size=10, agent_start_pos=(1, 8), goal_pos=(8, 1))
        obs, _ = env.reset()
        assert "mission" in obs

    def test_step_terminated_always_false(self):
        env = FixedGoal(size=10, agent_start_pos=(1, 8), goal_pos=(8, 1))
        env.reset()
        _, _, terminated, _, _ = env.step(0)
        assert terminated is False

    def test_reward_on_goal(self):
        env = FixedGoal(size=10, agent_start_pos=(1, 8), goal_pos=(8, 1))
        env.reset()
        env.agent_pos = env.goal_pos
        assert env._reward() == 0

    def test_reward_not_on_goal(self):
        env = FixedGoal(size=10, agent_start_pos=(1, 8), goal_pos=(8, 1))
        env.reset()
        agent_cell = env.grid.get(*env.agent_pos)
        if agent_cell is None or agent_cell.type != "goal":
            assert env._reward() == -1

    def test_step_includes_mission(self):
        env = FixedGoal(size=10, agent_start_pos=(1, 8), goal_pos=(8, 1))
        env.reset()
        obs, _, _, _, _ = env.step(0)
        assert "mission" in obs


class TestMakeFixedGoalEnv:
    def test_wraps_in_rgb_obs_wrapper(self):
        env = make_fixed_goal_env(size=10, goal_pos=(8, 1))
        assert isinstance(env, RGBImgObsWrapper)

    def test_kwargs_passed_through(self):
        env = make_fixed_goal_env(size=10, goal_pos=(8, 1), agent_start_dir=2)
        env.reset()
        assert env.unwrapped.agent_start_dir == 2
