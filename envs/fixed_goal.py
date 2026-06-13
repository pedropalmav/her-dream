import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgObsWrapper

DIRECTIONS = {0: "east", 1: "south", 2: "west", 3: "north"}


class FixedGoal(MiniGridEnv):
    def __init__(
        self,
        size: int = 10,
        agent_start_pos: tuple[int, int] = (1, 8),
        agent_start_dir: int = 0,
        max_steps: int = 100,
        render_mode: str = "rgb_array",
        goal_pos: tuple[int, int] = (8, 1),
        **kwargs,
    ):

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos
        self.size = size
        self._goal_image_generator = None

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(Goal(), *self.goal_pos)
        self._put_agent()

    def _put_agent(self):
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def _build_mission(self):
        ax, ay = self.agent_pos
        direction = DIRECTIONS[self.agent_dir]
        gx, gy = self.goal_pos
        return f"agent at ({ax},{ay}) facing {direction}. goal at ({gx},{gy})"

    def random_mission(self, rng: np.random.RandomState = None) -> str:
        """Return a mission string with randomly sampled [ax, ay, direction, gx, gy].

        All positions are sampled uniformly from the grid interior (1 to size-2),
        independently — agent and goal may coincide, matching the training distribution.
        """
        if rng is None:
            rng = np.random.RandomState()
        interior = np.arange(1, self.size - 1)
        ax, ay = rng.choice(interior), rng.choice(interior)
        gx, gy = self.goal_pos
        direction = DIRECTIONS[rng.randint(0, 4)]
        return f"agent at ({ax},{ay}) facing {direction}. goal at ({gx},{gy})"

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs["mission"] = self._build_mission()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward = self._reward()
        obs["mission"] = self._build_mission()
        # We use truncated to signal episode end instead of terminated
        terminated = False
        return obs, reward, terminated, truncated, info

    def _reward(self):
        agent_cell = self.grid.get(*self.agent_pos)
        return 0 if agent_cell is not None and agent_cell.type == "goal" else -1

    def goal_observation(self):
        """Observation of a synthetic goal state for goal_sample="image".

        Green square at this episode's goal position, agent at a random cell.
        """
        if self._goal_image_generator is None:
            self._goal_image_generator = GoalImageGenerator(self.size)
        return self._goal_image_generator.observation(self.goal_pos)


def make_fixed_goal_env(
    size: int = 10,
    agent_start_pos: tuple[int, int] = (1, 8),
    agent_start_dir: int = 0,
    max_steps: int = 100,
    goal_pos: tuple[int, int] = (8, 1),
    **kwargs,
):
    env = FixedGoal(
        size=size,
        agent_start_pos=agent_start_pos,
        agent_start_dir=agent_start_dir,
        goal_pos=goal_pos,
        max_steps=max_steps,
        **kwargs,
    )
    return RGBImgObsWrapper(env)


class GoalImageGenerator:
    """Render observations of synthetic goal states with an auxiliary FixedGoal env.

    Used by goal_sample="image": at episode start, the live env is asked for the
    observation of a state reachable in the current episode — the green square
    at the episode's goal position, the agent at a random interior cell — and
    the frozen world model encodes that image into the goal z
    (Dreamer.observe_goal). The auxiliary env goes through the same
    RGBImgObsWrapper render pipeline as the training envs, so the image matches
    what the agent would actually observe in that state.
    """

    def __init__(self, size: int):
        self.size = size
        self._env = None

    def observation(self, goal_pos, rng: np.random.RandomState = None):
        """Return {"image": (H, W, 3) uint8, "direction": (4,) one-hot float32}.

        Agent position and direction are sampled uniformly from the grid
        interior; agent and goal may coincide (the success state), matching
        the training distribution.
        """
        if rng is None:
            rng = np.random.RandomState()
        interior = np.arange(1, self.size - 1)
        agent_pos = (int(rng.choice(interior)), int(rng.choice(interior)))
        agent_dir = int(rng.randint(0, 4))
        if self._env is None:
            self._env = make_fixed_goal_env(size=self.size)
        core = self._env.unwrapped
        core.goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        core.agent_start_pos = agent_pos
        core.agent_start_dir = agent_dir
        obs, _ = self._env.reset()
        direction = np.zeros(4, dtype=np.float32)
        direction[obs["direction"]] = 1.0
        return {"image": obs["image"], "direction": direction}


if __name__ == "__main__":
    from minigrid.manual_control import ManualControl

    size = 10
    env = make_fixed_goal_env(size=size, max_steps=2 * size, render_mode="human")

    manual_control = ManualControl(env)
    manual_control.start()
