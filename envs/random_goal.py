from typing import Tuple
import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal
from minigrid.wrappers import RGBImgObsWrapper


class RandomGoal(MiniGridEnv):

    def __init__(
        self,
        size: int = 10,
        agent_start_pos: Tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        max_steps: int = 100,
        render_mode: str = "rgb_array",
        **kwargs,
    ):

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

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

        self.place_obj(Goal())
        self._put_agent()

    def _put_agent(self):
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward = self._reward()
        # We use truncated to signal episode end instead of terminated
        terminated = False
        return obs, reward, terminated, truncated, info

    def _reward(self):
        agent_cell = self.grid.get(*self.agent_pos)
        return 0 if agent_cell is not None and agent_cell.type == "goal" else -1


def make_random_goal_env(
    size: int = 10,
    agent_start_pos: Tuple[int, int] = (1, 1),
    agent_start_dir: int = 0,
    max_steps: int = 100,
    **kwargs,
):
    env = RandomGoal(
        size=size,
        agent_start_pos=agent_start_pos,
        agent_start_dir=agent_start_dir,
        max_steps=max_steps,
        **kwargs,
    )
    env = RGBImgObsWrapper(env)
    return env


if __name__ == "__main__":
    from minigrid.manual_control import ManualControl

    size = 10
    env = make_random_goal_env(size=size, max_steps=2 * size, render_mode="human")

    manual_control = ManualControl(env)
    manual_control.start()
