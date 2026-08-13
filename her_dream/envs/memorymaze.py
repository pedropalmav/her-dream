import gymnasium as gym
import numpy as np

_TASKS = ("9x9", "11x11", "13x13", "15x15")


class MemoryMaze(gym.Env):
    """Adapter over the `memory_maze` dm_env tasks.

    Builds the dm_env task directly instead of going through the gym registry
    (`gym.make("memory_maze:MemoryMaze-<task>-v0")`). `memory_maze` only registers
    its ids with the unmaintained `gym` package, whose passive env checker calls
    `np.bool8` — removed in numpy 2 — so every `step()` raised. Constructing the
    task by hand keeps this class's contract unchanged: `step` returns the
    old-style 4-tuple and `reset` returns the obs dict, which is what the wrapper
    chain in `envs/__init__.py` expects.

    `gym` stays an install requirement — `memory_maze/__init__.py` imports it and
    re-raises if absent — but it is never called from here, so the `np.bool8`
    path is no longer reachable.

    `image_only_obs=True` matches the `MemoryMaze-<task>-v0` registration, so the
    observation is the raw (H, W, 3) uint8 camera image.
    """

    def __init__(self, task, size=(64, 64), seed=0):
        if task not in _TASKS:
            raise ValueError(f"Unknown memory maze task {task!r}; available: {list(_TASKS)}")
        # Imported lazily: `memory_maze/__init__.py` defaults MUJOCO_GL to egl and
        # pulls in dm_control, which initialises a renderer at import time. Keeping
        # it in here matches the other suites and leaves `import her_dream.envs`
        # renderer-free.
        from memory_maze import tasks

        self._env = getattr(tasks, f"memory_maze_{task}")(image_only_obs=True, seed=seed)
        self._size = size

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        img_shape = self._size + (3,)
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        })

    @property
    def action_space(self):
        return gym.spaces.Discrete(self._env.action_spec().num_values)

    def step(self, action):
        time_step = self._env.step(action)
        done = time_step.last()
        # dm_env signals a true terminal with discount 0; a time-limit truncation
        # keeps discount 1, which is the distinction `is_terminal` carries.
        is_terminal = bool(done and time_step.discount == 0)
        obs = {
            "image": time_step.observation,
            "is_first": False,
            "is_last": done,
            "is_terminal": is_terminal,
        }
        return obs, time_step.reward or 0.0, done, {"is_terminal": is_terminal}

    def reset(self):
        time_step = self._env.reset()
        return {
            "image": time_step.observation,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
