import numpy as np


class GoalImageGenerator:
    """Render observations of synthetic goal states with an auxiliary goal env.

    Used by goal_sample="image": at episode start, the observation of a state
    reachable in the current episode — the green square at the episode's goal
    position, the agent at a random interior cell — is rendered, and the frozen
    world model encodes that image into the goal z (Dreamer.encode_observation).

    The auxiliary env is built lazily from the injected `env_factory` (a
    zero-arg callable returning an RGBImgObsWrapper-wrapped goal env that honours
    `goal_pos` / `agent_start_pos`, e.g. fixed_goal.make_fixed_goal_env). It goes
    through the same RGBImgObsWrapper render pipeline as the training envs, so
    the image matches what the agent would actually observe in that state.
    """

    def __init__(self, env_factory):
        self._env_factory = env_factory
        self._env = None

    def observation(self, goal_pos, rng: np.random.RandomState = None):
        """Return {"image": (H, W, 3) uint8, "direction": (4,) one-hot float32}.

        Agent position and direction are sampled uniformly from the grid
        interior; agent and goal may coincide (the success state), matching
        the training distribution.
        """
        if rng is None:
            rng = np.random.RandomState()
        if self._env is None:
            self._env = self._env_factory()
        core = self._env.unwrapped
        interior = np.arange(1, core.size - 1)
        agent_pos = (int(rng.choice(interior)), int(rng.choice(interior)))
        agent_dir = int(rng.randint(0, 4))
        core.goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        core.agent_start_pos = agent_pos
        core.agent_start_dir = agent_dir
        obs, _ = self._env.reset()
        direction = np.zeros(4, dtype=np.float32)
        direction[obs["direction"]] = 1.0
        return {"image": obs["image"], "direction": direction}
