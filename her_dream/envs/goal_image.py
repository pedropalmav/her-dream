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

    On an env with hazards the *whole* layout has to be reproduced, not just the
    goal: `lava_pos` pins the auxiliary env's lava to the live episode's cells
    (cookie-env's `LavaGrid.lava_pos`). Without it the auxiliary env resamples its
    own lava and the rendered state is one the live episode cannot reach — the same
    unreachable-goal failure that buffer-sampled goals have on random_goal.
    """

    def __init__(self, env_factory):
        self._env_factory = env_factory
        self._env = None

    def observation(self, goal_pos, rng: np.random.RandomState = None, agent_pos=None, agent_dir=None, lava_pos=None):
        """Return {"image": (H, W, 3) uint8, "direction": (4,) one-hot float32}.

        Agent position and direction are sampled uniformly from the grid
        interior; agent and goal may coincide (the success state), matching
        the training distribution. Pass `agent_pos` / `agent_dir` to pin either
        one instead of sampling it — used by the evaluation experiments, which
        need the ground-truth target state that the rendered goal encodes
        (e.g. the agent standing on the green square).

        `lava_pos` pins the auxiliary env's lava to the live episode's cells. A
        sampled agent is then drawn from the interior *minus* those cells: unlike
        the green square, standing on lava is a death state, not a state worth
        handing the agent as a goal. A pinned `agent_pos` on lava is left to raise
        in the env — the caller asked for a state that cannot exist.
        """
        if rng is None:
            rng = np.random.RandomState()
        if self._env is None:
            self._env = self._env_factory()
        core = self._env.unwrapped
        interior = np.arange(1, core.size - 1)
        if agent_pos is not None:
            agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
        elif lava_pos:
            # Only the hazard case filters, so the lava-free path keeps drawing the
            # two coordinates independently and stays bit-identical.
            lava = {(int(x), int(y)) for x, y in lava_pos}
            free = [(int(x), int(y)) for x in interior for y in interior if (int(x), int(y)) not in lava]
            agent_pos = free[int(rng.randint(0, len(free)))]
        else:
            agent_pos = (int(rng.choice(interior)), int(rng.choice(interior)))
        agent_dir = int(rng.randint(0, 4)) if agent_dir is None else int(agent_dir)
        core.goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        core.agent_start_pos = agent_pos
        core.agent_start_dir = agent_dir
        if lava_pos is not None:
            core.lava_pos = [(int(x), int(y)) for x, y in lava_pos]
        obs, _ = self._env.reset()
        direction = np.zeros(4, dtype=np.float32)
        direction[obs["direction"]] = 1.0
        return {"image": obs["image"], "direction": direction}
