"""Reusable posterior rollout driver.

`posterior_rollout` encapsulates the loop shared by most experiments: reset an
env, run the RSSM posterior chain (`obs_step`) step by step under a pluggable
`policy`, and yield a `Step` per visited state. Callers record only the fields
they need from each `Step`, keeping experiment-specific accumulation local.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from gymnasium.utils import seeding

from .envs import unwrap_env
from .obs import onehot_action, preprocess_obs


@dataclass
class Step:
    """One visited state of a posterior rollout."""

    t: int
    stoch: torch.Tensor  # (1, S, K)
    deter: torch.Tensor  # (1, D)
    logit: torch.Tensor  # (1, S, K)
    obs: dict  # raw obs dict (numpy) returned by the env at this step
    env: Any  # the (wrapped) env, for ground-truth reads via unwrap_env
    done: bool  # whether env.step returned done leading into this state


def random_policy(rng: np.random.Generator):
    """Uniform-random one-hot action policy.

    `rng` may be a numpy Generator (`.integers`) or a legacy RandomState
    (`.randint`); both are supported.
    """
    draw = getattr(rng, "integers", None) or rng.randint

    def policy(t: int, n_actions: int, state) -> np.ndarray:
        return onehot_action(int(draw(n_actions)), n_actions)

    return policy


def scripted_policy(action_indices: list):
    """Replay a fixed sequence of action indices."""

    def policy(t: int, n_actions: int, state) -> np.ndarray:
        return onehot_action(int(action_indices[t]), n_actions)

    return policy


def actor_policy(agent, goal: torch.Tensor, eval: bool = True):
    """Act with the trained actor, conditioned on a fixed (1, S, K) one-hot goal.

    Mirrors `Dreamer.act`'s policy branch: it reuses the agent's own
    `ActorCritic.policy` (so the `[feat, flat(goal)]` layout stays in one place)
    and takes its `mode` when `eval`, its sample otherwise. Unlike `act`, it
    reads the latent state from the rollout driver rather than keeping its own,
    so the posterior chain stays the single source of truth.
    """

    def policy(t: int, n_actions: int, state) -> np.ndarray:
        stoch, deter = state
        feat = agent.rssm.get_feat(stoch, deter)
        dist = agent.ac.policy(feat, goal)
        action = dist.mode if eval else dist.rsample()
        return action[0].detach().cpu().numpy().astype(np.float32)

    return policy


@torch.no_grad()
def posterior_rollout(agent, env, policy, device: str, max_steps: int, *, seed: int | None = None):
    """Yield a `Step` at every visited state of a single-env posterior rollout.

    Processes the reset observation first (t=0, is_first=True), then applies
    `policy(t, n_actions, state)` — where `state` is the current `(stoch, deter)`
    — and steps the env until `done` or `max_steps` actions have been taken.
    When `seed` is given, the base env's RNG is pre-seeded so the reset (and
    hence goal placement) is reproducible.
    """
    if seed is not None:
        base = unwrap_env(env)
        base._np_random, _ = seeding.np_random(seed)

    obs = env.reset()
    n_actions = env.action_space.shape[0]

    stoch, deter = agent.rssm.initial(1)
    prev_action = torch.zeros(1, n_actions, device=device)

    embed = agent.encoder(preprocess_obs(obs, device))
    is_first = torch.tensor([True], dtype=torch.bool, device=device)
    stoch, deter, logit = agent.rssm.obs_step(stoch, deter, prev_action, embed, is_first)
    yield Step(0, stoch, deter, logit, obs, env, done=False)

    for t in range(1, max_steps + 1):
        act_np = policy(t - 1, n_actions, (stoch, deter))
        obs, _reward, done, _info = env.step(act_np)
        prev_action = torch.as_tensor(act_np, device=device).unsqueeze(0)

        embed = agent.encoder(preprocess_obs(obs, device))
        is_first = torch.tensor([False], dtype=torch.bool, device=device)
        stoch, deter, logit = agent.rssm.obs_step(stoch, deter, prev_action, embed, is_first)
        yield Step(t, stoch, deter, logit, obs, env, done=bool(done))

        if done:
            break
