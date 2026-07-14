"""Load a trained agent from a run directory (`.hydra/config.yaml` + latest.pt)."""

import pathlib

import torch
from omegaconf import OmegaConf

from dreamer import Dreamer
from envs import make_env
from rewards import make_reward


def load_agent(
    logdir,
    device: str | None = None,
    *,
    require_mission_text: bool = False,
    strict: bool = True,
    freeze: bool = False,
):
    """Rebuild the Dreamer agent saved under `logdir` and load its weights.

    Returns ``(agent, config, env_cfg)`` where ``env_cfg`` is the resolved env
    config (with `device` injected) suitable for `make_env` / `env_factory`.

    Backfills config keys added after older runs were trained (`wm_only`,
    `goal_imag_horizon`) so old checkpoints load consistently.
    """
    logdir = pathlib.Path(logdir)
    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")
    device = device or config.device
    config.device = device

    # Backfill defaults for keys added after this run was trained.
    OmegaConf.set_struct(config.model, False)
    if "wm_only" not in config.model:
        config.model.wm_only = False
    if "goal_imag_horizon" not in config.model:
        config.model.goal_imag_horizon = int(config.model.imag_horizon)

    if require_mission_text and not config.mission_text:
        raise ValueError("This experiment requires mission_text=True (the run must have a TextEncoderGRU).")

    resolved = OmegaConf.to_container(config, resolve=True)
    resolved["env"]["device"] = device
    env_cfg = OmegaConf.create(resolved["env"])

    probe_env = make_env(env_cfg, 0)
    obs_space, act_space = probe_env.observation_space, probe_env.action_space

    reward_function = make_reward(config)
    agent = Dreamer(config.model, obs_space, act_space, reward_function=reward_function).to(device)

    checkpoint = torch.load(logdir / "latest.pt", map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=strict)
    agent.eval()
    if freeze:
        for p in agent.parameters():
            p.requires_grad_(False)

    print(f"Checkpoint loaded: {logdir / 'latest.pt'}  (env: {env_cfg.task}, device: {device})")
    return agent, config, env_cfg


def env_factory(config):
    """Return a zero-arg factory building a single env like the training env."""
    env_cfg = config.env if hasattr(config, "env") else config
    return lambda: make_env(env_cfg, 0)
