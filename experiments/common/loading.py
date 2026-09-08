"""Load a trained agent from a run directory (`.hydra/config.yaml` + latest.pt)."""

import pathlib

import torch
from omegaconf import OmegaConf

import her_dream.tools as tools
from her_dream import goals
from her_dream.dreamer import Dreamer
from her_dream.envs import make_env
from her_dream.rewards import make_reward


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
    `goal_imag_horizon`, and the goal-type descriptors) so old checkpoints load
    consistently.
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

    # The goal-type descriptors (state_repr / goal_repr / scope) arrived with the
    # goal_type config group; runs trained before it have none, and every consumer
    # that builds a GoalSpec — Dreamer, but also the GoalConditioned wrapper — would
    # raise on the missing key. They are fanned out per section, so backfill each
    # one this loader constructs, reading the config group as the source of truth.
    if "goal_type" in config:
        descriptors = goals.default_descriptors(config.goal_type)
        for section in ("model", "env"):
            OmegaConf.set_struct(config[section], False)
            if "state_repr" not in config[section]:
                for key, val in descriptors.items():
                    config[section][key] = val

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
    agent.load_state_dict(tools.migrate_agent_state_dict(checkpoint["agent_state_dict"]), strict=strict)
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
