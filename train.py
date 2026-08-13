"""Unified training entry point.

A single `main` covers the three training pipelines that used to live in
`train.py`, `post_train.py` and `distill_text.py`. The mode is selected purely
from the (Hydra) config:

  - **From-scratch training** (default): `load_from=null`. Trains the world
    model and actor/critic end-to-end. With `wm_only=True` only the WM is
    trained (random actions).

        python3 train.py logdir=./logdir/run/01

  - **Policy post-training on a frozen WM**: `load_from=<prev-logdir>`. Loads a
    checkpoint and trains actor/critic on top, freezing the WM when
    `freeze_wm=True` (the intended use).

        python3 train.py \
            load_from=./logdir/wm_only/01 \
            logdir=./logdir/post_train/01 \
            freeze_wm=True buffer=her

  - **Text-encoder distillation on a frozen WM**: `train_text_only=True`
    together with `load_from=<prev-logdir>` and `mission_text=True`. Trains only
    the `TextEncoderGRU` (KL against the frozen RSSM posterior); the world model
    and actor/critic stay fixed.

        python3 train.py \
            train_text_only=True load_from=./logdir/wm_only/01 \
            logdir=./logdir/distill/01 mission_text=True

`load_from` points at a directory that must contain a `latest.pt` checkpoint.
Any other field (env, buffer type, goal_sample, trainer.steps, ...) can be
overridden on the command line — the resulting run is independent of the source
logdir.
"""

import atexit
import pathlib
import warnings

import hydra
import torch

import her_dream
import her_dream.tools as tools
from her_dream.buffers import make_buffer
from her_dream.dreamer import Dreamer
from her_dream.envs import make_envs
from her_dream.rewards import make_reward
from her_dream.trainer import OnlineTrainer

warnings.filterwarnings("ignore")
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# Hydra configs ship inside the installed her_dream package.
_CONFIG_DIR = str(pathlib.Path(her_dream.__file__).parent / "configs")


def validate_config(config):
    """Validate the mode-selecting flags before any heavy setup."""
    if config.train_text_only:
        if config.load_from is None:
            raise ValueError(
                "train_text_only=True requires load_from=<path-to-prev-logdir>. "
                "The directory must contain a latest.pt checkpoint (e.g. a wm_only run)."
            )
        if not config.mission_text:
            raise ValueError("train_text_only=True requires mission_text=True.")
        # Distillation freezes the WM and actor/critic and only optimizes the
        # text encoder; the other freeze modes are mutually exclusive with it.
        config.wm_only = False
        config.freeze_wm = False
    elif config.load_from is not None:
        # Policy post-training on a (frozen) world model.
        if config.wm_only:
            raise ValueError("load_from with wm_only=True is not supported (post-training trains the policy).")
        if not config.freeze_wm:
            print(
                "[train] WARNING: load_from is set but freeze_wm=False — running a "
                "full fine-tune instead of a frozen-WM post-training. Set "
                "freeze_wm=True for the intended behavior."
            )

    if getattr(config, "reset_actor_critic", False) and config.load_from is None:
        raise ValueError("reset_actor_critic=True only makes sense together with load_from=<path-to-prev-logdir>.")

    if config.env.goal_sample == "text":
        assert config.mission_text, "goal_sample='text' requires mission_text=True so the agent owns a TextEncoderGRU."


# Actor/critic parameters, including the frozen inference clones and the slow
# critic target. `reset_actor_critic=True` drops all of them from the checkpoint.
_ACTOR_CRITIC_PREFIXES = (
    "actor.",
    "value.",
    "_slow_value.",
    "_frozen_actor.",
    "_frozen_value.",
    "_frozen_slow_value.",
)

# Heads a vanilla (non goal-conditioned) Dreamer trains but this agent no longer
# builds: the goal reward function replaces the learned reward head, and the
# continuation head went with it. Present in old checkpoints, absent here.
_STALE_HEAD_PREFIXES = ("reward.", "cont.", "_frozen_reward.", "_frozen_cont.")


def _world_model_state_dict(agent, ckpt_sd):
    """Keep only the world-model tensors of `ckpt_sd`, leaving actor/critic at init.

    Needed to post-train on a checkpoint whose actor/critic were built *without*
    a goal input — e.g. the vanilla-Dreamer crafter runs, whose first actor layer
    is `(units, feat_size)` while a goal-conditioned agent needs
    `(units, feat_size + goal_size)`. A strict load fails on those tensors even
    though the world model itself is perfectly compatible.

    This is deliberately not a blanket `strict=False`: a silent partial load of
    the *world model* would be indistinguishable from a correct one (the run
    would train happily on a half-random encoder). Anything dropped or missing
    outside the actor/critic and the stale vanilla heads raises instead.
    """
    model_sd = agent.state_dict()
    kept, dropped_ac, dropped_stale = {}, [], []
    for key, value in ckpt_sd.items():
        if key.startswith(_ACTOR_CRITIC_PREFIXES):
            dropped_ac.append(key)
        elif key not in model_sd:
            if not key.startswith(_STALE_HEAD_PREFIXES):
                raise ValueError(
                    f"Checkpoint key {key!r} has no counterpart in the agent and is not a "
                    "known stale head. The checkpoint does not match this world model; "
                    "refusing to load it partially."
                )
            dropped_stale.append(key)
        elif model_sd[key].shape != value.shape:
            raise ValueError(
                f"Shape mismatch on world-model tensor {key!r}: checkpoint {tuple(value.shape)} "
                f"vs agent {tuple(model_sd[key].shape)}. Check the model config used by the "
                "source run (<load_from>/.hydra/config.yaml), e.g. model.rssm.obs_use_deter."
            )
        else:
            kept[key] = value

    missing = [k for k in model_sd if k not in kept and not k.startswith(_ACTOR_CRITIC_PREFIXES)]
    if missing:
        raise ValueError(
            f"{len(missing)} world-model tensors are absent from the checkpoint, e.g. "
            f"{missing[:5]}. Refusing to load it partially."
        )

    print(f"[train] reset_actor_critic=True: loading {len(kept)} world-model tensors.")
    print(f"[train]   re-initialized (actor/critic): {len(dropped_ac)} tensors, {_summarize(dropped_ac)}")
    if dropped_stale:
        print(f"[train]   discarded (stale vanilla heads): {len(dropped_stale)} tensors, {_summarize(dropped_stale)}")
    return kept


def _summarize(keys):
    """Top-level module names of `keys`, for logging."""
    return sorted({k.split(".")[0] for k in keys})


def load_checkpoint(agent, config):
    """Load a checkpoint into `agent` and re-establish the frozen clones.

    `load_state_dict` updates the live parameters, so the `_frozen_*` clones
    built at construction time still hold the random init. `clone_and_freeze`
    re-syncs them, then the appropriate `_apply_*` re-disables gradients.

    With `reset_actor_critic=True` only the world model is loaded and the
    actor/critic keep their fresh init (see `_world_model_state_dict`).
    """
    ckpt_path = pathlib.Path(config.load_from).expanduser() / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}.")
    print("Loading agent state_dict from", config.load_from)
    state = torch.load(ckpt_path, map_location=config.device)
    ckpt_sd = state["agent_state_dict"]
    if getattr(config, "reset_actor_critic", False):
        agent.load_state_dict(_world_model_state_dict(agent, ckpt_sd), strict=False)
    else:
        agent.load_state_dict(ckpt_sd)
    agent.clone_and_freeze()
    if agent.train_text_only:
        agent._apply_train_text_only()
    elif agent.freeze_wm:
        agent._apply_freeze_wm()

    trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    total = sum(p.numel() for p in agent.parameters())
    print(f"Trainable params after load: {trainable:,} / {total:,}")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="configs")
def main(config):
    validate_config(config)

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.make_logger(config.logger, logdir)
    # save config
    logger.log_hydra_config(config)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    reward_function = make_reward(config)
    replay_buffer = make_buffer(config, reward_function)

    print("Build agent.")
    agent = Dreamer(config.model, obs_space, act_space, reward_function=reward_function).to(config.device)
    # Share the RSSM with the buffer so HER relabeling can build the goal
    # distribution for goal_type in {log_prob, prob}.
    replay_buffer.rssm = agent.rssm

    if config.load_from is not None:
        load_checkpoint(agent, config)

    policy_trainer = OnlineTrainer(
        config.trainer,
        replay_buffer,
        logger,
        logdir,
        train_envs,
        eval_envs,
        reward_function=reward_function,
    )
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    main()
