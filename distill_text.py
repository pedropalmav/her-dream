"""Text-encoder distillation on top of a frozen world model.

Loads an existing Dreamer checkpoint (e.g. one produced by `train.py` with
`wm_only=True`) and trains ONLY the `TextEncoderGRU`, keeping the world model
and actor-critic frozen. The text encoder learns to predict the (frozen) RSSM
posterior from the mission text via the `text_kl` loss.

This is "option 3": a dedicated distillation phase against a stationary target,
run between world-model pretraining (`wm_only`) and policy training
(`freeze_wm`). The resulting checkpoint can then be fed to `post_train.py` so
the policy trains with an already-trained, frozen text encoder.

Typical usage:

    python3 distill_text.py \
        load_from=./logdir/wm_only_random_mission/01 \
        logdir=./logdir/distill_text/01 \
        mission_text=True \
        env=random_goal \
        trainer.steps=200000

Then train the policy on top of the distilled checkpoint:

    python3 post_train.py \
        load_from=./logdir/distill_text/01 \
        logdir=./logdir/post_train/01 \
        freeze_wm=True mission_text=True \
        env=fixed_goal env.goal_sample=text buffer=her

`load_from` points at a directory that contains `latest.pt`. Data is collected
with random actions (the policy is irrelevant here), so only the world model and
mission text matter. Goals are not needed for distillation; prefer a
`goal_sample` that does not depend on the (still untrained) text encoder.
"""

import atexit
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from buffers import Buffer, HERBuffer
from dreamer import Dreamer
from envs import make_envs
from rewards import make_reward
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
torch.set_float32_matmul_precision("high")


def make_buffer(config, reward_function):
    match config.buffer.type:
        case "her":
            return HERBuffer(config.buffer, reward_function)
        case "normal":
            return Buffer(config.buffer)
        case _:
            raise ValueError(f"Tipo de buffer no soportado: {config.buffer.type}")


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    if config.load_from is None:
        raise ValueError(
            "distill_text.py requires `load_from=<path-to-prev-logdir>`. "
            "The directory must contain a latest.pt checkpoint (e.g. a wm_only run)."
        )
    if not config.mission_text:
        raise ValueError("distill_text.py requires mission_text=True.")

    # Force the distillation mode on and disable the incompatible modes so the
    # agent freezes the WM/actor-critic and only optimizes the text encoder.
    config.train_text_only = True
    config.wm_only = False
    config.freeze_wm = False

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)
    print("Loading agent state_dict from", config.load_from)

    logger = tools.Logger(logdir)
    logger.log_hydra_config(config)

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    reward_function = make_reward(config)
    replay_buffer = make_buffer(config, reward_function)

    print("Build agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
        reward_function=reward_function,
    ).to(config.device)

    # Load the (trained) world model and the rest of the checkpoint. The text
    # encoder weights in the source checkpoint are typically untrained — they are
    # just the starting point for distillation here.
    ckpt_path = pathlib.Path(config.load_from).expanduser() / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}.")
    state = torch.load(ckpt_path, map_location=config.device)
    agent.load_state_dict(state["agent_state_dict"])
    agent.clone_and_freeze()
    agent._apply_train_text_only()

    trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    total = sum(p.numel() for p in agent.parameters())
    print(f"Trainable params (text encoder only): {trainable:,} / {total:,}")

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
