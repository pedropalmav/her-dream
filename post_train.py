"""Policy post-training on top of a frozen world model.

Loads an existing Dreamer checkpoint (e.g. one produced by `train.py` with
`wm_only=True`) and runs `OnlineTrainer` with the world-model parameters
frozen. Only the actor and critic are updated.

Typical usage:

    python3 post_train.py \
        load_from=./logdir/wm_only_random_mission/01 \
        logdir=./logdir/post_train/01 \
        freeze_wm=True wm_only=False \
        env=fixed_goal env.goal_sample=text mission_text=True \
        buffer=her trainer.steps=500000

`load_from` points at a directory that contains `latest.pt`. Any other field
(env, buffer type, goal_sample, trainer.steps, ...) can be overridden on the
command line — the resulting run is independent of the source logdir.
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
from trainer import OnlineTrainer

from rewards import make_reward

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
            "post_train.py requires `load_from=<path-to-prev-logdir>`. "
            "The directory must contain a latest.pt checkpoint."
        )
    if not config.freeze_wm:
        print(
            "[post_train] WARNING: freeze_wm=False — running a full fine-tune "
            "instead of a frozen-WM post-training. Set freeze_wm=True for the "
            "intended behavior."
        )
    if config.wm_only:
        raise ValueError("post_train.py is incompatible with wm_only=True.")

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

    if config.env.goal_sample == "text":
        assert config.mission_text, (
            "goal_sample='text' requires mission_text=True so the agent owns a TextEncoderGRU."
        )

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

    # Load WM (and actor/critic init) from the source logdir, then re-establish
    # the storage sharing between `self.*` and `self._frozen_*` clones.
    ckpt_path = pathlib.Path(config.load_from).expanduser() / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}.")
    state = torch.load(ckpt_path, map_location=config.device)
    agent.load_state_dict(state["agent_state_dict"])
    agent.clone_and_freeze()
    if agent.freeze_wm:
        agent._apply_freeze_wm()

    trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    total = sum(p.numel() for p in agent.parameters())
    print(f"Trainable params after freeze: {trainable:,} / {total:,}")

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
