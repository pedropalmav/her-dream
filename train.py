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
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# TODO: move this funciton to buffer module __init__.py
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
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    if config.env.goal_sample == "text":
        assert config.mission_text, (
            "goal_sample='text' requires mission_text=True so the agent owns a TextEncoderGRU."
        )

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    reward_function = make_reward(config)
    replay_buffer = make_buffer(config, reward_function)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
        reward_function=reward_function,
    ).to(config.device)

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
