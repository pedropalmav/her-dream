import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.append(str(pathlib.Path(__file__).parent))

from dreamer import Dreamer
from envs import make_envs
from envs.random_goal import RandomGoal
from envs.wrappers import encode_mission
from train import reward_function


@torch.no_grad()
def goal_from_random_mission(text_encoder, env_size, device, B, deterministic=False, rng=None):
    """
    Build B random mission strings, encode them, run through the text encoder,
    and return a one-hot goal sampled (or argmax) from the first z-group.

    Returns:
        goal: (B, K) float32 one-hot tensor.
    """
    # (B, L) int8 token ids
    missions = np.stack([
        encode_mission(RandomGoal.random_mission(env_size, rng)) for _ in range(B)
    ])
    mission_t = torch.as_tensor(missions, device=device)
    mission_t = mission_t.unsqueeze(1)              # (B, 1, L)

    logits = text_encoder(mission_t)                # (B, 1, S, K)
    first_group = logits[:, 0, 0, :]               # (B, K)
    K = first_group.shape[-1]

    if deterministic:
        idx = first_group.argmax(-1)
    else:
        idx = torch.distributions.Categorical(logits=first_group).sample()

    return F.one_hot(idx, K).float()               # (B, K)


def run_eval(agent, eval_envs, device, K, env_size, deterministic):
    B = eval_envs.env_num
    rng = np.random.RandomState()

    done = torch.ones(B, dtype=torch.bool, device=device)
    once_done = torch.zeros(B, dtype=torch.bool, device=device)
    steps = torch.zeros(B, dtype=torch.int32, device=device)
    returns = torch.zeros(B, dtype=torch.float32, device=device)
    success = torch.zeros(B, dtype=torch.bool, device=device)

    agent_state = agent.get_initial_state(B)
    act = agent_state["prev_action"].clone()
    current_goals = torch.zeros(B, K, device=device)

    while not once_done.all():
        steps += ~done * ~once_done

        act_cpu = act.detach().to("cpu")
        done_cpu = done.detach().to("cpu")
        trans_cpu, done_cpu = eval_envs.step(act_cpu, done_cpu)
        trans = trans_cpu.to(device, non_blocking=True)
        done = done_cpu.to(device)

        # At episode start, compute goal from a freshly sampled random mission
        is_first = trans["is_first"][:, 0].bool()   # (B,)
        if is_first.any():
            new_goals = goal_from_random_mission(
                agent.text_encoder, env_size, device, B, deterministic, rng
            )
            current_goals = torch.where(
                is_first.unsqueeze(-1), new_goals, current_goals
            )

        # Override the environment's random goal with the text-encoder goal
        trans["goal"] = current_goals

        act, agent_state = agent.act(trans, agent_state, eval=True)

        new_reward = reward_function(agent_state["stoch"], current_goals)
        step_reward = new_reward[:, 0]
        returns += step_reward * ~once_done
        # Goal reached when reward == 0 (latent first row matches goal)
        success |= (step_reward == 0) & ~once_done
        once_done |= done

    return returns.cpu(), steps.float().cpu(), success.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir", type=str, required=True,
        help="Checkpoint directory (must contain latest.pt and .hydra/config.yaml)",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g. cpu, cuda:0)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use argmax instead of sampling from the text encoder distribution",
    )
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir).expanduser()
    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")

    # Same override as train.py line 74
    OmegaConf.update(config, "env.stochastic_classes", OmegaConf.select(config, "model.discrete"), merge=True)

    if args.device is not None:
        OmegaConf.update(config, "device", args.device, merge=True)

    device = torch.device(OmegaConf.select(config, "device"))
    env_size = int(OmegaConf.select(config, "env.env_size"))
    K = int(OmegaConf.select(config, "model.discrete"))

    OmegaConf.update(config, "env.eval_episode_num", args.episodes, merge=True)
    _, eval_envs, obs_space, act_space = make_envs(config.env)

    agent = Dreamer(config.model, obs_space, act_space, reward_function=reward_function).to(device)
    ckpt = torch.load(logdir / "latest.pt", map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()

    mode = "deterministic (argmax)" if args.deterministic else "stochastic (sample)"
    print(f"Running {args.episodes} episodes | goal mode: {mode} | K={K} | grid size={env_size}")

    returns, lengths, success = run_eval(agent, eval_envs, device, K, env_size, args.deterministic)

    n_success = int(success.sum())
    print(f"\nResults over {args.episodes} episodes:")
    print(f"  Score   : {returns.mean():.3f} ± {returns.std():.3f}")
    print(f"  Length  : {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"  Success : {n_success}/{args.episodes} ({100.0 * n_success / args.episodes:.1f}%)")

    for env in eval_envs.envs:
        env.close()


if __name__ == "__main__":
    # python3 eval_text_goal.py --logdir logdir/goal_dreamer_with_text/04 --episodes 10 --device cpu
    # python3 eval_text_goal.py --logdir logdir/goal_dreamer_with_text/04 --episodes 10 --device cpu --deterministic 
    main()
