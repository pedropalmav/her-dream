import math
import pathlib
import sys
import warnings

import hydra
import torch
from datetime import datetime
import tools
from dreamer import Dreamer
from envs import make_envs
from rewards import make_reward

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="eval_configs")
def main(config):
    if "goal_index" in config.env and config.goal_type != "first_row":
        raise ValueError("goal_index is only supported for goal_type 'first_row'")

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logdir = pathlib.Path(f"evals/{now}").expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    logger.log_hydra_config(config)

    print("Create envs.")
    env, _, obs_space, act_space = make_envs(config.env)

    reward_function = make_reward(config)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
        reward_function=reward_function,
    ).to(config.device)

    weights_dir = pathlib.Path(config.logdir)
    agent.load_state_dict(
        torch.load(weights_dir / "latest.pt", map_location=config.device)[
            "agent_state_dict"
        ]
    )
    agent.eval()

    video_cache = []
    num_episodes = 6
    for i in range(num_episodes):
        print(f"Episode {i + 1}")
        episode_cache = run_episode(env, agent, reward_function)
        video_cache.append(episode_cache["image"][:1])

    if video_cache:
        logger.video("eval_video", tools.to_np(make_video_grid(video_cache)))

    logger.write(num_episodes * config.env.time_limit)


def run_episode(env, agent, reward_function):
    done = torch.ones(env.env_num, dtype=torch.bool, device=agent.device)
    once_done = torch.zeros(env.env_num, dtype=torch.bool, device=agent.device)
    agent_state = agent.get_initial_state(env.env_num)
    action = agent_state["prev_action"].clone()
    step = 0
    cache = []
    while not once_done.all():
        action_cpu = action.detach().to("cpu")
        done_cpu = done.detach().to("cpu")
        trans_cpu, done_cpu = env.step(action_cpu, done_cpu)

        trans = trans_cpu.to(agent.device, non_blocking=True)
        done = done_cpu.to(agent.device)
        trans["action"] = action
        cache.append(trans.clone())

        action, agent_state = agent.act(trans, agent_state, eval=True)
        new_reward = reward_function(agent_state["stoch"], trans["goal"])
        print("step:", step, "reward:", new_reward.item())

        once_done |= done
        step += 1

    return torch.stack(cache, dim=1)


def make_video_grid(panels):
    N = len(panels)
    n_cols = math.ceil(math.sqrt(N))
    n_rows = math.ceil(N / n_cols)

    blank = torch.zeros_like(panels[0])
    while len(panels) < n_rows * n_cols:
        panels.append(blank)

    rows = []
    for r in range(n_rows):
        row_panels = panels[r * n_cols : (r + 1) * n_cols]
        rows.append(torch.cat(row_panels, dim=3))  # concat horizontal (W)
    grid = torch.cat(rows, dim=2)  # concat vertical (H)

    return grid


def interventions_video(agent, cache):
    initial = agent.get_initial_state(1)
    data = cache[:1]
    initial = (initial["stoch"], initial["deter"])
    with torch.no_grad():
        data = agent.preprocess(data)
        B = min(data["action"].shape[0], 6)

        embedding = agent.encoder(data)

        post_stoch, post_deter, _ = agent.rssm.observe(
            embedding[:B],
            data["action"][:B],
            tuple(val[:B] for val in initial),
            data["is_first"][:B],
        )
        recon = agent.decoder(post_stoch, post_deter)["image"].mode[:B]

        intervened = []
        for i in range(agent.rssm._discrete):
            stoch, deter = post_stoch.clone(), post_deter.clone()
            stoch[:, :, 0, :] = 0
            stoch[:, :, 0, i] = 1
            recon_i = agent.decoder(stoch, deter)["image"].mode[:B]
            intervened.append(recon_i)

        all_panels = [
            data["image"][:B],
            recon,
        ] + intervened  # lista de N tensores (B, T, H, W, C)
        N = len(all_panels)  # 2 + S

        n_cols = math.ceil(math.sqrt(N))
        n_rows = math.ceil(N / n_cols)

        H, W, C = all_panels[0].shape[2], all_panels[0].shape[3], all_panels[0].shape[4]
        blank = torch.zeros_like(all_panels[0])
        while len(all_panels) < n_rows * n_cols:
            all_panels.append(blank)

        rows = []
        for r in range(n_rows):
            row_panels = all_panels[r * n_cols : (r + 1) * n_cols]
            rows.append(torch.cat(row_panels, dim=3))  # concat horizontal (W)
        grid = torch.cat(rows, dim=2)  # concat vertical (H)

        return grid


if __name__ == "__main__":
    main()
