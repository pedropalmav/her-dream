import math
import pathlib
import warnings
from datetime import datetime

import hydra
import torch
from tensordict import TensorDict

import her_dream
import her_dream.tools as tools
from her_dream.dreamer import Dreamer
from her_dream.envs import make_env
from her_dream.rewards import make_reward

warnings.filterwarnings("ignore")
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# Hydra configs ship inside the installed her_dream package.
_CONFIG_DIR = str(pathlib.Path(her_dream.__file__).parent / "configs")


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="eval_configs")
def main(eval_cfg):
    from omegaconf import OmegaConf

    experiment_dir = pathlib.Path(eval_cfg.experiment_dir)
    train_cfg = OmegaConf.load(experiment_dir / ".hydra/config.yaml")
    config = OmegaConf.merge(train_cfg, {"env": {"time_limit": eval_cfg.eval_time_limit}})

    if "goal_index" in config.env and config.goal_type != "first_row":
        raise ValueError("goal_index is only supported for goal_type 'first_row'")

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logdir = pathlib.Path(f"evals/{now}").expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    print("Logdir", logdir)

    logger = tools.make_logger(eval_cfg.logger, logdir)
    logger.log_hydra_config(config)

    print("Create env.")
    env = make_env(config.env, 0)
    obs_space = env.observation_space
    act_space = env.action_space

    reward_function = make_reward(config)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
        reward_function=reward_function,
    ).to(config.device)

    agent.load_state_dict(torch.load(experiment_dir / "latest.pt", map_location=config.device)["agent_state_dict"])
    agent.eval()

    manual_ctrl = ManualControl(act_space.shape[0]) if eval_cfg.manual_control else None

    video_cache = []
    for i in range(eval_cfg.num_episodes):
        print(f"Episode {i + 1}")
        episode_cache = run_episode(env, agent, reward_function, manual_ctrl)
        if episode_cache is None:
            break
        video_cache.append(episode_cache["image"][:1])

    if manual_ctrl is not None:
        manual_ctrl.close()

    if video_cache:
        logger.video("eval_video", tools.to_np(make_video_grid(video_cache)))

    logger.write(len(video_cache) * config.env.time_limit)


class ManualControl:
    """Pygame window that intercepts keyboard input and returns one-hot actions.

    Key bindings (mirrors minigrid's ManualControl):
        ← / →     turn left / right
        ↑         move forward
        space      toggle
        page up    pickup
        page down  drop
        enter      done
        escape     quit episode
    """

    _KEY_TO_ACTION = {
        "left": 0,
        "right": 1,
        "up": 2,
        "page up": 3,
        "page down": 4,
        "space": 5,
        "return": 6,
    }

    def __init__(self, n_actions: int, screen_size: int = 512):
        import pygame

        self._pg = pygame
        self.n_actions = n_actions
        pygame.init()
        pygame.display.set_caption(
            "Manual Control — arrows: move, space: toggle, pgup/dn: pick/drop, enter: done, esc: quit"
        )
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        self.clock = pygame.time.Clock()

    def render(self, image):
        """Display the current observation image (H, W, C) uint8."""
        pg = self._pg
        surf = pg.surfarray.make_surface(image.transpose(1, 0, 2))
        surf = pg.transform.scale(surf, self.screen.get_size())
        self.screen.blit(surf, (0, 0))
        pg.display.flip()

    def get_action(self):
        """Block until a valid key is pressed.

        Returns a (1, n_actions) one-hot float32 tensor, or None if the user
        closed the window or pressed Escape.
        """
        pg = self._pg
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type == pg.KEYDOWN:
                    key_name = pg.key.name(event.key)
                    if key_name == "escape":
                        return None
                    if key_name in self._KEY_TO_ACTION:
                        idx = self._KEY_TO_ACTION[key_name]
                        action = torch.zeros(1, self.n_actions)
                        action[0, idx] = 1.0
                        return action
                    print(f"unbound key: {key_name!r}")
            self.clock.tick(30)

    def close(self):
        self._pg.quit()


def run_episode(env, agent, reward_function, manual_control=None):
    obs = env.reset()
    done = False
    agent_state = agent.get_initial_state(1)
    action = agent_state["prev_action"].clone()
    step = 0
    cache = []

    while True:
        trans_cpu = TensorDict(
            {k: torch.as_tensor(v, device="cpu").unsqueeze(0) for k, v in obs.items()},
            batch_size=(1,),
            device="cpu",
        ).pin_memory()

        trans = trans_cpu.to(agent.device, non_blocking=True)
        trans["action"] = action
        cache.append(trans.clone())

        agent_action, agent_state, _ = agent.act(trans, agent_state, eval=True)
        new_reward = reward_function(agent_state["stoch"], trans["goal"])
        print("step:", step, "reward:", new_reward.item())

        if manual_control is not None:
            manual_control.render(obs["image"])

        if done:
            break

        if manual_control is not None:
            action = manual_control.get_action()
            if action is None:
                return None
            action = action.to(agent.device)
            agent_state["prev_action"] = action
        else:
            action = agent_action

        obs, _, done, _ = env.step(tools.to_np(action[0]))
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

    return torch.cat(rows, dim=2)  # concat vertical (H)


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

        blank = torch.zeros_like(all_panels[0])
        while len(all_panels) < n_rows * n_cols:
            all_panels.append(blank)

        rows = []
        for r in range(n_rows):
            row_panels = all_panels[r * n_cols : (r + 1) * n_cols]
            rows.append(torch.cat(row_panels, dim=3))  # concat horizontal (W)
        return torch.cat(rows, dim=2)  # concat vertical (H)


if __name__ == "__main__":
    main()
