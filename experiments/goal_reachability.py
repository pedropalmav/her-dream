"""Is an imagined goal-z actually reachable by the z-reward, even with the perfect plan?

`imagine_goal` produces a goal by rolling the WM prior 15 random-action steps from z0.
The "reachable by construction" claim is about the STATE. But the reward compares the
z SAMPLE (all 32 one-hot groups, exactly). This script executes the *same* 15 actions
in the real env (an oracle plan) and measures how close the resulting posterior z gets
to the imagined goal z, under three lenses:

  A) sample match  (goal_type=full):      argmax(post_z) vs argmax(imag_z), per group
  B) mode match    (goal_type=argmax_full): argmax(post_logit) vs argmax(imag_logit)
  C) sampling floor: two independent samples from the SAME posterior logits
                     -> the irreducible Gumbel noise; an upper bound on full-match.

If even the oracle plan can't reproduce the goal z (A), no policy can: the goal is
unreachable in z-space by construction, regardless of green-square fidelity.
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from tensordict.base import TensorDictBase  # noqa: E402

# MPS pin_memory stub (harmless on CPU), kept for parity with local smoke tests.
TensorDictBase.pin_memory = lambda self, *a, **k: self

import her_dream.goals as goals  # noqa: E402
from experiments.common import onehot_action  # noqa: E402
from her_dream.dreamer import Dreamer  # noqa: E402
from her_dream.envs import make_env  # noqa: E402
from her_dream.rewards import make_reward  # noqa: E402

# Hydra configs now ship inside the her_dream package (next to goals.py).
_CONFIG_DIR = str(pathlib.Path(goals.__file__).parent / "configs")

CKPT = "logdir/random_goal/wm_only_randomgoal/01/latest.pt"
OUT = pathlib.Path("experiments/out")


def build(env_name, device="cpu"):
    """Build env + Dreamer and load the frozen random_goal world model."""
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        config = compose(
            config_name="configs",
            overrides=[
                f"env={env_name}",
                "model=size12M",
                f"device={device}",
                "goal_type=full",
                "mission_text=False",
                "env.goal_sample=imagination",
            ],
        )
    env = make_env(config.env, 0)
    reward_function = make_reward(config)
    agent = Dreamer(config.model, env.observation_space, env.action_space, reward_function=reward_function).to(device)
    state = torch.load(CKPT, map_location=device)
    agent.load_state_dict(state["agent_state_dict"], strict=False)
    agent.eval()
    for p in agent.parameters():
        p.requires_grad_(False)
    return config, env, agent


def rand_action(n, rng):
    return onehot_action(rng.integers(n), n)


def embed_of(agent, image_np, dir_np):
    """Encode a single observation -> (1, E)."""
    dev = agent.device
    img = torch.as_tensor(np.asarray(image_np), dtype=torch.float32, device=dev)[None]  # (1,80,80,3)
    d = torch.as_tensor(np.asarray(dir_np), dtype=torch.float32, device=dev)[None]  # (1,4)
    return agent.encoder(agent.preprocess({"image": img, "direction": d}))


@torch.no_grad()
def dual_rollout(agent, env, H, rng):
    """Posterior rollout (real obs) and prior rollout (imagined), SAME actions, from same z0."""
    rssm, dev, A = agent.rssm, agent.device, agent.act_dim
    obs = env.reset()
    z, h = rssm.initial(1)
    pa = torch.zeros(1, A, device=dev)
    z, h, lg = rssm.obs_step(
        z, h, pa, embed_of(agent, obs["image"], obs["direction"]), torch.ones(1, dtype=torch.bool, device=dev)
    )
    post_z, post_lg = [z], [lg]
    iz, ih = z.clone(), h.clone()
    imag_z, imag_lg = [iz], [lg]
    for _ in range(H):
        u = rand_action(A, rng)
        ut = torch.as_tensor(u, device=dev)[None]
        iz, ih, ilg = rssm.img_step(iz, ih, ut)
        imag_z.append(iz)
        imag_lg.append(ilg)
        obs, _, _, _ = env.step(u)
        z, h, lg = rssm.obs_step(
            z, h, ut, embed_of(agent, obs["image"], obs["direction"]), torch.zeros(1, dtype=torch.bool, device=dev)
        )
        post_z.append(z)
        post_lg.append(lg)
    return post_z, post_lg, imag_z, imag_lg


def groups_eq(za, zb):
    """# of the 32 one-hot groups that match (by argmax), and whether all match."""
    a, b = za.argmax(-1), zb.argmax(-1)  # (1,32)
    eq = a == b
    return int(eq.sum()), bool(eq.all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="random_goal")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--resets", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"Build model + env on {args.device}...")
    _, env, agent = build(args.env, args.device)
    S = agent.rssm._stoch  # 32 groups

    H = args.horizon
    # aligned same-plan: real exec vs imagined, per step
    samp_groups = np.zeros(H + 1)  # A: argmax(z) match
    samp_full = np.zeros(H + 1)
    mode_groups = np.zeros(H + 1)  # B: argmax(logit) match
    mode_full = np.zeros(H + 1)
    floor_groups = np.zeros(H + 1)  # C: two samples from same posterior logits
    floor_full = np.zeros(H + 1)
    # per-step approach to the FINAL imagined goal z (the actual goal)
    pvg_groups = np.zeros(H + 1)  # real execution vs final goal
    pvg_full = np.zeros(H + 1)
    ivg_groups = np.zeros(H + 1)  # imagined trajectory vs its own final goal
    # goal reachability: best match of any real step vs the FINAL imagined goal z
    best_vs_goal = []
    full_reach = 0

    for _ in range(args.resets):
        post_z, post_lg, imag_z, imag_lg = dual_rollout(agent, env, H, rng)
        goal = imag_z[H]
        best = 0
        reached = False
        for t in range(H + 1):
            ng, nf = groups_eq(post_z[t], imag_z[t])
            samp_groups[t] += ng
            samp_full[t] += nf
            mg = (post_lg[t].argmax(-1) == imag_lg[t].argmax(-1)).sum().item()
            mode_groups[t] += mg
            mode_full[t] += mg == S
            # sampling floor: resample z from the same posterior logits
            z2 = agent.rssm.get_dist(post_lg[t]).rsample()
            fg, ff = groups_eq(post_z[t], z2)
            floor_groups[t] += fg
            floor_full[t] += ff
            # per-step approach to the final imagined goal
            gg, gf = groups_eq(post_z[t], goal)
            pvg_groups[t] += gg
            pvg_full[t] += gf
            ivg_groups[t] += groups_eq(imag_z[t], goal)[0]
            best = max(best, gg)
            reached = reached or gf
        best_vs_goal.append(best)
        full_reach += reached

    n = args.resets
    print(f"\n=== Aligned same-plan: real execution vs imagined, averaged over {n} resets ===")
    print(" step | A:sample grp/32 | A:full% | B:mode grp/32 | B:full% | C:floor grp/32 | C:full%")
    for t in range(H + 1):
        print(
            f" {t:4d} |     {samp_groups[t] / n:5.1f}      |  {100 * samp_full[t] / n:4.0f}%  |"
            f"    {mode_groups[t] / n:5.1f}     |  {100 * mode_full[t] / n:4.0f}% |"
            f"     {floor_groups[t] / n:5.1f}      |  {100 * floor_full[t] / n:4.0f}%"
        )

    print("\n=== Reachability of the FINAL imagined goal z (the actual goal) ===")
    print(f" best groups matched over the episode (mean of {n}): {np.mean(best_vs_goal):.1f} / {S}")
    print(f" episodes with ANY exact full-z match to the goal: {full_reach}/{n} ({100 * full_reach / n:.0f}%)")

    # ---- step-by-step plots ----
    steps = np.arange(H + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: full-match RATE per step (does the reward fire?)
    ax1.plot(steps, 100 * samp_full / n, "-o", label="A: sample  (goal_type=full)")
    ax1.plot(steps, 100 * mode_full / n, "-s", label="B: mode  (argmax_full)")
    ax1.plot(steps, 100 * floor_full / n, "-^", label="C: sampling floor (same state)")
    ax1.set_xlabel("aligned step t")
    ax1.set_ylabel("full-z match rate (%)")
    ax1.set_ylim(-2, 102)
    ax1.legend()
    ax1.set_title("Would the reward fire?  (real exec vs imagined)")

    # Panel 2: mean groups matched per step (how close, out of 32)
    ax2.plot(steps, samp_groups / n, "-o", label="A: sample (full)")
    ax2.plot(steps, mode_groups / n, "-s", label="B: mode (argmax_full)")
    ax2.plot(steps, floor_groups / n, "-^", label="C: sampling floor")
    ax2.axhline(S, color="gray", ls="--", lw=1, label="32 = full match")
    ax2.set_xlabel("aligned step t")
    ax2.set_ylabel(f"mean groups matched / {S}")
    ax2.set_ylim(0, S + 1)
    ax2.legend()
    ax2.set_title("How close?  (real exec vs imagined)")

    # Panel 3: per-step approach to the FINAL imagined goal z
    ax3.plot(steps, pvg_groups / n, "-o", label="real execution vs goal")
    ax3.plot(steps, ivg_groups / n, "-s", label="imagined traj vs goal")
    ax3.axhline(S, color="gray", ls="--", lw=1, label="32 = full match")
    ax3.set_xlabel("step t")
    ax3.set_ylabel(f"mean groups matched / {S}")
    ax3.set_ylim(0, S + 1)
    ax3.legend()
    ax3.set_title(f"Approach to FINAL goal z (full-reach {100 * full_reach / n:.0f}%)")

    fig.suptitle(f"{args.env}: reachability of imagined goal-z, step by step (n={n} resets, 0% full-z reach)")
    fig.tight_layout()
    path = OUT / f"goal_reachability_{args.env}.png"
    fig.savefig(path, dpi=120)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
