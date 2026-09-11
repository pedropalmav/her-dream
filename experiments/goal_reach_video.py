"""Watch a goal-conditioned agent chase a *known* goal — on video.

The reward curves say these `max_cosine` fixed_goal runs learned to reach goals,
but that verdict lives entirely in z-space (cosine of two logit vectors). This
script turns it into something you can watch: it builds the goal from a rendered
synthetic state — the agent standing ON the green square — encodes it with the
frozen WM (`Dreamer.encode_observation`), then lets the trained actor act on that
goal z and records the rollout as a GIF next to the goal image it is chasing.

Because we render the goal state ourselves, the ground-truth target `(x, y)` is
known, so "did it arrive?" is a real question, not a z-space one — the same idea
as layer 2 (POLICY, target=square) of `goal_observation_eval.py`, but rendered to
video instead of reduced to a bar chart. It reuses that branch's plumbing:
`GoalImageObservation.goal_observation(agent_pos, agent_dir)` renders the pinned
state (present on every fixed_goal env, regardless of the run's goal_sample), and
`actor_policy` drives the trained actor from the rollout's own latent state.

The layout is pinned on every reset: the green square at a fixed cell (default:
the top-right corner) and the agent start at the opposite bottom-left corner. So
random_goal — which otherwise scatters the goal each episode — reproduces the
exact fixed_goal scenario across the whole diagonal, and the live square the agent
sees matches the goal it is told to chase. Override the square with `--goal-pos`.

With the layout and the (deterministic) actor fixed, the only thing that varies is
the direction the goal z encodes, so the script sweeps all four facings (0-3), one
GIF per direction. Restrict it to a single facing with `--goal-dir`.

Usage — the two max_cosine fixed_goal runs:

    uv run python3 experiments/goal_reach_video.py \
        --logdir logdir/full_train/fixed_goal/max_cosine_her \
                 logdir/full_train/fixed_goal/max_cosine_normal \
        --max-steps 120 --device cuda

The same test on the random_goal models (goal forced to the top-right corner):

    uv run python3 experiments/goal_reach_video.py \
        --logdir logdir/full_train/random_goal/max_cosine_her \
                 logdir/full_train/random_goal/max_cosine_normal \
        --max-steps 120 --device cuda

Local smoke test (CPU, ~1 min):

    uv run python3 experiments/goal_reach_video.py \
        --logdir logdir/full_train/random_goal/max_cosine_her \
        --goal-dir 0 --max-steps 40 --device cpu
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import imageio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from minigrid.core.grid import Grid  # noqa: E402
from minigrid.core.world_object import Goal  # noqa: E402

from experiments.common import (  # noqa: E402
    actor_policy,
    add_common_args,
    agent_pose,
    encode_goal_image,
    goal_pos,
    load_agent,
    posterior_rollout,
    random_policy,
    reward_of,
    save_fig,
    unwrap_env,
)
from her_dream import goals  # noqa: E402
from her_dream.envs import make_env  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "out"


def pin_layout(base, goal_cell, agent_cell):
    """Pin the green square and the agent's start cell on every reset.

    random_goal drops the goal at a random interior cell each reset (and, with
    agent_start_random, the agent too); fixed_goal reads the goal from config.
    Overriding the instance's `_gen_grid` pins both for either env, so the live
    episode the agent navigates matches the goal we encode (agent standing on the
    top-right square, starting from the bottom-left) — the fixed_goal test
    transplanted onto the random_goal model, rather than a randomly-placed square.
    """
    gx, gy = int(goal_cell[0]), int(goal_cell[1])
    base.agent_start_pos = (int(agent_cell[0]), int(agent_cell[1]))

    def _gen_grid(width, height):
        base.grid = Grid(width, height)
        base.grid.wall_rect(0, 0, width, height)
        base.put_obj(Goal(), gx, gy)
        # goal_position reads `_goal_pos` (random_goal) or `goal_pos` (fixed_goal);
        # set both so either env's property returns the pinned cell.
        base._goal_pos = (gx, gy)
        base.goal_pos = (gx, gy)
        base._put_agent()  # reads base.agent_start_pos / agent_start_dir

    base._gen_grid = _gen_grid


def build_goal(agent, env, target_pos, target_dir):
    """Encode `(target_pos, target_dir)` into this run's goal z, keeping the image.

    The goal tensor is produced by the shared `encode_goal_image` (raw logit for
    max_cosine, via `goals.goal_from_latent`); we re-render the same deterministic
    state once more only to keep the uint8 RGB image for the reference panel.
    `get_wrapper_attr` reaches `goal_observation` through the TimeLimit/Dtype
    wrappers that no longer forward attribute access.
    """
    goal = encode_goal_image(agent, env, target_pos, target_dir)
    goal_img = env.get_wrapper_attr("goal_observation")(agent_pos=target_pos, agent_dir=target_dir)["image"]
    return goal, goal_img


def compose_frame(live_img, goal_img, title):
    """Two panels (live rollout | encoded goal) with a status line, as uint8 RGB."""
    fig, (ax_live, ax_goal) = plt.subplots(1, 2, figsize=(6.4, 3.4))
    ax_live.imshow(live_img)
    ax_live.set_title("live rollout", fontsize=9)
    ax_goal.imshow(goal_img)
    ax_goal.set_title("goal (agent on square)", fontsize=9)
    for ax in (ax_live, ax_goal):
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def rollout_frames(agent, spec, env, policy, goal, goal_img, square, device, max_steps, seed):
    """Drive `policy`, composing one frame per step; report arrival + per-step reward."""
    base = unwrap_env(env)
    frames, rewards, reached_at, min_dist, steps_on_square = [], [], None, None, 0

    for step in posterior_rollout(agent, env, policy, device, max_steps=max_steps, seed=seed):
        (ax, ay), _dir = agent_pose(base)
        dist = abs(ax - square[0]) + abs(ay - square[1])
        min_dist = dist if min_dist is None else min(min_dist, dist)
        on_square = (ax, ay) == (int(square[0]), int(square[1]))
        if on_square:
            steps_on_square += 1
            if reached_at is None:
                reached_at = step.t

        rewards.append(reward_of(agent, spec, step.stoch, step.logit, goal))
        title = f"step {step.t:3d}" + ("  |  ON SQUARE" if on_square else "")
        frames.append(compose_frame(step.obs["image"], goal_img, title))

    return frames, {
        "reached": reached_at is not None,
        "reached_at": reached_at,
        "min_dist": int(min_dist),
        "steps_on_square": steps_on_square,
        "rewards": rewards,
    }


def run_checkpoint(logdir, stem, goal_target, directions, max_steps, fps, device, seed, outdir):
    """Evaluate one run: one GIF per goal-facing direction, plus a printed summary.

    The goal square and agent start are pinned, and the actor is deterministic, so
    the only thing that varies is the direction the goal z encodes — hence one
    rollout per direction rather than per (identical) episode.
    """
    agent, config, env_cfg = load_agent(logdir, device)
    if config.env.goal_sample != "buffer":
        print(f"  note: goal_sample={config.env.goal_sample} (this script encodes an image goal regardless)")
    env = make_env(env_cfg, 0)
    # Same sub-config Dreamer builds its spec from, so reward routing matches the run.
    spec = goals.make_goal_spec(config.model)
    base = unwrap_env(env)
    # Pin the green square (default: top-right corner) and the agent start (bottom-left
    # corner) so random_goal stops scattering them and every run reproduces the
    # fixed_goal scenario across the whole diagonal.
    square = tuple(int(v) for v in (goal_target if goal_target is not None else (env_cfg.env_size - 2, 1)))
    agent_start = (1, int(env_cfg.env_size) - 2)
    pin_layout(base, square, agent_start)
    rng = np.random.default_rng(seed)

    rewards_by_dir, reached = {}, {}
    for d in directions:
        # max_steps=0 seeds and resets the env so the (pinned) layout is applied
        # before any action; the policy is never called on this priming pass.
        next(posterior_rollout(agent, env, random_policy(rng), device, max_steps=0, seed=seed))
        assert tuple(int(v) for v in goal_pos(base)) == square, "goal pin did not take"

        goal, goal_img = build_goal(agent, env, square, d)
        frames, info = rollout_frames(
            agent, spec, env, actor_policy(agent, goal), goal, goal_img, square, device, max_steps, seed
        )
        rewards_by_dir[d] = info["rewards"]
        reached[d] = info["reached"]

        path = outdir / f"{stem}_dir{d}.gif"
        imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
        verdict = f"reached at step {info['reached_at']}" if info["reached"] else f"never (min dist {info['min_dist']})"
        print(f"  [{stem} dir{d}] square {square} start {agent_start}: {verdict}  ->  {path.name}")

    hits = sum(reached.values())
    print(f"  == {stem}: reached the square facing {hits}/{len(directions)} of the tested directions ==")
    return rewards_by_dir


def plot_rewards(all_rewards, path):
    """One panel per model; a reward-vs-step curve per goal-facing direction."""
    models = list(all_rewards)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False, sharey=True)
    for ax, name in zip(axes[0], models):
        for d, rewards in sorted(all_rewards[name].items()):
            ax.plot(rewards, lw=1.3, label=f"dir {d}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("reward (goal = agent on square)")
    fig.suptitle("Per-direction reward chasing the encoded goal")
    fig.tight_layout()
    save_fig(fig, path)


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__, conflict_handler="resolve"))
    parser.add_argument("--logdir", required=True, nargs="+", help="One or more run dirs to evaluate")
    parser.add_argument("--max-steps", type=int, default=120, help="Rollout length (episode never terminates early)")
    parser.add_argument(
        "--goal-dir",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Test only this goal-facing direction (default: sweep all four, 0-3)",
    )
    parser.add_argument(
        "--goal-pos",
        type=int,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Cell to pin the green square to (default: top-right corner, size-2 1)",
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    directions = [args.goal_dir] if args.goal_dir is not None else [0, 1, 2, 3]
    all_rewards = {}
    for logdir in args.logdir:
        # Include the parent dir (fixed_goal / random_goal) so the two same-named
        # runs don't overwrite each other's GIFs when sharing an --out.
        stem = f"{pathlib.Path(logdir).parent.name}_{pathlib.Path(logdir).name}"
        print(f"\n=== {logdir} ===")
        all_rewards[stem] = run_checkpoint(
            logdir, stem, args.goal_pos, directions, args.max_steps, args.fps, device, args.seed, outdir
        )
    tag = "_".join(sorted({pathlib.Path(p).parent.name for p in args.logdir}))
    plot_rewards(all_rewards, outdir / f"goal_reach_reward_{tag}.png")
    print(f"\n  GIFs + reward plot written to {outdir}")


if __name__ == "__main__":
    main()
