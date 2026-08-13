"""How often does a trained agent actually reach the state its goal-z stands for?

Every verdict so far is either a z-space score or a 50-episode spot check whose
layout is resampled on every invocation, so two checkpoints are never asked the
same questions and a rerun does not reproduce its own number. This script fixes
the questions: a frozen suite of 1000 `(initial_state, goal_state)` pairs
(`experiments/common/pairs.py`), identical across executions and across models,
turning "success rate" into one comparable number per checkpoint.

A pair is a ground-truth navigation task on the goal-grid room: spawn pose in,
target pose out. There is one suite per env (`--suite`, auto-detected from the
run's task), because each env poses a different question:

  * `random_goal` — the tasks are **sampled**: random spawn pose, and a goal cell
    drawn **independently of the green square**, so this measures
    goal-conditioning ("go to the state this z stands for") rather than the
    built-in reach-the-square task the env rewards. Every pair is at least 4
    actions from the spawn, so none is solvable by standing still.
  * `fixed_goal` — the layout never moves, so the tasks are **enumerated**: from
    the one spawn the agent trained at, every interior cell in every facing (the
    `sweep` block: 252 poses on a 10x10 room), giving a complete map of the
    goal-conditioning instead of a sample. The remaining 748 pairs are sampled
    spawn/goal poses — spawns the agent never trained from — and the two blocks
    are reported separately so in- and out-of-distribution do not pool.

The layout is dictated per pair via `fixed_goal_cfg` — the controllable FixedGoal
env honours both the spawn and the square — and the goal z comes from
`encode_goal_image`: the env renders the target state (green square at the pair's
position, agent at the target cell) and the frozen WM encodes it. Because we
render it, the state behind the goal z is known, so "did it arrive?" is a
ground-truth question.

Three measurements per pair:

  1. POLICY — the trained actor conditioned on the goal z. Success = the agent
     matches the full target pose (cell + facing); reaching the cell with the
     wrong facing is recorded separately. The rollout stops at the first pose
     match — that is the success event, nothing after it counts.
  2. RANDOM — the same driver on uniform-random actions: the floor a policy must
     beat to have learned anything.
  3. ORACLE — a BFS shortest path to the target pose, then the posterior z at
     arrival vs the goal z, against the sampling floor (a second draw from the
     same logits). If the oracle cannot reproduce the goal z, no policy can, and
     the policy number for that run says nothing about the policy.

Usage — the trained checkpoints, on the GPU box:

    uv run python3 experiments/success_rate.py --logdir logdir/full_train/fixed_goal/* --device cuda

    uv run python3 experiments/success_rate.py \
        --logdir logdir/random_goal/posttrain_frozenwm_herbuf_goalbuf/01 \
                 logdir/random_goal/posttrain_frozenwm_normalbuf_goalbuf/01 \
        --device cuda

Results go to `experiments/out/success_rate_<suite>.{json,png}`, plus a per-cell
success map (`..._map.png`) whenever the suite has a sweep block. Runs whose room
layout differs from the suite's are skipped rather than silently mixed in.

Cost: a failed pair burns the whole step budget (default: the run's
`env.time_limit`, 1000), so the worst case is ~1M WM steps per policy per model,
doubled by the random baseline. Successes stop early, `--max-steps` shortens the
budget, and the JSON is written after every checkpoint so a long run is not lost.

Local smoke test (CPU, ~1 min):

    uv run python3 experiments/success_rate.py \
        --logdir logdir/full_train/fixed_goal/max_cosine_normal \
        --pairs 20 --max-steps 100 --device cpu
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.common import (  # noqa: E402
    actor_policy,
    add_common_args,
    agent_pose,
    bfs_plan,
    dump_json,
    encode_goal_image,
    fixed_goal_cfg,
    groups_matched,
    load_agent,
    make_suite,
    oracle_z_check,
    posterior_rollout,
    random_policy,
    reward_of,
    save_fig,
    unwrap_env,
)
from her_dream import goals  # noqa: E402
from her_dream.envs import make_env  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "out"

# Success rate is reported per distance band as well as overall: the interesting
# failure signature is whether it decays with how far the goal is. The 1-3 band
# only fills for the fixed_goal sweep, which enumerates the near-spawn poses the
# sampled suite filters out.
BUCKETS = [(1, 3), (4, 8), (9, 14), (15, 10**6)]

# The blocks a suite can be made of (see experiments/common/pairs.py).
BLOCKS = ["sweep", "random"]


def bucket_label(plan_len: int) -> str:
    """The distance band a pair belongs to, e.g. '4-8' or '15+'."""
    for lo, hi in BUCKETS:
        if lo <= plan_len <= hi:
            return f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
    raise ValueError(f"plan_len {plan_len} outside the buckets")


# ------------------------------------------------------------------ the drive


def drive(agent, spec, env, policy, goal, target_pos, target_dir, device, max_steps, seed):
    """Run `policy` on the goal z until the target pose is reached or the budget runs out.

    Records the ground truth (did it arrive, when, how close did it get) next to
    the z-space view (how many groups matched, what reward the run's own goal_type
    would have paid) and the per-step confusion between the two: false positives
    (reward fires at the wrong pose) say the z does not identify the state; false
    negatives (at the pose, no reward) say there was no signal to learn from.
    """
    base = unwrap_env(env)
    S = goal.shape[1]
    steps_to_cell, steps_to_pose = None, None
    min_dist, groups_seen, rewards = None, [], []
    conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for step in posterior_rollout(agent, env, policy, device, max_steps=max_steps, seed=seed):
        pos, direction = agent_pose(base)
        dist = abs(pos[0] - target_pos[0]) + abs(pos[1] - target_pos[1])
        min_dist = dist if min_dist is None else min(min_dist, dist)
        if pos == target_pos and steps_to_cell is None:
            steps_to_cell = step.t
        pose_match = (pos, direction) == (target_pos, target_dir)
        if pose_match and steps_to_pose is None:
            steps_to_pose = step.t

        matched = groups_matched(step.stoch, goal)
        groups_seen.append(matched)
        rewards.append(reward_of(agent, spec, step.stoch, step.logit, goal))
        z_match = matched == S
        conf["tp" if (z_match and pose_match) else "fp" if z_match else "fn" if pose_match else "tn"] += 1

        if pose_match:
            break  # the success event: nothing after it counts

    return {
        "success": steps_to_pose is not None,
        "reached_cell": steps_to_cell is not None,
        "steps_to_pose": steps_to_pose,
        "steps_to_cell": steps_to_cell,
        "min_dist": min_dist,
        "best_groups": max(groups_seen),
        "mean_reward": float(np.mean(rewards)),
        "steps_run": len(groups_seen) - 1,
        "confusion": conf,
    }


def run_pair(agent, spec, env_cfg, pair, device, max_steps, rng, base_seed, skip_random, skip_oracle):
    """Evaluate one benchmark pair: build its layout and goal, then run the drives."""
    size = int(env_cfg.env_size)
    (spawn_pos, spawn_dir) = tuple(pair["spawn"][0]), pair["spawn"][1]
    (target_pos, target_dir) = tuple(pair["goal"][0]), pair["goal"][1]

    episode_cfg = fixed_goal_cfg(env_cfg, goal_pos=pair["square"], agent_start_pos=spawn_pos, agent_start_dir=spawn_dir)
    env = make_env(episode_cfg, 0)
    # The green square sits at the pair's cell in BOTH the goal image and the live
    # episode, so the goal is consistent with the episode by construction; only the
    # target cell is independent of the square.
    goal = encode_goal_image(agent, env, target_pos, target_dir)
    # Reproducible resets; the layout itself is dictated by episode_cfg.
    seed = base_seed + pair["index"]

    record = {
        "index": pair["index"],
        "spawn": pair["spawn"],
        "goal": pair["goal"],
        "square": pair["square"],
        "plan_len": pair["plan_len"],
        "on_square": pair["on_square"],
        "block": pair["block"],
        "policy": drive(
            agent, spec, env, actor_policy(agent, goal), goal, target_pos, target_dir, device, max_steps, seed
        ),
    }
    if not skip_random:
        record["random"] = drive(
            agent, spec, env, random_policy(rng), goal, target_pos, target_dir, device, max_steps, seed
        )
    if not skip_oracle:
        # The full-pose plan, unlike the cell-only distance the suite filters on.
        plan = bfs_plan(size, spawn_pos, spawn_dir, target_pos, target_dir)
        record["oracle"] = oracle_z_check(agent, spec, env, goal, target_pos, target_dir, plan, device, seed=seed)
    return record


# ------------------------------------------------------------- aggregation


def _rates(records, key):
    """Headline numbers for one driver ('policy' or 'random') over the suite."""
    runs = [r[key] for r in records]
    steps = [r["steps_to_pose"] for r in runs if r["steps_to_pose"] is not None]
    return {
        "success_rate": float(np.mean([r["success"] for r in runs])),
        "reach_cell_rate": float(np.mean([r["reached_cell"] for r in runs])),
        "mean_steps_to_pose": float(np.mean(steps)) if steps else None,
        "median_steps_to_pose": float(np.median(steps)) if steps else None,
        "mean_min_dist": float(np.mean([r["min_dist"] for r in runs])),
        "mean_best_groups": float(np.mean([r["best_groups"] for r in runs])),
        "mean_reward": float(np.mean([r["mean_reward"] for r in runs])),
    }


def summarize(records, S, has_random, has_oracle):
    """Aggregate the per-pair records into the reported summary."""
    summary = {"pairs": len(records), "groups_total": S, "policy": _rates(records, "policy")}

    by_bucket = {}
    for label in [bucket_label(lo) for lo, _ in BUCKETS]:
        group = [r for r in records if bucket_label(r["plan_len"]) == label]
        by_bucket[label] = {
            "pairs": len(group),
            "success_rate": float(np.mean([r["policy"]["success"] for r in group])) if group else None,
            "reach_cell_rate": float(np.mean([r["policy"]["reached_cell"] for r in group])) if group else None,
        }
    summary["policy_by_distance"] = by_bucket

    # The blocks answer different questions — the fixed_goal sweep is the spawn
    # the agent trained from, the random block is one it never saw — so pooling
    # them would hide exactly the in- vs out-of-distribution gap.
    summary["policy_by_block"] = {
        block: {"pairs": len(group), **_rates(group, "policy")}
        for block in BLOCKS
        if (group := [r for r in records if r["block"] == block])
    }

    conf = {k: sum(r["policy"]["confusion"][k] for r in records) for k in ("tp", "fp", "fn", "tn")}
    z_fires = conf["tp"] + conf["fp"]
    at_target = conf["tp"] + conf["fn"]
    summary["confusion"] = {
        **conf,
        # Of the steps where the reward fired, how many were at the wrong pose?
        "false_positive_rate": float(conf["fp"] / z_fires) if z_fires else None,
        # Of the steps at the target pose, how many produced no reward?
        "false_negative_rate": float(conf["fn"] / at_target) if at_target else None,
    }

    if has_random:
        summary["random"] = _rates(records, "random")
    if has_oracle:
        oracles = [r["oracle"] for r in records]
        summary["oracle"] = {
            "arrived_rate": float(np.mean([o["arrived"] for o in oracles])),
            "mean_groups": float(np.mean([o["groups"] for o in oracles])),
            "full_match_rate": float(np.mean([o["full"] for o in oracles])),
            "mean_reward": float(np.mean([o["reward"] for o in oracles])),
            "floor_mean_groups": float(np.mean([o["floor_groups"] for o in oracles])),
            "floor_full_rate": float(np.mean([o["floor_full"] for o in oracles])),
            "mean_plan_len": float(np.mean([o["plan_len"] for o in oracles])),
        }
    return summary


# -------------------------------------------------------------- evaluation


def evaluate(agent, config, env_cfg, pairs, suite, device, max_steps, seed, skip_random, skip_oracle):
    """Run the whole suite against one already-loaded checkpoint, or None if it cannot be driven."""
    if "goal" not in str(env_cfg.task):
        print(f"  ! {env_cfg.task} is not a goal-grid task; the layout override will not apply")
    if agent.threshold_bins > 0:
        # Skipping rather than aborting: the natural invocation is a glob over a
        # whole family of runs, and one log_prob run in it must not cost the rest.
        print(
            f"  ! skipped: goal_type={config.goal_type} has threshold_bins>0, which the shared "
            "actor_policy cannot drive — see experiments/common/rollout.py"
        )
        return None
    size = int(env_cfg.env_size)
    spec = goals.make_goal_spec(config.model)
    S = agent.rssm._stoch
    budget = int(env_cfg.time_limit) if max_steps is None else int(max_steps)
    rng = np.random.default_rng(seed)

    records = []
    for i, pair in enumerate(pairs):
        records.append(
            run_pair(agent, spec, env_cfg, pair, device, budget, rng, suite["seed"], skip_random, skip_oracle)
        )
        if (i + 1) % 50 == 0:
            done = float(np.mean([r["policy"]["success"] for r in records]))
            print(f"    {i + 1}/{len(pairs)} pairs — success {100 * done:.1f}%")

    summary = summarize(records, S, not skip_random, not skip_oracle)
    meta = {
        "goal_type": config.get("goal_type"),
        "goal_sample": config.env.get("goal_sample"),
        "task": env_cfg.task,
        "env_size": size,
        "max_steps": budget,
    }
    line = (
        f"  success {100 * summary['policy']['success_rate']:.1f}% "
        f"(cell {100 * summary['policy']['reach_cell_rate']:.1f}%)"
    )
    blocks = summary["policy_by_block"]
    if len(blocks) > 1:
        line += " [" + ", ".join(f"{b} {100 * blocks[b]['success_rate']:.1f}%" for b in BLOCKS if b in blocks) + "]"
    if not skip_random:
        line += f" vs random {100 * summary['random']['success_rate']:.1f}%"
    if not skip_oracle:
        oracle = summary["oracle"]
        line += f" | oracle full {100 * oracle['full_match_rate']:.0f}% ({oracle['mean_groups']:.1f}/{S} groups)"
    print(line)
    return {"meta": meta, "summary": summary, "pairs": records}


# ------------------------------------------------------------------ plotting


def _run_label(name: str) -> str:
    """A short axis label for a run dir (the seed-parent when the leaf is a digit)."""
    return name.split("/")[-2] if name.split("/")[-1].isdigit() else name.split("/")[-1]


def plot(all_results, path, skip_random):
    """Two panels: success rate per run vs the random floor, and its decay with distance."""
    names = list(all_results)
    if not names:
        return
    summaries = [all_results[n]["summary"] for n in names]
    x = np.arange(len(names))
    labels = [_run_label(n) for n in names]

    fig, (ax_rate, ax_dist) = plt.subplots(1, 2, figsize=(13, 5))

    width = 0.4 if not skip_random else 0.6
    policy_x = x - (width / 2 if not skip_random else 0)
    ax_rate.bar(
        policy_x,
        [100 * s["policy"]["success_rate"] for s in summaries],
        width,
        label="trained policy",
    )
    if not skip_random:
        ax_rate.bar(
            x + width / 2, [100 * s["random"]["success_rate"] for s in summaries], width, label="random actions"
        )
    # When the suite has both blocks, mark them on the pooled bar: the sweep is
    # the spawn the agent trained from, the random block one it never saw.
    two_blocks = any(len(s["policy_by_block"]) > 1 for s in summaries)
    for block, marker in (("sweep", "^"), ("random", "v")) if two_blocks else ():
        rates = [s["policy_by_block"].get(block, {}).get("success_rate") for s in summaries]
        ax_rate.scatter(
            policy_x,
            [np.nan if r is None else 100 * r for r in rates],
            marker=marker,
            color="black",
            zorder=3,
            s=28,
            label=f"{block} block",
        )
    ax_rate.set_ylabel("pairs solved (%)")
    ax_rate.set_title("Success rate on the fixed suite\n(agent reaches the goal cell AND facing)")

    bucket_labels = [bucket_label(lo) for lo, _ in BUCKETS]
    for name, summary in zip(names, summaries):
        # An empty bucket is nan, not 0: the line breaks instead of claiming a
        # measured failure where nothing was measured.
        rates = [summary["policy_by_distance"][b]["success_rate"] for b in bucket_labels]
        ax_dist.plot(
            bucket_labels, [np.nan if r is None else 100 * r for r in rates], marker="o", label=_run_label(name)
        )
    ax_dist.set_xlabel("BFS distance from spawn to goal cell (actions)")
    ax_dist.set_ylabel("pairs solved (%)")
    ax_dist.set_title("Does success decay with distance?")

    ax_rate.set_xticks(x)
    ax_rate.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    for ax in (ax_rate, ax_dist):
        ax.legend(fontsize=8)
    fig.suptitle(f"Goal-conditioned success rate — fixed {summaries[0]['pairs']}-pair suite")
    fig.tight_layout()
    save_fig(fig, path)


def plot_map(all_results, suite, path):
    """Where in the room does the sweep succeed? One heatmap per run.

    Only the enumerated block can be drawn this way — it visits every interior
    cell in every facing, so each cell has a well-defined "fraction of the four
    facings solved". The aggregate number says 37%; this says whether that is a
    ring around the spawn, a gradient with distance, or a handful of lucky cells.
    """
    names = [n for n in all_results if any(r["block"] == "sweep" for r in all_results[n]["pairs"])]
    if not names:
        return
    size = suite["size"]
    spawn_pos, square = tuple(suite["spawn"][0]), tuple(suite["square"])

    cols = min(4, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.6 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        # grid[y][x]: nan where nothing was measured (the spawn cell, or a
        # truncated --pairs run), so it reads as blank instead of as failure.
        grid = np.full((size - 2, size - 2), np.nan)
        solved = {}
        for record in all_results[name]["pairs"]:
            if record["block"] != "sweep":
                continue
            cell = tuple(record["goal"][0])
            solved.setdefault(cell, []).append(record["policy"]["success"])
        for (cx, cy), hits in solved.items():
            grid[cy - 1, cx - 1] = 100 * float(np.mean(hits))

        im = ax.imshow(grid, origin="upper", cmap="viridis", vmin=0, vmax=100)
        ax.text(spawn_pos[0] - 1, spawn_pos[1] - 1, "S", ha="center", va="center", color="red", fontweight="bold")
        ax.text(square[0] - 1, square[1] - 1, "■", ha="center", va="center", color="lime")
        ax.set_xticks(range(size - 2), [str(i + 1) for i in range(size - 2)], fontsize=6)
        ax.set_yticks(range(size - 2), [str(i + 1) for i in range(size - 2)], fontsize=6)
        ax.set_title(_run_label(name), fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, label="% of facings solved")
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    fig.suptitle("Which goal cells are reachable from the training spawn (S)? — green square ■")
    fig.tight_layout()
    save_fig(fig, path)


# --------------------------------------------------------------- suite choice


def suite_for(env_cfg) -> str:
    """The suite a run's task calls for."""
    return "fixed_goal" if str(env_cfg.task).startswith("fixed-goal") else "random_goal"


def run_layout(env_cfg) -> dict:
    """The room a run was trained in: size, spawn pose, green square.

    `random_goal` configs carry no `goal_pos_*` — the square moves every episode —
    so `square` is None there, and only the enumerated suite (which needs a fixed
    one) ever reads it.
    """
    square = None
    if env_cfg.get("goal_pos_x") is not None:
        square = (int(env_cfg.goal_pos_x), int(env_cfg.goal_pos_y))
    return {
        "size": int(env_cfg.env_size),
        "spawn": ((int(env_cfg.agent_start_pos_x), int(env_cfg.agent_start_pos_y)), int(env_cfg.agent_start_dir)),
        "square": square,
    }


def layout_mismatch(layout, suite) -> str | None:
    """Why this run cannot be scored on `suite`, or None if it can.

    The room size always matters (the suite enumerates or samples cells in it);
    the spawn and square matter only for the enumerated suite, which was built
    against one specific layout and would otherwise be asking a different
    question of this run than of the first.
    """
    if layout["size"] != suite["size"]:
        return f"env_size={layout['size']} differs from the suite's {suite['size']}"
    if "sweep" not in suite:
        return None
    if layout["square"] is None:
        return "no fixed green square (goal_pos_*), which the enumerated suite is built against"
    spawn = [list(layout["spawn"][0]), layout["spawn"][1]]
    if spawn != suite["spawn"]:
        return f"spawn {spawn} differs from the suite's {suite['spawn']}"
    if list(layout["square"]) != suite["square"]:
        return f"green square {list(layout['square'])} differs from the suite's {suite['square']}"
    return None


def main():
    # conflict_handler: this script takes several logdirs, overriding the single
    # --logdir that add_common_args defines (--device / --seed come from there).
    parser = add_common_args(argparse.ArgumentParser(description=__doc__, conflict_handler="resolve"))
    parser.add_argument("--logdir", required=True, nargs="+", help="One or more run dirs to evaluate")
    parser.add_argument(
        "--suite",
        choices=["auto", "random_goal", "fixed_goal"],
        default="auto",
        help="Which benchmark suite to run (default: from the first run's task)",
    )
    parser.add_argument("--pairs", type=int, default=None, help="Evaluate only the first N pairs (smoke tests)")
    parser.add_argument("--max-steps", type=int, default=None, help="Step budget per pair (default: env.time_limit)")
    parser.add_argument("--skip-random", action="store_true", help="Skip the random-action baseline")
    parser.add_argument("--skip-oracle", action="store_true", help="Skip the BFS-oracle z-check")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    all_results = {}
    pairs, suite, json_path = None, None, None
    for logdir in args.logdir:
        print(f"\n=== {logdir} ===")
        agent, config, env_cfg = load_agent(logdir, args.device)
        layout = run_layout(env_cfg)
        if pairs is None:
            # The suite is a pure function of the room (size, and for fixed_goal
            # the spawn/square it enumerates against): build it once from the
            # first run and reuse it for every checkpoint — that shared identity
            # is exactly what makes the numbers comparable.
            name = args.suite if args.suite != "auto" else suite_for(env_cfg)
            pairs, suite = make_suite(name, layout["size"], spawn=layout["spawn"], square=layout["square"])
            if args.pairs:
                pairs = pairs[: args.pairs]
            suite["evaluated"] = len(pairs)
            json_path = outdir / f"success_rate_{name}.json"
            print(
                f"  suite: {name}, {len(pairs)} pairs, {layout['size']}x{layout['size']} room, "
                f"seed {suite['seed']}, min plan {suite['min_plan_len']}"
                + (f", sweep {suite['sweep']} from {suite['spawn']}" if "sweep" in suite else "")
            )
        elif (mismatch := layout_mismatch(layout, suite)) is not None:
            print(f"  ! skipped: {mismatch}")
            continue

        result = evaluate(
            agent,
            config,
            env_cfg,
            pairs,
            suite,
            args.device,
            args.max_steps,
            args.seed,
            args.skip_random,
            args.skip_oracle,
        )
        if result is None:
            continue
        all_results[logdir] = result
        # Written after every checkpoint so a long multi-model run is not lost.
        dump_json({"suite": suite, "runs": all_results}, json_path)

    if not all_results:
        print("\n  nothing evaluated")
        return
    print(f"\n  JSON: {json_path}")
    plot(all_results, outdir / f"success_rate_{suite['name']}.png", args.skip_random)
    if "sweep" in suite:
        plot_map(all_results, suite, outdir / f"success_rate_{suite['name']}_map.png")


if __name__ == "__main__":
    main()
