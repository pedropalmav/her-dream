"""Do goal-conditioned agents actually REACH the state their goal-z stands for?

Every score-based verdict so far is measured in z-space, which is exactly what
cannot be trusted: `posttrain_randomstart_goalimag_rowbyrow` reaches score ≈ -290
while the videos show the agent walking AWAY from the green square. The score says
"the z matches"; the task says "the agent is nowhere near the goal". To tell those
apart you need a goal whose ground-truth state you know.

`goal_sample="image"` gives exactly that: the env renders a synthetic state — the
green square at the CURRENT episode's position, the agent at a chosen cell — and
the frozen WM encodes it into z (`Dreamer.encode_observation`). Because we render
it, we know the target state `(x, y, dir)` behind the goal z, so "did it arrive?"
becomes a ground-truth question rather than a z-space one.

Two target modes (`--target`):
  square — the target state is the agent standing ON the green square: the real task.
  random — the target state is the agent at a random interior cell: reaching an
           arbitrary state, which is what goal-conditioning actually promises.
In both, the green square sits at its true position for the episode, so the goal
is consistent with the live episode by construction (unlike buffer-sampled goals).

Three layers, each answering a different question:

  1. ORACLE — is the goal-z even attainable? A BFS shortest path drives the agent
     to the target state; at arrival we compare the posterior z against the goal z.
     If the ORACLE can't reproduce the goal z, no policy can, and the run's score
     is measuring something other than the task. Control: the sampling floor (two
     samples from the same posterior logits) — the irreducible Gumbel noise, an
     upper bound on any exact-match rate. Note the goal z is encoded as if it were
     step 0 (`is_first=True`, no history) while the agent's z at arrival carries a
     history, so this layer also prices in that context mismatch.

  2. POLICY — does the trained actor arrive? It acts conditioned on the goal z;
     we record ground-truth arrival, steps taken and closest approach, against a
     random-action baseline (the floor a policy must beat to have learned anything).

  3. CONFUSION — does the z mean what you think? Per step, cross "the reward fires"
     (full z match) with "the agent is at the target pose". False positives (reward
     fires, wrong cell) say the z does not identify position; false negatives
     (arrived, no reward) say there was no signal to learn from. This is the direct
     test of the random-goal position-in-z hypothesis.

Usage — the six frozen-WM runs of the current investigation, on the GPU box:

    uv run python3 experiments/goal_observation_eval.py \
        --logdir logdir/post_train_from_wm_only/her_buffer_goals \
                 logdir/post_train_from_wm_only/normal_buffer_goals \
                 logdir/random_goal/posttrain_frozenwm_herbuf_goalbuf/01 \
                 logdir/random_goal/posttrain_frozenwm_normalbuf_goalbuf/01 \
                 logdir/random_goal/posttrain_randomstart_goalimag_rowbyrow/02 \
                 logdir/random_goal/posttrain_randomstart_goalimag_rowbyrow/03 \
        --episodes 50 --target both --device cuda

Those six are three contrasts. The first two (fixed_goal, `full` + buffer goals) are
the positive control: post-training works there, score ≈ -430/-500. The next two are
that same recipe on random_goal, where it never leaves the -1001 floor. The last two
are the `row_by_row` + imagination recipe that does learn, plateauing at ≈ -290 while
the videos show the agent ignoring the green square — the contradiction this script
exists to settle.

NOT `.../posttrain_randomstart_goalimag_rowbyrow/01`, despite the name: that run
predates the argmax reward fix and trained against a dead gradient (its `train/rew`
has 4 distinct values, all ≈ -1). The third healthy seed is
`posttrain_randomstart_goalimag_rowbyrow_fixedrew/01`, which has logs but no
`latest.pt` synced locally — add it if you fetch its checkpoint.

Since every run above freezes its WM, layer 1 (the oracle) really measures the three
underlying WMs (`wm_only_random_mission/01`, `wm_only_randomgoal/01`,
`wm_only_randomstart/01`), not six independent things; runs sharing a WM should give
the same layer-1 numbers, which doubles as a sanity check on the eval itself.

Local smoke test (CPU, no CUDA needed, ~1 min):

    uv run python3 experiments/goal_observation_eval.py \
        --logdir logdir/random_goal/posttrain_frozenwm_normalbuf_goalbuf/01 \
        --episodes 2 --target square --max-steps 40 --device cpu
"""

import argparse
import pathlib
import sys
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from minigrid.core.constants import DIR_TO_VEC  # noqa: E402

from experiments.common import (  # noqa: E402
    actor_policy,
    add_common_args,
    agent_pose,
    dump_json,
    encode_goal_image,
    fixed_goal_cfg,
    groups_matched,
    load_agent,
    posterior_rollout,
    random_policy,
    reward_of,
    save_fig,
    scripted_policy,
    unwrap_env,
)
from her_dream import goals  # noqa: E402
from her_dream.envs import make_env  # noqa: E402

# MiniGrid action indices (minigrid.core.actions.Actions).
TURN_LEFT, TURN_RIGHT, FORWARD = 0, 1, 2

OUT = pathlib.Path(__file__).resolve().parent / "out"


# ---------------------------------------------------------------- BFS oracle


def _walkable(size: int, x: int, y: int) -> bool:
    """Can the agent stand on cell (x, y)? The goal-grid rooms are a bordered box
    with a single overlappable goal tile and no interior walls, so every interior
    cell is walkable — no grid read (hence no env reset) needed."""
    return 1 <= x <= size - 2 and 1 <= y <= size - 2


def _successors(size: int, state):
    """(action, next_state) for the three navigation actions from `state`."""
    x, y, d = state
    yield TURN_LEFT, (x, y, (d - 1) % 4)
    yield TURN_RIGHT, (x, y, (d + 1) % 4)
    dx, dy = DIR_TO_VEC[d]
    nx, ny = x + int(dx), y + int(dy)
    if _walkable(size, nx, ny):
        yield FORWARD, (nx, ny, d)


def bfs_plan(size, start_pos, start_dir, target_pos, target_dir):
    """Shortest action sequence from the start pose to the target pose.

    BFS over `(x, y, dir)` rather than over cells: the target direction is part
    of the goal state (the WM encoder sees `direction`, so the goal z fixes it),
    and turning costs steps. `target_dir=None` accepts any facing. Returns the
    list of action indices, or None if unreachable.
    """
    start = (int(start_pos[0]), int(start_pos[1]), int(start_dir))
    tx, ty = int(target_pos[0]), int(target_pos[1])

    def is_target(s):
        return (s[0], s[1]) == (tx, ty) and (target_dir is None or s[2] == int(target_dir))

    parent = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if is_target(state):
            plan = []
            while parent[state] is not None:
                state, action = parent[state]
                plan.append(action)
            return plan[::-1]
        for action, nxt in _successors(size, state):
            if nxt not in parent:
                parent[nxt] = (state, action)
                queue.append(nxt)
    return None


# ------------------------------------------------------------ goal machinery


def _sample_cell(size: int, rng) -> tuple:
    """A uniformly random interior cell `(x, y)`."""
    interior = np.arange(1, size - 1)
    return int(rng.choice(interior)), int(rng.choice(interior))


def _episode_layout(mode: str, env_cfg, rng):
    """Read this episode's layout off the config and pick the goal-image target.

    `evaluate` has already sampled the spawn and green-square positions into
    `env_cfg` and built the env from it, so here we just read them back and choose
    the target the goal image depicts: the green square (`square` mode, the real
    task) or an arbitrary interior cell (`random` mode). Returns `(size, spawn_pos,
    spawn_dir, square, target_pos, target_dir)`.
    """
    size = int(env_cfg.env_size)
    spawn_pos = (int(env_cfg.agent_start_pos_x), int(env_cfg.agent_start_pos_y))
    spawn_dir = int(env_cfg.agent_start_dir)
    square = (int(env_cfg.goal_pos_x), int(env_cfg.goal_pos_y))

    if mode == "square":
        target_pos = square
    elif mode == "random":
        target_pos = _sample_cell(size, rng)
    else:
        raise ValueError(f"unknown target mode: {mode}")
    return size, spawn_pos, spawn_dir, square, target_pos, int(rng.integers(4))


# --------------------------------------------------------------- the layers


def _oracle(agent, spec, env, seed, goal, target_pos, target_dir, plan, device):
    """Layer 1: drive the BFS plan for real; measure the z at arrival vs the goal z."""
    last = list(posterior_rollout(agent, env, scripted_policy(plan), device, max_steps=len(plan), seed=seed))[-1]
    base = unwrap_env(env)
    pose = agent_pose(base)
    S = goal.shape[1]

    matched = groups_matched(last.stoch, goal)
    # Sampling floor: a second draw from the SAME posterior logits. Any exact-match
    # rate above this is signal; at or below it, the metric is measuring Gumbel noise.
    resample = agent.rssm.get_dist(last.logit).rsample()
    return {
        "plan_len": len(plan),
        "arrived": pose == (target_pos, target_dir),
        "groups": matched,
        "full": matched == S,
        "reward": reward_of(agent, spec, last.stoch, last.logit, goal),
        "floor_groups": groups_matched(last.stoch, resample),
        "floor_full": groups_matched(last.stoch, resample) == S,
    }


def _drive(agent, spec, env, seed, policy, target_pos, target_dir, goal, device, max_steps):
    """Layers 2+3: run `policy` on the goal z, recording ground truth and z match."""
    base = unwrap_env(env)
    S = goal.shape[1]
    steps_to_pos, steps_to_pose = None, None
    min_dist, groups_seen, rewards = None, [], []
    conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for step in posterior_rollout(agent, env, policy, device, max_steps=max_steps, seed=seed):
        pos, direction = agent_pose(base)
        dist = abs(pos[0] - target_pos[0]) + abs(pos[1] - target_pos[1])
        min_dist = dist if min_dist is None else min(min_dist, dist)
        if pos == target_pos and steps_to_pos is None:
            steps_to_pos = step.t
        if (pos, direction) == (target_pos, target_dir) and steps_to_pose is None:
            steps_to_pose = step.t

        matched = groups_matched(step.stoch, goal)
        groups_seen.append(matched)
        rewards.append(reward_of(agent, spec, step.stoch, step.logit, goal))
        z_match = matched == S
        state_match = (pos, direction) == (target_pos, target_dir)
        conf["tp" if (z_match and state_match) else "fp" if z_match else "fn" if state_match else "tn"] += 1

    return {
        "reached_pos": steps_to_pos is not None,
        "reached_pose": steps_to_pose is not None,
        "steps_to_pos": steps_to_pos,
        "steps_to_pose": steps_to_pose,
        "min_dist": min_dist,
        "best_groups": max(groups_seen),
        "mean_reward": float(np.mean(rewards)),
        "confusion": conf,
    }


def run_episode(agent, spec, env, task, env_cfg, mode, seed, rng, max_steps, device):
    """One episode: read the chosen layout off the config, build the goal, run the layers."""
    size, spawn_pos, spawn_dir, square, target_pos, target_dir = _episode_layout(mode, env_cfg, rng)

    plan = bfs_plan(size, spawn_pos, spawn_dir, target_pos, target_dir)
    if plan is None:
        return None
    goal = encode_goal_image(agent, env, target_pos, target_dir)

    return {
        "seed": seed,
        "task": task,
        "spawn": [list(spawn_pos), spawn_dir],
        "target": [list(target_pos), target_dir],
        "green_square": [int(v) for v in square],
        "oracle": _oracle(agent, spec, env, seed, goal, target_pos, target_dir, plan, device),
        "policy": _drive(
            agent, spec, env, seed, actor_policy(agent, goal), target_pos, target_dir, goal, device, max_steps
        ),
        "random": _drive(agent, spec, env, seed, random_policy(rng), target_pos, target_dir, goal, device, max_steps),
    }


# ------------------------------------------------------------- aggregation


def summarize(episodes, S):
    """Aggregate per-episode records into the headline numbers."""
    conf = {k: sum(e["policy"]["confusion"][k] for e in episodes) for k in ("tp", "fp", "fn", "tn")}
    z_fires = conf["tp"] + conf["fp"]
    at_target = conf["tp"] + conf["fn"]
    return {
        "episodes": len(episodes),
        "oracle": {
            "arrived_rate": float(np.mean([e["oracle"]["arrived"] for e in episodes])),
            "mean_groups": float(np.mean([e["oracle"]["groups"] for e in episodes])),
            "groups_total": S,
            "full_match_rate": float(np.mean([e["oracle"]["full"] for e in episodes])),
            "mean_reward": float(np.mean([e["oracle"]["reward"] for e in episodes])),
            "floor_mean_groups": float(np.mean([e["oracle"]["floor_groups"] for e in episodes])),
            "floor_full_rate": float(np.mean([e["oracle"]["floor_full"] for e in episodes])),
            "mean_plan_len": float(np.mean([e["oracle"]["plan_len"] for e in episodes])),
        },
        "policy": {
            "reach_pos_rate": float(np.mean([e["policy"]["reached_pos"] for e in episodes])),
            "reach_pose_rate": float(np.mean([e["policy"]["reached_pose"] for e in episodes])),
            "mean_min_dist": float(np.mean([e["policy"]["min_dist"] for e in episodes])),
            "mean_best_groups": float(np.mean([e["policy"]["best_groups"] for e in episodes])),
            "mean_reward": float(np.mean([e["policy"]["mean_reward"] for e in episodes])),
        },
        "random": {
            "reach_pos_rate": float(np.mean([e["random"]["reached_pos"] for e in episodes])),
            "reach_pose_rate": float(np.mean([e["random"]["reached_pose"] for e in episodes])),
            "mean_min_dist": float(np.mean([e["random"]["min_dist"] for e in episodes])),
        },
        "confusion": {
            **conf,
            # Of the steps where the reward fired, how many were at the wrong pose?
            "false_positive_rate": float(conf["fp"] / z_fires) if z_fires else None,
            # Of the steps at the target pose, how many produced no reward?
            "false_negative_rate": float(conf["fn"] / at_target) if at_target else None,
        },
    }


def evaluate(logdir, modes, episodes, max_steps, device, seed):
    """Run every mode for one checkpoint."""
    agent, config, env_cfg = load_agent(logdir, device)
    # 1. Was the run fixed-goal or random-goal? The eval always drives a
    #    controllable fixed-goal GoalGrid (it can dictate both the spawn and the green
    #    square), so remember the real task and reproduce the run's distribution by
    #    how we sample the layout below.
    task = env_cfg.task
    random_goal = "fixed" not in task
    size = int(env_cfg.env_size)
    # Same sub-config Dreamer builds its own spec from, so the eval routes states
    # and goals exactly as the run did while training.
    spec = goals.make_goal_spec(config.model)
    S = agent.rssm._stoch
    results = {}
    for mode in modes:
        rng = np.random.default_rng(seed)
        records = []
        for i in range(episodes):
            # 2.1 Sample the layout: the agent (pos, dir) always, plus the green
            #     square for random-goal runs (fixed-goal keeps its own goal_pos).
            spawn_pos = _sample_cell(size, rng)
            spawn_dir = int(rng.integers(4))
            square = _sample_cell(size, rng) if random_goal else None
            # 2.2 Build a fixed-goal env carrying this episode's layout.
            episode_cfg = fixed_goal_cfg(
                env_cfg, goal_pos=square, agent_start_pos=spawn_pos, agent_start_dir=spawn_dir
            )
            env = make_env(episode_cfg, 0)
            # 2.3 run_episode reads the layout back off episode_cfg.
            record = run_episode(agent, spec, env, task, episode_cfg, mode, seed + i, rng, max_steps, device)
            if record is not None:
                records.append(record)
        if not records:
            continue
        results[mode] = {"summary": summarize(records, S), "episodes": records}
        s = results[mode]["summary"]
        print(
            f"  [{mode}] oracle {s['oracle']['mean_groups']:.1f}/{S} groups "
            f"(full {100 * s['oracle']['full_match_rate']:.0f}%, floor "
            f"{s['oracle']['floor_mean_groups']:.1f}/{S}) | policy reach "
            f"{100 * s['policy']['reach_pos_rate']:.0f}% vs random "
            f"{100 * s['random']['reach_pos_rate']:.0f}%"
        )
    return results, {"goal_type": config.goal_type, "goal_sample": config.env.goal_sample, "task": task}


# ------------------------------------------------------------------ plotting


def _env_kind(meta) -> str:
    """Which goal-grid env a run trained on: 'fixed_goal' or 'random_goal'."""
    return "fixed_goal" if "fixed" in meta["task"] else "random_goal"


def _run_label(name: str) -> str:
    """A short axis label for a run dir (the seed-parent when the leaf is a digit)."""
    return name.split("/")[-2] if name.split("/")[-1].isdigit() else name.split("/")[-1]


def _panel_oracle(ax, summaries, x, S):
    """Layer 1: how close the z at the target is to the goal z, vs the sampling floor."""
    ax.bar(x - 0.2, [s["oracle"]["mean_groups"] for s in summaries], 0.4, label="oracle at target")
    ax.bar(x + 0.2, [s["oracle"]["floor_mean_groups"] for s in summaries], 0.4, label="sampling floor")
    ax.axhline(S, color="gray", ls="--", lw=1, label=f"{S} = full match")
    ax.set_ylabel(f"groups matched / {S}")
    ax.set_title("Is the goal-z attainable?\n(BFS oracle standing on the target)")


def _panel_policy(ax, summaries, x):
    """Layer 2: how often the trained policy reaches the target cell, vs random actions."""
    ax.bar(x - 0.2, [100 * s["policy"]["reach_pos_rate"] for s in summaries], 0.4, label="trained policy")
    ax.bar(x + 0.2, [100 * s["random"]["reach_pos_rate"] for s in summaries], 0.4, label="random actions")
    ax.set_ylabel("episodes reaching the target cell (%)")
    ax.set_title("Does the policy arrive?\n(ground truth, vs random baseline)")


def plot(all_results, mode, path):
    """Two rows — fixed_goal (top) vs random_goal (bottom) — each with two panels:
    is the goal-z attainable, and does the policy arrive."""
    names = [n for n in all_results if mode in all_results[n]["results"]]
    if not names:
        return

    rows = [
        ("fixed_goal", "fixed goal (green square fixed)"),
        ("random_goal", "random goal (green square moves)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), squeeze=False)

    for (kind, kind_label), (ax_oracle, ax_policy) in zip(rows, axes):
        group = [n for n in names if _env_kind(all_results[n]["meta"]) == kind]
        if not group:
            for ax in (ax_oracle, ax_policy):
                ax.text(0.5, 0.5, f"no {kind} runs", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            continue

        summaries = [all_results[n]["results"][mode]["summary"] for n in group]
        x = np.arange(len(group))
        labels = [_run_label(n) for n in group]
        S = summaries[0]["oracle"]["groups_total"]

        _panel_oracle(ax_oracle, summaries, x, S)
        _panel_policy(ax_policy, summaries, x)
        for ax in (ax_oracle, ax_policy):
            ax.set_ylabel(f"{kind_label}\n{ax.get_ylabel()}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
            ax.legend(fontsize=8)

    fig.suptitle(f"Observation-goal evaluation — target: {mode}")
    fig.tight_layout()
    save_fig(fig, path)


def main():
    # conflict_handler: this script takes several logdirs, overriding the single
    # --logdir that add_common_args defines (--device / --seed come from there).
    parser = add_common_args(argparse.ArgumentParser(description=__doc__, conflict_handler="resolve"))
    parser.add_argument("--logdir", required=True, nargs="+", help="One or more run dirs to evaluate")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--target", default="both", choices=["square", "random", "both"])
    parser.add_argument("--max-steps", type=int, default=200, help="Cap on policy/random rollout length")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    modes = ["square", "random"] if args.target == "both" else [args.target]
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    all_results = {}
    for logdir in args.logdir:
        print(f"\n=== {logdir} ===")
        results, meta = evaluate(logdir, modes, args.episodes, args.max_steps, args.device, args.seed)
        all_results[logdir] = {"meta": meta, "results": results}

    dump_json(all_results, outdir / "goal_observation_eval.json")
    print(f"\n  JSON: {outdir / 'goal_observation_eval.json'}")
    for mode in modes:
        plot(all_results, mode, outdir / f"goal_observation_eval_{mode}.png")


if __name__ == "__main__":
    main()
