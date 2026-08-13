"""The frozen `(initial_state, goal_state)` benchmark suites.

A *pair* is one ground-truth navigation task on a goal-grid room: spawn the
agent at a known pose, ask it to reach a known pose. Success rates measured on a
suite are comparable across models and across reruns because a suite is a pure
function of its parameters — no env, no torch, no checkpoint, no wall clock.

Two suites, one per env, because the question each env poses is different:

  * **`random_goal`** (`generate_pairs`) — the layout moves every episode, so the
    tasks are *sampled*: random spawn pose, random goal pose, and a green square
    drawn **independently of the goal**, so the task is "reach the state this
    goal-z stands for", not the built-in reach-the-square task the env rewards.
    A pair is kept only when the goal is at least `min_plan_len` actions away, so
    an agent cannot score by standing still or dithering next to its spawn.
  * **`fixed_goal`** (`generate_fixed_goal_pairs`) — the layout never moves, so
    the tasks can be *enumerated*: from the one spawn the agent trained at, every
    interior cell in every facing (the **sweep** block, 252 poses on a 10x10
    room). That is a complete map of the goal-conditioning rather than a sample
    of it. The remaining pairs are drawn at random as above, to also measure
    spawns the agent never trained from.

`PAIRS_SEED` / `FIXED_SEED` are frozen on purpose: changing one silently
invalidates every success rate already recorded against that suite.
"""

import numpy as np

from .grid import bfs_plan, sample_cell

PAIRS_SEED = 20260811
FIXED_SEED = 20260812
N_PAIRS = 1000
MIN_PLAN_LEN = 4


def _record(index, spawn_pos, spawn_dir, goal_pos, goal_dir, square, plan_len, block):
    """One benchmark pair, in the schema both suites share."""
    return {
        "index": index,
        "spawn": [list(spawn_pos), int(spawn_dir)],
        "goal": [list(goal_pos), int(goal_dir)],
        "square": list(square),
        "plan_len": int(plan_len),
        "on_square": tuple(goal_pos) == tuple(square),
        "block": block,
    }


def generate_pairs(size: int, n: int = N_PAIRS, seed: int = PAIRS_SEED, min_plan_len: int = MIN_PLAN_LEN) -> list:
    """The `n` benchmark pairs for a `size`x`size` `random_goal` room, deterministically.

    Each record is::

        {"index": i,
         "spawn": [[x, y], dir],   # initial state: cell + facing
         "goal":  [[x, y], dir],   # goal state: cell + facing (the goal image's pose)
         "square": [x, y],         # green square, drawn independently of `goal`
         "plan_len": int,          # cell-only BFS distance from the spawn pose
         "on_square": bool,        # goal cell happens to be the green square
         "block": "random"}        # how the pair was drawn (see the module docstring)

    `plan_len` counts *actions* (turning costs a step) from the spawn pose to the
    goal cell with any facing, which is the distance the `min_plan_len` filter
    applies. The oracle plans the full pose separately and may be longer.

    Rejected draws re-draw the whole tuple — spawn, goal and square together —
    so the random stream stays a simple function of the accepted count.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    while len(pairs) < n:
        spawn_pos = sample_cell(size, rng)
        spawn_dir = int(rng.integers(4))
        goal_pos = sample_cell(size, rng)
        goal_dir = int(rng.integers(4))
        square = sample_cell(size, rng)

        plan = bfs_plan(size, spawn_pos, spawn_dir, goal_pos, target_dir=None)
        if plan is None or len(plan) < min_plan_len:
            continue

        pairs.append(_record(len(pairs), spawn_pos, spawn_dir, goal_pos, goal_dir, square, len(plan), "random"))
    return pairs


def sweep_pairs(size: int, spawn, square, start_index: int = 0) -> list:
    """Every interior cell in every facing, as goals from the one `spawn` pose.

    `spawn` is `((x, y), dir)`, `square` is `(x, y)`. The spawn **cell** is
    skipped entirely — all four of its poses — which is both what makes the count
    `(interior - 1) * 4` (252 on a 10x10 room) and what guarantees no pair is
    already solved at `t=0`.

    No `min_plan_len` filter: enumerating means enumerating, so the easy
    near-spawn poses are included and show up in the distance buckets.
    """
    (spawn_pos, spawn_dir) = (tuple(spawn[0]), int(spawn[1]))
    pairs = []
    for x in range(1, size - 1):
        for y in range(1, size - 1):
            if (x, y) == spawn_pos:
                continue
            # Cell-only distance: the same for all four facings of this cell.
            plan = bfs_plan(size, spawn_pos, spawn_dir, (x, y), target_dir=None)
            for goal_dir in range(4):
                index = start_index + len(pairs)
                pairs.append(_record(index, spawn_pos, spawn_dir, (x, y), goal_dir, square, len(plan), "sweep"))
    return pairs


def generate_fixed_goal_pairs(
    size: int,
    spawn,
    square,
    n: int = N_PAIRS,
    seed: int = FIXED_SEED,
    min_plan_len: int = MIN_PLAN_LEN,
) -> list:
    """The `fixed_goal` suite: the full sweep from `spawn`, then random pairs to `n`.

    The green square is **pinned** to `square` in every pair rather than drawn:
    in `fixed_goal` it cannot move, so goal-vs-square independence is automatic
    (the sweep visits every cell regardless of where the square sits) instead of
    something the sampler has to enforce.

    The random block draws spawn and goal poses freely — those are spawns the
    agent never trained from, so it measures generalisation and is reported
    separately from the sweep.
    """
    pairs = sweep_pairs(size, spawn, square)
    seen = {(tuple(p["spawn"][0]), p["spawn"][1], tuple(p["goal"][0]), p["goal"][1]) for p in pairs}

    rng = np.random.default_rng(seed)
    while len(pairs) < n:
        spawn_pos = sample_cell(size, rng)
        spawn_dir = int(rng.integers(4))
        goal_pos = sample_cell(size, rng)
        goal_dir = int(rng.integers(4))

        key = (spawn_pos, spawn_dir, goal_pos, goal_dir)
        if key in seen:
            continue  # keep the n tasks distinct: no sweep pair re-run by chance
        plan = bfs_plan(size, spawn_pos, spawn_dir, goal_pos, target_dir=None)
        if plan is None or len(plan) < min_plan_len:
            continue

        seen.add(key)
        pairs.append(_record(len(pairs), spawn_pos, spawn_dir, goal_pos, goal_dir, square, len(plan), "random"))
    return pairs


def suite_meta(size: int, n: int = N_PAIRS, seed: int = PAIRS_SEED, min_plan_len: int = MIN_PLAN_LEN) -> dict:
    """The identity of a suite, to be stored alongside any result it produced."""
    return {"size": int(size), "n": int(n), "seed": int(seed), "min_plan_len": int(min_plan_len)}


def make_suite(name: str, size: int, *, spawn=None, square=None, n: int = N_PAIRS) -> tuple:
    """Build the named suite, returning `(pairs, meta)`.

    `meta` describes the suite completely enough that a result file says which
    questions produced it — including, for `fixed_goal`, the layout the sweep was
    enumerated against, so a caller can refuse to score two runs whose rooms
    differ. `spawn` (`((x, y), dir)`) and `square` (`(x, y)`) are read off the
    evaluated run's env config and are required for `fixed_goal` only.
    """
    if name == "random_goal":
        return generate_pairs(size, n=n), {**suite_meta(size, n=n), "name": name}
    if name == "fixed_goal":
        if spawn is None or square is None:
            raise ValueError("the fixed_goal suite needs the run's spawn pose and green-square position")
        pairs = generate_fixed_goal_pairs(size, spawn, square, n=n)
        meta = {
            **suite_meta(size, n=n, seed=FIXED_SEED),
            "name": name,
            "spawn": [list(spawn[0]), int(spawn[1])],
            "square": list(square),
            "sweep": sum(p["block"] == "sweep" for p in pairs),
        }
        return pairs, meta
    raise ValueError(f"unknown suite {name!r} (expected 'random_goal' or 'fixed_goal')")
