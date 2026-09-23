"""Grid geometry for the goal-grid rooms: walkability, pose successors, BFS.

The goal-grid envs (`fixed_goal` / `random_goal`) are bordered boxes with a
single overlappable goal tile and no interior walls, so every one of these
helpers is a pure function of the room `size` — no env instance, no reset, no
checkpoint. That is what lets the benchmark suite (`pairs.py`) be generated
offline and the oracle plan be computed before the episode runs.
"""

from collections import deque

import numpy as np
from minigrid.core.constants import DIR_TO_VEC

# MiniGrid action indices (minigrid.core.actions.Actions).
TURN_LEFT, TURN_RIGHT, FORWARD = 0, 1, 2


def walkable(size: int, x: int, y: int) -> bool:
    """Can the agent stand on cell (x, y)? Every interior cell of the bordered
    box is walkable (the goal tile can be overlapped), so this is just bounds."""
    return 1 <= x <= size - 2 and 1 <= y <= size - 2


def successors(size: int, state):
    """(action, next_state) for the three navigation actions from `state`."""
    x, y, d = state
    yield TURN_LEFT, (x, y, (d - 1) % 4)
    yield TURN_RIGHT, (x, y, (d + 1) % 4)
    dx, dy = DIR_TO_VEC[d]
    nx, ny = x + int(dx), y + int(dy)
    if walkable(size, nx, ny):
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
        for action, nxt in successors(size, state):
            if nxt not in parent:
                parent[nxt] = (state, action)
                queue.append(nxt)
    return None


def sample_cell(size: int, rng) -> tuple:
    """A uniformly random interior cell `(x, y)`."""
    interior = np.arange(1, size - 1)
    return int(rng.choice(interior)), int(rng.choice(interior))
