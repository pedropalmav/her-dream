"""PEG-style exploratory goal selection (Hu et al., ICLR 2023), adapted to z space.

PEG picks the training goal whose imagined pursuit leaves the achiever where the
explorer expects the most to discover, then runs a Go-Explore episode: the
achiever chases the goal for `go_steps`, the explorer takes over for the rest.

The original searches observation space with MPPI/CEM. Here candidates are drawn
at random in the RSSM's discrete latent space instead (`random_candidates`), a
single batch with no iterative refit. The rollout and scoring live on `Dreamer`
(`Dreamer.peg_goal` / `Dreamer._peg_score`); this module holds the pieces that do
not need the networks.
"""

import torch
import torch.nn.functional as F


def go_steps(go_ratio, time_limit, action_repeat):
    """Agent steps the achiever gets before the explorer takes over.

    `time_limit` is in env steps, like the `TimeLimit` wrapper's input, so the
    episode lasts `time_limit // action_repeat` agent steps.
    """
    if not 0.0 <= float(go_ratio) <= 1.0:
        raise ValueError(f"peg.go_ratio must be in [0, 1], got {go_ratio}.")
    return round(float(go_ratio) * (int(time_limit) // int(action_repeat)))


def explore_every_ep(value, goal_sample):
    """Resolve LEXA's pure-explorer episode cadence; `None` means the mode's default.

    PEG has no explorer-only episodes — every episode is Go-Explore, which
    already gives the explorer its share — so the default is 0 under
    goal_sample="peg" and LEXA's 2 otherwise. An explicit value always wins.
    """
    if value is not None:
        return int(value)
    return 0 if goal_sample == "peg" else 2


def planning_horizon(peg_cfg):
    """Imagined steps per candidate: `peg.horizon`, or the go-phase length if null.

    Matching the go-phase is PEG's choice: the plan scores the state the achiever
    would actually hand over to the explorer.
    """
    if peg_cfg.horizon is not None:
        return int(peg_cfg.horizon)
    return go_steps(peg_cfg.go_ratio, peg_cfg.time_limit, peg_cfg.action_repeat)


def random_candidates(spec, n, stoch, discrete, device):
    """Draw `n` random goals in the layout the achiever is conditioned on.

    One-hot goal types (`goal_repr` sample/argmax) get an independent uniform
    category per group; real-valued ones get standard-normal logits (softmaxed
    for `prob`), mirroring the `GoalConditioned` wrapper's random goals.

    Returns:
        (n, K) for `scope="first_row"`, else (n, S, K), float32 on `device`.
    """
    rows = 1 if spec.scope == "first_row" else stoch
    if spec.goal_repr in ("logit", "prob"):
        goal = torch.randn(n, rows, discrete, device=device)
        if spec.goal_repr == "prob":
            goal = F.softmax(goal, dim=-1)
    else:
        idx = torch.randint(0, discrete, (n, rows), device=device)
        goal = F.one_hot(idx, discrete).to(torch.float32)
    return goal[:, 0] if spec.scope == "first_row" else goal
