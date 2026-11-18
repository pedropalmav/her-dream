"""PEG-style exploratory goal selection (Hu et al., ICLR 2023), adapted to z space.

PEG picks the training goal whose imagined pursuit leaves the achiever where the
explorer expects the most to discover, then runs a Go-Explore episode: the
achiever chases the goal for `go_steps`, the explorer takes over for the rest.

The original searches observation space with MPPI/CEM. Here the search runs in
the RSSM's latent space: a single batch of random z (`random_candidates`) or of
replay-buffer z (drawn by the trainer), or PEG's MPPI/CEM over a Gaussian on the z *logits*, each sampled logit turned
into a goal by a `OneHotDist` draw (`refit` is the distribution update). The
rollout and scoring live on `Dreamer` (`Dreamer.peg_goal` / `Dreamer._peg_score`);
this module holds the pieces that do not need the networks.
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


SEARCHES = ("random", "buffer", "mppi", "cem")
# The searches that iterate an MPPI/CEM refit (`refit`); the others score one batch.
OPTIMIZED_SEARCHES = ("mppi", "cem")


def refit(method, samples, scores, mppi_gamma, elite_ratio):
    """One MPPI or CEM update of the per-env Gaussian over goal logits, as in PEG.

    Both refit the mean and std from scratch each iteration (no momentum, no
    std floor beyond CEM's 1e-6), exactly like PEG's `SubgoalPlanner`:

    - "mppi": weights `softmax(mppi_gamma * score)` over the candidates; the
      mean and std are the weighted mean and weighted (population) std.
    - "cem": the top `int(C * elite_ratio)` candidates (at least one); the mean
      and std are their mean and `sqrt(var + 1e-6)`.

    Args:
        samples: (N, C, *D) sampled logits.
        scores: (N, C) candidate scores, higher is better.

    Returns:
        (mean, std), each (N, *D).
    """
    N, C = scores.shape
    flat = samples.reshape(N, C, -1)
    if method == "mppi":
        weights = torch.softmax(mppi_gamma * scores, dim=-1).unsqueeze(-1)
        mean = (weights * flat).sum(dim=1)
        std = (weights * (flat - mean.unsqueeze(1)).square()).sum(dim=1).sqrt()
    elif method == "cem":
        n_elite = max(1, int(C * elite_ratio))
        idx = scores.topk(n_elite, dim=-1).indices
        elites = flat.gather(1, idx.unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
        mean = elites.mean(dim=1)
        std = (elites.var(dim=1, unbiased=False) + 1e-6).sqrt()
    else:
        raise ValueError(f"Unknown PEG refit method {method!r}; expected 'mppi' or 'cem'.")
    shape = (N, *samples.shape[2:])
    return mean.reshape(shape), std.reshape(shape)
