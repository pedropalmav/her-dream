"""
Linear probing of the world-model posterior `z`.

Research question
─────────────────
What interpretable concepts about the agent's state are *linearly decodable*
from the RSSM posterior latent? We train simple linear probes (logistic /
ridge) to predict known ground-truth concepts from the latent and read the
test accuracy as "how linearly decodable the concept is".

Environment
───────────
Designed for `fixed_goal`, where the green goal square is constant at (8,1),
so all latent variation reflects the agent's own state.

Caveat that shapes the reading of results: in the `fixed_goal` runs the agent
`direction` is an explicit MLP input to the WM encoder, whereas the agent x/y
position is only available through the image. So:
  • direction  → POSITIVE CONTROL: the probe should be near-perfect; if it is
    not, the pipeline is broken.
  • x / y      → the genuinely interesting targets ("does z recover position
    from pixels?").
  • goal x / y → constant in fixed_goal (no variance) → not probed here.

Representations
───────────────
For every concept we fit the probe on three latents to learn *where* the
concept lives:
  • stoch  z      flatten (S,K) → S*K   (the goal-matching latent; research focus)
  • deter  h      D                     (recurrent state)
  • full feat     get_feat = z ++ h     (S*K + D)

Stoch is additionally probed ROW BY ROW: one probe per stoch group s using only
that group's K-dim one-hot, giving S accuracies per concept that localize which
rows encode each concept (mirrors the per-slot framing of goal_position_in_z).
The aggregate stoch probe is kept as the "distributed upper bound": a concept
spread across several rows still shows up there even if no single row suffices.

Controls
────────
  • majority-class / mean-predictor baseline
  • shuffled-label probe (must land at chance) → calibrates leakage

Output
──────
Plots and JSON in {logdir}/experiments/z_probe/.

Probe family (`--probe`)
────────────────────────
  • linear (default): sklearn LogisticRegression / Ridge. Output → {logdir}/experiments/z_probe/.
  • mlp: a small PyTorch MLP trained on GPU (nonlinear probe). Output → {logdir}/experiments/z_probe_mlp/
    (a separate folder, so the linear results are never overwritten).

Usage
─────
uv run python -m experiments.z_probe \
    --logdir logdir/wm_only_fixed_goal_random_mission/01 \
    --device cpu

uv run python -m experiments.z_probe \
    --logdir logdir/wm_only_fixed_goal_random_mission/01 \
    --probe mlp --device cuda
"""

import argparse
import dataclasses
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from dreamer import Dreamer
from envs import make_env
from rewards import make_reward

# ─────────────────────────────────────────────────────────────────────────────
# Concepts. kind ∈ {"clf", "bin", "reg"}; describes the estimator + metric.
# ─────────────────────────────────────────────────────────────────────────────
CONCEPTS = {
    "dir": "clf",  # 0..3 — POSITIVE CONTROL (encoder input)
    "x": "clf",  # 1..8 — image-only
    "y": "clf",  # 1..8 — image-only
    "manhattan": "reg",  # |ax-gx| + |ay-gy|
    "dx": "reg",  # gx - ax
    "dy": "reg",  # gy - ay
    "on_goal": "bin",  # agent on goal cell (== reward 0)
}


# ─────────────────────────────────────────────────────────────────────────────
# Obs preprocessing (same convention as the other experiments)
# ─────────────────────────────────────────────────────────────────────────────


def _preprocess_obs(obs: dict, device: str) -> dict:
    """obs dict (numpy) → (1, ...) float tensors on device. uint8 images /255."""
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            t = torch.as_tensor(v, dtype=torch.float32, device=device)
            if v.dtype == np.uint8:
                t = t / 255.0
            out[k] = t.unsqueeze(0)
        elif isinstance(v, (bool, np.bool_)):
            out[k] = torch.tensor([[float(v)]], dtype=torch.float32, device=device)
    return out


def _unwrap(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def _goal_pos(base_env) -> np.ndarray:
    raw = getattr(base_env, "_goal_pos", None)
    if raw is None:
        raw = getattr(base_env, "goal_pos", None)
    if raw is None:
        raise AttributeError(f"No goal_pos / _goal_pos on {type(base_env).__name__}")
    return np.array(raw, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset collection: on-trajectory random rollouts, one row per visited state
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def collect_dataset(agent: Dreamer, env_cfg, episodes: int, max_steps: int, device: str, seed: int) -> dict:
    """Roll out `episodes` uniform-random-policy episodes and record, at EVERY
    visited state, the posterior latents aligned with the ground-truth agent
    state. On-trajectory (not a single-step sweep) so `deter` accumulates
    history and the stoch/deter/feat comparison is fair.
    """
    rng = np.random.RandomState(seed)

    stochs, deters, feats = [], [], []
    pos, dirs = [], []
    traj_ids = []

    for ep in range(episodes):
        env = make_env(env_cfg, 0)
        obs = env.reset()
        base = _unwrap(env)
        goal = _goal_pos(base)  # constant in fixed_goal
        n_actions = env.action_space.shape[0]

        stoch, deter = agent.rssm.initial(1)
        prev_action = torch.zeros(1, n_actions, device=device)

        def _record():
            feat = agent.rssm.get_feat(stoch, deter)
            stochs.append(stoch.squeeze(0).cpu().numpy())  # (S, K)
            deters.append(deter.squeeze(0).cpu().numpy())  # (D,)
            feats.append(feat.squeeze(0).cpu().numpy())  # (S*K + D,)
            b = _unwrap(env)
            ax, ay = b.agent_pos
            pos.append((int(ax), int(ay)))
            dirs.append(int(b.agent_dir))
            traj_ids.append(ep)

        # t = 0: process the reset observation
        obs_t = _preprocess_obs(obs, device)
        is_first = torch.tensor([True], dtype=torch.bool, device=device)
        embed = agent.encoder(obs_t)
        stoch, deter, _ = agent.rssm.obs_step(stoch, deter, prev_action, embed, is_first)
        _record()

        for _ in range(max_steps):
            act_idx = rng.randint(n_actions)
            act_np = np.zeros(n_actions, dtype=np.float32)
            act_np[act_idx] = 1.0
            obs, _r, done, _info = env.step(act_np)

            prev_action = torch.as_tensor(act_np, device=device).unsqueeze(0)
            obs_t = _preprocess_obs(obs, device)
            is_first = torch.tensor([False], dtype=torch.bool, device=device)
            embed = agent.encoder(obs_t)
            stoch, deter, _ = agent.rssm.obs_step(stoch, deter, prev_action, embed, is_first)
            _record()

            if done:
                break

        if (ep + 1) % 20 == 0:
            print(f"  episode {ep + 1}/{episodes}  (states so far: {len(pos)})")

    pos = np.array(pos, dtype=np.int32)  # (N, 2)
    dirs = np.array(dirs, dtype=np.int32)  # (N,)
    gx, gy = int(goal[0]), int(goal[1])

    labels = {
        "dir": dirs,
        "x": pos[:, 0],
        "y": pos[:, 1],
        "manhattan": np.abs(pos[:, 0] - gx) + np.abs(pos[:, 1] - gy),
        "dx": gx - pos[:, 0],
        "dy": gy - pos[:, 1],
        "on_goal": ((pos[:, 0] == gx) & (pos[:, 1] == gy)).astype(np.int32),
    }

    return {
        "stoch": np.stack(stochs).astype(np.float32),  # (N, S, K)
        "deter": np.stack(deters).astype(np.float32),  # (N, D)
        "feat": np.stack(feats).astype(np.float32),  # (N, S*K + D)
        "labels": labels,
        "pos": pos,
        "goal": np.array([gx, gy], dtype=np.int32),
        "traj_ids": np.array(traj_ids, dtype=np.int32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Probe fitting
# ─────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ProbeConfig:
    """Selects the probe family and its hyperparameters."""

    type: str = "linear"  # "linear" | "mlp"
    device: str = "cpu"
    hidden: tuple = (256, 256)
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLPProbe:
    """A small MLP with an sklearn-like fit/predict interface, trained on GPU.

    Full-batch Adam for a fixed number of epochs. Inputs are standardized on the
    train split (zero-variance columns — e.g. unused one-hot categories — get
    unit scale). Classification labels are mapped to a contiguous 0..C-1 range
    over the classes seen in training and mapped back on predict.
    """

    def __init__(self, kind: str, cfg: ProbeConfig, seed: int = 0):
        self.kind = kind
        self.cfg = cfg
        self.seed = seed

    def _standardize_fit(self, X: np.ndarray):
        self.mu_ = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        self.sd_ = sd

    def _to_device(self, X: np.ndarray) -> torch.Tensor:
        Xn = (X - self.mu_) / self.sd_
        return torch.as_tensor(Xn, dtype=torch.float32, device=self.cfg.device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.seed)
        self._standardize_fit(X)
        Xd = self._to_device(X)

        if self.kind == "reg":
            out_dim = 1
            yd = torch.as_tensor(y, dtype=torch.float32, device=self.cfg.device).view(-1, 1)
            loss_fn = nn.MSELoss()
        else:  # clf | bin
            self.classes_, y_idx = np.unique(y, return_inverse=True)
            out_dim = len(self.classes_)
            yd = torch.as_tensor(y_idx, dtype=torch.long, device=self.cfg.device)
            loss_fn = nn.CrossEntropyLoss()

        self.net_ = _MLP(X.shape[1], out_dim, self.cfg.hidden).to(self.cfg.device)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.net_.train()
        for _ in range(self.cfg.epochs):
            opt.zero_grad()
            loss_fn(self.net_(Xd), yd).backward()
            opt.step()
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net_.eval()
        out = self.net_(self._to_device(X))
        if self.kind == "reg":
            return out.squeeze(-1).cpu().numpy()
        return self.classes_[out.argmax(-1).cpu().numpy()]


def _linear_estimator(kind: str, scale: bool):
    est = Ridge(alpha=1.0) if kind == "reg" else LogisticRegression(max_iter=2000, C=1.0)
    return make_pipeline(StandardScaler(), est) if scale else est


def build_estimator(kind: str, scale: bool, pcfg: ProbeConfig):
    """Return a fresh estimator with .fit/.predict for the selected probe family.
    (`scale` is only meaningful for the linear probe; the MLP always standardizes.)"""
    if pcfg.type == "mlp":
        return TorchMLPProbe(kind, pcfg)
    return _linear_estimator(kind, scale)


def _dummy(kind: str):
    return DummyRegressor(strategy="mean") if kind == "reg" else DummyClassifier(strategy="most_frequent")


def _metrics(kind: str, y_true, y_pred) -> dict:
    if kind == "reg":
        return {"r2": float(r2_score(y_true, y_pred)), "mse": float(mean_squared_error(y_true, y_pred))}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _score(kind: str, metrics: dict) -> float:
    """Single scalar used for ranking / plotting (higher = better)."""
    return metrics["r2"] if kind == "reg" else metrics["balanced_accuracy"]


def fit_probe(X_tr, y_tr, X_te, y_te, kind: str, scale: bool, pcfg: ProbeConfig) -> dict:
    """Fit one probe; return metrics + baseline + shuffled control."""
    est = build_estimator(kind, scale, pcfg).fit(X_tr, y_tr)
    probe = _metrics(kind, y_te, est.predict(X_te))

    base = _dummy(kind).fit(X_tr, y_tr)
    baseline = _metrics(kind, y_te, base.predict(X_te))

    rng = np.random.RandomState(0)
    y_shuf = y_tr.copy()
    rng.shuffle(y_shuf)
    shuf = build_estimator(kind, scale, pcfg).fit(X_tr, y_shuf)
    shuffled = _metrics(kind, y_te, shuf.predict(X_te))

    return {"probe": probe, "baseline": baseline, "shuffled": shuffled, "score": _score(kind, probe)}


def _degenerate(y_tr, y_te, kind: str) -> bool:
    """Skip probes with no signal to learn/measure (single-class labels)."""
    if kind == "reg":
        return np.unique(y_tr).size < 2 or np.unique(y_te).size < 2
    return np.unique(y_tr).size < 2 or np.unique(y_te).size < 2


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────


def run_probes(data: dict, test_frac: float, seed: int, top_k_max: int, pcfg: ProbeConfig) -> dict:
    labels = data["labels"]
    traj_ids = data["traj_ids"]
    N, S, K = data["stoch"].shape

    stoch_flat = data["stoch"].reshape(N, S * K)
    reps = {  # name → (X, needs_scaling)
        "stoch": (stoch_flat, False),
        "deter": (data["deter"], True),
        "feat": (data["feat"], True),
    }

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr_idx, te_idx = next(splitter.split(stoch_flat, groups=traj_ids))
    print(
        f"\nSplit by trajectory: {len(tr_idx)} train / {len(te_idx)} test states "
        f"({np.unique(traj_ids[tr_idx]).size}/{np.unique(traj_ids[te_idx]).size} episodes)"
    )

    results = {"n_states": int(N), "S": int(S), "K": int(K), "probe": pcfg.type, "skipped": []}

    # ── Aggregate probes: concept × representation ───────────────────────────
    aggregate = {}
    for concept, kind in CONCEPTS.items():
        y = labels[concept]
        if _degenerate(y[tr_idx], y[te_idx], kind):
            results["skipped"].append(concept)
            print(f"  [skip] '{concept}': single-class labels (no variance to probe)")
            continue
        aggregate[concept] = {}
        for rep_name, (X, scale) in reps.items():
            r = fit_probe(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], kind, scale, pcfg)
            aggregate[concept][rep_name] = r
        base = _score(kind, aggregate[concept]["stoch"]["baseline"])
        shuf = _score(kind, aggregate[concept]["stoch"]["shuffled"])
        print(
            f"  {concept:10s} ({kind}):  stoch={aggregate[concept]['stoch']['score']:.3f}  "
            f"deter={aggregate[concept]['deter']['score']:.3f}  feat={aggregate[concept]['feat']['score']:.3f}  "
            f"(baseline {base:.3f}, shuffled {shuf:.3f})"
        )
    results["aggregate"] = aggregate

    # ── Per-row stoch probes: concept × row ──────────────────────────────────
    print("\nPer-row stoch probes (which of the 32 rows encode each concept)...")
    per_row = {}
    for concept, kind in CONCEPTS.items():
        if concept in results["skipped"]:
            continue
        y = labels[concept]
        row_scores = np.full(S, np.nan)
        for s in range(S):
            Xs = data["stoch"][:, s, :]  # (N, K) one-hot → no scaling
            r = fit_probe(Xs[tr_idx], y[tr_idx], Xs[te_idx], y[te_idx], kind, False, pcfg)
            row_scores[s] = r["score"]
        per_row[concept] = row_scores.tolist()
        top = np.argsort(-row_scores)[:5]
        print(f"  {concept:10s}: top rows {top.tolist()}  scores {np.round(row_scores[top], 3).tolist()}")
    results["per_row"] = per_row

    # ── Optional top-k cumulative curve (rows added best-first) ───────────────
    kmax = min(top_k_max, S)
    print(f"\nTop-k cumulative stoch probes (up to k={kmax})...")
    top_k = {}
    for concept, kind in CONCEPTS.items():
        if concept in results["skipped"]:
            continue
        y = labels[concept]
        order = np.argsort(-np.array(per_row[concept]))  # best rows first
        curve = []
        for k in range(1, kmax + 1):
            rows = order[:k]
            Xk = data["stoch"][:, rows, :].reshape(N, k * K)
            r = fit_probe(Xk[tr_idx], y[tr_idx], Xk[te_idx], y[te_idx], kind, False, pcfg)
            curve.append(r["score"])
        top_k[concept] = {"order": order.tolist(), "curve": curve}
    results["top_k"] = top_k

    # Stash test-set predictions for confusion matrices (position + direction).
    conf = {}
    for concept in ("x", "y", "dir"):
        if concept in results["skipped"]:
            continue
        X, scale = reps["stoch"]
        est = build_estimator(CONCEPTS[concept], scale, pcfg).fit(X[tr_idx], labels[concept][tr_idx])
        conf[concept] = {
            "y_true": labels[concept][te_idx].tolist(),
            "y_pred": est.predict(X[te_idx]).tolist(),
        }
    results["_confusion"] = conf
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {path}")


def plot_all(data: dict, results: dict, outdir: pathlib.Path):
    concepts = [c for c in CONCEPTS if c not in results["skipped"]]
    S = results["S"]

    # ── Accuracy/score by concept × representation, with controls ────────────
    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(concepts)), 5), constrained_layout=True)
    reps = ["stoch", "deter", "feat"]
    colors = {"stoch": "firebrick", "deter": "steelblue", "feat": "seagreen"}
    x = np.arange(len(concepts))
    w = 0.22
    for i, rep in enumerate(reps):
        vals = [results["aggregate"][c][rep]["score"] for c in concepts]
        ax.bar(x + (i - 1) * w, vals, width=w, color=colors[rep], label=rep)
    base = [_score(CONCEPTS[c], results["aggregate"][c]["stoch"]["baseline"]) for c in concepts]
    shuf = [_score(CONCEPTS[c], results["aggregate"][c]["stoch"]["shuffled"]) for c in concepts]
    ax.plot(x, base, "k_", markersize=18, markeredgewidth=2, label="majority/mean baseline")
    ax.plot(x, shuf, "x", color="0.4", markersize=8, label="shuffled control")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({CONCEPTS[c]})" for c in concepts])
    ax.set_ylabel("balanced acc. (clf/bin)  /  R² (reg)")
    probe_name = "MLP" if results.get("probe") == "mlp" else "Linear"
    ax.set_title(f"{probe_name} decodability of each concept from the WM latent")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, outdir / "accuracy_by_concept_and_representation.png")

    # ── Per-row heatmap: concept × row ───────────────────────────────────────
    mat = np.array([results["per_row"][c] for c in concepts])  # (C, S)
    fig, ax = plt.subplots(figsize=(max(10, 0.3 * S), 0.6 * len(concepts) + 2), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(S))
    ax.set_xlabel("stoch row s")
    ax.set_yticks(np.arange(len(concepts)))
    ax.set_yticklabels(concepts)
    ax.set_title("Per-row probe score (which rows encode each concept)")
    plt.colorbar(im, ax=ax, fraction=0.025, label="score")
    _save(fig, outdir / "per_row_probe_heatmap.png")

    # ── Top-k cumulative curves ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for c in concepts:
        curve = results["top_k"][c]["curve"]
        ax.plot(np.arange(1, len(curve) + 1), curve, marker=".", label=c)
    ax.set_xlabel("number of top rows (best-first)")
    ax.set_ylabel("score")
    ax.set_title("Cumulative decodability vs. number of stoch rows")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, outdir / "top_k_rows_curve.png")

    # ── Confusion matrices (stoch probe) for x, y, dir ───────────────────────
    conf = results.get("_confusion", {})
    present = [c for c in ("x", "y", "dir") if c in conf]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 4), constrained_layout=True)
        for ax, c in zip(np.atleast_1d(axes), present):
            yt = np.array(conf[c]["y_true"])
            yp = np.array(conf[c]["y_pred"])
            classes = np.unique(np.concatenate([yt, yp]))
            idx = {v: i for i, v in enumerate(classes)}
            m = np.zeros((len(classes), len(classes)), dtype=int)
            for t, p in zip(yt, yp):
                m[idx[t], idx[p]] += 1
            im = ax.imshow(m, cmap="Blues")
            ax.set_title(f"{c} (stoch probe)")
            ax.set_xlabel("pred")
            ax.set_ylabel("true")
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels(classes)
            plt.colorbar(im, ax=ax, fraction=0.046)
        _save(fig, outdir / "confusion_xy_dir.png")

    # ── Cell-visitation coverage ─────────────────────────────────────────────
    pos, goal = data["pos"], data["goal"]
    # infer grid extent from labels (interior 1..size-2); pad by 1 for walls
    size = int(max(pos.max(), goal.max())) + 2
    grid = np.zeros((size, size))
    for x_, y_ in pos:
        grid[y_, x_] += 1
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(grid, cmap="magma")
    ax.plot(goal[0], goal[1], "g*", markersize=16, label="goal")
    ax.set_title("State (cell) visitation under random policy")
    ax.legend()
    plt.colorbar(im, ax=ax, fraction=0.046, label="# visits")
    _save(fig, outdir / "cell_visitation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def _load_agent(logdir: pathlib.Path, device: str | None):
    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")
    device = device or config.device
    config.device = device

    # Backfill defaults for keys added after this run was trained.
    OmegaConf.set_struct(config.model, False)
    if "wm_only" not in config.model:
        config.model.wm_only = False
    if "goal_imag_horizon" not in config.model:
        config.model.goal_imag_horizon = int(config.model.imag_horizon)

    resolved = OmegaConf.to_container(config, resolve=True)
    resolved["env"]["device"] = device
    env_cfg = OmegaConf.create(resolved["env"])

    probe_env = make_env(env_cfg, 0)
    obs_space, act_space = probe_env.observation_space, probe_env.action_space

    reward_function = make_reward(config)
    agent = Dreamer(config.model, obs_space, act_space, reward_function=reward_function).to(device)
    checkpoint = torch.load(logdir / "latest.pt", map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    print(f"Checkpoint loaded: {logdir / 'latest.pt'}  (env: {env_cfg.task}, device: {device})")
    return agent, env_cfg, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Run with .hydra/config.yaml and latest.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=200, help="Random-policy episodes to collect")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per collected episode")
    parser.add_argument("--test-frac", type=float, default=0.25, help="Fraction of trajectories held out")
    parser.add_argument("--top-k-max", type=int, default=16, help="Max rows in the top-k cumulative curve")
    parser.add_argument("--probe", choices=["linear", "mlp"], default="linear", help="Probe family")
    parser.add_argument("--mlp-hidden", type=int, nargs="+", default=[256, 256], help="MLP hidden layer sizes")
    parser.add_argument("--mlp-epochs", type=int, default=300, help="MLP training epochs (full-batch Adam)")
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-wd", type=float, default=1e-4, help="MLP weight decay")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logdir = pathlib.Path(args.logdir)

    agent, env_cfg, device = _load_agent(logdir, args.device)

    pcfg = ProbeConfig(
        type=args.probe,
        device=device,
        hidden=tuple(args.mlp_hidden),
        epochs=args.mlp_epochs,
        lr=args.mlp_lr,
        weight_decay=args.mlp_wd,
    )

    print(f"\nCollecting {args.episodes} random-policy episodes (≤{args.max_steps} steps each)...")
    data = collect_dataset(agent, env_cfg, args.episodes, args.max_steps, device, args.seed)
    print(f"Collected {data['stoch'].shape[0]} states; goal at {tuple(data['goal'].tolist())}.")

    print(
        f"\nProbe family: {pcfg.type}"
        + (f" (hidden={pcfg.hidden}, epochs={pcfg.epochs}, device={device})" if pcfg.type == "mlp" else "")
    )
    results = run_probes(data, args.test_frac, args.seed, args.top_k_max, pcfg)

    # Separate folder per probe family so results never overwrite each other.
    outname = "z_probe" if pcfg.type == "linear" else "z_probe_mlp"
    outdir = logdir / "experiments" / outname
    outdir.mkdir(parents=True, exist_ok=True)
    plot_all(data, results, outdir)

    results.pop("_confusion", None)  # predictions are large; keep the JSON lean
    results["config"] = {
        "probe": pcfg.type,
        "mlp_hidden": list(pcfg.hidden),
        "mlp_epochs": pcfg.epochs,
        "mlp_lr": pcfg.lr,
        "mlp_weight_decay": pcfg.weight_decay,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "goal": data["goal"].tolist(),
    }
    with open(outdir / "z_probe.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {outdir}")


if __name__ == "__main__":
    main()
