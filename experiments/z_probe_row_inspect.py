"""
Drill-down on a single (concept, stoch-row) pair from the z_probe study.

Motivation
──────────
In the MLP probe run, `on_goal` was decodable from stoch row 27 alone at
balanced accuracy ≈ 0.785, whereas the *linear* per-row probe on the same row
sat at ≈ 0.5. That is surprising: the per-row features are a K-way ONE-HOT, and
a linear logistic regression on a one-hot is already a per-category lookup
table — it should not be beaten by an MLP on the same input. This script figures
out what is going on for a given (--concept, --row), by default (on_goal, 27).

It reuses the exact dataset collection of experiments.z_probe (same seed ⇒ same
states/split) and reports:
  1. Class balance of the concept (how many positives, in train/test).
  2. The row's category → concept relationship: P(concept | row argmax = k) and
     support per category  → is there a category that "lights up" at the goal?
  3. A spatial map of the row's modal category per grid cell, next to where the
     concept is positive (the goal cell).
  4. A like-for-like probe comparison on that one row:
        linear · linear(class_weight=balanced) · MLP
     If the balanced linear probe ≈ MLP, the gap was class imbalance /
     regularization, not genuine nonlinearity.

Usage
─────
uv run python -m experiments.z_probe_row_inspect \
    --logdir logdir/wm_only_fixed_goal_random_mission/01 \
    --device cuda --concept on_goal --row 27
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from experiments.z_probe import CONCEPTS, ProbeConfig, TorchMLPProbe, _load_agent, collect_dataset


def _linear(balanced: bool):
    return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced" if balanced else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--concept", default="on_goal", choices=list(CONCEPTS))
    parser.add_argument("--row", type=int, default=27)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    concept, s = args.concept, args.row
    kind = CONCEPTS[concept]
    logdir = pathlib.Path(args.logdir)

    agent, env_cfg, device = _load_agent(logdir, args.device)
    print(f"\nCollecting {args.episodes} episodes (≤{args.max_steps} steps, seed {args.seed})...")
    data = collect_dataset(agent, env_cfg, args.episodes, args.max_steps, device, args.seed)
    N, S, K = data["stoch"].shape
    y = data["labels"][concept]
    pos, goal = data["pos"], data["goal"]
    row_cat = data["stoch"][:, s, :].argmax(1)  # (N,) argmax category of row s
    print(f"Collected {N} states; goal at {tuple(goal.tolist())}.")

    # ── 1. Class balance ─────────────────────────────────────────────────────
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    tr, te = next(splitter.split(data["stoch"].reshape(N, -1), groups=data["traj_ids"]))
    pos_all, pos_tr, pos_te = int((y == 1).sum()), int((y[tr] == 1).sum()), int((y[te] == 1).sum())
    print("\n" + "=" * 68)
    print(f"  Inspecting concept='{concept}' ({kind}) vs stoch row {s}")
    print("=" * 68)
    print(f"  positives: {pos_all}/{N} total  |  {pos_tr}/{len(tr)} train  |  {pos_te}/{len(te)} test")

    # ── 2. Category → concept relationship (only meaningful for clf/bin) ──────
    cat_support = np.array([(row_cat == k).sum() for k in range(K)])
    cat_pos_rate = np.array([y[row_cat == k].mean() if (row_cat == k).any() else np.nan for k in range(K)])
    print("\n  Per-category (row argmax) support and P(concept=1 | category):")
    for k in range(K):
        if cat_support[k] > 0:
            flag = "  <-- concentrates positives" if cat_pos_rate[k] > 5 * (pos_all / N) and cat_support[k] > 5 else ""
            print(f"    cat {k:2d}: n={cat_support[k]:5d}  P(pos)={cat_pos_rate[k]:.3f}{flag}")

    # ── 3. Probe comparison on this single row ───────────────────────────────
    Xs = data["stoch"][:, s, :]
    results_probes = {}
    if kind != "reg":
        lin = _linear(False).fit(Xs[tr], y[tr])
        linb = _linear(True).fit(Xs[tr], y[tr])
        mlp = TorchMLPProbe(kind, ProbeConfig(type="mlp", device=device)).fit(Xs[tr], y[tr])
        for name, est in [("linear", lin), ("linear_balanced", linb), ("mlp", mlp)]:
            yp = est.predict(Xs[te])
            ba = float(balanced_accuracy_score(y[te], yp))
            cm = confusion_matrix(y[te], yp).tolist()
            results_probes[name] = {"balanced_accuracy": ba, "confusion": cm}
            print(f"  {name:16s} balanced acc = {ba:.3f}   confusion={cm}")

    # ── Plots ────────────────────────────────────────────────────────────────
    outdir = logdir / "experiments" / "z_probe_mlp" / f"row_inspect_{concept}_r{s}"
    outdir.mkdir(parents=True, exist_ok=True)

    # (a) per-category support + positive rate
    fig, ax1 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax1.bar(np.arange(K), cat_support, color="0.8", label="support (count)")
    ax1.set_xlabel(f"row {s} argmax category k")
    ax1.set_ylabel("support", color="0.4")
    ax2 = ax1.twinx()
    ax2.plot(np.arange(K), cat_pos_rate, "o-", color="firebrick", label=f"P({concept}=1 | k)")
    ax2.axhline(pos_all / N, color="firebrick", ls="--", alpha=0.5, label="base rate")
    ax2.set_ylabel(f"P({concept}=1 | category)", color="firebrick")
    ax1.set_title(f"Row {s}: which category carries '{concept}'")
    fig.legend(loc="upper right", fontsize=8)
    fig.savefig(outdir / "category_positive_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (b) spatial: modal category of row s per cell, and concept-positive cells
    size = int(max(pos.max(), goal.max())) + 2
    modal = np.full((size, size), np.nan)
    posrate = np.full((size, size), np.nan)
    for x_ in range(size):
        for y_ in range(size):
            m = (pos[:, 0] == x_) & (pos[:, 1] == y_)
            if m.any():
                vals, cnts = np.unique(row_cat[m], return_counts=True)
                modal[y_, x_] = vals[cnts.argmax()]
                posrate[y_, x_] = y[m].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].imshow(modal, cmap="tab20", vmin=0, vmax=K - 1)
    axes[0].plot(goal[0], goal[1], "k*", markersize=16)
    axes[0].set_title(f"Modal category of row {s} per cell (★=goal)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label="argmax k")
    im1 = axes[1].imshow(posrate, cmap="magma", vmin=0, vmax=1)
    axes[1].plot(goal[0], goal[1], "g*", markersize=16)
    axes[1].set_title(f"P({concept}=1) per cell")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.savefig(outdir / "spatial_modal_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (c) probe comparison
    if results_probes:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        names = list(results_probes)
        ax.bar(
            names, [results_probes[n]["balanced_accuracy"] for n in names], color=["steelblue", "seagreen", "firebrick"]
        )
        ax.axhline(0.5, color="k", ls="--", alpha=0.5, label="chance")
        ax.set_ylabel("balanced accuracy")
        ax.set_title(f"'{concept}' from row {s} only: does class-balancing close the gap?")
        ax.legend()
        fig.savefig(outdir / "probe_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    out = {
        "concept": concept,
        "row": s,
        "positives": {"total": pos_all, "train": pos_tr, "test": pos_te, "N": int(N)},
        "category_support": cat_support.tolist(),
        "category_positive_rate": [None if np.isnan(v) else float(v) for v in cat_pos_rate],
        "probes": results_probes,
    }
    with open(outdir / "row_inspect.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {outdir}")


if __name__ == "__main__":
    main()
