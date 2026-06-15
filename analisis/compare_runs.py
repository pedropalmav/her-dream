#!/usr/bin/env python3
"""Compare two (or more) runs on the same axes.

Overlays the shared metrics of several runs:
  1. Recompensa de entrenamiento (media movil)   -- episode/score
  2. Recompensa de evaluacion (greedy, suavizada) -- episode/eval_score
  3. Supervivencia (largo de episodio, eval)       -- episode/eval_length
  4. Logros (eval final): barras agrupadas por run -- episode/eval_<achievement>

Usage:
  python3 compare_runs.py \
      ../logdir/original_wm_crafter/02 \
      ../logdir/z_without_history_wm_crafter/01 \
      --labels "original (con historia)" "sin historia (z)" \
      -o comparacion_original_vs_zsinhistoria.png

Run from any cwd; rundir paths are resolved as given. If --labels is omitted,
the run directory basename is used.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_run_analysis import (
    final_eval_achievements,
    load_metrics,
    moving_avg,
    series,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundirs", nargs="+", help="run directories with metrics.jsonl")
    ap.add_argument("--labels", nargs="*", default=None, help="legend label per run")
    ap.add_argument("-o", "--out", default=None, help="output png path")
    ap.add_argument("--title", default="Comparacion de runs", help="figure suptitle")
    ap.add_argument("--window", type=int, default=100, help="moving-average window (episodes)")
    args = ap.parse_args()

    runs = []
    for rd in args.rundirs:
        runs.append((rd, load_metrics(os.path.join(rd, "metrics.jsonl"))))

    if args.labels and len(args.labels) == len(runs):
        labels = args.labels
    else:
        labels = [os.path.basename(os.path.normpath(rd)) for rd, _ in runs]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "comparacion_runs.png"
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(args.title, fontsize=14, fontweight="bold")

    # --- Panel 1: train reward (moving avg only, for readability) ---
    ax = axes[0, 0]
    for i, (rd, rows) in enumerate(runs):
        xs, ys = series(rows, "episode/score")
        if not len(xs):
            continue
        ma = moving_avg(ys, args.window)
        ax.plot(xs[len(xs) - len(ma):], ma, color=colors[i % len(colors)], lw=2,
                label=labels[i])
    ax.set_title(f"Recompensa de entrenamiento (media movil {args.window} ep)")
    ax.set_xlabel("step de entorno")
    ax.set_ylabel("reward por episodio")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # Eval metrics are logged sparsely (~1 point per 10k steps, ~100 total), so a
    # 100-episode moving average is meaningless. Use a light smoothing instead and
    # always draw the full raw line.
    eval_win = max(1, min(5, args.window))

    # --- Panel 2: eval reward ---
    ax = axes[0, 1]
    for i, (rd, rows) in enumerate(runs):
        xs, ys = series(rows, "episode/eval_score")
        if not len(xs):
            continue
        c = colors[i % len(colors)]
        ax.plot(xs, ys, color=c, alpha=0.35, lw=1.0)
        ma = moving_avg(ys, eval_win)
        ax.plot(xs[len(xs) - len(ma):], ma, color=c, lw=2, label=labels[i])
    ax.set_title(f"Recompensa de evaluacion (greedy, suavizado {eval_win} ep)")
    ax.set_xlabel("step de entorno")
    ax.set_ylabel("eval_score (reward)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 3: survival (eval episode length) ---
    ax = axes[1, 0]
    for i, (rd, rows) in enumerate(runs):
        xs, ys = series(rows, "episode/eval_length")
        if not len(xs):
            continue
        c = colors[i % len(colors)]
        ax.plot(xs, ys, color=c, alpha=0.35, lw=1.0)
        ma = moving_avg(ys, eval_win)
        ax.plot(xs[len(xs) - len(ma):], ma, color=c, lw=2, label=labels[i])
    ax.set_title(f"Supervivencia (largo de episodio eval, suavizado {eval_win} ep)")
    ax.set_xlabel("step de entorno")
    ax.set_ylabel("longitud del episodio")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 4: grouped achievements (eval final) ---
    ax = axes[1, 1]
    per_run = [final_eval_achievements(rows) for _, rows in runs]
    # union of achievement names, ordered by the first run's ranking then the rest
    names = list(per_run[0].keys())
    for d in per_run[1:]:
        for n in d:
            if n not in names:
                names.append(n)
    if names:
        y = np.arange(len(names))[::-1]
        n_runs = len(runs)
        h = 0.8 / n_runs
        for i, d in enumerate(per_run):
            vals = np.asarray([d.get(n, 0.0) for n in names])
            disp = np.where(vals > 0, vals, 1e-3)
            offset = (i - (n_runs - 1) / 2) * h
            ax.barh(y + offset, disp, height=h, color=colors[i % len(colors)],
                    label=labels[i])
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xscale("log")
        ax.set_xlabel("conteo medio por episodio (eval final, log)")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Logros (eval final): comparacion")
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
