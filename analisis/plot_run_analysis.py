#!/usr/bin/env python3
"""Produce the 4-panel run analysis figure from a run's metrics.jsonl.

Panels:
  1. Recompensa de entrenamiento (train)  -- episode/score, raw + 100-ep moving avg
  2. Recompensa de evaluacion (greedy)    -- episode/eval_score
  3. Supervivencia (largo del episodio)   -- episode/length + episode/eval_length
  4. Logros (eval final, log)             -- episode/eval_<achievement>, green=achieved/red=never

Usage:
  python3 plot_run_analysis.py logdir/original_wm_crafter/02
  python3 plot_run_analysis.py logdir/z_without_history_wm_crafter/01 --title "Z sin historia - run 01"
  python3 plot_run_analysis.py <rundir> -o out.png
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path):
    """Return list of dicts from a metrics.jsonl file."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def series(rows, key):
    """Extract (steps, values) for every row that contains `key`."""
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None:
            xs.append(r["step"])
            ys.append(r[key])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def moving_avg(y, w):
    if len(y) < 1:
        return y
    w = max(1, min(w, len(y)))
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="valid")


def final_eval_achievements(rows):
    """Mean per-episode count of each achievement at the last eval step.

    Achievement columns are 'episode/eval_<name>' excluding score/length.
    Returns dict name -> mean count, sorted descending by count.
    """
    ach_keys = [
        k
        for k in {k for r in rows for k in r}
        if k.startswith("episode/eval_")
        and not k.endswith("eval_score")
        and not k.endswith("eval_length")
    ]
    eval_rows = [r for r in rows if any(k in r for k in ach_keys)]
    if not eval_rows:
        return {}
    last_step = max(r["step"] for r in eval_rows)
    final_rows = [r for r in eval_rows if r["step"] == last_step]
    out = {}
    for k in ach_keys:
        vals = [r[k] for r in final_rows if k in r and r[k] is not None]
        if vals:
            out[k.replace("episode/eval_", "")] = float(np.mean(vals))
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir", help="run directory containing metrics.jsonl")
    ap.add_argument("-o", "--out", default=None, help="output png path")
    ap.add_argument("--title", default=None, help="figure suptitle")
    ap.add_argument("--window", type=int, default=100, help="moving-average window (episodes)")
    args = ap.parse_args()

    metrics_path = os.path.join(args.rundir, "metrics.jsonl")
    rows = load_metrics(metrics_path)

    run_name = os.path.basename(os.path.normpath(args.rundir))
    max_step = max((r.get("step", 0) for r in rows), default=0)
    title = args.title or f"R2-Dreamer vanilla en Crafter - {run_name} ({max_step/1e6:.2f}M steps)"
    out = args.out or os.path.join(args.rundir, f"analisis_{run_name}.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.style.use("default")

    # --- Panel 1: train reward ---
    ax = axes[0, 0]
    xs, ys = series(rows, "episode/score")
    if len(xs):
        ax.plot(xs, ys, color="gray", alpha=0.35, lw=0.7, label="por episodio")
        ma = moving_avg(ys, args.window)
        ax.plot(xs[len(xs) - len(ma):], ma, color="tab:blue", lw=2,
                label=f"media movil ({args.window} ep)")
    ax.set_title("Recompensa de entrenamiento (train)")
    ax.set_xlabel("step de entorno")
    ax.set_ylabel("reward por episodio")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 2: eval reward ---
    ax = axes[0, 1]
    xs, ys = series(rows, "episode/eval_score")
    if len(xs):
        ax.plot(xs, ys, color="tab:orange", lw=1.2)
    ax.set_title("Recompensa de evaluacion (greedy)")
    ax.set_xlabel("step de entorno")
    ax.set_ylabel("eval_score (reward)")
    ax.grid(alpha=0.3)

    # --- Panel 3: survival / episode length ---
    ax = axes[1, 0]
    xs, ys = series(rows, "episode/length")
    if len(xs):
        ma = moving_avg(ys, args.window)
        ax.plot(xs, ys, color="purple", alpha=0.25, lw=0.7)
        ax.plot(xs[len(xs) - len(ma):], ma, color="purple", lw=2, label="train len (mm)")
    xe, ye = series(rows, "episode/eval_length")
    if len(xe):
        ax.plot(xe, ye, color="tab:green", alpha=0.7, lw=1.0, label="eval len")
    ax.set_title("Supervivencia (largo del episodio)")
    ax.set_xlabel("step de entorno")
    ax.set_ylabel("longitud del episodio")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 4: achievements ---
    ax = axes[1, 1]
    ach = final_eval_achievements(rows)
    if ach:
        names = list(ach.keys())
        vals = np.asarray([ach[n] for n in names])
        y = np.arange(len(names))[::-1]  # largest on top
        colors = ["tab:green" if v > 0 else "tab:red" for v in vals]
        # log scale needs positive; floor the zeros for display
        disp = np.where(vals > 0, vals, 1e-3)
        ax.barh(y, disp, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xscale("log")
        ax.set_xlabel("conteo medio por episodio (eval final, log)")
    ax.set_title(f"Logros @{max_step/1e6:.0f}M steps (verde=alcanzado, rojo=nunca)")
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
