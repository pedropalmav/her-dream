"""Figuras de las corridas de jul-26..ago-3 (goal_sample=image, joint end-to-end,
HER future vs final).

Complementa `plot_runs.py` (que cubre las fases históricas y **regenera** el CSV
maestro): este script es autocontenido, lee `metrics.jsonl` en vez de los eventos
de TensorBoard y sólo toca las corridas nuevas, así que se puede correr aunque el
`logdir/` local no tenga todas las corridas viejas.

Produce:
  - assets/goalimage_vs_imag.png   goal_sample=image vs imagination (sin/con `h`)
  - assets/joint_vs_posttrain.png  end-to-end vs post-train con WM congelado
  - assets/her_future_vs_final.png HER her_strategy=future vs final (item 40)

Uso:
    uv run python3 docs/experimentos/scripts/plot_goalimage_joint.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOGDIR = os.path.join(ROOT, "logdir")
ASSETS = os.path.join(ROOT, "docs", "experimentos", "assets")
FLOOR = -1001

# (run, etiqueta, color, estilo)
GOALIMAGE = [
    ("random_goal/posttrain_no_deter_rowbyrow_goalimage/01", "sin h · goal=image (item 36)", "#2ca02c", "-"),
    ("random_goal/posttrain_randomstart_no_deter_rowbyrow/01", "sin h · goal=imagination (item 33-base)", "#1f77b4", "-"),
    ("random_goal/posttrain_randomstart_goalimag_rowbyrow/03", "con h · goal=imagination (item 27)", "#d62728", "--"),
]

JOINT = [
    ("random_goal/posttrain_no_deter_rowbyrow_goalimage/01", "post-train sin h · goal=image", "#2ca02c", "-"),
    ("random_goal/posttrain_randomstart_no_deter_rowbyrow/01", "post-train sin h · imagination", "#1f77b4", "-"),
    ("random_goal/posttrain_randomstart_goalimag_rowbyrow/03", "post-train con h · imagination", "#d62728", "--"),
    ("random_goal/joint_no_deter_rowbyrow/01", "joint sin h (item 35)", "#9467bd", "-"),
    ("random_goal/joint_rowbyrow/01", "joint con h (item 37)", "#ff7f0e", "--"),
]


# A/B de `her_strategy` (item 40): solido = future, punteado = final; un color por
# fuente de goal, para que cada par sea comparable de un vistazo.
HERFUTURE = [
    ("random_goal/posttrain_no_deter_rowbyrow_goalimage_herfuture/01", "goal=image · future (item 40)", "#2ca02c", "-"),
    ("random_goal/posttrain_no_deter_rowbyrow_goalimage/01", "goal=image · final (item 36)", "#2ca02c", "--"),
    ("random_goal/posttrain_randomstart_no_deter_rowbyrow_herfuture/01", "goal=imagination · future (item 40)", "#1f77b4", "-"),
    ("random_goal/posttrain_randomstart_no_deter_rowbyrow/01", "goal=imagination · final", "#1f77b4", "--"),
]


def series(run, tag):
    path = os.path.join(LOGDIR, run, "metrics.jsonl")
    xs, ys = [], []
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if tag in d:
                xs.append(d["step"])
                ys.append(d[tag])
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def ma(y, k=25):
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def panel(ax, runs, tag, title):
    for run, label, color, ls in runs:
        try:
            x, y = series(run, tag)
        except FileNotFoundError:
            continue
        if len(y) == 0:
            continue
        ys = ma(y)
        xs = x[len(x) - len(ys):]
        ax.plot(xs, ys, label=f"{label}  (ma25 final={ys[-1]:.0f})", lw=1.8, color=color, ls=ls)
    ax.axhline(FLOOR, ls=":", c="grey", lw=1)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("score (media movil 25 episodios)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)


def figure(runs, fname, suptitle):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    panel(axes[0], runs, "episode/score", "entrenamiento (episode/score)")
    panel(axes[1], runs, "episode/eval_score", "evaluacion (episode/eval_score)")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    out = os.path.join(ASSETS, fname)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"escrito {out}")


def main():
    os.makedirs(ASSETS, exist_ok=True)
    figure(GOALIMAGE, "goalimage_vs_imag.png",
           "random_goal · row_by_row + HER · goal desde imagen vs desde imaginacion")
    figure(JOINT, "joint_vs_posttrain.png",
           "random_goal · row_by_row + HER · joint end-to-end vs post-train con WM congelado")
    figure(HERFUTURE, "her_future_vs_final.png",
           "random_goal · row_by_row · post-train sin h · HER her_strategy=future vs final")


if __name__ == "__main__":
    main()
