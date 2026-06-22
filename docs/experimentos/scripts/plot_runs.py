"""Genera las figuras de docs/experimentos/ a partir de los eventos de TensorBoard.

Lee todos los `logdir/**/events.out.tfevents*`, extrae la curva de `episode/score`
(o `episode/eval_score` si no hay score de entrenamiento) y produce:

  - assets/curvas_por_fase.png      curvas score-vs-step agrupadas por fase del proyecto
  - assets/score_final_barras.png   barra de score final (media últimos 10 episodios) por corrida
  - assets/resumen_corridas.csv     tabla cruda (run, n, step_final, last10, max)

Uso:
    uv run python3 docs/experimentos/scripts/plot_runs.py

Es la fuente reproducible de los gráficos del informe; si llegan corridas nuevas,
basta con re-ejecutarlo. El mapeo run->etiqueta/fase está en GROUPS más abajo.
"""

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ASSETS = os.path.join(ROOT, "docs", "experimentos", "assets")
FLOOR = -1001  # piso del time-limit (goal nunca alcanzado)

# run-substring -> (etiqueta legible, fase). El primero que matchea gana.
GROUPS = [
    ("goal_dreamer_with_text/03", ("F1 · WM-goal seed3", "Fase 1: el setup funciona")),
    ("goal_dreamer_with_text/04", ("F1 · WM-goal seed4", "Fase 1: el setup funciona")),
    ("goal_dreamer_with_text/01", ("F1 · WM-goal seed0 (corta)", "Fase 1: el setup funciona")),
    ("text_goal_sample_her_buffer", ("F2 · texto · HER · 32x16", "Fase 2: goal desde texto")),
    ("text_goal_sample_normal_buffer", ("F2 · texto · normal · 32x16", "Fase 2: goal desde texto")),
    ("text_goal_sample_8x8_goal_her_buffer", ("F2 · texto · HER · 8x8", "Fase 2: goal desde texto")),
    ("text_goal_sample_8x8_normal_buffer", ("F2 · texto · normal · 8x8", "Fase 2: goal desde texto")),
    ("post_train_from_wm_only/normal_buffer_goals", ("F3 · fixed · WM cong · normal · buffer", "Fase 3: aislar WM vs politica")),
    ("post_train_from_wm_only/her_buffer_goals", ("F3 · fixed · WM cong · HER · buffer", "Fase 3: aislar WM vs politica")),
    ("post_train_from_wm_only/herbuf_textgoal", ("F3 · fixed · WM cong · HER · texto", "Fase 3: aislar WM vs politica")),
    ("post_train_from_wm_only/normalbuf_randomgoal", ("F3 · fixed · WM cong · normal · random", "Fase 3: aislar WM vs politica")),
    ("post_train_from_distill", ("F3 · post-train desde destilado", "Fase 3: aislar WM vs politica")),
    ("random_goal/posttrain_frozenwm_normalbuf_goalbuf", ("RG · WM cong · normal · buffer", "random_goal vs fixed_goal")),
    ("random_goal/posttrain_frozenwm_herbuf_goalbuf", ("RG · WM cong · HER · buffer", "random_goal vs fixed_goal")),
    ("random_goal/posttrain_frozenwm_normalbuf_goalimag", ("RG · imagination · full", "random_goal vs fixed_goal")),
]

# Corridas con goal_type=prob: el score logueado es la SUMA de la recompensa de
# probabilidad in [0,1] -> piso 0, mas alto es mejor (escala distinta a las de
# arriba, cuyo piso es -1001). Van en su propia figura para no mezclar escalas.
PROB_GROUPS = [
    ("post_train_from_wm_only/04_imag_prob", "fixed_goal · imagination · prob · normal"),
    ("post_train_from_wm_only/05_imag_prob_her", "fixed_goal · imagination · prob · HER"),
    ("random_goal/posttrain_frozenwm_normalbuf_goalimag_prob", "random_goal · imagination · prob · normal"),
    ("random_goal/posttrain_frozenwm_herbuf_goalimag_prob", "random_goal · imagination · prob · HER"),
]


def load_series(event_file):
    ea = EventAccumulator(event_file, size_guidance={"scalars": 100000})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    for tag in ("episode/score", "episode/eval_score"):
        if tag in tags:
            ev = ea.Scalars(tag)
            if ev:
                return np.array([(e.step, e.value) for e in ev])
    return None


def classify(run):
    for sub, (label, phase) in GROUPS:
        if sub in run:
            return label, phase
    return None, None


def smooth(y, k=15):
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def main():
    os.makedirs(ASSETS, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "logdir", "**", "events.out.tfevents*"), recursive=True))
    runs = {}  # run -> series (concatena multiples event files de la misma corrida)
    for f in files:
        run = "/".join(os.path.relpath(f, os.path.join(ROOT, "logdir")).split("/")[:-1])
        s = load_series(f)
        if s is None or len(s) == 0:
            continue
        runs.setdefault(run, []).append(s)

    summary = []
    for run, parts in runs.items():
        s = np.concatenate(parts)
        s = s[np.argsort(s[:, 0])]
        summary.append((run, len(s), int(s[-1, 0]), float(s[-10:, 1].mean()), float(s[:, 1].max())))

    summary.sort(key=lambda r: r[0])
    with open(os.path.join(ASSETS, "resumen_corridas.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "n_puntos", "step_final", "score_last10", "score_max"])
        w.writerows(summary)

    # --- figura 1: curvas por fase ---
    phases = []
    for _, (_, ph) in GROUPS:
        if ph not in phases:
            phases.append(ph)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, phase in zip(axes.flat, phases):
        for sub, (label, ph) in GROUPS:
            if ph != phase:
                continue
            run = next((r for r in runs if sub in r), None)
            if run is None:
                continue
            s = np.concatenate(runs[run])
            s = s[np.argsort(s[:, 0])]
            x, y = s[:, 0], s[:, 1]
            ys = smooth(y)
            xs = x[len(x) - len(ys):]
            ax.plot(xs, ys, label=label, lw=1.6)
        ax.axhline(FLOOR, ls="--", c="grey", lw=1, label="piso -1001")
        ax.set_title(phase)
        ax.set_xlabel("step")
        ax.set_ylabel("score (media movil 15)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
    fig.suptitle("Curvas de entrenamiento por fase (score vs step)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "curvas_por_fase.png"), dpi=110)
    plt.close(fig)

    # --- figura 2: barras de score final ---
    bars = []
    for sub, (label, _) in GROUPS:
        run = next((r for r in runs if sub in r), None)
        if run is None:
            continue
        s = np.concatenate(runs[run])
        bars.append((label, float(s[:, 1][np.argsort(s[:, 0])][-10:].mean())))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(bars))))
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    colors = ["#2ca02c" if v > -600 else "#d62728" for v in vals]
    ax.barh(range(len(bars)), vals, color=colors)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(FLOOR, ls="--", c="grey", label="piso -1001")
    ax.axvline(-500, ls=":", c="black", label="umbral aprendizaje ~-500")
    # etiqueta numerica al final de cada barra (clave para el 0 de RG-imag-prob)
    for i, v in enumerate(vals):
        off = 12 if v >= 0 else -12
        ha = "left" if v >= 0 else "right"
        ax.text(v + off, i, f"{v:.0f}", va="center", ha=ha, fontsize=7,
                color=colors[i], fontweight="bold")
    ax.set_xlim(FLOOR - 60, max(vals) + 120)
    ax.invert_yaxis()
    ax.set_xlabel("score final (media ultimos 10 episodios)")
    ax.set_title("Que sirve y que no — reward {-1,0}, piso -1001 (excluye runs 'prob')")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "score_final_barras.png"), dpi=110)
    plt.close(fig)

    # --- figura 3: runs con goal_type=prob (escala propia, piso 0) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for sub, label in PROB_GROUPS:
        run = next((r for r in runs if sub in r), None)
        if run is None:
            continue
        s = np.concatenate(runs[run])
        s = s[np.argsort(s[:, 0])]
        ys = smooth(s[:, 1])
        xs = s[:, 0][len(s) - len(ys):]
        last10 = s[-10:, 1].mean()
        ax.plot(xs, ys, lw=1.8, label=f"{label}  (last10={last10:.0f})")
    ax.axhline(0, ls="--", c="grey", lw=1, label="piso 0 = no aprende (min de prob)")
    ax.set_xlabel("step")
    ax.set_ylabel("score = suma reward prob ∈ [0,1] · step")
    ax.set_title("goal_type=prob + imagination: fixed aprende, random no (piso 0)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "prob_runs.png"), dpi=110)
    plt.close(fig)

    print(f"Figuras y CSV escritos en {ASSETS}")
    for row in summary:
        print(f"{row[0]:62s} n={row[1]:5d} step={row[2]:8d} last10={row[3]:9.1f} max={row[4]:9.1f}")


if __name__ == "__main__":
    main()
