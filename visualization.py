import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_run(run_dir: Path):
    records = []
    for line in (run_dir / "metrics.jsonl").read_text().strip().split("\n"):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    steps = [r["step"] for r in records if "episode/score" in r]
    scores = [r["episode/score"] for r in records if "episode/score" in r]
    return np.array(steps), np.array(scores)


def moving_avg(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_runs(exp_dir: Path, outdir: Path, title: str = "Goal conditioned Dreamer"):
    exp_dir = exp_dir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([d for d in exp_dir.iterdir() if (d / "metrics.jsonl").exists()])
    if not run_dirs:
        run_dirs = [exp_dir]

    print(f"Encontradas {len(run_dirs)} run(s): {[d.name for d in run_dirs]}")

    all_steps, all_scores = [], []
    for run_dir in run_dirs:
        steps, scores = load_run(run_dir)
        all_steps.append(steps)
        all_scores.append(scores)

    for i, s in enumerate(all_steps[1:], 1):
        assert np.array_equal(
            all_steps[0], s
        ), f"Run {run_dirs[i].name} tiene steps distintos a run {run_dirs[0].name}"

    steps = all_steps[0]

    # Suavizado por run, luego agregación
    window = max(1, len(steps) // 30)
    smoothed = np.stack(
        [moving_avg(scores, window) for scores in all_scores]
    )  # (n_runs, n_points_smooth)
    steps_smooth = steps[window - 1 :]
    mean = smoothed.mean(axis=0)
    std = smoothed.std(axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))

    # for scores in all_scores:
    #     ax.plot(steps, scores, alpha=0.08, color="#0022ff", linewidth=0.6)

    ax.fill_between(
        steps_smooth,
        mean - std,
        mean + std,
        alpha=0.2,
        color="#0022ff",
        linewidth=0,
    )
    ax.plot(
        steps_smooth,
        mean,
        color="#0022ff",
        linewidth=1.8,
        label=f"Mean ± Std (n={len(run_dirs)})",
    )

    ax.set_xlabel("Steps")
    ax.set_ylabel("Episode score")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.7)
    ax.grid(color="#eeeeee")
    fig.tight_layout()

    out_path = outdir / "performance.png"
    fig.savefig(out_path, dpi=150)
    print("Saved:", out_path)


if __name__ == "__main__":
    plot_runs(
        exp_dir=Path("logdir/2855"),
        outdir=Path("plots/2855"),
        title="Goal conditioned Dreamer",
    )
