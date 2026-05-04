import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List


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


def plot_runs(
    exp_dirs: Union[Path, List[Path]],
    outdir: Path,
    labels: Union[str, List[str]] = None,
    title: str = "Goal conditioned Dreamer",
):
    if isinstance(exp_dirs, Path):
        exp_dirs = [exp_dirs]
    if labels is None:
        labels = [d.name for d in exp_dirs]
    elif isinstance(labels, str):
        labels = [labels]

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = plt.cm.tab10.colors

    for idx, (exp_dir, label) in enumerate(zip(exp_dirs, labels)):
        exp_dir = exp_dir.expanduser()

        run_dirs = sorted(
            [d for d in exp_dir.iterdir() if (d / "metrics.jsonl").exists()]
        )
        if not run_dirs:
            if (exp_dir / "metrics.jsonl").exists():
                run_dirs = [exp_dir]
            else:
                print(f"No metrics found for {exp_dir}")
                continue

        print(
            f"Encontradas {len(run_dirs)} run(s) para {label}: {[d.name for d in run_dirs]}"
        )

        all_steps, all_scores = [], []
        for run_dir in run_dirs:
            steps, scores = load_run(run_dir)
            if len(steps) > 0:
                all_steps.append(steps)
                all_scores.append(scores)

        if not all_steps:
            continue

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

        color = colors[idx % len(colors)]

        ax.fill_between(
            steps_smooth,
            mean - std,
            mean + std,
            alpha=0.2,
            color=color,
            linewidth=0,
        )
        ax.plot(
            steps_smooth,
            mean,
            color=color,
            linewidth=1.8,
            label=f"{label} (n={len(run_dirs)})",
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
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Plot experiment runs")
    parser.add_argument(
        "--paths",
        "-p",
        nargs="+",
        required=True,
        help="Paths to experiment directories (one or more)",
    )
    parser.add_argument(
        "--labels",
        "-l",
        nargs="+",
        help="Labels for each provided path (optional)",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default="plots",
        help="Output directory for saved figures",
    )
    parser.add_argument(
        "--title",
        "-t",
        default="Reward vs steps",
        help="Plot title",
    )

    args = parser.parse_args()

    exp_dirs = [Path(p) for p in args.paths]
    labels = args.labels
    if labels is None:
        labels = [d.name for d in exp_dirs]
    elif len(labels) != len(exp_dirs):
        print(
            "Warning: number of labels doesn't match number of paths; adjusting",
            file=sys.stderr,
        )
        if len(labels) < len(exp_dirs):
            labels = labels + [d.name for d in exp_dirs[len(labels) :]]
        else:
            labels = labels[: len(exp_dirs)]

    outdir = Path(args.outdir)

    plot_runs(exp_dirs=exp_dirs, outdir=outdir, labels=labels, title=args.title)
