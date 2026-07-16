"""Output helpers: figure saving, per-experiment output dir, JSON dumping."""

import json
import pathlib

import matplotlib.pyplot as plt


def save_fig(fig, path) -> None:
    """Save a figure at 150 dpi (tight bbox), close it, and print the path."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {path}")


def experiment_outdir(logdir, name: str) -> pathlib.Path:
    """`<logdir>/experiments/<name>`, created if missing."""
    outdir = pathlib.Path(logdir) / "experiments" / name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def dump_json(obj, path) -> None:
    """Write `obj` as indented JSON to `path`."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
