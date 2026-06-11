"""
viz/replay_check.py — Chequeo de determinismo de una trayectoria guardada.

Toma una trayectoria guardada por `save_trajectory` (p.ej. la del builder de
`crafter_dash.py`), re-ejecuta **exactamente la misma secuencia de acciones**
N veces, cada vez en un env fresco con la misma semilla, y genera un GIF con
una grilla (referencia guardada + las N ejecuciones) donde el borde de cada
panel es verde si el frame coincide exactamente con la referencia en ese paso
y rojo si difiere. También imprime, por ejecución, el primer paso de
divergencia (si lo hay).

Nota sobre semillas: la semilla efectiva de Crafter es `config.seed + id`
(fijada al construir `crafter.Env`, ver `envs/__init__.py::make_env`). El
`seed` del metadata solo setea el `_np_random` de gymnasium en el wrapper
interno (el mismo hack que usa `collect_trajectory`), que Crafter ignora; se
replica igualmente para reproducir el procedimiento original al pie de la letra.

Uso:
    uv run python -m viz.replay_check \
        --traj-dir logdir/original_wm_crafter/02/trajectories/Pareciera_funcionar \
        --runs 10 --fps 8
"""

import argparse
import json
import math
import pathlib

import numpy as np
from gymnasium.utils import seeding
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from envs import make_env
from viz.common import _onehot

# Layout del GIF
_SCALE = 2  # 64×64 → 128×128
_LABEL_H = 16  # franja de etiqueta sobre cada panel
_BORDER = 2  # grosor del borde verde/rojo
_PAD = 4  # separación entre paneles
_HEADER_H = 22  # barra superior con t / acción / divergencias
_GREEN = (46, 125, 50)
_RED = (198, 40, 40)
_BG = (250, 250, 250)


def replay(config, actions: list[int], seed: int, n_steps: int) -> tuple[np.ndarray, int | None]:
    """
    Ejecuta `actions` en un env fresco. Devuelve `(frames, early_done)` con
    `frames` de forma (n_steps+1, H, W, 3); si el episodio termina antes de
    consumir todas las acciones, rellena con el último frame y reporta el paso.
    """
    env = make_env(config.env, 0)
    base = env
    while hasattr(base, "env"):
        base = base.env
    base._np_random, _ = seeding.np_random(seed)

    n_act = env.action_space.shape[0]
    obs = env.reset()
    frames = [np.array(obs["image"], dtype=np.uint8)]
    early_done = None

    for t, act in enumerate(actions, start=1):
        obs, _, done, _ = env.step(_onehot(int(act), n_act))
        frames.append(np.array(obs["image"], dtype=np.uint8))
        if done and t < n_steps:
            early_done = t
            break

    while len(frames) < n_steps + 1:
        frames.append(frames[-1].copy())
    return np.stack(frames), early_done


def _panel(img: np.ndarray, label: str, ok: bool) -> Image.Image:
    """Frame 64×64 → panel con etiqueta arriba y borde verde/rojo."""
    h, w = img.shape[:2]
    pw, ph = w * _SCALE + 2 * _BORDER, h * _SCALE + 2 * _BORDER + _LABEL_H
    color = _GREEN if ok else _RED
    panel = Image.new("RGB", (pw, ph), color)
    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, pw - 1, _LABEL_H - 1], fill=color)
    draw.text((4, 2), label, fill="white")
    big = Image.fromarray(img).resize((w * _SCALE, h * _SCALE), Image.NEAREST)
    panel.paste(big, (_BORDER, _LABEL_H + _BORDER))
    return panel


def make_grid_gif(
    out_path: pathlib.Path,
    ref_frames: np.ndarray,
    runs_frames: list[np.ndarray],
    actions_named: list[str],
    fps: float,
    cols: int = 4,
) -> None:
    """GIF: por paso, grilla [referencia + runs] con bordes según coincidencia."""
    n_panels = 1 + len(runs_frames)
    rows = math.ceil(n_panels / cols)
    T = ref_frames.shape[0] - 1

    sample = _panel(ref_frames[0], "ref", True)
    pw, ph = sample.size
    W = cols * pw + (cols + 1) * _PAD
    H = _HEADER_H + rows * ph + (rows + 1) * _PAD

    gif_frames = []
    for t in range(T + 1):
        canvas = Image.new("RGB", (W, H), _BG)
        draw = ImageDraw.Draw(canvas)

        matches = [np.array_equal(fr[t], ref_frames[t]) for fr in runs_frames]
        n_diff = sum(not m for m in matches)
        header = f"t = {t} / {T}    accion: {actions_named[t]}    runs distintos a ref: {n_diff}/{len(runs_frames)}"
        draw.text((_PAD + 2, 5), header, fill=(40, 40, 40))

        panels = [("ref", ref_frames[t], True)] + [
            (f"run {i + 1}", fr[t], matches[i]) for i, fr in enumerate(runs_frames)
        ]
        for idx, (label, img, ok) in enumerate(panels):
            r, c = divmod(idx, cols)
            x = _PAD + c * (pw + _PAD)
            y = _HEADER_H + _PAD + r * (ph + _PAD)
            canvas.paste(_panel(img, label, ok), (x, y))
        gif_frames.append(canvas)

    duration_ms = int(round(1000.0 / max(fps, 0.1)))
    gif_frames[0].save(
        out_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def main():
    parser = argparse.ArgumentParser(description="Re-ejecuta N veces las acciones de una trayectoria guardada y compara frames.")
    parser.add_argument(
        "--traj-dir",
        default="logdir/original_wm_crafter/02/trajectories/Pareciera_funcionar",
        help="Carpeta de la trayectoria (actions.json + metadata.json + arrays.npz)",
    )
    parser.add_argument("--runs", type=int, default=10, help="Número de re-ejecuciones")
    parser.add_argument("--fps", type=float, default=8.0, help="FPS del GIF")
    parser.add_argument("--out", default=None, help="Ruta del GIF de salida (default: <traj-dir>/replays_x<N>.gif)")
    args = parser.parse_args()

    traj_dir = pathlib.Path(args.traj_dir)
    meta = json.load(open(traj_dir / "metadata.json"))
    acts = json.load(open(traj_dir / "actions.json"))
    arr = np.load(traj_dir / "arrays.npz")

    actions = [int(a) for a in acts["actions"]]
    seed = int(meta.get("seed", 0))
    logdir = pathlib.Path(meta["logdir"])
    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")

    ref_frames = arr["images"]  # (T+1, H, W, 3)
    n_steps = len(actions)
    names = meta.get("action_names", {})
    actions_named = ["reset"] + [names.get(str(a), str(a)) for a in actions]

    print(f"Trayectoria: {traj_dir.name}  ({n_steps} acciones, seed metadata={seed}, crafter seed={config.seed})")
    print(f"Re-ejecutando {args.runs} veces…")

    runs_frames = []
    for i in range(args.runs):
        frames, early = replay(config, actions, seed, n_steps)
        runs_frames.append(frames)
        eq = np.all([np.array_equal(frames[t], ref_frames[t]) for t in range(n_steps + 1)])
        if eq:
            verdict = "idéntico a la referencia"
        else:
            first = next(t for t in range(n_steps + 1) if not np.array_equal(frames[t], ref_frames[t]))
            verdict = f"DIVERGE de la referencia desde t={first}"
        extra = f"  (done temprano en t={early})" if early is not None else ""
        print(f"  run {i + 1:2d}: {verdict}{extra}")

    # ¿Las N re-ejecuciones son idénticas entre sí?
    all_same = all(np.array_equal(runs_frames[0], fr) for fr in runs_frames[1:])
    print(f"\nLas {args.runs} re-ejecuciones son {'idénticas entre sí' if all_same else 'DISTINTAS entre sí'}.")

    out_path = pathlib.Path(args.out) if args.out else traj_dir / f"replays_x{args.runs}.gif"
    make_grid_gif(out_path, ref_frames, runs_frames, actions_named, args.fps)
    print(f"GIF → {out_path}  ({n_steps + 1} frames)")


if __name__ == "__main__":
    main()
