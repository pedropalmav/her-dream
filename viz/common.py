"""
viz/common.py — Código compartido por las visualizaciones Dash del WM.

Reúne las constantes de acción, los helpers de obs/imagen, la ejecución de
trayectorias, los helpers de figuras Plotly y la carga de checkpoint, para que
tanto `traj_viz_dash.py` (replay con slider) como `interactive_dash.py`
(construcción con flechas) compartan el mismo núcleo sin duplicarlo.
"""

import base64
import io
import pathlib

import numpy as np
import plotly.graph_objects as go
import torch
from dash import html
from gymnasium.utils import seeding
from omegaconf import OmegaConf

from dreamer import Dreamer
from envs import make_env, make_envs
from rewards import make_reward

# ─────────────────────────────────────────────────────────────────────────────
# Acción constants
# ─────────────────────────────────────────────────────────────────────────────
ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2

ACT_NAMES = {
    None: "reset",
    0: "turn_left",
    1: "turn_right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}
ACT_SYMBOLS = {
    None: "●",
    0: "←",
    1: "→",
    2: "↑",
    3: "P",
    4: "D",
    5: "T",
    6: "✓",
}
ACT_COLORS = {
    None: "#9E9E9E",
    0: "#2196F3",
    1: "#FF9800",
    2: "#4CAF50",
    3: "#9C27B0",
    4: "#F44336",
    5: "#00BCD4",
    6: "#795548",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers obs / imagen
# ─────────────────────────────────────────────────────────────────────────────


def _preprocess_obs(obs: dict, device: str) -> dict:
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


def _onehot(idx: int, n: int) -> np.ndarray:
    a = np.zeros(n, dtype=np.float32)
    a[idx] = 1.0
    return a


def _arr_to_b64(arr: np.ndarray) -> str:
    """numpy uint8 H×W×3 → data URI PNG (PIL si disponible, si no matplotlib)."""
    try:
        from PIL import Image

        img = Image.fromarray(arr.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    except ImportError:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(2, 2), dpi=64)
        ax.imshow(arr.astype(np.uint8))
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Correr trayectoria y guardar todos los pasos
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def collect_trajectory(
    agent,
    env_factory,
    actions: list[int],
    device: str,
    seed: int = 0,
) -> dict:
    """
    Ejecuta las acciones en un env fresco y devuelve arrays crudos por paso.

    El env es determinista dado el seed, así que re-ejecutar la misma lista de
    acciones reproduce exactamente la misma trayectoria (esto es lo que permite
    al builder interactivo recalcular todo en cada tecla y soportar undo, y que
    `save_trajectory` reconstruya idéntico lo que se ve en pantalla).

    En cada paso se registra tanto el **posterior** `p(z | deter, obs)` como el
    **prior** `p(z | deter)`. Comparten el mismo `deter` (la transición
    determinista no usa la observación), así que el prior es la predicción
    one-step del world model *antes* de ver la imagen, y el posterior su
    corrección tras observarla.

    Retorna (numpy crudo, sin serializar)
    -------------------------------------
    {
      "images":      list[np.uint8 (H,W,3)],
      "post_probs":  list[np.float32 (S,K)],   # posterior  p(k|s)
      "post_z":      list[np.float32 (S,K)],   # stoch muestreado (aprox one-hot)
      "prior_probs": list[np.float32 (S,K)],   # prior      p(k|s)
      "prior_z":     list[np.float32 (S,K)],
      "actions":     list[int | None],         # None en t=0 (reset)
      "done":        bool,
      "T":           int,
    }
    """
    env = env_factory()
    base = env
    while hasattr(base, "env"):
        base = base.env
    base._np_random, _ = seeding.np_random(seed)
    obs = env.reset()

    n_act = env.action_space.shape[0]
    stoch, deter = agent.rssm.initial(1)
    prev_action = torch.zeros(1, n_act, device=device)

    out: dict = {
        "images": [],
        "post_probs": [],
        "post_z": [],
        "prior_probs": [],
        "prior_z": [],
        "actions": [],
    }
    ended = False

    def _record(stoch, deter, post_logit, img_arr, action):
        prior_z, prior_logit = agent.rssm.prior(deter)
        post_probs = agent.rssm.get_dist(post_logit).base_dist.probs
        prior_probs = agent.rssm.get_dist(prior_logit).base_dist.probs
        out["post_probs"].append(post_probs.squeeze(0).cpu().numpy())
        out["post_z"].append(stoch.squeeze(0).cpu().numpy())
        out["prior_probs"].append(prior_probs.squeeze(0).cpu().numpy())
        out["prior_z"].append(prior_z.squeeze(0).cpu().numpy())
        out["images"].append(np.array(img_arr, dtype=np.uint8))
        out["actions"].append(action)

    # t = 0: obs del reset
    obs_t = _preprocess_obs(obs, device)
    embed = agent.encoder(obs_t)
    stoch, deter, logit = agent.rssm.obs_step(
        stoch,
        deter,
        prev_action,
        embed,
        torch.tensor([True], dtype=torch.bool, device=device),
    )
    _record(stoch, deter, logit, obs["image"], None)

    for act_idx in actions:
        act_np = _onehot(act_idx, n_act)
        obs, _, done, _ = env.step(act_np)
        prev_action = torch.as_tensor(act_np, device=device).unsqueeze(0)

        obs_t = _preprocess_obs(obs, device)
        embed = agent.encoder(obs_t)
        stoch, deter, logit = agent.rssm.obs_step(
            stoch,
            deter,
            prev_action,
            embed,
            torch.tensor([False], dtype=torch.bool, device=device),
        )
        _record(stoch, deter, logit, obs["image"], act_idx)

        if done:
            print(f"  [warn] episodio terminó en paso {len(out['actions']) - 1}")
            ended = True
            break

    out["done"] = ended
    out["T"] = len(out["actions"]) - 1
    return out


def run_trajectory(
    agent,
    env_factory,
    actions: list[int],
    device: str,
    seed: int = 0,
) -> dict:
    """
    Wrapper de `collect_trajectory` serializado para los `dcc.Store` de Dash.

    Mantiene las claves históricas (`probs`/`zs` = posterior) y añade el prior
    (`prior_probs`/`prior_zs`). Las imágenes se devuelven como data URIs PNG.
    """
    raw = collect_trajectory(agent, env_factory, actions, device, seed)
    return {
        "probs": [p.tolist() for p in raw["post_probs"]],
        "zs": [z.tolist() for z in raw["post_z"]],
        "prior_probs": [p.tolist() for p in raw["prior_probs"]],
        "prior_zs": [z.tolist() for z in raw["prior_z"]],
        "images": [_arr_to_b64(im) for im in raw["images"]],
        "actions": raw["actions"],
        "done": raw["done"],
        "T": raw["T"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Guardar trayectoria (acciones, estados, frames, gif)
# ─────────────────────────────────────────────────────────────────────────────


def _safe_name(name: str | None) -> str:
    """Sanitiza un nombre para usarlo como carpeta (sin separadores ni espacios)."""
    if not name:
        return ""
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in str(name).strip()]
    return "".join(keep).strip("._") or ""


def _write_gif(path, raw: dict, names: dict | None = None, fps: float = 2.0) -> None:
    """
    Escribe un GIF con, por cada paso: la imagen (con la acción justo debajo),
    el heatmap del `z prior` y el del `z posterior`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    names = names or ACT_NAMES
    T = raw["T"]
    frames: list = []

    for t in range(T + 1):
        img = raw["images"][t]
        act = raw["actions"][t]
        prior = raw["prior_probs"][t]  # (S, K)
        post = raw["post_probs"][t]  # (S, K)

        fig, axes = plt.subplots(
            1, 3, figsize=(9.0, 4.0), dpi=100, gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]}
        )
        ax_img, ax_pr, ax_po = axes

        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(f"t = {t} / {T}", fontsize=11)
        # La acción, justo debajo de la imagen.
        ax_img.set_xlabel(names.get(act, "?"), fontsize=13, fontweight="bold", color="#222", labelpad=8)

        for ax, mat, ttl in ((ax_pr, prior, "z prior  p(k|s)"), (ax_po, post, "z posterior  p(k|s)")):
            im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_title(ttl, fontsize=10)
            ax.set_xlabel("clase k", fontsize=9)
            ax.set_ylabel("slot s", fontsize=9)
        fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.02)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(Image.fromarray(buf[..., :3].copy()))
        plt.close(fig)

    duration_ms = int(round(1000.0 / max(fps, 0.1)))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def save_trajectory(
    agent,
    env_factory,
    actions: list[int],
    device: str,
    out_root,
    seed: int = 0,
    name: str | None = None,
    names: dict | None = None,
    fps: float = 2.0,
) -> pathlib.Path:
    """
    Re-ejecuta la trayectoria y la guarda en `<out_root>/trajectories/<name>/`.

    Genera, de forma que sea reconstruible y analizable:
      - `actions.json`     — secuencia de acciones (índices + nombres) y metadatos.
      - `metadata.json`    — name, seed, logdir, T, done, timestamp, action map.
      - `arrays.npz`       — prior/posterior probs y z, acciones (-1 = reset).
      - `frames/step_*.png`— cada observación cruda.
      - `trajectory.gif`   — imagen + acción + z prior + z posterior por paso.

    `actions` es la lista de índices del camino (sin el reset). `names` es el
    diccionario de nombres de acción del entorno (Crafter, MiniGrid, …).

    Retorna el `Path` de la carpeta creada.
    """
    import datetime
    import json

    from PIL import Image

    names = names or ACT_NAMES
    raw = collect_trajectory(agent, env_factory, actions, device, seed)

    name = _safe_name(name) or f"traj_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out_dir = pathlib.Path(out_root) / "trajectories" / name
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Acciones como enteros (-1 marca el reset en t=0).
    act_idx = [(-1 if a is None else int(a)) for a in raw["actions"]]
    act_names = [names.get(a, "reset" if a is None else "?") for a in raw["actions"]]

    # Frames PNG crudos.
    for t, img in enumerate(raw["images"]):
        Image.fromarray(img).save(frames_dir / f"step_{t:03d}.png")

    # Arrays para reconstrucción / análisis.
    np.savez_compressed(
        out_dir / "arrays.npz",
        post_probs=np.stack(raw["post_probs"]).astype(np.float32),
        post_z=np.stack(raw["post_z"]).astype(np.float32),
        prior_probs=np.stack(raw["prior_probs"]).astype(np.float32),
        prior_z=np.stack(raw["prior_z"]).astype(np.float32),
        images=np.stack(raw["images"]).astype(np.uint8),
        actions=np.array(act_idx, dtype=np.int64),
    )

    # Acciones (el camino reproducible: lista sin el reset inicial).
    with open(out_dir / "actions.json", "w") as f:
        json.dump(
            {
                "actions": [int(a) for a in actions],
                "actions_named": [names.get(int(a), "?") for a in actions],
                "per_step_actions": act_idx,
                "per_step_actions_named": act_names,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    metadata = {
        "name": name,
        "seed": int(seed),
        "logdir": str(out_root),
        "T": int(raw["T"]),
        "num_actions": len(actions),
        "done": bool(raw["done"]),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "image_shape": list(raw["images"][0].shape),
        "stoch_shape": list(raw["post_probs"][0].shape),
        "action_names": {str(k): v for k, v in names.items()},
        "fps": fps,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    _write_gif(out_dir / "trajectory.gif", raw, names=names, fps=fps)

    print(f"Trayectoria guardada → {out_dir}")
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Helpers para figuras Plotly
# ─────────────────────────────────────────────────────────────────────────────

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=50, r=20, t=30, b=50),
    font=dict(family="Arial, sans-serif", size=11),
)


def _empty_fig(msg: str = "Run a trajectory first") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        xaxis_visible=False,
        yaxis_visible=False,
        annotations=[
            {
                "text": msg,
                "showarrow": False,
                "font": {"size": 13, "color": "#888"},
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
            }
        ],
    )
    return fig


def _probs_fig(probs_sk: np.ndarray) -> go.Figure:
    """probs_sk: (S, K) → heatmap de probabilidades."""
    fig = go.Figure(
        go.Heatmap(
            z=probs_sk,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(title="p(k|s)", thickness=14, len=0.85, tickfont=dict(size=10)),
        )
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Posterior  p(k | slot s)", font=dict(size=12)),
        xaxis=dict(title="Clase k", tickfont=dict(size=9)),
        yaxis=dict(title="Slot s", autorange="reversed", tickfont=dict(size=9)),
    )
    return fig


def _z_fig(zs_sk: np.ndarray) -> go.Figure:
    """zs_sk: (S, K) aprox one-hot → heatmap de valores reales del stoch."""
    S, K = zs_sk.shape
    fig = go.Figure(
        go.Heatmap(
            z=zs_sk,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar=dict(title="z value", thickness=14, len=0.85, tickfont=dict(size=10)),
        )
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Stoch z  (aprox one-hot)", font=dict(size=12)),
        xaxis=dict(title="Clase k", tickfont=dict(size=9)),
        yaxis=dict(title="Slot s", autorange="reversed", tickfont=dict(size=9)),
    )
    return fig


def _action_strip(
    actions: list,
    current_t: int,
    clickable: bool = False,
    symbols: dict | None = None,
    colors: dict | None = None,
    names: dict | None = None,
) -> list:
    """
    Lista de Div para el strip de acciones, resaltando current_t.

    Si `clickable=True`, cada celda lleva un id pattern-matching
    {"type": "step", "index": i} y `n_clicks`, para poder seleccionar
    cualquier paso del historial al hacer click.

    `symbols` / `colors` / `names` permiten inyectar la metadata de acciones de
    otro entorno (p.ej. Crafter, con 17 acciones). Por defecto usa las de MiniGrid.
    """
    symbols = symbols or ACT_SYMBOLS
    colors = colors or ACT_COLORS
    names = names or ACT_NAMES
    items = []
    for i, act in enumerate(actions):
        sym = symbols.get(act, "?")
        color = colors.get(act, "#888")
        is_cur = i == current_t
        cell_kwargs = dict(
            title=f"t={i}  {names.get(act, '')}",
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "justifyContent": "center",
                "width": "38px",
                "height": "52px",
                "background": color if is_cur else "#f5f5f5",
                "color": "white" if is_cur else "#333",
                "border": f"2px solid {color}",
                "borderRadius": "6px",
                "flexShrink": "0",
                "fontWeight": "bold" if is_cur else "normal",
                "boxShadow": "0 2px 6px rgba(0,0,0,.25)" if is_cur else "none",
                "transition": "all .15s",
                "cursor": "pointer" if clickable else "default",
            },
        )
        if clickable:
            cell_kwargs["id"] = {"type": "step", "index": i}
            cell_kwargs["n_clicks"] = 0
        items.append(
            html.Div(
                [
                    html.Div(sym, style={"fontSize": "18px", "lineHeight": "1"}),
                    html.Div(str(i), style={"fontSize": "9px", "color": "#bbb" if is_cur else "#888"}),
                ],
                **cell_kwargs,
            )
        )
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Carga de checkpoint
# ─────────────────────────────────────────────────────────────────────────────


def load_agent(logdir, device: str | None = None):
    """
    Carga config + checkpoint de un run y construye el agente en modo eval.

    Soporta tanto los runs goal-conditioned (RandomGoal, …) como world models
    "vanilla" sin objetivo (p.ej. Crafter, `logdir/original_wm_crafter/*`). Para
    estos últimos el checkpoint no trae las claves de goal en su config y su
    actor/critic tienen otra dimensión de entrada (sin concatenar el goal), así
    que se rellenan defaults y se cargan solo los tensores cuya forma coincide.
    El visualizador únicamente usa `encoder` + `rssm`, de modo que basta con que
    esos módulos del world model carguen correctamente.

    Retorna `(agent, env_factory, device)`.
    """
    logdir = pathlib.Path(logdir)
    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")
    device = device or config.device
    config.device = device
    OmegaConf.set_struct(config, False)
    OmegaConf.set_struct(config.model, False)
    OmegaConf.set_struct(config.env, False)
    config.env.device = device

    # Detecta un checkpoint sin objetivo (Crafter vanilla, etc.): le faltan las
    # claves de goal que `Dreamer` espera. Rellena defaults inocuos para que el
    # constructor no falle; el reward goal-conditioned no se usa en el viz.
    has_goal = "goal_type" in config.model
    model_defaults = {
        "wm_only": False,
        "goal_type": "first_row",
        "mission_text": False,
        "prob_threshold": 0.5,
        "prob_threshold_step": 0.05,
    }
    for key, val in model_defaults.items():
        if key not in config.model:
            config.model[key] = val

    reward_fn = make_reward(config) if "goal_type" in config else None
    _, _, obs_space, act_space = make_envs(config.env)

    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
        reward_function=reward_fn,
    ).to(device)

    ckpt = torch.load(logdir / "latest.pt", map_location=device)
    state = ckpt["agent_state_dict"]
    own = agent.state_dict()
    # Carga solo los tensores presentes y con forma coincidente: descarta
    # módulos extra del checkpoint (_frozen_*) y actor/critic con otra dimensión
    # de goal. Lo que importa para el viz —encoder y rssm— sí coincide.
    loadable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    agent.load_state_dict(loadable, strict=False)

    wm_missing = [
        k for k in own if (k.startswith("encoder.") or k.startswith("rssm.")) and k not in loadable
    ]
    if wm_missing:
        print(f"[warn] {len(wm_missing)} claves del world model NO se cargaron, p.ej. {wm_missing[:3]}")
    agent.eval()
    skipped = len(state) - len(loadable)
    tag = "" if has_goal else "  (WM vanilla sin goal)"
    print(f"Checkpoint cargado → {logdir / 'latest.pt'}  [{len(loadable)} tensores, {skipped} omitidos]{tag}")

    def env_factory():
        return make_env(config.env, 0)

    return agent, env_factory, device
