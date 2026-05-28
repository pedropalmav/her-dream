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
import torch
import plotly.graph_objects as go
from dash import html
from gymnasium.utils import seeding
from omegaconf import OmegaConf

from dreamer import Dreamer
from envs import make_env, make_envs
from rewards import make_reward


# ─────────────────────────────────────────────────────────────────────────────
# Acción constants
# ─────────────────────────────────────────────────────────────────────────────
ACT_LEFT    = 0
ACT_RIGHT   = 1
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
def run_trajectory(
    agent,
    env_factory,
    actions: list[int],
    device: str,
    seed: int = 0,
) -> dict:
    """
    Ejecuta las acciones en un env fresco y devuelve todos los pasos.

    El env es determinista dado el seed, así que re-ejecutar la misma lista de
    acciones reproduce exactamente la misma trayectoria (esto es lo que permite
    al builder interactivo recalcular todo en cada tecla y soportar undo).

    Retorna
    -------
    {
      "probs":   list[list[list[float]]],  # (T+1, S, K) — posterior probs
      "zs":      list[list[list[float]]],  # (T+1, S, K) — stoch (aprox one-hot)
      "images":  list[str],                # data URIs PNG
      "actions": list[int | None],         # None en t=0 (reset)
      "done":    bool,                      # True si el episodio terminó antes
      "T":       int                        # índice final = len-1
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
    prev_action  = torch.zeros(1, n_act, device=device)

    probs_all: list = []
    zs_all:     list = []
    imgs_all:   list = []
    act_all:    list = []
    ended = False

    def _record(s, probs, img_arr, action):
        probs_all.append(probs.squeeze(0).cpu().numpy().tolist())
        zs_all.append(s.squeeze(0).cpu().numpy().tolist())
        imgs_all.append(_arr_to_b64(np.array(img_arr)))
        act_all.append(action)

    # t = 0: obs del reset
    obs_t = _preprocess_obs(obs, device)
    embed = agent.encoder(obs_t)
    stoch, deter, logit = agent.rssm.obs_step(
        stoch, deter, prev_action, embed,
        torch.tensor([True], dtype=torch.bool, device=device),
    )
    probs = agent.rssm.get_dist(logit).base_dist.probs
    _record(stoch, probs, obs["image"], None)

    for act_idx in actions:
        act_np = _onehot(act_idx, n_act)
        obs, _, done, _ = env.step(act_np)
        prev_action = torch.as_tensor(act_np, device=device).unsqueeze(0)

        obs_t = _preprocess_obs(obs, device)
        embed = agent.encoder(obs_t)
        stoch, deter, logit = agent.rssm.obs_step(
            stoch, deter, prev_action, embed,
            torch.tensor([False], dtype=torch.bool, device=device),
        )
        probs = agent.rssm.get_dist(logit).base_dist.probs
        _record(stoch, probs, obs["image"], act_idx)

        if done:
            print(f"  [warn] episodio terminó en paso {len(act_all) - 1}")
            ended = True
            break

    return {
        "probs":   probs_all,
        "zs":      zs_all,
        "images":  imgs_all,
        "actions": act_all,
        "done":    ended,
        "T":       len(act_all) - 1,
    }


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
        annotations=[{
            "text": msg, "showarrow": False,
            "font": {"size": 13, "color": "#888"},
            "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
        }],
    )
    return fig


def _probs_fig(probs_sk: np.ndarray) -> go.Figure:
    """probs_sk: (S, K) → heatmap de probabilidades."""
    fig = go.Figure(go.Heatmap(
        z=probs_sk,
        colorscale="Blues",
        zmin=0, zmax=1,
        colorbar=dict(title="p(k|s)", thickness=14, len=0.85,
                      tickfont=dict(size=10)),
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Posterior  p(k | slot s)", font=dict(size=12)),
        xaxis=dict(title="Clase k", tickfont=dict(size=9)),
        yaxis=dict(title="Slot s", autorange="reversed",
                   tickfont=dict(size=9)),
    )
    return fig


def _z_fig(zs_sk: np.ndarray) -> go.Figure:
    """zs_sk: (S, K) aprox one-hot → heatmap de valores reales del stoch."""
    S, K = zs_sk.shape
    fig = go.Figure(go.Heatmap(
        z=zs_sk,
        colorscale="Viridis",
        zmin=0, zmax=1,
        colorbar=dict(title="z value", thickness=14, len=0.85,
                      tickfont=dict(size=10)),
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Stoch z  (aprox one-hot)", font=dict(size=12)),
        xaxis=dict(title="Clase k", tickfont=dict(size=9)),
        yaxis=dict(title="Slot s", autorange="reversed",
                   tickfont=dict(size=9)),
    )
    return fig


def _action_strip(actions: list, current_t: int, clickable: bool = False) -> list:
    """
    Lista de Div para el strip de acciones, resaltando current_t.

    Si `clickable=True`, cada celda lleva un id pattern-matching
    {"type": "step", "index": i} y `n_clicks`, para poder seleccionar
    cualquier paso del historial al hacer click.
    """
    items = []
    for i, act in enumerate(actions):
        sym   = ACT_SYMBOLS.get(act, "?")
        color = ACT_COLORS.get(act, "#888")
        is_cur = (i == current_t)
        cell_kwargs = dict(
            title=f"t={i}  {ACT_NAMES.get(act, '')}",
            style={
                "display":        "flex",
                "flexDirection":  "column",
                "alignItems":     "center",
                "justifyContent": "center",
                "width":          "38px",
                "height":         "52px",
                "background":     color        if is_cur else "#f5f5f5",
                "color":          "white"      if is_cur else "#333",
                "border":         f"2px solid {color}",
                "borderRadius":   "6px",
                "flexShrink":     "0",
                "fontWeight":     "bold"       if is_cur else "normal",
                "boxShadow":      "0 2px 6px rgba(0,0,0,.25)" if is_cur else "none",
                "transition":     "all .15s",
                "cursor":         "pointer"    if clickable else "default",
            },
        )
        if clickable:
            cell_kwargs["id"] = {"type": "step", "index": i}
            cell_kwargs["n_clicks"] = 0
        items.append(
            html.Div(
                [
                    html.Div(sym,   style={"fontSize": "18px", "lineHeight": "1"}),
                    html.Div(str(i), style={"fontSize": "9px", "color": "#bbb"
                                            if is_cur else "#888"}),
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

    Retorna `(agent, env_factory, device)`.
    """
    logdir = pathlib.Path(logdir)
    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")
    device = device or config.device
    config.device = device

    # Compatibilidad con checkpoints antiguos
    if "wm_only" not in config.model:
        OmegaConf.set_struct(config.model, False)
        config.model.wm_only = False

    reward_fn = make_reward(config)
    _, _, obs_space, act_space = make_envs(config.env)

    agent = Dreamer(
        config.model, obs_space, act_space,
        reward_function=reward_fn,
    ).to(device)

    ckpt = torch.load(logdir / "latest.pt", map_location=device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    print(f"Checkpoint cargado → {logdir / 'latest.pt'}")

    env_factory = lambda: make_env(config.env, 0)
    return agent, env_factory, device
