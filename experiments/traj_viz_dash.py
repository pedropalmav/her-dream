"""
experiments/traj_viz_dash.py — Dash app para visualizar trayectorias paso a paso.

Uso:
    python -m experiments.traj_viz_dash \\
        --logdir logdir/wm_only_random_mission/01 \\
        --device cpu \\
        --seed 0 \\
        --port 8050
"""

import argparse
import base64
import io
import pathlib

import numpy as np
import torch
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
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
# Builders de trayectoria
# ─────────────────────────────────────────────────────────────────────────────

def make_actions(exp_type: str, n_spin: int, n_forward: int) -> list[int]:
    """Construye la lista de action_indices según el experimento elegido."""

    def spin(k: int, d: str = "left") -> list[int]:
        return [ACT_LEFT if d == "left" else ACT_RIGHT] * k

    def loop(side: int, turn: str = "right") -> list[int]:
        t = ACT_RIGHT if turn == "right" else ACT_LEFT
        out: list[int] = []
        for _ in range(4):
            out.extend([ACT_FORWARD] * side)
            out.append(t)
        return out

    side = max(1, n_forward)
    k    = max(0, n_spin)

    if exp_type == "spin":
        return spin(k)
    elif exp_type == "loop_right":
        return loop(side, "right")
    elif exp_type == "loop_left":
        return loop(side, "left")
    elif exp_type == "loop_right_and_spin":
        return loop(side, "right") + spin(k)
    elif exp_type == "loop_left_and_spin":
        return loop(side, "left") + spin(k)
    return []


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

    Retorna
    -------
    {
      "logits":  list[list[list[float]]],  # (T+1, S, K) — logits crudos
      "zs":      list[list[list[float]]],  # (T+1, S, K) — stoch (aprox one-hot)
      "images":  list[str],                # data URIs PNG
      "actions": list[int | None],         # None en t=0 (reset)
      "T":       int                       # índice final = len-1
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
            break

    return {
        "probs":   probs_all,
        "zs":      zs_all,
        "images":  imgs_all,
        "actions": act_all,
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


def _action_strip(actions: list, current_t: int) -> list:
    """Lista de Div para el strip de acciones, resaltando current_t."""
    items = []
    for i, act in enumerate(actions):
        sym   = ACT_SYMBOLS.get(act, "?")
        color = ACT_COLORS.get(act, "#888")
        is_cur = (i == current_t)
        items.append(
            html.Div(
                [
                    html.Div(sym,   style={"fontSize": "18px", "lineHeight": "1"}),
                    html.Div(str(i), style={"fontSize": "9px", "color": "#bbb"
                                            if is_cur else "#888"}),
                ],
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
                },
            )
        )
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Dash app
# ─────────────────────────────────────────────────────────────────────────────

_SIDEBAR_W = "260px"
_CTRL_BG   = "#F7F9FC"
_ACCENT    = "#4C72B0"


def _label(text: str) -> html.Label:
    return html.Label(text, style={
        "fontWeight": "600", "fontSize": "12px",
        "color": "#444", "marginBottom": "4px", "display": "block",
    })


def create_app(agent, env_factory, device: str, seed: int) -> Dash:
    app = Dash(__name__, title="Trajectory Visualizer")
    app.layout = html.Div(
        style={
            "fontFamily": "Arial, sans-serif",
            "background": "#FAFAFA",
            "minHeight":  "100vh",
        },
        children=[
            # ── Stores ──────────────────────────────────────────────────────
            dcc.Store(id="traj-store"),

            # ── Header ──────────────────────────────────────────────────────
            html.Div(
                "🧠  Trajectory Visualizer",
                style={
                    "background":  _ACCENT,
                    "color":       "white",
                    "padding":     "10px 24px",
                    "fontSize":    "18px",
                    "fontWeight":  "bold",
                    "letterSpacing": ".5px",
                },
            ),

            # ── Body ─────────────────────────────────────────────────────
            html.Div(
                style={"display": "flex", "height": "calc(100vh - 44px)"},
                children=[

                    # ── Sidebar controls ──────────────────────────────────
                    html.Div(
                        style={
                            "width":        _SIDEBAR_W,
                            "minWidth":     _SIDEBAR_W,
                            "background":   _CTRL_BG,
                            "borderRight":  "1px solid #DDE2EA",
                            "padding":      "20px 16px",
                            "overflowY":    "auto",
                            "display":      "flex",
                            "flexDirection":"column",
                            "gap":          "18px",
                        },
                        children=[
                            html.Div([
                                _label("Experiment type"),
                                dcc.Dropdown(
                                    id="exp-type",
                                    options=[
                                        {"label": "⟳  Spin",          "value": "spin"},
                                        {"label": "□→  Loop Right",    "value": "loop_right"},
                                        {"label": "←□  Loop Left",     "value": "loop_left"},
                                        {"label": "□+⟳  Loop Left + Spin", "value": "loop_left_and_spin"},
                                        {"label": "□+⟳  Loop Right + Spin", "value": "loop_right_and_spin"},
                                    ],
                                    value="spin",
                                    clearable=False,
                                    style={"fontSize": "13px"},
                                ),
                            ]),
                            html.Div([
                                _label("Número de spins  (k)"),
                                dcc.Input(
                                    id="n-spin",
                                    type="number",
                                    value=4,
                                    min=0, max=256, step=1,
                                    style={
                                        "width": "100%", "padding": "6px 10px",
                                        "border": "1px solid #CDD2DA",
                                        "borderRadius": "5px", "fontSize": "14px",
                                    },
                                ),
                                html.Div(
                                    "Múltiplo de 4 para volver a la dir. original",
                                    style={"fontSize": "10px", "color": "#888",
                                           "marginTop": "3px"},
                                ),
                            ]),
                            html.Div([
                                _label("Número de forwards  (lado del loop)"),
                                dcc.Input(
                                    id="n-forward",
                                    type="number",
                                    value=2,
                                    min=1, max=20, step=1,
                                    style={
                                        "width": "100%", "padding": "6px 10px",
                                        "border": "1px solid #CDD2DA",
                                        "borderRadius": "5px", "fontSize": "14px",
                                    },
                                ),
                            ]),
                            html.Button(
                                "▶  Run",
                                id="run-btn",
                                n_clicks=0,
                                style={
                                    "width":          "100%",
                                    "padding":        "10px 0",
                                    "background":     _ACCENT,
                                    "color":          "white",
                                    "border":         "none",
                                    "borderRadius":   "6px",
                                    "fontSize":       "15px",
                                    "fontWeight":     "bold",
                                    "cursor":         "pointer",
                                    "letterSpacing":  ".4px",
                                },
                            ),
                            # Status / info
                            html.Div(
                                id="status-msg",
                                style={
                                    "fontSize":     "12px",
                                    "color":        "#555",
                                    "background":   "white",
                                    "border":       "1px solid #DDE2EA",
                                    "borderRadius": "5px",
                                    "padding":      "8px 10px",
                                    "minHeight":    "36px",
                                },
                            ),
                            # Legend
                            html.Div(
                                [
                                    html.Div("Símbolos de acción",
                                             style={"fontWeight": "bold",
                                                    "fontSize": "11px",
                                                    "color": "#666",
                                                    "marginBottom": "6px"}),
                                    *[
                                        html.Div(
                                            [
                                                html.Span(
                                                    ACT_SYMBOLS[k],
                                                    style={
                                                        "display":       "inline-block",
                                                        "width":         "24px",
                                                        "textAlign":     "center",
                                                        "background":    ACT_COLORS[k],
                                                        "color":         "white",
                                                        "borderRadius":  "3px",
                                                        "marginRight":   "6px",
                                                        "fontSize":      "13px",
                                                    },
                                                ),
                                                html.Span(ACT_NAMES[k],
                                                          style={"fontSize": "11px",
                                                                 "color": "#444"}),
                                            ],
                                            style={"marginBottom": "3px",
                                                   "display": "flex",
                                                   "alignItems": "center"},
                                        )
                                        for k in [None, 0, 1, 2]
                                    ],
                                ],
                                style={
                                    "background":   "white",
                                    "border":       "1px solid #DDE2EA",
                                    "borderRadius": "5px",
                                    "padding":      "10px",
                                    "marginTop":    "auto",
                                },
                            ),
                        ],
                    ),

                    # ── Main area ─────────────────────────────────────────
                    html.Div(
                        style={
                            "flex":       "1",
                            "overflowY":  "auto",
                            "padding":    "16px 20px",
                            "display":    "flex",
                            "flexDirection": "column",
                            "gap":        "12px",
                        },
                        children=[

                            # ── Timestep slider ───────────────────────────
                            html.Div(
                                style={
                                    "background":   "white",
                                    "borderRadius": "8px",
                                    "boxShadow":    "0 1px 4px rgba(0,0,0,.08)",
                                    "padding":      "14px 24px 10px",
                                },
                                children=[
                                    html.Div(
                                        "Timestep",
                                        style={"fontWeight": "bold",
                                               "fontSize": "13px",
                                               "color": "#444",
                                               "marginBottom": "6px"},
                                    ),
                                    dcc.Slider(
                                        id="t-slider",
                                        min=0, max=0, step=1, value=0,
                                        marks={},
                                        tooltip={"placement": "bottom",
                                                 "always_visible": True},
                                        updatemode="drag",
                                    ),
                                ],
                            ),

                            # ── Plots row ─────────────────────────────────
                            html.Div(
                                style={
                                    "display":   "flex",
                                    "gap":       "12px",
                                    "flexWrap":  "wrap",
                                },
                                children=[
                                    # Observation
                                    html.Div(
                                        style={
                                            "background":   "white",
                                            "borderRadius": "8px",
                                            "boxShadow":    "0 1px 4px rgba(0,0,0,.08)",
                                            "padding":      "12px",
                                            "flex":         "0 0 auto",
                                            "display":      "flex",
                                            "flexDirection":"column",
                                            "alignItems":   "center",
                                            "gap":          "8px",
                                        },
                                        children=[
                                            html.Div("Observation",
                                                     style={"fontWeight": "bold",
                                                            "fontSize": "13px",
                                                            "color": "#444"}),
                                            html.Img(
                                                id="obs-img",
                                                style={
                                                    "imageRendering": "pixelated",
                                                    "width":  "256px",
                                                    "height": "256px",
                                                    "border": "1px solid #DDE2EA",
                                                    "borderRadius": "4px",
                                                    "background": "#eee",
                                                },
                                            ),
                                            html.Div(
                                                id="action-label",
                                                style={
                                                    "fontSize":   "14px",
                                                    "fontWeight": "bold",
                                                    "color":      "#333",
                                                    "background": "#F0F4FF",
                                                    "borderRadius": "5px",
                                                    "padding":    "5px 14px",
                                                    "minWidth":   "160px",
                                                    "textAlign":  "center",
                                                },
                                            ),
                                        ],
                                    ),
                                    # Probs heatmap
                                    html.Div(
                                        style={
                                            "background":   "white",
                                            "borderRadius": "8px",
                                            "boxShadow":    "0 1px 4px rgba(0,0,0,.08)",
                                            "padding":      "12px",
                                            "flex":         "1 1 320px",
                                            "minWidth":     "300px",
                                        },
                                        children=[
                                            dcc.Graph(
                                                id="probs-heatmap",
                                                style={"height": "340px"},
                                                config={"displayModeBar": False},
                                            ),
                                        ],
                                    ),
                                    # Z heatmap
                                    html.Div(
                                        style={
                                            "background":   "white",
                                            "borderRadius": "8px",
                                            "boxShadow":    "0 1px 4px rgba(0,0,0,.08)",
                                            "padding":      "12px",
                                            "flex":         "1 1 320px",
                                            "minWidth":     "300px",
                                        },
                                        children=[
                                            dcc.Graph(
                                                id="z-heatmap",
                                                style={"height": "340px"},
                                                config={"displayModeBar": False},
                                            ),
                                        ],
                                    ),
                                ],
                            ),

                            # ── Action timeline strip ─────────────────────
                            html.Div(
                                style={
                                    "background":   "white",
                                    "borderRadius": "8px",
                                    "boxShadow":    "0 1px 4px rgba(0,0,0,.08)",
                                    "padding":      "12px 16px",
                                },
                                children=[
                                    html.Div(
                                        "Action timeline",
                                        style={"fontWeight": "bold",
                                               "fontSize": "13px",
                                               "color": "#444",
                                               "marginBottom": "8px"},
                                    ),
                                    html.Div(
                                        id="action-strip",
                                        style={
                                            "display":    "flex",
                                            "gap":        "4px",
                                            "overflowX":  "auto",
                                            "padding":    "4px 2px 8px",
                                            "alignItems": "center",
                                        },
                                    ),
                                ],
                            ),

                        ],
                    ),
                ],
            ),
        ],
    )

    # ── Callback 1: ejecutar trayectoria ────────────────────────────────────
    @app.callback(
        Output("traj-store",  "data"),
        Output("t-slider",    "max"),
        Output("t-slider",    "marks"),
        Output("t-slider",    "value"),
        Output("status-msg",  "children"),
        Input("run-btn",      "n_clicks"),
        State("exp-type",     "value"),
        State("n-spin",       "value"),
        State("n-forward",    "value"),
        prevent_initial_call=True,
    )
    def _run(n_clicks, exp_type, n_spin, n_forward):
        n_spin    = int(n_spin    or 0)
        n_forward = int(n_forward or 1)
        actions   = make_actions(exp_type, n_spin, n_forward)
        print(f"[run] exp={exp_type}  n_spin={n_spin}  n_fwd={n_forward}  "
              f"len(actions)={len(actions)}")

        data = run_trajectory(agent, env_factory, actions, device, seed=seed)
        T    = data["T"]

        # Marcas en el slider cada ~10% del recorrido
        step_mark = max(1, T // 10)
        marks     = {i: str(i) for i in range(0, T + 1, step_mark)}
        status    = (
            f"✓  {len(actions)} acciones  →  T = {T + 1} pasos\n"
            f"Experimento: {exp_type}  |  spins={n_spin}  fwd={n_forward}"
        )
        return data, T, marks, 0, status

    # ── Callback 2: actualizar visualización según timestep ─────────────────
    @app.callback(
        Output("obs-img",      "src"),
        Output("action-label", "children"),
        Output("probs-heatmap","figure"),
        Output("z-heatmap",    "figure"),
        Output("action-strip", "children"),
        Input("t-slider",      "value"),
        Input("traj-store",    "data"),
    )
    def _update(t, data):
        empty = _empty_fig()
        if data is None:
            return "", "—", empty, empty, []

        t       = int(t or 0)
        imgs    = data["images"]
        probs   = np.array(data["probs"],  dtype=np.float32)  # (T+1, S, K)
        zs      = np.array(data["zs"],      dtype=np.float32)  # (T+1, S, K)
        actions = data["actions"]
        T       = data["T"]
        t       = min(t, T)

        # Imagen
        img_src = imgs[t]

        # Etiqueta de acción
        act = actions[t]
        color = ACT_COLORS.get(act, "#888")
        act_label = html.Span(
            f"{ACT_SYMBOLS.get(act, '?')}  {ACT_NAMES.get(act, '')}",
            style={"color": color},
        )

        # Figuras
        fig_probs = _probs_fig(probs[t])
        fig_z     = _z_fig(zs[t])

        # Strip
        strip = _action_strip(actions, t)

        return img_src, act_label, fig_probs, fig_z, strip

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dash app para visualizar trayectorias del World Model."
    )
    parser.add_argument("--logdir",  required=True,
                        help="Directorio del run (debe contener latest.pt y .hydra/)")
    parser.add_argument("--device",  default=None,
                        help="cpu / cuda (por defecto usa config.device)")
    parser.add_argument("--seed",    type=int, default=0,
                        help="Seed para el reset del env en cada ejecución")
    parser.add_argument("--port",    type=int, default=8050,
                        help="Puerto del servidor Dash")
    parser.add_argument("--debug",   action="store_true",
                        help="Activar modo debug de Dash")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)
    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")
    device = args.device or config.device
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

    print(f"\nIniciando Dash en  http://localhost:{args.port}\n")
    app = create_app(agent, env_factory, device, args.seed)
    app.run(debug=args.debug, port=args.port)