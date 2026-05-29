"""
viz/traj_viz_dash.py — Dash app para visualizar trayectorias paso a paso.

Eliges un experimento predefinido (spin / loop), lo corres entero y recorres los
pasos con un slider (replay de solo lectura). Para construir una trayectoria a
mano con las flechas, usa `experiments/viz/interactive_dash.py`.

Uso:
    uv run python -m viz.traj_viz_dash \
        --logdir logdir/wm_only_random_mission/01 \
        --device cpu \
        --seed 0 \
        --port 8050
"""

import argparse

import numpy as np
from dash import Dash, dcc, html, Input, Output, State

from viz.common import (
    ACT_LEFT, ACT_RIGHT, ACT_FORWARD,
    ACT_NAMES, ACT_SYMBOLS, ACT_COLORS,
    run_trajectory, load_agent,
    _empty_fig, _probs_fig, _z_fig, _action_strip,
)


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

    if exp_type == "spin_left":
        return spin(k, "left")
    elif exp_type == "spin_right":
        return spin(k, "right")
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
                                        {"label": "←⟳  Spin Left",          "value": "spin_left"},
                                        {"label": "⟳→  Spin Right",          "value": "spin_right"},
                                        {"label": "□→  Loop Right",    "value": "loop_right"},
                                        {"label": "←□  Loop Left",     "value": "loop_left"},
                                        {"label": "□+⟳  Loop Left + Spin", "value": "loop_left_and_spin"},
                                        {"label": "□+⟳  Loop Right + Spin", "value": "loop_right_and_spin"},
                                    ],
                                    value="spin_left",
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

    agent, env_factory, device = load_agent(args.logdir, args.device)

    print(f"\nIniciando Dash en  http://localhost:{args.port}\n")
    app = create_app(agent, env_factory, device, args.seed)
    app.run(debug=args.debug, port=args.port)
