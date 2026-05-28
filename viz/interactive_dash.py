"""
experiments/viz/interactive_dash.py — Builder interactivo de trayectorias.

A diferencia de `traj_viz_dash.py` (que corre un experimento predefinido y lo
recorres con un slider), aquí partes de un reset al azar y vas **añadiendo
acciones una a una con las flechas del teclado** (o con los botones). En cada
paso ves la imagen, la acción, las probs del posterior y la z, y la tira de
acciones de abajo muestra el historial del camino que vas construyendo —
clickea cualquier celda para inspeccionar ese paso pasado.

Teclas
──────
    ←  turn_left      →  turn_right     ↑  forward
    p  pickup         d  drop           t  toggle      Enter  done
    Backspace  deshace último paso      r  reset (vacía el camino)
    [  paso anterior  ]  paso siguiente  (solo navega, no modifica el camino)

Modelo de estado
────────────────
El env es determinista por seed, así que en vez de mantener un env "vivo" en el
servidor guardamos solo la lista de acciones en un `dcc.Store` y re-ejecutamos
la trayectoria completa en cada cambio. Esto hace trivial el undo y evita estado
mutable frágil.

Uso:
    uv run python -m viz.interactive_dash \
        --logdir logdir/wm_only_random_mission/01 \
        --device cpu \
        --seed 0 \
        --port 8051
"""

import argparse

import numpy as np
from dash import Dash, dcc, html, Input, Output, State, ALL, ctx, no_update
from dash.exceptions import PreventUpdate

from viz.common import (
    ACT_NAMES, ACT_SYMBOLS, ACT_COLORS,
    run_trajectory, load_agent,
    _empty_fig, _probs_fig, _z_fig, _action_strip,
)


# ─────────────────────────────────────────────────────────────────────────────
# Estilo
# ─────────────────────────────────────────────────────────────────────────────

_SIDEBAR_W = "280px"
_CTRL_BG   = "#F7F9FC"
_ACCENT    = "#3A7D5C"   # verde para distinguirlo del replay (azul)

# Acciones que aparecen como botones (todas las del Discrete(7) de MiniGrid).
_BUTTON_ACTS = [0, 1, 2, 3, 4, 5, 6]

# Mapeo tecla → id de botón (consumido por el script de teclado del index_string).
_KEYMAP = {
    "ArrowLeft": "btn-act-0",
    "ArrowRight": "btn-act-1",
    "ArrowUp": "btn-act-2",
    "p": "btn-act-3", "P": "btn-act-3",
    "d": "btn-act-4", "D": "btn-act-4",
    "t": "btn-act-5", "T": "btn-act-5",
    "Enter": "btn-act-6",
    "Backspace": "btn-undo",
    "r": "btn-reset", "R": "btn-reset",
    "[": "btn-prev",
    "]": "btn-next",
}

_INDEX_STRING = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
        (function () {
            var KEYMAP = __KEYMAP__;
            document.addEventListener("keydown", function (e) {
                var tag = (e.target.tagName || "").toLowerCase();
                if (tag === "input" || tag === "textarea" || tag === "select") return;
                var id = KEYMAP[e.key];
                if (!id) return;
                var el = document.getElementById(id);
                if (el && !el.disabled) {
                    e.preventDefault();
                    el.click();
                }
            });
        })();
        </script>
    </body>
</html>"""


def _label(text: str) -> html.Label:
    return html.Label(text, style={
        "fontWeight": "600", "fontSize": "12px",
        "color": "#444", "marginBottom": "4px", "display": "block",
    })


def _act_button(act: int) -> html.Button:
    """Botón para añadir una acción `act` al camino."""
    return html.Button(
        [
            html.Span(ACT_SYMBOLS[act], style={"fontSize": "20px"}),
            html.Span(ACT_NAMES[act], style={"fontSize": "10px", "marginTop": "2px"}),
        ],
        id=f"btn-act-{act}",
        n_clicks=0,
        title=f"{ACT_NAMES[act]}",
        style={
            "display":        "flex",
            "flexDirection":  "column",
            "alignItems":     "center",
            "justifyContent": "center",
            "height":         "58px",
            "background":     "white",
            "color":          ACT_COLORS[act],
            "border":         f"2px solid {ACT_COLORS[act]}",
            "borderRadius":   "8px",
            "fontWeight":     "bold",
            "cursor":         "pointer",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dash app
# ─────────────────────────────────────────────────────────────────────────────

def create_app(agent, env_factory, device: str, seed: int) -> Dash:
    import json

    app = Dash(__name__, title="Interactive Trajectory Builder")
    app.index_string = _INDEX_STRING.replace("__KEYMAP__", json.dumps(_KEYMAP))

    app.layout = html.Div(
        style={
            "fontFamily": "Arial, sans-serif",
            "background": "#FAFAFA",
            "minHeight":  "100vh",
        },
        children=[
            # ── Stores ──────────────────────────────────────────────────────
            dcc.Store(id="actions-store", data=[]),   # lista de action_idx del camino
            dcc.Store(id="traj-store"),               # datos completos re-ejecutados
            dcc.Store(id="view-store", data=0),       # timestep que se está viendo

            # ── Header ──────────────────────────────────────────────────────
            html.Div(
                "🎮  Interactive Trajectory Builder",
                style={
                    "background":  _ACCENT,
                    "color":       "white",
                    "padding":     "10px 24px",
                    "fontSize":    "18px",
                    "fontWeight":  "bold",
                    "letterSpacing": ".5px",
                },
            ),

            html.Div(
                style={"display": "flex", "height": "calc(100vh - 44px)"},
                children=[

                    # ── Sidebar ───────────────────────────────────────────
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
                            "gap":          "16px",
                        },
                        children=[
                            html.Div([
                                _label("Añadir acción  (o usa el teclado)"),
                                html.Div(
                                    [_act_button(a) for a in _BUTTON_ACTS],
                                    style={
                                        "display":             "grid",
                                        "gridTemplateColumns": "1fr 1fr 1fr",
                                        "gap":                 "6px",
                                    },
                                ),
                            ]),
                            html.Div(
                                [
                                    html.Button(
                                        "⤺  Undo", id="btn-undo", n_clicks=0,
                                        style={
                                            "flex": "1", "padding": "9px 0",
                                            "background": "#fff",
                                            "color": "#B23A48",
                                            "border": "2px solid #B23A48",
                                            "borderRadius": "7px",
                                            "fontWeight": "bold", "cursor": "pointer",
                                        },
                                    ),
                                    html.Button(
                                        "⟲  Reset", id="btn-reset", n_clicks=0,
                                        style={
                                            "flex": "1", "padding": "9px 0",
                                            "background": "#fff",
                                            "color": "#555",
                                            "border": "2px solid #999",
                                            "borderRadius": "7px",
                                            "fontWeight": "bold", "cursor": "pointer",
                                        },
                                    ),
                                ],
                                style={"display": "flex", "gap": "6px"},
                            ),
                            # botones ocultos para navegación por teclado [ ]
                            html.Div(
                                [
                                    html.Button("prev", id="btn-prev", n_clicks=0),
                                    html.Button("next", id="btn-next", n_clicks=0),
                                ],
                                style={"display": "none"},
                            ),
                            # Status
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
                                    "whiteSpace":   "pre-line",
                                },
                            ),
                            # Cheatsheet de teclas
                            html.Div(
                                [
                                    html.Div("Teclado",
                                             style={"fontWeight": "bold",
                                                    "fontSize": "11px",
                                                    "color": "#666",
                                                    "marginBottom": "6px"}),
                                    html.Div("← → ↑  mover / girar",
                                             style={"fontSize": "11px", "color": "#444"}),
                                    html.Div("p d t Enter  pickup/drop/toggle/done",
                                             style={"fontSize": "11px", "color": "#444"}),
                                    html.Div("Backspace  deshacer   ·   r  reset",
                                             style={"fontSize": "11px", "color": "#444"}),
                                    html.Div("[  ]  navegar pasos",
                                             style={"fontSize": "11px", "color": "#444"}),
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
                            # Plots row
                            html.Div(
                                style={"display": "flex", "gap": "12px",
                                       "flexWrap": "wrap"},
                                children=[
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
                                                    "background": "#EAF6EF",
                                                    "borderRadius": "5px",
                                                    "padding":    "5px 14px",
                                                    "minWidth":   "160px",
                                                    "textAlign":  "center",
                                                },
                                            ),
                                        ],
                                    ),
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

                            # Historial del camino (strip clickeable)
                            html.Div(
                                style={
                                    "background":   "white",
                                    "borderRadius": "8px",
                                    "boxShadow":    "0 1px 4px rgba(0,0,0,.08)",
                                    "padding":      "12px 16px",
                                },
                                children=[
                                    html.Div(
                                        "Camino construido  (clic en un paso para verlo)",
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
                                            "minHeight":  "60px",
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

    # ── Callback 1: modificar el camino (añadir / undo / reset) ─────────────
    @app.callback(
        Output("actions-store", "data"),
        [Input(f"btn-act-{a}", "n_clicks") for a in _BUTTON_ACTS]
        + [Input("btn-undo", "n_clicks"), Input("btn-reset", "n_clicks")],
        State("actions-store", "data"),
        prevent_initial_call=True,
    )
    def _modify(*args):
        actions = list(args[-1] or [])
        trig = ctx.triggered_id
        if trig is None:
            raise PreventUpdate

        if trig == "btn-reset":
            return []
        if trig == "btn-undo":
            return actions[:-1] if actions else no_update
        if isinstance(trig, str) and trig.startswith("btn-act-"):
            act = int(trig.rsplit("-", 1)[1])
            return actions + [act]
        raise PreventUpdate

    # ── Callback 2: re-ejecutar la trayectoria cuando cambia el camino ──────
    @app.callback(
        Output("traj-store", "data"),
        Output("view-store", "data"),
        Input("actions-store", "data"),
    )
    def _replay(actions):
        actions = list(actions or [])
        data = run_trajectory(agent, env_factory, actions, device, seed=seed)
        # Al añadir/quitar, saltamos a mostrar el último paso del camino.
        return data, data["T"]

    # ── Callback 3: navegar el historial (clic en strip / teclas [ ]) ───────
    @app.callback(
        Output("view-store", "data", allow_duplicate=True),
        Input({"type": "step", "index": ALL}, "n_clicks"),
        Input("btn-prev", "n_clicks"),
        Input("btn-next", "n_clicks"),
        State("view-store", "data"),
        State("traj-store", "data"),
        prevent_initial_call=True,
    )
    def _navigate(step_clicks, prev_clicks, next_clicks, view, data):
        # Evita disparos espurios al recrearse los botones del strip (n_clicks=0).
        trig_val = ctx.triggered[0]["value"]
        if not trig_val:
            raise PreventUpdate

        T = (data or {}).get("T", 0)
        view = int(view or 0)
        trig = ctx.triggered_id

        if trig == "btn-prev":
            return max(0, view - 1)
        if trig == "btn-next":
            return min(T, view + 1)
        if isinstance(trig, dict) and trig.get("type") == "step":
            return min(T, int(trig["index"]))
        raise PreventUpdate

    # ── Callback 4: render del paso seleccionado ────────────────────────────
    @app.callback(
        Output("obs-img",       "src"),
        Output("action-label",  "children"),
        Output("probs-heatmap", "figure"),
        Output("z-heatmap",     "figure"),
        Output("action-strip",  "children"),
        Output("status-msg",    "children"),
        Input("view-store",     "data"),
        Input("traj-store",     "data"),
    )
    def _render(view, data):
        empty = _empty_fig("Pulsa una flecha para empezar")
        if data is None:
            return "", "—", empty, empty, [], "Cargando…"

        actions = data["actions"]
        probs   = np.array(data["probs"], dtype=np.float32)  # (T+1, S, K)
        zs      = np.array(data["zs"],    dtype=np.float32)
        T       = data["T"]
        t       = min(int(view or 0), T)

        act = actions[t]
        color = ACT_COLORS.get(act, "#888")
        act_label = html.Span(
            f"t={t}   {ACT_SYMBOLS.get(act, '?')}  {ACT_NAMES.get(act, '')}",
            style={"color": color},
        )

        strip = _action_strip(actions, t, clickable=True)

        n_steps = len(actions) - 1  # acciones reales (t=0 es el reset)
        status = f"Pasos en el camino: {n_steps}\nViendo t = {t} / {T}"
        if data.get("done"):
            status += "\n⚠ el episodio terminó (done)"

        return (
            data["images"][t],
            act_label,
            _probs_fig(probs[t]),
            _z_fig(zs[t]),
            strip,
            status,
        )

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Builder interactivo de trayectorias del World Model."
    )
    parser.add_argument("--logdir",  required=True,
                        help="Directorio del run (debe contener latest.pt y .hydra/)")
    parser.add_argument("--device",  default=None,
                        help="cpu / cuda (por defecto usa config.device)")
    parser.add_argument("--seed",    type=int, default=0,
                        help="Seed para el reset del env")
    parser.add_argument("--port",    type=int, default=8051,
                        help="Puerto del servidor Dash")
    parser.add_argument("--debug",   action="store_true",
                        help="Activar modo debug de Dash")
    args = parser.parse_args()

    agent, env_factory, device = load_agent(args.logdir, args.device)

    print(f"\nIniciando Dash en  http://localhost:{args.port}\n")
    app = create_app(agent, env_factory, device, args.seed)
    app.run(debug=args.debug, port=args.port)
