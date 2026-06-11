"""
viz/crafter_dash.py — Builder interactivo de trayectorias para Crafter.

Versión de `interactive_dash.py` adaptada al world model de Crafter
(`logdir/original_wm_crafter/*`): partes de un reset y vas **añadiendo acciones
una a una** (con el teclado o los botones). En cada paso ves el frame de Crafter,
la acción, el posterior `p(k | slot s)` y la `z` muestreada, y la tira de abajo
muestra el camino construido — clickea cualquier celda para inspeccionar ese paso.

A diferencia de MiniGrid (7 acciones), Crafter tiene 17 acciones discretas. El
checkpoint de Crafter es un world model "vanilla" (sin objetivo), por lo que
`viz.common.load_agent` carga solo los pesos del world model (encoder + rssm),
que es justo lo que necesita esta visualización del posterior.

Teclas
──────
    ← → ↑ ↓   mover (left/right/up/down)
    Espacio   do (interactuar)        Tab   sleep
    r place_stone   t place_table   f place_furnace   p place_plant
    1/2/3   wood/stone/iron pickaxe   4/5/6   wood/stone/iron sword
    0       noop
    Backspace  deshace último paso    [ ]  navegar pasos

Uso:
    uv run python -m viz.crafter_dash \
        --logdir logdir/original_wm_crafter/02 \
        --device cpu \
        --seed 0 \
        --port 8052
"""

import argparse
import json

import numpy as np
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from viz.common import (
    _action_strip,
    _empty_fig,
    _probs_fig,
    _z_fig,
    load_agent,
    run_trajectory,
    save_trajectory,
)

# ─────────────────────────────────────────────────────────────────────────────
# Metadata de acciones de Crafter (17 acciones discretas)
# ─────────────────────────────────────────────────────────────────────────────

CRAFTER_NAMES = {
    None: "reset",
    0: "noop",
    1: "move_left",
    2: "move_right",
    3: "move_up",
    4: "move_down",
    5: "do",
    6: "sleep",
    7: "place_stone",
    8: "place_table",
    9: "place_furnace",
    10: "place_plant",
    11: "make_wood_pickaxe",
    12: "make_stone_pickaxe",
    13: "make_iron_pickaxe",
    14: "make_wood_sword",
    15: "make_stone_sword",
    16: "make_iron_sword",
}

CRAFTER_SYMBOLS = {
    None: "●",
    0: "·",
    1: "←",
    2: "→",
    3: "↑",
    4: "↓",
    5: "✋",
    6: "z",
    7: "▦",
    8: "▤",
    9: "♨",
    10: "❀",
    11: "⛏",
    12: "⛏",
    13: "⛏",
    14: "⚔",
    15: "⚔",
    16: "⚔",
}

# Paleta por familia de acción (los tres picos/espadas comparten símbolo pero se
# distinguen por tono: madera marrón · piedra gris · hierro azul).
CRAFTER_COLORS = {
    None: "#9E9E9E",
    0: "#BDBDBD",
    1: "#2196F3",
    2: "#1976D2",
    3: "#43A047",
    4: "#2E7D32",
    5: "#FB8C00",
    6: "#8E24AA",
    7: "#607D8B",
    8: "#795548",
    9: "#E64A19",
    10: "#7CB342",
    11: "#A1887F",
    12: "#78909C",
    13: "#5C6BC0",
    14: "#BCAAA4",
    15: "#90A4AE",
    16: "#7986CB",
}

# Acciones agrupadas para el panel de botones.
_ACT_GROUPS = [
    ("Movimiento", [1, 3, 2, 4, 0]),
    ("Interacción", [5, 6]),
    ("Construir", [7, 8, 9, 10]),
    ("Picos", [11, 12, 13]),
    ("Espadas", [14, 15, 16]),
]
_BUTTON_ACTS = [a for _, acts in _ACT_GROUPS for a in acts]

# Mapeo tecla → id de botón (consumido por el script del index_string).
_KEYMAP = {
    "ArrowLeft": "btn-act-1",
    "ArrowRight": "btn-act-2",
    "ArrowUp": "btn-act-3",
    "ArrowDown": "btn-act-4",
    " ": "btn-act-5",
    "Tab": "btn-act-6",
    "r": "btn-act-7",
    "R": "btn-act-7",
    "t": "btn-act-8",
    "T": "btn-act-8",
    "f": "btn-act-9",
    "F": "btn-act-9",
    "p": "btn-act-10",
    "P": "btn-act-10",
    "1": "btn-act-11",
    "2": "btn-act-12",
    "3": "btn-act-13",
    "4": "btn-act-14",
    "5": "btn-act-15",
    "6": "btn-act-16",
    "0": "btn-act-0",
    "Backspace": "btn-undo",
    "Delete": "btn-reset",
    "[": "btn-prev",
    "]": "btn-next",
}

_SIDEBAR_W = "320px"
_CTRL_BG = "#F7F9FC"
_ACCENT = "#3A7D5C"

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
    return html.Label(
        text,
        style={
            "fontWeight": "600",
            "fontSize": "12px",
            "color": "#444",
            "marginBottom": "4px",
            "display": "block",
        },
    )


def _act_button(act: int) -> html.Button:
    """Botón para añadir una acción `act` al camino."""
    return html.Button(
        [
            html.Span(CRAFTER_SYMBOLS[act], style={"fontSize": "18px", "lineHeight": "1"}),
            html.Span(
                CRAFTER_NAMES[act].replace("_", " "),
                style={"fontSize": "8px", "marginTop": "2px", "lineHeight": "1.05", "textAlign": "center"},
            ),
        ],
        id=f"btn-act-{act}",
        n_clicks=0,
        title=CRAFTER_NAMES[act],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center",
            "height": "52px",
            "background": "white",
            "color": CRAFTER_COLORS[act],
            "border": f"2px solid {CRAFTER_COLORS[act]}",
            "borderRadius": "8px",
            "fontWeight": "bold",
            "cursor": "pointer",
            "padding": "2px",
        },
    )


def _act_group(title: str, acts: list[int]) -> html.Div:
    return html.Div([
        _label(title),
        html.Div(
            [_act_button(a) for a in acts],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(5, 1fr)",
                "gap": "5px",
            },
        ),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dash app
# ─────────────────────────────────────────────────────────────────────────────


def create_app(agent, env_factory, device: str, seed: int, logdir: str = ".") -> Dash:
    app = Dash(__name__, title="Crafter Trajectory Builder")
    app.index_string = _INDEX_STRING.replace("__KEYMAP__", json.dumps(_KEYMAP))

    app.layout = html.Div(
        style={
            "fontFamily": "Arial, sans-serif",
            "background": "#FAFAFA",
            "minHeight": "100vh",
        },
        children=[
            dcc.Store(id="actions-store", data=[]),
            dcc.Store(id="traj-store"),
            dcc.Store(id="view-store", data=0),
            html.Div(
                "🌲  Crafter Trajectory Builder",
                style={
                    "background": _ACCENT,
                    "color": "white",
                    "padding": "10px 24px",
                    "fontSize": "18px",
                    "fontWeight": "bold",
                    "letterSpacing": ".5px",
                },
            ),
            html.Div(
                style={"display": "flex", "height": "calc(100vh - 44px)"},
                children=[
                    # ── Sidebar ───────────────────────────────────────────
                    html.Div(
                        style={
                            "width": _SIDEBAR_W,
                            "minWidth": _SIDEBAR_W,
                            "background": _CTRL_BG,
                            "borderRight": "1px solid #DDE2EA",
                            "padding": "20px 16px",
                            "overflowY": "auto",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "14px",
                        },
                        children=[
                            *[_act_group(title, acts) for title, acts in _ACT_GROUPS],
                            html.Div(
                                [
                                    html.Button(
                                        "⤺  Undo",
                                        id="btn-undo",
                                        n_clicks=0,
                                        style={
                                            "flex": "1",
                                            "padding": "9px 0",
                                            "background": "#fff",
                                            "color": "#B23A48",
                                            "border": "2px solid #B23A48",
                                            "borderRadius": "7px",
                                            "fontWeight": "bold",
                                            "cursor": "pointer",
                                        },
                                    ),
                                    html.Button(
                                        "⟲  Reset",
                                        id="btn-reset",
                                        n_clicks=0,
                                        style={
                                            "flex": "1",
                                            "padding": "9px 0",
                                            "background": "#fff",
                                            "color": "#555",
                                            "border": "2px solid #999",
                                            "borderRadius": "7px",
                                            "fontWeight": "bold",
                                            "cursor": "pointer",
                                        },
                                    ),
                                ],
                                style={"display": "flex", "gap": "6px"},
                            ),
                            html.Div(
                                [
                                    html.Button("prev", id="btn-prev", n_clicks=0),
                                    html.Button("next", id="btn-next", n_clicks=0),
                                ],
                                style={"display": "none"},
                            ),
                            # ── Guardar trayectoria ───────────────────────
                            html.Div(
                                [
                                    _label("Guardar trayectoria"),
                                    dcc.Input(
                                        id="traj-name",
                                        type="text",
                                        placeholder="nombre (opcional)…",
                                        debounce=False,
                                        style={
                                            "width": "100%",
                                            "padding": "7px 9px",
                                            "border": "1px solid #CDD2DA",
                                            "borderRadius": "5px",
                                            "fontSize": "13px",
                                            "marginBottom": "6px",
                                            "boxSizing": "border-box",
                                        },
                                    ),
                                    html.Button(
                                        "💾  Guardar trayectoria",
                                        id="btn-save",
                                        n_clicks=0,
                                        style={
                                            "width": "100%",
                                            "padding": "9px 0",
                                            "background": _ACCENT,
                                            "color": "white",
                                            "border": "none",
                                            "borderRadius": "7px",
                                            "fontWeight": "bold",
                                            "cursor": "pointer",
                                        },
                                    ),
                                    dcc.Loading(
                                        html.Div(
                                            id="save-msg",
                                            style={
                                                "fontSize": "11px",
                                                "color": "#3A7D5C",
                                                "marginTop": "6px",
                                                "whiteSpace": "pre-line",
                                                "wordBreak": "break-all",
                                            },
                                        ),
                                        type="circle",
                                    ),
                                ],
                                style={
                                    "background": "white",
                                    "border": "1px solid #DDE2EA",
                                    "borderRadius": "5px",
                                    "padding": "10px",
                                },
                            ),
                            html.Div(
                                id="status-msg",
                                style={
                                    "fontSize": "12px",
                                    "color": "#555",
                                    "background": "white",
                                    "border": "1px solid #DDE2EA",
                                    "borderRadius": "5px",
                                    "padding": "8px 10px",
                                    "minHeight": "36px",
                                    "whiteSpace": "pre-line",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Teclado",
                                        style={"fontWeight": "bold", "fontSize": "11px", "color": "#666", "marginBottom": "6px"},
                                    ),
                                    html.Div("← → ↑ ↓  mover", style={"fontSize": "11px", "color": "#444"}),
                                    html.Div("Espacio do · Tab sleep", style={"fontSize": "11px", "color": "#444"}),
                                    html.Div("r/t/f/p  stone/table/furnace/plant", style={"fontSize": "11px", "color": "#444"}),
                                    html.Div("1·2·3 picos  ·  4·5·6 espadas", style={"fontSize": "11px", "color": "#444"}),
                                    html.Div("0 noop · Backspace undo · [ ] navegar", style={"fontSize": "11px", "color": "#444"}),
                                ],
                                style={
                                    "background": "white",
                                    "border": "1px solid #DDE2EA",
                                    "borderRadius": "5px",
                                    "padding": "10px",
                                    "marginTop": "auto",
                                },
                            ),
                        ],
                    ),
                    # ── Main area ─────────────────────────────────────────
                    html.Div(
                        style={
                            "flex": "1",
                            "overflowY": "auto",
                            "padding": "16px 20px",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "12px",
                        },
                        children=[
                            html.Div(
                                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                                children=[
                                    html.Div(
                                        style={
                                            "background": "white",
                                            "borderRadius": "8px",
                                            "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
                                            "padding": "12px",
                                            "flex": "0 0 auto",
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "alignItems": "center",
                                            "gap": "8px",
                                        },
                                        children=[
                                            html.Div(
                                                "Observation",
                                                style={"fontWeight": "bold", "fontSize": "13px", "color": "#444"},
                                            ),
                                            html.Img(
                                                id="obs-img",
                                                style={
                                                    "imageRendering": "pixelated",
                                                    "width": "256px",
                                                    "height": "256px",
                                                    "border": "1px solid #DDE2EA",
                                                    "borderRadius": "4px",
                                                    "background": "#eee",
                                                },
                                            ),
                                            html.Div(
                                                id="action-label",
                                                style={
                                                    "fontSize": "14px",
                                                    "fontWeight": "bold",
                                                    "color": "#333",
                                                    "background": "#EAF6EF",
                                                    "borderRadius": "5px",
                                                    "padding": "5px 14px",
                                                    "minWidth": "180px",
                                                    "textAlign": "center",
                                                },
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        style={
                                            "background": "white",
                                            "borderRadius": "8px",
                                            "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
                                            "padding": "12px",
                                            "flex": "1 1 320px",
                                            "minWidth": "300px",
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
                                            "background": "white",
                                            "borderRadius": "8px",
                                            "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
                                            "padding": "12px",
                                            "flex": "1 1 320px",
                                            "minWidth": "300px",
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
                            html.Div(
                                style={
                                    "background": "white",
                                    "borderRadius": "8px",
                                    "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
                                    "padding": "12px 16px",
                                },
                                children=[
                                    html.Div(
                                        "Camino construido  (clic en un paso para verlo)",
                                        style={
                                            "fontWeight": "bold",
                                            "fontSize": "13px",
                                            "color": "#444",
                                            "marginBottom": "8px",
                                        },
                                    ),
                                    html.Div(
                                        id="action-strip",
                                        style={
                                            "display": "flex",
                                            "gap": "4px",
                                            "overflowX": "auto",
                                            "padding": "4px 2px 8px",
                                            "alignItems": "center",
                                            "minHeight": "60px",
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

    # ── Callback de guardado ────────────────────────────────────────────────
    @app.callback(
        Output("save-msg", "children"),
        Input("btn-save", "n_clicks"),
        State("traj-name", "value"),
        State("actions-store", "data"),
        prevent_initial_call=True,
    )
    def _save(n_clicks, name, actions):
        actions = list(actions or [])
        if not actions:
            return "⚠ Camino vacío: añade alguna acción antes de guardar."
        try:
            out_dir = save_trajectory(
                agent,
                env_factory,
                actions,
                device,
                out_root=logdir,
                seed=seed,
                name=name,
                names=CRAFTER_NAMES,
            )
        except Exception as exc:  # noqa: BLE001 — mostrar el error en la UI
            return f"❌ Error al guardar: {exc}"
        return f"✓ Guardado en\n{out_dir}\n({len(actions)} acciones · gif + npz + frames)"

    # ── Callback 2: re-ejecutar la trayectoria cuando cambia el camino ──────
    @app.callback(
        Output("traj-store", "data"),
        Output("view-store", "data"),
        Input("actions-store", "data"),
    )
    def _replay(actions):
        actions = list(actions or [])
        data = run_trajectory(agent, env_factory, actions, device, seed=seed)
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
        Output("obs-img", "src"),
        Output("action-label", "children"),
        Output("probs-heatmap", "figure"),
        Output("z-heatmap", "figure"),
        Output("action-strip", "children"),
        Output("status-msg", "children"),
        Input("view-store", "data"),
        Input("traj-store", "data"),
    )
    def _render(view, data):
        empty = _empty_fig("Pulsa una acción para empezar")
        if data is None:
            return "", "—", empty, empty, [], "Cargando…"

        actions = data["actions"]
        probs = np.array(data["probs"], dtype=np.float32)  # (T+1, S, K)
        zs = np.array(data["zs"], dtype=np.float32)
        T = data["T"]
        t = min(int(view or 0), T)

        act = actions[t]
        color = CRAFTER_COLORS.get(act, "#888")
        act_label = html.Span(
            f"t={t}   {CRAFTER_SYMBOLS.get(act, '?')}  {CRAFTER_NAMES.get(act, '')}",
            style={"color": color},
        )

        strip = _action_strip(
            actions,
            t,
            clickable=True,
            symbols=CRAFTER_SYMBOLS,
            colors=CRAFTER_COLORS,
            names=CRAFTER_NAMES,
        )

        n_steps = len(actions) - 1
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
    parser = argparse.ArgumentParser(description="Builder interactivo de trayectorias del World Model de Crafter.")
    parser.add_argument("--logdir", default="logdir/original_wm_crafter/02", help="Directorio del run (latest.pt + .hydra/)")
    parser.add_argument("--device", default=None, help="cpu / cuda / mps (por defecto usa config.device)")
    parser.add_argument("--seed", type=int, default=0, help="Seed para el reset del env")
    parser.add_argument("--port", type=int, default=8052, help="Puerto del servidor Dash")
    parser.add_argument("--debug", action="store_true", help="Activar modo debug de Dash")
    args = parser.parse_args()

    agent, env_factory, device = load_agent(args.logdir, args.device)

    print(f"\nIniciando Dash en  http://localhost:{args.port}\n")
    app = create_app(agent, env_factory, device, args.seed, logdir=args.logdir)
    app.run(debug=args.debug, port=args.port)
