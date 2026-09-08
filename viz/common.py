"""
viz/common.py — Código compartido por las visualizaciones Dash del WM.

Reúne las constantes de acción, los helpers de obs/imagen, la ejecución de
trayectorias, los helpers de figuras Plotly y la carga de checkpoint, para que
tanto `traj_viz_dash.py` (replay con slider) como `interactive_dash.py`
(construcción con flechas) compartan el mismo núcleo sin duplicarlo.
"""

import base64
import copy
import io
import pathlib

import numpy as np
import plotly.graph_objects as go
import torch
from dash import html
from gymnasium.utils import seeding
from omegaconf import OmegaConf

import her_dream.tools as tools
from her_dream.dreamer import Dreamer
from her_dream.envs import make_env, make_envs
from her_dream.rewards import make_reward

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
# Builder incremental con estado vivo
# ─────────────────────────────────────────────────────────────────────────────


class TrajectoryBuilder:
    """
    Constructor incremental y con estado de una trayectoria.

    A diferencia de `collect_trajectory` (que re-ejecuta todo el camino desde
    t=0 en cada llamada), este builder mantiene **un único env vivo** y el estado
    del RSSM, y avanza **exactamente un paso por acción**. Nunca reproduce el
    historial, así que:

    - La estocasticidad del entorno (zombies, etc.) se realiza una sola vez.
    - El `stoch z` se muestrea una sola vez por paso y queda congelado, de modo
      que el posterior/prior y la trayectoria latente del pasado no cambian al
      añadir acciones nuevas.

    El `undo` restaura un snapshot (`deepcopy` del env + estado latente), que es
    barato y tampoco reproduce acciones. La navegación (ver pasos pasados) solo
    lee de los registros acumulados.
    """

    def __init__(self, agent, env_factory, device, seed: int = 0):
        self.agent = agent
        self.env_factory = env_factory
        self.device = device
        self.seed = seed
        self.reset()

    # ── operaciones ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def reset(self) -> None:
        """Crea un env fresco y registra el paso inicial (t=0)."""
        env = self.env_factory()
        base = env
        while hasattr(base, "env"):
            base = base.env
        base._np_random, _ = seeding.np_random(self.seed)
        obs = env.reset()

        self._env = env
        self._n_act = env.action_space.shape[0]
        self._stoch, self._deter = self.agent.rssm.initial(1)
        self._done = False

        self._raw = {
            "images": [],
            "post_probs": [],
            "post_z": [],
            "prior_probs": [],
            "prior_z": [],
            "actions": [],
        }
        self._b64: list[str] = []
        self._actions: list[int] = []  # camino (sin el reset)
        self._undo: list = []  # snapshots (env, stoch, deter) por paso

        prev_action = torch.zeros(1, self._n_act, device=self.device)
        self._ingest(obs, prev_action, action=None, reset=True)

    @torch.no_grad()
    def step(self, action_idx: int) -> None:
        """Avanza un único paso con la acción dada (no-op si el episodio terminó)."""
        if self._done:
            return
        act_np = _onehot(int(action_idx), self._n_act)
        obs, _, done, _ = self._env.step(act_np)
        prev_action = torch.as_tensor(act_np, device=self.device).unsqueeze(0)
        self._actions.append(int(action_idx))
        self._ingest(obs, prev_action, action=int(action_idx), reset=False)
        if done:
            self._done = True
            print(f"  [warn] episodio terminó en paso {self.T}")

    def undo(self) -> None:
        """Deshace el último paso restaurando el snapshot previo (sin re-ejecutar)."""
        if len(self._undo) <= 1:  # no se puede deshacer el reset (t=0)
            return
        self._undo.pop()
        for v in self._raw.values():
            v.pop()
        self._b64.pop()
        if self._actions:
            self._actions.pop()
        env_snap, stoch, deter = self._undo[-1]
        self._env = copy.deepcopy(env_snap)
        self._stoch, self._deter = stoch.clone(), deter.clone()
        self._done = False

    # ── interno ──────────────────────────────────────────────────────────────

    def _ingest(self, obs, prev_action, action, reset: bool) -> None:
        obs_t = _preprocess_obs(obs, self.device)
        embed = self.agent.encoder(obs_t)
        stoch, deter, post_logit = self.agent.rssm.obs_step(
            self._stoch,
            self._deter,
            prev_action,
            embed,
            torch.tensor([reset], dtype=torch.bool, device=self.device),
        )
        prior_z, prior_logit = self.agent.rssm.prior(deter)
        post_probs = self.agent.rssm.get_dist(post_logit).base_dist.probs
        prior_probs = self.agent.rssm.get_dist(prior_logit).base_dist.probs

        img = np.array(obs["image"], dtype=np.uint8)
        self._raw["images"].append(img)
        self._raw["post_probs"].append(post_probs.squeeze(0).cpu().numpy())
        self._raw["post_z"].append(stoch.squeeze(0).cpu().numpy())
        self._raw["prior_probs"].append(prior_probs.squeeze(0).cpu().numpy())
        self._raw["prior_z"].append(prior_z.squeeze(0).cpu().numpy())
        self._raw["actions"].append(action)
        self._b64.append(_arr_to_b64(img))

        # Avanza el estado vivo y guarda el snapshot de ESTE paso para el undo.
        self._stoch, self._deter = stoch, deter
        self._undo.append((copy.deepcopy(self._env), stoch.clone(), deter.clone()))

    # ── accesores ──────────────────────────────────────────────────────────

    @property
    def T(self) -> int:
        return len(self._raw["actions"]) - 1

    @property
    def actions(self) -> list[int]:
        return list(self._actions)

    @property
    def raw(self) -> dict:
        """Datos crudos acumulados (numpy), aptos para `save_trajectory`."""
        return {**self._raw, "done": self._done, "T": self.T}

    def data(self) -> dict:
        """Snapshot serializable para los `dcc.Store` de Dash (mismo formato que `run_trajectory`)."""
        return {
            "probs": [p.tolist() for p in self._raw["post_probs"]],
            "zs": [z.tolist() for z in self._raw["post_z"]],
            "prior_probs": [p.tolist() for p in self._raw["prior_probs"]],
            "prior_zs": [z.tolist() for z in self._raw["prior_z"]],
            "images": list(self._b64),
            "actions": list(self._raw["actions"]),
            "done": self._done,
            "T": self.T,
        }

    def save(self, out_root, name: str | None = None, names: dict | None = None, fps: float = 2.0):
        """Guarda la trayectoria acumulada (sin re-ejecutar nada)."""
        return save_trajectory(self.raw, self.actions, out_root, seed=self.seed, name=name, names=names, fps=fps)


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
    el heatmap del `z prior`, el del `z posterior` y el `z` real que se muestreó
    (el stoch one-hot que el world model efectivamente propaga).
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
        z = raw["post_z"][t]  # (S, K) — stoch muestreado (aprox one-hot)

        fig, axes = plt.subplots(
            1, 4, figsize=(12.0, 4.0), dpi=100, gridspec_kw={"width_ratios": [1.15, 1.0, 1.0, 1.0]}
        )
        ax_img, ax_pr, ax_po, ax_z = axes

        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(f"t = {t} / {T}", fontsize=11)
        # La acción, justo debajo de la imagen.
        ax_img.set_xlabel(names.get(act, "?"), fontsize=13, fontweight="bold", color="#222", labelpad=8)

        # Distribuciones prior/posterior (probs en azul).
        for ax, mat, ttl in (
            (ax_pr, prior, "z prior  p(z | h_t)"),
            (ax_po, post, "z posterior  p(z | x_t, h_t)"),
        ):
            im_p = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_title(ttl, fontsize=10)
            ax.set_xlabel("clase k", fontsize=9)
            ax.set_ylabel("slot s", fontsize=9)
        fig.colorbar(im_p, ax=[ax_pr, ax_po], fraction=0.03, pad=0.02)

        # z real muestreado (one-hot) en verde, para distinguirlo de las probs.
        im_z = ax_z.imshow(z, cmap="Greens", vmin=0.0, vmax=1.0, aspect="auto")
        ax_z.set_title("z muestreado  (one-hot)", fontsize=10)
        ax_z.set_xlabel("clase k", fontsize=9)
        ax_z.set_ylabel("slot s", fontsize=9)
        fig.colorbar(im_z, ax=ax_z, fraction=0.045, pad=0.02)

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


def regenerate_gif(traj_dir, fps: float | None = None):
    """
    Regenera el `trajectory.gif` de una trayectoria ya guardada, leyendo
    `arrays.npz` (+ `metadata.json` para los nombres de acción y el fps). Útil
    para re-renderizar con el layout actual del gif (p.ej. tras añadir el panel
    del `z` muestreado) sin re-ejecutar el world model.
    """
    import json

    traj_dir = pathlib.Path(traj_dir)
    arr = np.load(traj_dir / "arrays.npz")
    meta_path = traj_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    # Nombres de acción: claves a int cuando se pueda; -1 marca el reset (t=0).
    names: dict = {}
    for k, v in (meta.get("action_names") or {}).items():
        try:
            names[int(k)] = v
        except (TypeError, ValueError):
            names[k] = v
    names.setdefault(-1, names.get("None", "reset"))
    if fps is None:
        fps = float(meta.get("fps", 2.0))

    T = int(arr["actions"].shape[0]) - 1
    raw = {
        "images": list(arr["images"]),
        "prior_probs": list(arr["prior_probs"]),
        "post_probs": list(arr["post_probs"]),
        "post_z": list(arr["post_z"]),
        "prior_z": list(arr["prior_z"]),
        "actions": [int(a) for a in arr["actions"]],
        "T": T,
    }
    _write_gif(traj_dir / "trajectory.gif", raw, names=names, fps=fps)
    print(f"GIF regenerado → {traj_dir / 'trajectory.gif'}  ({T + 1} frames)")
    return traj_dir / "trajectory.gif"


def save_trajectory(
    raw: dict,
    actions: list[int],
    out_root,
    seed: int = 0,
    name: str | None = None,
    names: dict | None = None,
    fps: float = 2.0,
) -> pathlib.Path:
    """
    Guarda una trayectoria ya colectada en `<out_root>/trajectories/<name>/`.

    No re-ejecuta nada: recibe `raw` (los arrays crudos acumulados, p.ej. de
    `TrajectoryBuilder.raw` o de `collect_trajectory`) y `actions` (el camino sin
    el reset). Genera, de forma reconstruible y analizable:
      - `actions.json`     — secuencia de acciones (índices + nombres) y metadatos.
      - `metadata.json`    — name, seed, logdir, T, done, timestamp, action map.
      - `arrays.npz`       — prior/posterior probs y z, acciones (-1 = reset).
      - `frames/step_*.png`— cada observación cruda.
      - `trajectory.gif`   — imagen + acción + z prior + z posterior por paso.

    `names` es el diccionario de nombres de acción del entorno (Crafter, …).

    Retorna el `Path` de la carpeta creada.
    """
    import datetime
    import json

    from PIL import Image

    names = names or ACT_NAMES

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
    # claves de goal que `Dreamer` espera. `backfill_config` rellena defaults
    # inocuos para que el constructor no falle (el reward goal-conditioned no se
    # usa en el viz) y reporta `wm_only=True` para esos checkpoints sin goal.
    config, wm_only = tools.backfill_config(config)
    has_goal = not wm_only

    reward_fn = None if wm_only else make_reward(config)
    _, _, obs_space, act_space = make_envs(config.env)

    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
        reward_function=reward_fn,
    ).to(device)

    ckpt = torch.load(logdir / "latest.pt", map_location=device)
    state = tools.migrate_agent_state_dict(ckpt["agent_state_dict"])
    own = agent.state_dict()
    # Carga solo los tensores presentes y con forma coincidente: descarta
    # módulos extra del checkpoint (_frozen_*) y actor/critic con otra dimensión
    # de goal. Lo que importa para el viz —encoder y rssm— sí coincide.
    loadable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    agent.load_state_dict(loadable, strict=False)

    wm_missing = [k for k in own if k.startswith(("encoder.", "rssm.")) and k not in loadable]
    if wm_missing:
        print(f"[warn] {len(wm_missing)} claves del world model NO se cargaron, p.ej. {wm_missing[:3]}")
    agent.eval()
    skipped = len(state) - len(loadable)
    tag = "" if has_goal else "  (WM vanilla sin goal)"
    print(f"Checkpoint cargado → {logdir / 'latest.pt'}  [{len(loadable)} tensores, {skipped} omitidos]{tag}")

    def env_factory():
        return make_env(config.env, 0)

    return agent, env_factory, device
