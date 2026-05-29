"""
Consistencia del z para un MISMO estado físico alcanzado por DIFERENTES trayectorias.

Pregunta de investigación
─────────────────────────
Si el agente termina en el mismo estado (misma posición, misma dirección, mismo goal y
misma observación visual), ¿el RSSM produce el mismo logit posterior y la misma z?
¿O la historia (la trayectoria) sesga la creencia interna?

Diseño
──────
Todas las trayectorias parten del mismo reset (mismo seed -> mismo goal, misma pose
inicial) y todas terminan en la pose de reset (`agent_start_pos`, `agent_start_dir`)
que se lee del env (compatible con RandomGoal y FixedGoal). La observación visual y
el goal son idénticos en el último paso. Si el WM fuese estrictamente Markoviano y
consistente, el logit posterior debería coincidir entre trayectorias salvo por el
ruido del straight-through Gumbel-Softmax.

Conjuntos de trayectorias
─────────────────────────
1. Spin in place        — solo girar a la izquierda k = 0, 4, 8, 12, 16, 20 veces.
                           (k múltiplo de 4 garantiza que la dirección final == inicial)
2. Loops por el mapa    — para cada lado n ∈ {1, ..., max_geom}, recorrer un cuadrado
                           de lado n con (forward × n, turn-right) × 4. El lado máximo
                           se calcula automáticamente para que el loop quepa en el grid
                           dada la pose inicial.
3. Reset puro           — referencia sin acciones (solo t=0 después de reset).

Métricas
────────
Sobre los logits posteriores del ÚLTIMO paso de cada trayectoria:
  • Distancia de Hamming inter-trayectoria sobre argmax_k.
  • KL simétrica entre cada par de logits.
  • Distancia coseno entre los feat = flatten(stoch) ++ deter.
  • Comparada con la variabilidad intra-estado por muestreo del mismo logit.
  • Entropía media del logit final por trayectoria.

Salida
──────
Plots y JSON en {logdir}/experiments/state_traj_consistency/.
"""
import argparse
import json
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.utils import seeding
from omegaconf import OmegaConf

from dreamer import Dreamer
from envs import make_envs, make_env
from rewards import make_reward


# ─────────────────────────────────────────────────────────────────────────────
# Acciones MiniGrid (índices en el espacio Discrete(7))
# ─────────────────────────────────────────────────────────────────────────────
ACT_LEFT    = 0
ACT_RIGHT   = 1
ACT_FORWARD = 2


def _loop(side: int, turn: str = "right") -> list[int]:
    """Cuadrado de lado `side`: 4 x (forward·side + turn). Cierra el bucle."""
    t = ACT_RIGHT if turn == "right" else ACT_LEFT
    out = []
    for _ in range(4):
        out.extend([ACT_FORWARD] * side)
        out.append(t)
    return out


def _spin(k: int, direction: str = "left") -> list[int]:
    """k giros en `direction` (left/right). k % 4 == 0 deja la orientación igual."""
    act = ACT_LEFT if direction == "left" else ACT_RIGHT
    return [act] * k


# Vectores de desplazamiento por dirección (MiniGrid: 0=east, 1=south, 2=west, 3=north).
_DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def _max_loop_side_from_pose(
    start_pos: tuple[int, int], start_dir: int, size: int, turn: str = "right",
) -> int:
    """
    Mayor `n` tal que el loop (forward·n + turn)×4 cabe en el interior [1, size-2]².

    Las cuatro direcciones recorridas son `D`, `D±1`, `D±2`, `D±3` (signo según
    el giro). Los desplazamientos opuestos (D y D+2) se cancelan, así que la
    extensión del cuadrado queda definida por D y D±1 únicamente.
    """
    ax, ay = start_pos
    free = {
        0: size - 2 - ax,   # east
        1: size - 2 - ay,   # south
        2: ax - 1,          # west
        3: ay - 1,          # north
    }
    d1 = start_dir % 4
    step = 1 if turn == "right" else -1
    d2 = (start_dir + step) % 4
    return max(0, min(free[d1], free[d2]))


def build_trajectories(
    start_pos: tuple[int, int],
    start_dir: int,
    size: int,
    max_loop_side: int | None = None,
) -> dict[str, list[int]]:
    """
    Trayectorias adaptadas a la pose de reset; todas devuelven el agente a
    `(start_pos, start_dir)`. Para los loops se prueban ambos sentidos de giro
    (right / left) y se usan los que caben geométricamente en el grid.
    """
    geom_right = _max_loop_side_from_pose(start_pos, start_dir, size, "right")
    geom_left  = _max_loop_side_from_pose(start_pos, start_dir, size, "left")
    if max_loop_side is not None:
        geom_right = min(geom_right, max_loop_side)
        geom_left  = min(geom_left,  max_loop_side)

    trajs: dict[str, list[int]] = {}
    trajs["reset_only"] = []                                # baseline (sin acciones)
    for k in (4, 8, 12, 16, 20):
        trajs[f"spin_left_{k}"] = _spin(k, "left")
    for k in (4, 8, 12):
        trajs[f"spin_right_{k}"] = _spin(k, "right")
    for n in range(1, geom_right + 1):
        trajs[f"loop_right_{n}"] = _loop(n, "right")
    for n in range(1, geom_left + 1):
        trajs[f"loop_left_{n}"] = _loop(n, "left")
    # Mezclas (usan el sentido de giro disponible, prefiriendo right).
    loop2_turn = "right" if geom_right >= 2 else ("left" if geom_left >= 2 else None)
    loop3_turn = "right" if geom_right >= 3 else ("left" if geom_left >= 3 else None)
    if loop2_turn is not None:
        trajs[f"spin8_then_loop2_{loop2_turn}"] = _spin(8) + _loop(2, loop2_turn)
    if loop3_turn is not None:
        trajs[f"loop3_{loop3_turn}_then_spin4"] = _loop(3, loop3_turn) + _spin(4)
    return trajs


def _read_start_pose(env_factory) -> tuple[tuple[int, int], int, int]:
    """Devuelve (start_pos, start_dir, size) leyendo el env base sin avanzar el RNG."""
    env = env_factory()
    base = env
    while hasattr(base, "env"):
        base = base.env
    pos = tuple(int(c) for c in base.agent_start_pos)
    return pos, int(base.agent_start_dir), int(base.size)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocesado obs -> tensor batch=1
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_obs(obs: dict, device: str) -> dict:
    """obs dict (numpy) -> tensores (1, ...) en device. uint8 imágenes se normalizan."""
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


def _onehot_action(idx: int, n_actions: int) -> np.ndarray:
    a = np.zeros(n_actions, dtype=np.float32)
    a[idx] = 1.0
    return a


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución de una trayectoria (acciones planificadas) en el WM
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_trajectory(
    agent: Dreamer,
    env_factory,
    action_indices: list[int],
    device: str,
    seed: int = 0,
) -> dict:
    """
    Ejecuta `action_indices` en un env nuevo (resetado con `seed`) y devuelve el
    posterior logit / z en cada paso, además del último estado físico observado.

    Devuelve
    ────────
    {
      "logits": (T+1, S, K),  # posterior tras procesar la obs en cada t (incluye reset)
      "zs":     (T+1, S, K),  # muestras stoch correspondientes
      "deter":  (T+1, D),
      "pos":    (T+1, 2),     # (ax, ay) del agente en cada paso
      "dir":    (T+1,),       # entero 0..3
      "goal_pos": (2,),       # posición del goal (constante)
      "images": (T+1, H, W, 3) numpy uint8  # observación visual en cada paso
      "last_image": (H, W, 3) numpy uint8
    }
    """
    env = env_factory()
    # Reset reproducible: los wrappers de este repo no aceptan `seed=` en reset(),
    # pero gymnasium consume seed sólo cuando es != None. Pre-inicializamos el
    # np_random del env base para que el reset() encadenado sea determinista
    # (misma posición de goal en cada trayectoria) sin tocar los wrappers.
    base = env
    while hasattr(base, "env"):
        base = base.env
    base._np_random, _ = seeding.np_random(seed)

    # Reset oficial vía la pipeline completa (Dtype / TimeLimit / GoalConditioned)
    obs = env.reset()

    n_actions = env.action_space.shape[0]

    stoch, deter = agent.rssm.initial(1)
    prev_action = torch.zeros(1, n_actions, device=device)

    logits, zs, deters, poses, dirs, images = [], [], [], [], [], []

    def _record(stoch_, deter_, logit_):
        logits.append(logit_.squeeze(0).cpu())
        zs.append(stoch_.squeeze(0).cpu())
        deters.append(deter_.squeeze(0).cpu())
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        ax, ay = base_env.agent_pos
        poses.append((int(ax), int(ay)))
        dirs.append(int(base_env.agent_dir))

    # t = 0: procesar obs del reset
    obs_t = _preprocess_obs(obs, device)
    is_first = torch.tensor([True], dtype=torch.bool, device=device)
    embed = agent.encoder(obs_t)
    stoch, deter, logit = agent.rssm.obs_step(stoch, deter, prev_action, embed, is_first)
    _record(stoch, deter, logit)

    last_image = obs["image"]
    images.append(np.array(last_image))

    for act_idx in action_indices:
        act_np = _onehot_action(act_idx, n_actions)
        obs, _r, done, _info = env.step(act_np)
        last_image = obs["image"]
        prev_action = torch.as_tensor(act_np, device=device).unsqueeze(0)

        obs_t = _preprocess_obs(obs, device)
        is_first = torch.tensor([False], dtype=torch.bool, device=device)
        embed = agent.encoder(obs_t)
        stoch, deter, logit = agent.rssm.obs_step(stoch, deter, prev_action, embed, is_first)
        _record(stoch, deter, logit)
        images.append(np.array(last_image))

        if done:
            print(f"  [warn] episodio terminó tempranamente en paso {len(logits)-1}")
            break

    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    # RandomGoal expone `_goal_pos` (asignado en place_obj), FixedGoal usa `goal_pos`.
    raw_goal = getattr(base_env, "_goal_pos", None)
    if raw_goal is None:
        raw_goal = getattr(base_env, "goal_pos", None)
    if raw_goal is None:
        raise AttributeError(
            f"No encuentro goal_pos ni _goal_pos en {type(base_env).__name__}"
        )
    goal_pos = np.array(raw_goal, dtype=np.int32)

    return {
        "logits":    torch.stack(logits, dim=0),     # (T+1, S, K)
        "zs":        torch.stack(zs, dim=0),         # (T+1, S, K)
        "deter":     torch.stack(deters, dim=0),     # (T+1, D)
        "pos":       np.array(poses, dtype=np.int32),
        "dir":       np.array(dirs, dtype=np.int32),
        "goal_pos":  goal_pos,
        "images":    np.stack(images, axis=0),       # (T+1, H, W, 3)
        "last_image": np.array(last_image),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Métricas entre logits/zs finales de N trayectorias
# ─────────────────────────────────────────────────────────────────────────────

def _kl_pairwise(agent, logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (N, S, K)
    Devuelve KL simétrica (N, N) promediada sobre slots S.
    """
    import distributions as dists
    N = logits.shape[0]
    out = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            kl_ij = dists.kl(logits[i:i+1], logits[j:j+1]).sum(-1).mean()
            kl_ji = dists.kl(logits[j:j+1], logits[i:i+1]).sum(-1).mean()
            out[i, j] = 0.5 * (kl_ij + kl_ji)
    return out


def _hamming_pairwise(modes: torch.Tensor) -> torch.Tensor:
    """modes: (N, S) -> (N, N) Hamming normalizada."""
    diff = (modes.unsqueeze(0) != modes.unsqueeze(1)).float()
    return diff.mean(-1)


def _cosine_pairwise(feat: torch.Tensor) -> torch.Tensor:
    """feat: (N, F) -> (N, N) coseno."""
    f = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
    return f @ f.T


def _intra_state_hamming(agent, logit: torch.Tensor, M: int) -> float:
    """Hamming intra-estado entre M muestras del mismo logit. logit: (S, K)."""
    l = logit.unsqueeze(0).expand(M, -1, -1)
    samples = agent.rssm.get_dist(l).rsample()        # (M, S, K)
    classes = samples.argmax(-1)
    mask = torch.triu(torch.ones(M, M, device=logit.device), diagonal=1).bool()
    diff = (classes.unsqueeze(0) != classes.unsqueeze(1)).float().mean(-1)
    return diff[mask].mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Principal
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_state_traj_consistency(
    agent: Dreamer,
    env_factory,
    trajectories: dict[str, list[int]],
    n_samples_intra: int = 50,
    seed: int = 0,
    outdir: Path = Path("results/state_traj_consistency"),
    gif_fps: int = 2,
):
    outdir.mkdir(parents=True, exist_ok=True)
    device = agent.device

    # ── Correr cada trayectoria ──────────────────────────────────────────────
    runs: dict[str, dict] = {}
    print(f"\nCorriendo {len(trajectories)} trayectorias (seed={seed}) …")
    for name, actions in trajectories.items():
        print(f"  · {name:25s}  len={len(actions):3d}")
        runs[name] = run_trajectory(agent, env_factory, actions, device, seed=seed)

    names = list(runs.keys())
    N = len(names)

    # ── Verificar que los estados FÍSICOS finales coinciden ──────────────────
    final_pos = np.stack([runs[n]["pos"][-1]  for n in names])   # (N, 2)
    final_dir = np.stack([runs[n]["dir"][-1]  for n in names])   # (N,)
    goal_pos  = np.stack([runs[n]["goal_pos"] for n in names])   # (N, 2)

    same_pos  = np.all(final_pos == final_pos[0],   axis=1)
    same_dir  = (final_dir == final_dir[0])
    same_goal = np.all(goal_pos == goal_pos[0],     axis=1)

    print("\nEstado físico final por trayectoria:")
    for i, n in enumerate(names):
        ok = "✓" if (same_pos[i] and same_dir[i] and same_goal[i]) else "✗"
        print(f"  {ok}  {n:25s}  pos={tuple(final_pos[i])}  dir={final_dir[i]}  goal={tuple(goal_pos[i])}")

    valid = same_pos & same_dir & same_goal
    if not valid.all():
        bad = [n for n, ok in zip(names, valid) if not ok]
        print(f"\n  [warn] descartando trayectorias con estado final distinto: {bad}")
        names = [n for n in names if n not in bad]
        N = len(names)
    if N < 2:
        raise RuntimeError("No quedan suficientes trayectorias con el mismo estado final.")

    # ── Apilar último logit/z/feat de las trayectorias válidas ───────────────
    last_logits = torch.stack([runs[n]["logits"][-1] for n in names], dim=0).to(device)  # (N,S,K)
    last_zs     = torch.stack([runs[n]["zs"][-1]     for n in names], dim=0).to(device)  # (N,S,K)
    last_deter  = torch.stack([runs[n]["deter"][-1]  for n in names], dim=0).to(device)  # (N,D)

    S, K = last_logits.shape[1], last_logits.shape[2]
    print(f"\n  Logits finales: shape=({N}, {S}, {K})")

    probs = agent.rssm.get_dist(last_logits).base_dist.probs   # (N, S, K)
    modes = probs.argmax(-1)                                   # (N, S)
    feat  = torch.cat([last_zs.reshape(N, -1), last_deter], dim=-1)  # (N, S*K + D)

    # ── Métricas inter-trayectoria ───────────────────────────────────────────
    ham_modes  = _hamming_pairwise(modes)                      # (N, N)  sobre modes
    ham_zs     = _hamming_pairwise(last_zs.argmax(-1))         # (N, N)  sobre z muestreado
    kl_logits  = _kl_pairwise(agent, last_logits)              # (N, N)
    cos_feat   = _cosine_pairwise(feat)                        # (N, N)

    mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
    H_per_traj  = -(probs * (probs + 1e-8).log()).sum(-1).mean(-1).cpu().numpy()  # (N,)
    pi_per_traj = probs.max(-1).values.mean(-1).cpu().numpy()                      # (N,)
    H_max = float(np.log(K))

    # ── Baseline: variabilidad intra-estado por muestreo ─────────────────────
    intra_hamming = np.array([
        _intra_state_hamming(agent, last_logits[i], n_samples_intra)
        for i in range(N)
    ])
    inter_hamming_modes = ham_modes[mask].cpu().numpy()
    inter_hamming_zs    = ham_zs[mask].cpu().numpy()

    print("\n" + "=" * 64)
    print("  Resultados — Consistencia del z para el mismo estado")
    print("=" * 64)
    print(f"  Trayectorias válidas:                  N = {N}")
    print(f"  Inter-traj  Hamming (sobre modes):     {inter_hamming_modes.mean():.4f}  "
          f"max={inter_hamming_modes.max():.4f}")
    print(f"  Inter-traj  Hamming (sobre z sample):  {inter_hamming_zs.mean():.4f}  "
          f"max={inter_hamming_zs.max():.4f}")
    print(f"  Intra-state Hamming (M={n_samples_intra}):           "
          f"{intra_hamming.mean():.4f}")
    print(f"  Inter-traj  KL simétrica media:        {kl_logits[mask].mean().item():.4f}")
    print(f"  Inter-traj  cos(feat) media:           {cos_feat[mask].mean().item():.4f}  "
          f"(1.0 = idéntico)")
    print(f"  H̄ logit final por traj:               {H_per_traj.mean():.4f}  "
          f"(H_max={H_max:.2f})")
    print(f"  π̄ peak prob por traj:                  {pi_per_traj.mean():.4f}")
    print("=" * 64)

    # ── Plots ────────────────────────────────────────────────────────────────
    _plot_all(
        names=names,
        last_logits=last_logits,
        probs=probs,
        modes=modes,
        last_zs=last_zs,
        ham_modes=ham_modes.cpu().numpy(),
        ham_zs=ham_zs.cpu().numpy(),
        kl_logits=kl_logits.cpu().numpy(),
        cos_feat=cos_feat.cpu().numpy(),
        H_per_traj=H_per_traj,
        intra_hamming=intra_hamming,
        inter_hamming_modes=inter_hamming_modes,
        runs=runs,
        outdir=outdir,
        H_max=H_max,
    )

    # ── GIFs de loops y spin (obs ‖ probs ‖ z) ───────────────────────────────
    make_loop_spin_gifs(agent, runs, outdir, fps=gif_fps)

    # ── JSON ─────────────────────────────────────────────────────────────────
    results = {
        "seed": seed,
        "names": names,
        "N": N, "S": S, "K": K,
        "n_samples_intra": n_samples_intra,
        "final_pos": final_pos.tolist(),
        "final_dir": final_dir.tolist(),
        "goal_pos":  goal_pos[0].tolist(),
        "ham_modes_mean": float(inter_hamming_modes.mean()),
        "ham_modes_max":  float(inter_hamming_modes.max()),
        "ham_zs_mean":    float(inter_hamming_zs.mean()),
        "intra_hamming_mean": float(intra_hamming.mean()),
        "kl_sym_mean":    float(kl_logits[mask].mean().item()),
        "cos_feat_mean":  float(cos_feat[mask].mean().item()),
        "H_per_traj":     H_per_traj.tolist(),
        "H_max":          H_max,
        "pi_per_traj":    pi_per_traj.tolist(),
        "traj_lengths":   {n: len(trajectories[n]) for n in names},
    }
    with open(outdir / "state_traj_consistency.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Guardado en {outdir}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# GIFs por trayectoria: observación ‖ probs posterior ‖ z muestreado
# ─────────────────────────────────────────────────────────────────────────────

_DIR_NAME = {0: "east", 1: "south", 2: "west", 3: "north"}


def _render_gif_frame(image, probs_sk, z_sk, *, step, n_steps, name, dir_, pos):
    """Un frame del gif: obs (RGB) | probs posterior (S×K) | z one-hot (S×K)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), constrained_layout=True)

    axes[0].imshow(image)
    axes[0].set_title(f"observación   t={step}/{n_steps}\n"
                      f"pos={tuple(int(p) for p in pos)}  dir={_DIR_NAME[int(dir_)]}")
    axes[0].axis("off")

    im1 = axes[1].imshow(probs_sk, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                         interpolation="nearest")
    axes[1].set_title("probs posterior  p(z|·)")
    axes[1].set_xlabel("clase k"); axes[1].set_ylabel("slot s")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label="p_sk")

    im2 = axes[2].imshow(z_sk, aspect="auto", cmap="Greys", vmin=0, vmax=1,
                         interpolation="nearest")
    axes[2].set_title("z muestreado  (one-hot)")
    axes[2].set_xlabel("clase k"); axes[2].set_ylabel("slot s")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, label="z_sk")

    fig.suptitle(name, fontsize=13, fontweight="bold")
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def _make_trajectory_gif(agent, name, run, outdir, fps=2):
    """Crea {outdir}/gif_{name}.gif con un frame por paso de la trayectoria."""
    import imageio.v2 as imageio

    logits_t = run["logits"]                       # (T+1, S, K)
    probs_t  = agent.rssm.get_dist(
        logits_t.to(agent.device)).base_dist.probs.cpu().numpy()  # (T+1, S, K)
    z_t      = run["zs"].cpu().numpy()             # (T+1, S, K)  one-hot
    images   = run["images"]                       # (T+1, H, W, 3)
    n_steps  = len(images) - 1

    frames = [
        _render_gif_frame(
            images[t], probs_t[t], z_t[t],
            step=t, n_steps=n_steps, name=name,
            dir_=run["dir"][t], pos=run["pos"][t],
        )
        for t in range(len(images))
    ]
    path = outdir / f"gif_{name}.gif"
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"  GIF:  {path}  ({len(frames)} frames)")


def make_loop_spin_gifs(agent, runs, outdir, fps=2):
    """Genera un gif por cada trayectoria de loop / spin."""
    gif_dir = outdir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    selected = [n for n in runs if n.startswith(("loop", "spin"))]
    print(f"\nGenerando {len(selected)} gifs (loops + spin) en {gif_dir} …")
    for name in selected:
        _make_trajectory_gif(agent, name, runs[name], gif_dir, fps=fps)


def _plot_all(
    *, names, last_logits, probs, modes, last_zs, ham_modes, ham_zs, kl_logits,
    cos_feat, H_per_traj, intra_hamming, inter_hamming_modes, runs, outdir, H_max,
):
    N, S, K = probs.shape
    short = [n.replace("_", "\n", 1) for n in names]

    # ── Heatmap pairwise: Hamming sobre modes ────────────────────────────────
    fig, ax = plt.subplots(figsize=(0.6 * N + 4, 0.55 * N + 3),
                           constrained_layout=True)
    im = ax.imshow(ham_modes, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(N)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N)); ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Hamming inter-trayectoria (sobre argmax_k logit)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Hamming")
    _save(fig, outdir / "pairwise_hamming_modes.png")

    # ── Heatmap pairwise: KL simétrica ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(0.6 * N + 4, 0.55 * N + 3),
                           constrained_layout=True)
    im = ax.imshow(kl_logits, cmap="viridis")
    ax.set_xticks(range(N)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N)); ax.set_yticklabels(names, fontsize=8)
    ax.set_title("KL simétrica inter-trayectoria (logits posteriores)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="KL (nats)")
    _save(fig, outdir / "pairwise_kl.png")

    # ── Heatmap pairwise: cos(feat) ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(0.6 * N + 4, 0.55 * N + 3),
                           constrained_layout=True)
    im = ax.imshow(cos_feat, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(N)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N)); ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Cos(feat) inter-trayectoria  (z ‖ deter)")
    plt.colorbar(im, ax=ax, fraction=0.046, label="cos")
    _save(fig, outdir / "pairwise_cosine_feat.png")

    # ── Comparativa inter vs intra Hamming ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(inter_hamming_modes, bins=20, alpha=0.7, color="steelblue",
            label=f"inter-traj media={inter_hamming_modes.mean():.3f}")
    ax.hist(intra_hamming, bins=20, alpha=0.7, color="coral",
            label=f"intra-state media={intra_hamming.mean():.3f}")
    ax.set_xlabel("Hamming normalizada")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Variabilidad inter-trayectoria vs intra-estado")
    ax.legend()
    _save(fig, outdir / "inter_vs_intra_hamming.png")

    # ── Entropía media por trayectoria ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * N + 2), 5),
                           constrained_layout=True)
    ax.bar(range(N), H_per_traj, color="steelblue", alpha=0.8)
    ax.axhline(H_max, color="r", linestyle="--", label=f"H_max={H_max:.2f}")
    ax.set_xticks(range(N)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("H media (nats)")
    ax.set_title("Entropía media del logit final por trayectoria")
    ax.legend()
    _save(fig, outdir / "entropy_per_trajectory.png")

    # ── Heatmap de probs por trayectoria (S × K) ─────────────────────────────
    probs_np = probs.cpu().numpy()
    for i, name in enumerate(names):
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        im = ax.imshow(probs_np[i], aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Logit final como prob — {name}")
        ax.set_xlabel("Clase k")
        ax.set_ylabel("Slot s")
        plt.colorbar(im, ax=ax, fraction=0.046, label="p_sk")
        _save(fig, outdir / f"probs_{name}.png")

    # ── Evolución de entropía durante cada trayectoria ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name in names:
        logits_t = runs[name]["logits"]                  # (T+1, S, K)
        p = torch.softmax(logits_t, dim=-1)
        H_t = -(p * (p + 1e-8).log()).sum(-1).mean(-1).numpy()
        ax.plot(H_t, label=name, alpha=0.85)
    ax.axhline(H_max, color="r", linestyle="--", alpha=0.6, label=f"H_max={H_max:.2f}")
    ax.set_xlabel("Paso de la trayectoria")
    ax.set_ylabel("H media del logit posterior")
    ax.set_title("Entropía posterior a lo largo de cada trayectoria")
    ax.legend(fontsize=7, loc="best", ncol=2)
    _save(fig, outdir / "entropy_over_time.png")

    # ── Mapa de modos: para cada slot s, qué clase k es la moda en cada traj ─
    modes_np = modes.cpu().numpy()                        # (N, S)
    fig_h = max(4, 0.35 * N)
    fig, ax = plt.subplots(figsize=(min(20, 0.25 * S + 2), fig_h),
                           constrained_layout=True)
    im = ax.imshow(modes_np, aspect="auto", cmap="tab20",
                   interpolation="nearest", vmin=0, vmax=K - 1)
    ax.set_yticks(range(N)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Slot s")
    ax.set_title("Clase modal por slot (filas = trayectorias)")
    plt.colorbar(im, ax=ax, fraction=0.025, label="argmax_k")
    _save(fig, outdir / "modes_per_trajectory.png")

    # ── Mapa del z muestreado: clase one-hot del stoch real en cada slot ─────
    zs_np = last_zs.argmax(-1).cpu().numpy()              # (N, S)
    fig, ax = plt.subplots(figsize=(min(20, 0.25 * S + 2), fig_h),
                           constrained_layout=True)
    im = ax.imshow(zs_np, aspect="auto", cmap="tab20",
                   interpolation="nearest", vmin=0, vmax=K - 1)
    ax.set_yticks(range(N)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Slot s")
    ax.set_title("z muestreado por slot (filas = trayectorias)")
    plt.colorbar(im, ax=ax, fraction=0.025, label="argmax_k z")
    _save(fig, outdir / "zs_per_trajectory.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    uv run python -m experiments.state_traj_consistency \
        --logdir logdir/wm_only_random_mission/01 \
        --device cpu \
        --seed 0 \
        --n_samples 50
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir",    required=True)
    parser.add_argument("--device",    default=None)
    parser.add_argument("--seed",      type=int, default=0,
                        help="Seed para el reset del env (fija la posición del goal)")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="Muestras por estado para la variabilidad intra-state")
    parser.add_argument("--max_loop",  type=int, default=None,
                        help="Lado máximo del loop. Si se omite, se usa el máximo "
                             "geométrico que cabe dado el agent_start_pos del env.")
    parser.add_argument("--gif_fps",   type=int, default=2,
                        help="FPS de los gifs por trayectoria (loops y spin).")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)

    config        = OmegaConf.load(logdir / ".hydra" / "config.yaml")
    device        = args.device or config.device
    config.device = device

    # Backfill defaults for keys added after this run was trained.
    if "wm_only" not in config.model:
        OmegaConf.set_struct(config.model, False)
        config.model.wm_only = False

    reward_function = make_reward(config)
    _, eval_envs, obs_space, act_space = make_envs(config.env)

    agent = Dreamer(
        config.model, obs_space, act_space,
        reward_function=reward_function,
    ).to(device)

    checkpoint = torch.load(logdir / "latest.pt", map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    print(f"Checkpoint cargado: {logdir / 'latest.pt'}")

    # El env_factory de cada trayectoria crea un env independiente para que
    # cada trayectoria empiece desde un reset limpio.
    env_factory = lambda: make_env(config.env, 0)

    start_pos, start_dir, size = _read_start_pose(env_factory)
    dir_name = {0: "east", 1: "south", 2: "west", 3: "north"}[start_dir]
    print(f"Pose inicial del env: pos={start_pos}, dir={start_dir} ({dir_name}), "
          f"interior={size - 2}x{size - 2}")

    trajectories = build_trajectories(
        start_pos     = start_pos,
        start_dir     = start_dir,
        size          = size,
        max_loop_side = args.max_loop,
    )

    run_state_traj_consistency(
        agent          = agent,
        env_factory    = env_factory,
        trajectories   = trajectories,
        n_samples_intra= args.n_samples,
        seed           = args.seed,
        outdir         = logdir / "experiments" / "state_traj_consistency",
        gif_fps        = args.gif_fps,
    )
