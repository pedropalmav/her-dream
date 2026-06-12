"""
¿Codifica el posterior z la posición del cuadrado verde (goal)?

Pregunta de investigación
─────────────────────────
En random-goal, los goals muestreados del buffer vienen de episodios donde el
cuadrado verde estaba en OTRA celda. Si algunos slots de z codifican la posición
del cuadrado, un match z-full (`goal_type=full`) contra un goal de otro episodio
es inalcanzable por construcción: esos slots nunca van a coincidir. Esto
explicaría por qué el post-training funciona en fixed-goal (-430/-500) pero se
queda en -1001 en random-goal con la misma receta.

Diseño
──────
FixedGoal y RandomGoal generan observaciones idénticas (mismo grid, misma
cadena de wrappers); la única diferencia es cómo se elige la celda del goal.
Usamos FixedGoal con el config del run de random-goal para barrer TODAS las
celdas interiores como posición del cuadrado, manteniendo fija la pose del
agente, la secuencia de acciones (spin que vuelve a la pose inicial) y todo lo
demás. Para cada posición se corre la misma trayectoria y se toma el posterior
del último paso, con R repeticiones (distinta semilla de Gumbel) para medir la
variabilidad intra-posición de base.

Métricas
────────
Sobre el argmax_k del posterior final, por slot s:
  • Variation ratio INTER-posición (sobre el modo de consenso por posición)
    vs INTRA-posición (entre repeticiones de la misma posición). Slots con
    inter >> intra codifican la posición del cuadrado.
  • Mapa espacial: clase de consenso del slot en función de la celda del goal.
Sobre el reward real (`full_goal_reward`, sample vs sample):
  • Colisión por slot c_s(p,p') = Σ_k p̄[p,s,k]·p̄[p',s,k] y
    log10 P(match z-full) = Σ_s log10 c_s, comparando pares de la MISMA
    posición (diagonal) vs posiciones DISTINTAS (fuera de diagonal). Esto
    cuantifica directamente la hipótesis en unidades del reward.

Salida
──────
Plots y JSON en {logdir}/experiments/goal_position_in_z/.

Uso
───
uv run python -m experiments.goal_position_in_z \
    --logdir logdir/random_goal/wm_only_randomgoal/01 \
    --device cpu
"""

import argparse
import json
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from dreamer import Dreamer
from envs import make_env
from rewards import make_reward

ACT_LEFT = 0


# ─────────────────────────────────────────────────────────────────────────────
# Recolección de observaciones por posición del goal
# ─────────────────────────────────────────────────────────────────────────────


def make_fixed_goal_cfg(env_cfg_resolved: dict, goal_pos: tuple[int, int]):
    """Config de env fixed-goal con el cuadrado en `goal_pos`, resto idéntico."""
    cfg = OmegaConf.create(dict(env_cfg_resolved))
    OmegaConf.set_struct(cfg, False)
    cfg.task = "fixed-goal_"
    cfg.goal_pos_x = int(goal_pos[0])
    cfg.goal_pos_y = int(goal_pos[1])
    return cfg


def collect_obs_sequence(env_cfg_resolved: dict, goal_pos: tuple[int, int], action_indices: list[int]) -> dict:
    """
    Corre `action_indices` en un FixedGoal con el cuadrado en `goal_pos` y
    devuelve la secuencia de observaciones (numpy) y la pose del agente.
    El env es determinista dado el plan de acciones, así que una sola pasada
    basta para todas las repeticiones.
    """
    env = make_env(make_fixed_goal_cfg(env_cfg_resolved, goal_pos), 0)
    obs = env.reset()
    n_actions = env.action_space.shape[0]

    obs_seq = [obs]
    for act_idx in action_indices:
        act = np.zeros(n_actions, dtype=np.float32)
        act[act_idx] = 1.0
        obs, _r, done, _info = env.step(act)
        obs_seq.append(obs)
        if done:
            raise RuntimeError(f"Episodio terminó tempranamente con goal en {goal_pos}")

    base = env
    while hasattr(base, "env"):
        base = base.env
    return {
        "obs_seq": obs_seq,
        "n_actions": n_actions,
        "agent_pos": tuple(int(c) for c in base.agent_pos),
        "agent_dir": int(base.agent_dir),
        "image0": np.array(obs_seq[0]["image"]),
    }


def _stack_obs(per_pos_seqs: list[list[dict]], t: int, repeats: int, device: str) -> dict:
    """Apila la obs del paso `t` de cada posición y repite R veces → (P·R, ...)."""
    keys = per_pos_seqs[0][t].keys()
    out = {}
    for k in keys:
        vals = [seq[t][k] for seq in per_pos_seqs]
        if isinstance(vals[0], np.ndarray):
            arr = np.stack(vals, axis=0)
            tens = torch.as_tensor(arr, dtype=torch.float32, device=device)
            if arr.dtype == np.uint8:
                tens = tens / 255.0
            out[k] = tens.repeat_interleave(repeats, dim=0)
        elif isinstance(vals[0], (bool, np.bool_)):
            tens = torch.tensor([[float(v)] for v in vals], dtype=torch.float32, device=device)
            out[k] = tens.repeat_interleave(repeats, dim=0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Rollout posterior batcheado: (posiciones × repeticiones) en un solo batch
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def batched_posterior(
    agent: Dreamer,
    per_pos_seqs: list[list[dict]],
    action_indices: list[int],
    n_actions: int,
    repeats: int,
    device: str,
) -> dict:
    """
    Procesa la misma secuencia de obs para P posiciones × R repeticiones.
    Las repeticiones comparten observaciones pero muestrean Gumbel independiente
    en cada obs_step, lo que da la variabilidad intra-posición de base.

    Devuelve logits/probs del posterior en t=0 y en el último paso:
      logits_t0, logits_last: (P, R, S, K)
    """
    P = len(per_pos_seqs)
    B = P * repeats
    T = len(action_indices)

    stoch, deter = agent.rssm.initial(B)
    prev_action = torch.zeros(B, n_actions, device=device)

    logits_t0 = None
    for t in range(T + 1):
        obs_t = _stack_obs(per_pos_seqs, t, repeats, device)
        embed = agent.encoder(obs_t)
        is_first = torch.full((B,), t == 0, dtype=torch.bool, device=device)
        stoch, deter, logit = agent.rssm.obs_step(stoch, deter, prev_action, embed, is_first)
        if t == 0:
            logits_t0 = logit.clone()
        if t < T:
            act = torch.zeros(B, n_actions, device=device)
            act[:, action_indices[t]] = 1.0
            prev_action = act

    S, K = logit.shape[-2], logit.shape[-1]
    return {
        "logits_t0": logits_t0.reshape(P, repeats, S, K).cpu(),
        "logits_last": logit.reshape(P, repeats, S, K).cpu(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────


def _variation_ratio(classes: np.ndarray) -> float:
    """1 - frecuencia de la clase más común. 0 = todas iguales. classes: (N,)."""
    _, counts = np.unique(classes, return_counts=True)
    return 1.0 - counts.max() / len(classes)


def slot_metrics(probs: torch.Tensor) -> dict:
    """
    probs: (P, R, S, K). Variación inter vs intra posición por slot, sobre modos.
    """
    P, R, S, K = probs.shape
    modes = probs.argmax(-1).numpy()  # (P, R, S)

    # Consenso por posición: clase mayoritaria entre las R repeticiones.
    consensus = np.zeros((P, S), dtype=np.int64)
    for p in range(P):
        for s in range(S):
            vals, counts = np.unique(modes[p, :, s], return_counts=True)
            consensus[p, s] = vals[counts.argmax()]

    inter_vr = np.array([_variation_ratio(consensus[:, s]) for s in range(S)])
    intra_vr = np.array([np.mean([_variation_ratio(modes[p, :, s]) for p in range(P)]) for s in range(S)])
    n_classes = np.array([len(np.unique(consensus[:, s])) for s in range(S)])

    return {"modes": modes, "consensus": consensus, "inter_vr": inter_vr, "intra_vr": intra_vr, "n_classes": n_classes}


def match_probability(probs: torch.Tensor) -> dict:
    """
    probs: (P, R, S, K). Probabilidad de que dos muestras one-hot independientes
    coincidan en TODOS los slots (= reward `full` distinto de -1), usando la
    prob media por posición p̄[p] = mean_R probs[p].

      c_s(p, p') = Σ_k p̄[p,s,k] · p̄[p',s,k]          (colisión por slot)
      log10 P(match | p, p') = Σ_s log10 c_s(p, p')

    Diagonal = goal de la MISMA posición del cuadrado (caso fixed-goal),
    fuera de diagonal = goal de OTRA posición (caso random-goal + buffer).
    """
    pbar = probs.mean(dim=1)  # (P, S, K)
    collision = torch.einsum("psk,qsk->pqs", pbar, pbar)  # (P, P, S)
    log10_match = torch.log10(collision.clamp_min(1e-30)).sum(-1)  # (P, P)

    P = pbar.shape[0]
    eye = torch.eye(P, dtype=torch.bool)
    coll_same = collision[eye]  # (P, S)
    coll_cross = collision[~eye].reshape(P * (P - 1), -1)  # (P·(P-1), S)

    return {
        "collision": collision.numpy(),
        "log10_match": log10_match.numpy(),
        "log10_match_same": log10_match[eye].numpy(),
        "log10_match_cross": log10_match[~eye].numpy(),
        "coll_per_slot_same": coll_same.mean(0).numpy(),  # (S,)
        "coll_per_slot_cross": coll_cross.mean(0).numpy(),  # (S,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {path}")


def plot_all(
    *,
    positions: list[tuple[int, int]],
    agent_pos: tuple[int, int],
    env_size: int,
    sm: dict,
    mp: dict,
    images0: np.ndarray,
    K: int,
    outdir: Path,
):
    P, S = sm["consensus"].shape
    inter, intra = sm["inter_vr"], sm["intra_vr"]
    order = np.argsort(-inter)

    # ── Inter vs intra variation ratio por slot ──────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, 0.3 * S), 5), constrained_layout=True)
    x = np.arange(S)
    ax.bar(x - 0.2, inter, width=0.4, color="firebrick", label="inter-posición (consenso)")
    ax.bar(x + 0.2, intra, width=0.4, color="steelblue", label="intra-posición (Gumbel)")
    ax.set_xlabel("Slot s")
    ax.set_ylabel("Variation ratio del modo")
    ax.set_title("¿Qué slots cambian cuando solo se mueve el cuadrado verde?")
    ax.set_xticks(x)
    ax.legend()
    _save(fig, outdir / "inter_vs_intra_variation_per_slot.png")

    # ── Consenso (posiciones × slots) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(min(20, 0.3 * S + 2), max(6, 0.12 * P)), constrained_layout=True)
    im = ax.imshow(sm["consensus"], aspect="auto", cmap="tab20", interpolation="nearest", vmin=0, vmax=K - 1)
    ax.set_xlabel("Slot s")
    ax.set_ylabel("Posición del cuadrado (orden y,x)")
    ax.set_title("Clase modal de consenso por slot, una fila por posición del cuadrado")
    plt.colorbar(im, ax=ax, fraction=0.03, label="argmax_k")
    _save(fig, outdir / "consensus_modes_positions_x_slots.png")

    # ── Mapas espaciales de los slots más sensibles ──────────────────────────
    top = order[:8]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, s in zip(axes.flat, top):
        grid = np.full((env_size, env_size), np.nan)
        for (gx, gy), cls in zip(positions, sm["consensus"][:, s]):
            grid[gy, gx] = cls
        im = ax.imshow(grid, cmap="tab20", interpolation="nearest", vmin=0, vmax=K - 1)
        ax.plot(agent_pos[0], agent_pos[1], "r*", markersize=14)
        ax.set_title(f"slot {s}  inter={inter[s]:.2f}  intra={intra[s]:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Clase modal del slot según la celda del cuadrado verde (★ = agente)", fontsize=13)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, label="argmax_k")
    _save(fig, outdir / "spatial_maps_top_slots.png")

    # ── Colisión por slot: misma posición vs cruzada ─────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, 0.3 * S), 5), constrained_layout=True)
    ax.bar(x - 0.2, mp["coll_per_slot_same"], width=0.4, color="seagreen", label="misma posición")
    ax.bar(x + 0.2, mp["coll_per_slot_cross"], width=0.4, color="darkorange", label="posición distinta")
    ax.axhline(1.0 / K, color="k", linestyle="--", alpha=0.6, label=f"azar 1/K={1 / K:.3f}")
    ax.set_xlabel("Slot s")
    ax.set_ylabel("Colisión  Σ_k p̄·p̄'")
    ax.set_yscale("log")
    ax.set_title("Prob. de que el slot coincida entre dos muestras (escala log)")
    ax.set_xticks(x)
    ax.legend()
    _save(fig, outdir / "collision_per_slot_same_vs_cross.png")

    # ── Matriz log10 P(match z-full) ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    im = ax.imshow(mp["log10_match"], cmap="viridis")
    ax.set_xlabel("Posición del cuadrado (goal muestreado)")
    ax.set_ylabel("Posición del cuadrado (episodio actual)")
    ax.set_title("log10 P(match z-full, 32 slots) entre posiciones")
    plt.colorbar(im, ax=ax, fraction=0.046, label="log10 P")
    _save(fig, outdir / "log10_match_matrix.png")

    # ── Histograma same vs cross ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.hist(
        mp["log10_match_same"],
        bins=30,
        alpha=0.75,
        color="seagreen",
        label=f"misma posición  media={mp['log10_match_same'].mean():.1f}",
    )
    ax.hist(
        mp["log10_match_cross"],
        bins=30,
        alpha=0.75,
        color="darkorange",
        label=f"posición distinta  media={mp['log10_match_cross'].mean():.1f}",
    )
    ax.set_xlabel("log10 P(match z-full)")
    ax.set_ylabel("Frecuencia (pares)")
    ax.set_title("Alcanzabilidad del reward `full`: goal del mismo episodio vs de otro")
    ax.legend()
    _save(fig, outdir / "hist_log10_match.png")

    # ── Observaciones de muestra ─────────────────────────────────────────────
    n_show = min(6, len(images0))
    idx = np.linspace(0, len(images0) - 1, n_show).astype(int)
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3.4), constrained_layout=True)
    for ax, i in zip(np.atleast_1d(axes), idx):
        ax.imshow(images0[i])
        ax.set_title(f"goal={positions[i]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Misma pose del agente, solo se mueve el cuadrado verde")
    _save(fig, outdir / "sample_observations.png")


# ─────────────────────────────────────────────────────────────────────────────
# Principal
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_goal_position_in_z(
    agent: Dreamer,
    env_cfg_resolved: dict,
    action_indices: list[int],
    repeats: int,
    device: str,
    outdir: Path,
):
    outdir.mkdir(parents=True, exist_ok=True)

    size = int(env_cfg_resolved["env_size"])
    agent_start = (int(env_cfg_resolved["agent_start_pos_x"]), int(env_cfg_resolved["agent_start_pos_y"]))
    positions = [(x, y) for y in range(1, size - 1) for x in range(1, size - 1) if (x, y) != agent_start]

    print(f"\nBarrido de {len(positions)} posiciones del cuadrado verde (interior {size - 2}x{size - 2}, agente fijo en {agent_start})")
    print(f"Trayectoria por posición: {len(action_indices)} acciones, {repeats} repeticiones (Gumbel)")

    # ── Observaciones por posición (env determinista, una pasada) ────────────
    per_pos = [collect_obs_sequence(env_cfg_resolved, goal_pos, action_indices) for goal_pos in positions]
    poses = {(d["agent_pos"], d["agent_dir"]) for d in per_pos}
    assert len(poses) == 1, f"La pose final del agente difiere entre posiciones del goal: {poses}"
    n_actions = per_pos[0]["n_actions"]
    images0 = np.stack([d["image0"] for d in per_pos])

    # ── Posterior batcheado ──────────────────────────────────────────────────
    out = batched_posterior(agent, [d["obs_seq"] for d in per_pos], action_indices, n_actions, repeats, device)
    probs_last = agent.rssm.get_dist(out["logits_last"].to(device)).base_dist.probs.cpu()  # (P, R, S, K)
    probs_t0 = agent.rssm.get_dist(out["logits_t0"].to(device)).base_dist.probs.cpu()

    P, R, S, K = probs_last.shape

    # ── Métricas ─────────────────────────────────────────────────────────────
    sm = slot_metrics(probs_last)
    sm_t0 = slot_metrics(probs_t0)
    mp = match_probability(probs_last)

    margin = 0.05
    sensitive = np.where(sm["inter_vr"] > sm["intra_vr"] + margin)[0]
    order = np.argsort(-sm["inter_vr"])

    print("\n" + "=" * 72)
    print("  Resultados — ¿z codifica la posición del cuadrado verde?")
    print("=" * 72)
    print(f"  Slots sensibles a la posición (inter > intra + {margin}):  {len(sensitive)}/{S}")
    print(f"    → {sensitive.tolist()}")
    print("  Top 10 slots por variación inter-posición:")
    for s in order[:10]:
        print(
            f"    slot {s:2d}: inter={sm['inter_vr'][s]:.3f}  intra={sm['intra_vr'][s]:.3f}  "
            f"clases distintas={sm['n_classes'][s]:2d}  "
            f"colisión cross={mp['coll_per_slot_cross'][s]:.3f} (same={mp['coll_per_slot_same'][s]:.3f})"
        )
    print(f"\n  log10 P(match z-full)  misma posición:    {mp['log10_match_same'].mean():.2f}")
    print(f"  log10 P(match z-full)  posición distinta: {mp['log10_match_cross'].mean():.2f}")
    same_steps = 10 ** -mp["log10_match_same"].mean()
    cross_steps = 10 ** -mp["log10_match_cross"].mean()
    print(f"  → pasos esperados hasta un match: {same_steps:.1e} (misma) vs {cross_steps:.1e} (distinta)")
    print("=" * 72)

    plot_all(
        positions=positions,
        agent_pos=agent_start,
        env_size=size,
        sm=sm,
        mp=mp,
        images0=images0,
        K=K,
        outdir=outdir,
    )

    results = {
        "P": P,
        "R": R,
        "S": S,
        "K": K,
        "positions": [list(p) for p in positions],
        "agent_start": list(agent_start),
        "n_actions_traj": len(action_indices),
        "inter_vr": sm["inter_vr"].tolist(),
        "intra_vr": sm["intra_vr"].tolist(),
        "n_classes_per_slot": sm["n_classes"].tolist(),
        "sensitive_slots": sensitive.tolist(),
        "inter_vr_t0": sm_t0["inter_vr"].tolist(),
        "intra_vr_t0": sm_t0["intra_vr"].tolist(),
        "coll_per_slot_same": mp["coll_per_slot_same"].tolist(),
        "coll_per_slot_cross": mp["coll_per_slot_cross"].tolist(),
        "log10_match_same_mean": float(mp["log10_match_same"].mean()),
        "log10_match_cross_mean": float(mp["log10_match_cross"].mean()),
        "consensus_modes": sm["consensus"].tolist(),
    }
    with open(outdir / "goal_position_in_z.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Guardado en {outdir}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True, help="Run con .hydra/config.yaml y latest.pt (WM a inspeccionar)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--repeats", type=int, default=8, help="Repeticiones por posición (ruido Gumbel)")
    parser.add_argument(
        "--spin", type=int, default=4, help="Giros a la izquierda antes de medir (múltiplo de 4 = misma pose)"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    logdir = pathlib.Path(args.logdir)

    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")
    device = args.device or config.device
    config.device = device

    # Backfill defaults for keys added after this run was trained.
    if "wm_only" not in config.model:
        OmegaConf.set_struct(config.model, False)
        config.model.wm_only = False

    resolved = OmegaConf.to_container(config, resolve=True)
    resolved["env"]["device"] = device
    env_cfg_resolved = resolved["env"]

    # Espacios de obs/acción desde un env igual al de entrenamiento.
    probe_env = make_env(OmegaConf.create(env_cfg_resolved), 0)
    obs_space, act_space = probe_env.observation_space, probe_env.action_space

    reward_function = make_reward(config)
    agent = Dreamer(config.model, obs_space, act_space, reward_function=reward_function).to(device)
    checkpoint = torch.load(logdir / "latest.pt", map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    print(f"Checkpoint cargado: {logdir / 'latest.pt'}")

    run_goal_position_in_z(
        agent=agent,
        env_cfg_resolved=env_cfg_resolved,
        action_indices=[ACT_LEFT] * args.spin,
        repeats=args.repeats,
        device=device,
        outdir=logdir / "experiments" / "goal_position_in_z",
    )
