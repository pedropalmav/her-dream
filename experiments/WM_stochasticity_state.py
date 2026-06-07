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

# ─────────────────────────────────────────────────────────────────────────────
# Recolección de estados finales de trayectorias
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def collect_final_states(
    agent: Dreamer,
    env_factory,
    n_trajectories: int,
    traj_len: int,
    device: str,
) -> dict:
    """
    Corre `n_trajectories` trayectorias de `traj_len` pasos con acciones aleatorias
    y, para cada trayectoria, devuelve únicamente el logit posterior y la muestra z
    del ÚLTIMO estado alcanzado.

    Retorna
    -------
    logits : (N, S, K)   logits posteriores en el último paso de cada trayectoria
    zs     : (N, S, K)   muestras one-hot correspondientes al último paso
    """
    last_logits = []
    last_zs = []

    for n in range(n_trajectories):
        env = env_factory()
        obs = env.reset()

        # Estado inicial del RSSM
        stoch, deter = agent.rssm.initial(1)

        act_space = env.action_space
        n_actions = act_space.shape[0]
        action = torch.zeros(1, n_actions, device=device)

        logit = None
        for t in range(traj_len):
            obs_t = _preprocess_obs(obs, device)
            is_first = torch.tensor([t == 0], dtype=torch.bool, device=device)

            embed = agent.encoder(obs_t)  # (1, E)
            stoch, deter, logit = agent.rssm.obs_step(
                stoch,
                deter,
                action,
                embed,
                is_first,
            )  # (1,S,K),(1,D),(1,S,K)

            # Acción aleatoria one-hot
            act_idx = np.random.randint(n_actions)
            act_np = np.zeros(n_actions, dtype=np.float32)
            act_np[act_idx] = 1.0
            action = torch.as_tensor(act_np, device=device).unsqueeze(0)

            obs, _, done, _ = env.step(act_np)
            if done:
                break

        # Guardar solo el último estado
        last_logits.append(logit.squeeze(0))  # (S, K)
        last_zs.append(stoch.squeeze(0))  # (S, K)

        if (n + 1) % 10 == 0:
            print(f"  Trayectoria {n + 1}/{n_trajectories}")

    logits = torch.stack(last_logits, dim=0)  # (N, S, K)
    zs = torch.stack(last_zs, dim=0)  # (N, S, K)
    return {"logits": logits, "zs": zs}


def _preprocess_obs(obs: dict, device: str) -> dict:
    """
    Convierte el obs del env (dict de numpy) a tensores (1, ...) en `device`.
    Solo incluye las claves que el encoder espera.
    """
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            t = torch.as_tensor(v, dtype=torch.float32, device=device)
            if v.dtype == np.uint8:
                t = t / 255.0
            out[k] = t.unsqueeze(0)  # (1, ...)
        elif isinstance(v, (bool, np.bool_)):
            out[k] = torch.tensor([[float(v)]], dtype=torch.float32, device=device)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Experimentos sobre el último estado posterior (N, S, K)
# ─────────────────────────────────────────────────────────────────────────────

# ── Exp 1 — Diversidad inter-trayectoria ─────────────────────────────────────
# d(i,j)  = (1/S) * sum_s 1[ mode_i_s != mode_j_s ]
# D_inter = 2/(N(N-1)) * sum_{i<j} d(i,j)


def exp1_inter_traj_diversity(agent, logits: torch.Tensor) -> dict:
    """
    logits : (N, S, K)
    """
    N = logits.shape[0]
    probs = agent.rssm.get_dist(logits).base_dist.probs  # (N, S, K)
    modes = probs.argmax(-1)  # (N, S)

    diff = (modes.unsqueeze(0) != modes.unsqueeze(1)).float()  # (N, N, S)
    ham = diff.mean(-1)  # (N, N)
    mask = torch.triu(torch.ones(N, N, device=logits.device), diagonal=1).bool()
    D_inter = ham[mask].mean().item()

    return {
        "D_inter": float(D_inter),
        "ham_matrix": ham.cpu().numpy(),  # (N, N)
    }


# ── Exp 2 — Varianza intra-estado (sampling sobre el mismo logit) ────────────
# delta(m1,m2) = (1/S) * sum_s 1[ argmax z^m1_s != argmax z^m2_s ]
# D_intra^n    = 2/(M(M-1)) * sum_{m1<m2} delta(m1,m2)
# D_intra      = (1/N) * sum_n D_intra^n


def exp2_intra_state_variance(
    agent,
    logits: torch.Tensor,
    M: int,
) -> dict:
    """
    logits : (N, S, K)   logits del último estado por trayectoria
    M      : muestras por estado
    """
    N = logits.shape[0]
    device = logits.device

    mask = torch.triu(torch.ones(M, M, device=device), diagonal=1).bool()
    D_intra_per_state = np.zeros(N)

    for i in range(N):
        l_i = logits[i].unsqueeze(0).expand(M, -1, -1)  # (M, S, K)
        samples = agent.rssm.get_dist(l_i).rsample()  # (M, S, K)
        classes = samples.argmax(-1)  # (M, S)
        diff = (classes.unsqueeze(0) != classes.unsqueeze(1)).float()
        ham = diff.mean(-1)  # (M, M)
        D_intra_per_state[i] = ham[mask].mean().item()

    return {
        "D_intra": float(D_intra_per_state.mean()),
        "D_intra_per_state": D_intra_per_state,  # (N,)
    }


# ── Exp 3 — Entropía de los logits posteriores ───────────────────────────────
# H^n_s = - sum_k p^n_sk * log p^n_sk


def exp3_posterior_entropy(agent, logits: torch.Tensor) -> dict:
    """
    logits : (N, S, K)
    """
    probs = agent.rssm.get_dist(logits).base_dist.probs
    H = -(probs * (probs + 1e-8).log()).sum(-1)  # (N, S)
    H_np = H.cpu().numpy()
    K = logits.shape[-1]
    H_max = float(np.log(K))

    return {
        "H_global": float(H_np.mean()),
        "H_max": H_max,
        "H_per_slot": H_np.mean(axis=0),  # (S,)
        "H_per_traj": H_np.mean(axis=1),  # (N,)
        "H_all": H_np,  # (N, S)
        "frac_peaked": float((H_np < 0.1 * H_max).mean()),
        "frac_flat": float((H_np > 0.9 * H_max).mean()),
    }


# ── Exp 4 — Peak probability ──────────────────────────────────────────────────
# pi^n_s = max_k p^n_sk


def exp4_peak_probability(agent, logits: torch.Tensor) -> dict:
    """
    logits : (N, S, K)
    """
    probs = agent.rssm.get_dist(logits).base_dist.probs
    pi = probs.max(-1).values.cpu().numpy()  # (N, S)
    K = logits.shape[-1]

    return {
        "pi_bar": float(pi.mean()),
        "pi_chance": 1.0 / K,
        "pi_all": pi,  # (N, S)
    }


# ── Exp 5 — Heatmaps de probabilidad por trayectoria ─────────────────────────


def exp5_probability_heatmaps(agent, logits: torch.Tensor, n_show: int = 4) -> np.ndarray:
    """
    logits : (N, S, K)
    Devuelve (n_show, S, K) con la distribución por trayectoria (con unimix del RSSM).
    """
    probs = agent.rssm.get_dist(logits[:n_show]).base_dist.probs
    return probs.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_wm_stochasticity(
    agent,
    env_factory,
    n_trajectories: int = 200,
    traj_len: int = 50,
    n_samples_per_state: int = 50,
    n_show_heatmaps: int = 4,
    outdir: Path = Path("results/wm_stoch"),
):
    outdir.mkdir(parents=True, exist_ok=True)
    device = agent.device

    # ── Recolectar último estado de cada trayectoria ─────────────────────────
    print(
        f"\nRecolectando {n_trajectories} trayectorias de longitud {traj_len} "
        f"(quedándonos con el último estado de cada una) …"
    )
    data = collect_final_states(agent, env_factory, n_trajectories, traj_len, device)
    logits = data["logits"]  # (N, S, K)
    N, S, K = logits.shape
    print(f"  Shape logits: {tuple(logits.shape)}  (N={N}, S={S}, K={K})")

    # ── Experimentos ──────────────────────────────────────────────────────────
    print("Exp 1: diversidad inter-trayectoria …")
    r1 = exp1_inter_traj_diversity(agent, logits)

    print("Exp 2: varianza intra-estado …")
    r2 = exp2_intra_state_variance(agent, logits, n_samples_per_state)

    print("Exp 3: entropía posterior …")
    r3 = exp3_posterior_entropy(agent, logits)

    print("Exp 4: peak probability …")
    r4 = exp4_peak_probability(agent, logits)

    print("Exp 5: heatmaps de probabilidad …")
    r5 = exp5_probability_heatmaps(agent, logits, n_show=n_show_heatmaps)

    # ── Consola ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Resultados — World Model z Stochasticity (último estado)")
    print("=" * 60)
    print(f"  [Exp 1] D_inter  (Hamming inter-traj):  {r1['D_inter']:.4f}")
    print(f"  [Exp 2] D_intra  (Hamming intra-state): {r2['D_intra']:.4f}  (ideal ≪ D_inter)")
    print(f"  [Exp 3] H global (entropía posterior):  {r3['H_global']:.4f}  (H_max={r3['H_max']:.2f})")
    print(f"  [Exp 3] Slots peaked (<10% H_max):      {r3['frac_peaked'] * 100:.1f}%")
    print(f"  [Exp 3] Slots planos (>90% H_max):      {r3['frac_flat'] * 100:.1f}%")
    print(f"  [Exp 4] pi_bar   (peak prob):           {r4['pi_bar']:.4f}  (chance={r4['pi_chance']:.4f})")
    print("=" * 60)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_all(r1, r2, r3, r4, r5, S, N, outdir)

    # ── JSON ──────────────────────────────────────────────────────────────────
    results = {
        "n_trajectories": N,
        "S": S,
        "K": K,
        "exp1": {
            "D_inter": r1["D_inter"],
        },
        "exp2": {
            "D_intra": r2["D_intra"],
            "D_intra_per_state": r2["D_intra_per_state"].tolist(),
        },
        "exp3": {
            "H_global": r3["H_global"],
            "H_max": r3["H_max"],
            "H_per_slot": r3["H_per_slot"].tolist(),
            "H_per_traj": r3["H_per_traj"].tolist(),
            "frac_peaked": r3["frac_peaked"],
            "frac_flat": r3["frac_flat"],
        },
        "exp4": {
            "pi_bar": r4["pi_bar"],
            "pi_chance": r4["pi_chance"],
        },
    }
    with open(outdir / "wm_stoch_results.json", "w") as f:
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


def _plot_all(r1, r2, r3, r4, r5, S, N, outdir):
    H_max = r3["H_max"]

    # ── Exp 1 — Histograma de d(i,j) ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pairs = r1["ham_matrix"][mask]
    ax.hist(pairs, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(r1["D_inter"], color="r", linestyle="--", label=f"media={r1['D_inter']:.3f}")
    ax.set_title("Exp 1 — Distribución de d(i,j) (Hamming inter-trayectoria)")
    ax.set_xlabel("Hamming por par")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    _save(fig, outdir / "exp1_hist_inter_traj.png")

    # ── Exp 2 — Distribución D_intra por trayectoria ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.hist(r2["D_intra_per_state"], bins=40, color="coral", edgecolor="white")
    ax.axvline(r2["D_intra"], color="darkred", linestyle="--", label=f"D_intra media={r2['D_intra']:.4f}")
    ax.axvline(r1["D_inter"], color="steelblue", linestyle="--", label=f"D_inter={r1['D_inter']:.3f}")
    ax.set_title("Exp 2 — Distribución de D_intra por trayectoria\n(varianza intra entre M muestras del mismo logit)")
    ax.set_xlabel("D_intra^n")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    _save(fig, outdir / "exp2_hist_intra_state.png")

    # ── Exp 3 — Boxplot H por slot ───────────────────────────────────────────
    box_w = max(14, S * 0.45)
    fig, ax = plt.subplots(figsize=(box_w, 6), constrained_layout=True)
    ax.boxplot(
        r3["H_all"],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="steelblue", linewidth=2),
    )
    ax.axhline(H_max, color="r", linestyle="--", label=f"H_max={H_max:.2f}")
    ax.set_title("Exp 3 — Distribución H por slot")
    ax.set_xlabel("Slot s")
    ax.set_ylabel("Entropía (nats)")
    ax.legend()
    _save(fig, outdir / "exp3_box_H_per_slot.png")

    # ── Exp 3 — H_s media por slot (barras) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(range(S), r3["H_per_slot"], color="steelblue", alpha=0.8)
    ax.axhline(H_max, color="r", linestyle="--", label=f"H_max={H_max:.2f}")
    ax.axhline(r3["H_global"], color="orange", linestyle="--", label=f"H̄={r3['H_global']:.3f}")
    ax.set_title("Exp 3 — H_s media por slot")
    ax.set_xlabel("Slot s")
    ax.set_ylabel("H_s (nats)")
    ax.legend()
    _save(fig, outdir / "exp3_bar_H_per_slot.png")

    # ── Exp 3 — Histograma global de H^n_s ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(r3["H_all"].flatten(), bins=50, color="steelblue", edgecolor="white")
    ax.axvline(H_max, color="r", linestyle="--", label=f"H_max={H_max:.2f}")
    ax.axvline(r3["H_global"], color="orange", linestyle="--", label=f"H̄={r3['H_global']:.3f}")
    ax.set_title("Exp 3 — Histograma global de H^n_s")
    ax.set_xlabel("Entropía (nats)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    _save(fig, outdir / "exp3_hist_H_global.png")

    # ── Exp 3 — Heatmap H^n_s (trayectorias × slots) ─────────────────────────
    hm_w = max(12, S * 0.4)
    hm_h = min(20, max(8, N * 0.06))
    fig, ax = plt.subplots(figsize=(hm_w, hm_h), constrained_layout=True)
    im = ax.imshow(r3["H_all"], aspect="auto", cmap="hot", vmin=0, vmax=H_max, interpolation="nearest")
    ax.set_title("Exp 3 — Heatmap H^n_s (trayectorias × slots)")
    ax.set_xlabel("Slot s")
    ax.set_ylabel("Trayectoria n")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="H (nats)")
    _save(fig, outdir / "exp3_heatmap_H.png")

    # ── Exp 4 — Distribución de π^n_s ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(r4["pi_all"].flatten(), bins=50, color="mediumseagreen", edgecolor="white")
    ax.axvline(r4["pi_bar"], color="darkgreen", linestyle="--", label=f"π̄={r4['pi_bar']:.3f}")
    ax.axvline(r4["pi_chance"], color="r", linestyle="--", label=f"1/K={r4['pi_chance']:.3f}")
    ax.set_title("Exp 4 — Distribución π^n_s (peak prob)")
    ax.set_xlabel("max_k p^n_sk")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    _save(fig, outdir / "exp4_hist_peak_prob.png")

    # ── Exp 4 — Heatmap π^n_s ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(r4["pi_all"], aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_title("Exp 4 — Heatmap π^n_s (trayectorias × slots)")
    ax.set_xlabel("Slot s")
    ax.set_ylabel("Trayectoria n")
    plt.colorbar(im, ax=ax, fraction=0.046, label="π^n_s")
    _save(fig, outdir / "exp4_heatmap_peak_prob.png")

    # ── Exp 5 — Heatmaps de probabilidad por trayectoria ─────────────────────
    n_show = r5.shape[0]
    for idx in range(n_show):
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        im = ax.imshow(r5[idx], aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Exp 5 — P^{idx} (S×K) — trayectoria {idx}")
        ax.set_xlabel("Clase k")
        ax.set_ylabel("Slot s")
        plt.colorbar(im, ax=ax, fraction=0.046, label="p^n_sk")
        _save(fig, outdir / f"exp5_prob_traj_{idx}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    uv run python -m experiments.WM_stochasticity_state \
        --logdir logdir/text_goal_sample_normal_buffer/01 \
        --device cpu \
        --n_traj 200 \
        --traj_len 50 \
        --n_samples 50
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_traj", type=int, default=200, help="Número de trayectorias aleatorias")
    parser.add_argument(
        "--traj_len", type=int, default=50, help="Pasos por trayectoria (solo se guarda el último estado)"
    )
    parser.add_argument("--n_samples", type=int, default=50, help="Muestras del posterior por estado (Exp 2)")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)

    # ── Config ────────────────────────────────────────────────────────────────
    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")
    device = args.device or config.device
    config.device = device

    # Backfill defaults for keys added after this run was trained.
    if "wm_only" not in config.model:
        OmegaConf.set_struct(config.model, False)
        config.model.wm_only = False

    # ── Agente ───────────────────────────────────────────────────────────────
    reward_function = make_reward(config)

    from envs import make_envs

    _, eval_envs, obs_space, act_space = make_envs(config.env)

    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
        reward_function=reward_function,
    ).to(device)

    checkpoint = torch.load(logdir / "latest.pt", map_location=device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    print(f"Checkpoint cargado: {logdir / 'latest.pt'}")

    # ── Factory de env individual ─────────────────────────────────────────────
    def env_factory():
        return make_env(config.env, 0)

    # ── Correr ───────────────────────────────────────────────────────────────
    run_wm_stochasticity(
        agent=agent,
        env_factory=env_factory,
        n_trajectories=args.n_traj,
        traj_len=args.traj_len,
        n_samples_per_state=args.n_samples,
        outdir=logdir / "experiments" / "wm_stoch",
    )
