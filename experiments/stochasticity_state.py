import argparse
import json
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from dreamer import Dreamer
from envs import make_envs, make_env
from rewards import make_reward


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades compartidas
# ─────────────────────────────────────────────────────────────────────────────

def get_logits_and_probs(agent, tokens: torch.Tensor):
    """
    tokens : (N, 1, L, V)
    returns
      logits : (N, S, K)
      probs  : (N, S, K)   softmax de logits
    """
    logits = agent.text_encoder(tokens)[:, 0]   # (N, S, K)
    probs = F.softmax(logits, dim=-1)
    return logits, probs


def encode_missions(env, n: int, device: str) -> torch.Tensor:
    """Devuelve tokens (N, 1, L, V) listos para el encoder."""
    raw = np.stack(
        [env.encoded_random_mission()() for _ in range(n)]
    ).astype(np.float32)                          # (N, L, V)
    tokens = torch.as_tensor(raw, device=device)  # (N, L, V)
    return tokens.unsqueeze(1)                    # (N, 1, L, V)


# ─────────────────────────────────────────────────────────────────────────────
# Experimento principal: corre los 5 sub-experimentos sobre el text encoder
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def exp_text_encoder_stochasticity(
    agent: Dreamer,
    env,
    n_missions: int = 200,
    n_samples_per_mission: int = 50,
    n_show_heatmaps: int = 4,
    outdir: Path = Path("results/text_stoch"),
):
    """
    Análisis completo de la estocasticidad del text encoder.

    Para `n_missions` misiones aleatorias se calculan, en orden:
      Exp 1 — Diversidad inter-misión (Hamming sobre modos).
      Exp 2 — Varianza intra-misión   (Hamming entre M muestras de la misma misión).
      Exp 3 — Entropía de los logits   (global, por slot, fracciones peaked/flat).
      Exp 4 — Peak probability         (concentración del modo).
      Exp 5 — Heatmaps de probabilidad para varias misiones.

    Responde: ¿la variabilidad de z viene del muestreo o los logits ya son peaked?
    ¿Las distintas misiones producen modos distintos o el encoder colapsa?
    """
    outdir.mkdir(parents=True, exist_ok=True)
    device = agent.device

    # ── Preparar tokens ───────────────────────────────────────────────────────
    print(f"Generando {n_missions} misiones …")
    tokens = encode_missions(env, n_missions, device)            # (N,1,L,V)
    logits, probs = get_logits_and_probs(agent, tokens)          # (N,S,K)

    S, K = logits.shape[1], logits.shape[2]
    print(f"  Slots S={S}, Clases K={K}")

    # ── Exp 1: diversidad inter-misión (Hamming sobre modos) ─────────────────
    # d(i,j)  = (1/S) * sum_s 1[ mode_i_s != mode_j_s ]
    # D_inter = 2/(N(N-1)) * sum_{i<j} d(i,j)
    print("Exp 1: diversidad inter-misión …")
    modes = probs.argmax(dim=-1)                                 # (N, S)
    diff = (modes.unsqueeze(0) != modes.unsqueeze(1)).float()    # (N, N, S)
    ham_inter = diff.mean(-1)                                    # (N, N)
    mask_inter = torch.triu(
        torch.ones(n_missions, n_missions, device=device), diagonal=1
    ).bool()
    D_inter = ham_inter[mask_inter].mean().item()
    ham_matrix = ham_inter.cpu().numpy()

    # ── Exp 2: varianza intra-misión (Hamming entre M muestras) ──────────────
    # delta(m1,m2) = (1/S) * sum_s 1[ argmax z^m1_s != argmax z^m2_s ]
    # D_intra^i    = 2/(M(M-1)) * sum_{m1<m2} delta(m1,m2)
    # D_intra      = (1/N) * sum_i D_intra^i
    print("Exp 2: varianza intra-misión …")
    D_intra_per_mission = np.zeros(n_missions)
    M = n_samples_per_mission
    mask_intra = torch.triu(torch.ones(M, M, device=device), diagonal=1).bool()

    for i in range(n_missions):
        t_i = tokens[i:i+1].expand(M, -1, -1, -1)                # (M, 1, L, V)
        logits_i = agent.text_encoder(t_i)[:, 0]                 # (M, S, K)
        samples = agent.rssm.get_dist(logits_i).rsample()                     # (M, S, K) one-hot
        classes = samples.argmax(-1)                             # (M, S)
        diff_i = (classes.unsqueeze(0) != classes.unsqueeze(1)).float()
        ham_i = diff_i.mean(-1)                                  # (M, M)
        D_intra_per_mission[i] = ham_i[mask_intra].mean().item()

    D_intra = D_intra_per_mission.mean()

    # ── Exp 3: entropía de los logits ────────────────────────────────────────
    # H^i_s = - sum_k p^i_sk * log(p^i_sk)
    print("Exp 3: entropía de logits …")
    H_all_t = -(probs * (probs + 1e-8).log()).sum(-1)            # (N, S)
    H_all = H_all_t.cpu().numpy()
    H_per_slot = H_all.mean(0)                                   # (S,)
    H_global = float(H_all.mean())
    H_max = float(np.log(K))
    frac_peaked = float((H_all < 0.1 * H_max).mean())
    frac_flat = float((H_all > 0.9 * H_max).mean())

    # ── Exp 4: peak probability ──────────────────────────────────────────────
    # pi^i_s = max_k p^i_sk
    print("Exp 4: peak probability …")
    pi_all = probs.max(-1).values.cpu().numpy()                  # (N, S)
    pi_bar = float(pi_all.mean())
    pi_chance = 1.0 / K

    # ── Exp 5: heatmaps de probabilidad ──────────────────────────────────────
    print("Exp 5: heatmaps de probabilidad …")
    prob_heatmaps = probs[:n_show_heatmaps].cpu().numpy()        # (n_show, S, K)

    # ── Consola ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Resultados — Text Encoder Stochasticity")
    print("=" * 55)
    print(f"  [Exp 1] D_inter  (Hamming modos):     {D_inter:.4f}  (0=colapso, 1=máx diversidad)")
    print(f"  [Exp 2] D_intra  (Hamming sampleo):   {D_intra:.4f}  (ideal ≪ D_inter)")
    print(f"  [Exp 3] H global (entropía logits):   {H_global:.4f}  (H_max={H_max:.2f})")
    print(f"  [Exp 3] Slots peaked (<10% H_max):    {frac_peaked * 100:.1f}%")
    print(f"  [Exp 3] Slots planos (>90% H_max):    {frac_flat * 100:.1f}%")
    print(f"  [Exp 4] pi_bar   (peak prob):         {pi_bar:.4f}  (chance={pi_chance:.4f})")
    print("=" * 55)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_all(
        ham_matrix=ham_matrix,
        D_inter=D_inter,
        D_intra=D_intra,
        D_intra_per_mission=D_intra_per_mission,
        H_all=H_all,
        H_per_slot=H_per_slot,
        H_global=H_global,
        H_max=H_max,
        pi_all=pi_all,
        pi_bar=pi_bar,
        pi_chance=pi_chance,
        prob_heatmaps=prob_heatmaps,
        S=S,
        N=n_missions,
        outdir=outdir,
    )

    # ── Guardar JSON ──────────────────────────────────────────────────────────
    results = {
        "n_missions": n_missions,
        "n_samples": n_samples_per_mission,
        "S": S,
        "K": K,
        "exp1": {"D_inter": D_inter},
        "exp2": {
            "D_intra": float(D_intra),
            "D_intra_per_mission": D_intra_per_mission.tolist(),
        },
        "exp3": {
            "H_global": H_global,
            "H_max": H_max,
            "H_per_slot": H_per_slot.tolist(),
            "frac_peaked": frac_peaked,
            "frac_flat": frac_flat,
        },
        "exp4": {
            "pi_bar": pi_bar,
            "pi_chance": pi_chance,
        },
    }
    with open(outdir / "text_stoch_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Guardado en {outdir}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot guardado: {path}")


def _plot_all(
    *,
    ham_matrix, D_inter, D_intra, D_intra_per_mission,
    H_all, H_per_slot, H_global, H_max,
    pi_all, pi_bar, pi_chance,
    prob_heatmaps, S, N, outdir,
):
    """Genera un PNG independiente por cada gráfico."""

    # ── Exp 1 — Histograma de d(i,j) (sin heatmap inter-misión) ──────────────
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pairs = ham_matrix[mask]
    ax.hist(pairs, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(D_inter, color="r", linestyle="--", label=f"media={D_inter:.3f}")
    ax.set_title("Exp 1 — Distribución de d(i,j)")
    ax.set_xlabel("Hamming por par")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)
    _save(fig, outdir / "exp1_hist_inter_mission.png")

    # ── Exp 2 — Distribución de D_intra^i ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.hist(D_intra_per_mission, bins=40, color="coral", edgecolor="white")
    ax.axvline(D_intra, color="darkred", linestyle="--",
               label=f"D_intra media={D_intra:.4f}")
    ax.axvline(D_inter, color="steelblue", linestyle="--",
               label=f"D_inter={D_inter:.3f}")
    ax.set_title("Exp 2 — Distribución de D_intra por misión\n"
                 "(varianza intra entre M muestras)")
    ax.set_xlabel("D_intra^i  (Hamming entre muestras de la misma misión)")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)
    _save(fig, outdir / "exp2_hist_intra_mission.png")

    # ── Exp 3 — Distribución H por slot (boxplot, agrandado) ─────────────────
    # Ancho proporcional a S para que se vea cada slot
    box_w = max(14, S * 0.45)
    fig, ax = plt.subplots(figsize=(box_w, 7), constrained_layout=True)
    ax.boxplot(H_all, patch_artist=True,
               boxprops=dict(facecolor="lightblue", alpha=0.7),
               medianprops=dict(color="steelblue", linewidth=2))
    ax.axhline(H_max, color="r", linestyle="--", label=f"H_max={H_max:.2f}")
    ax.set_title("Exp 3 — Distribución H por slot")
    ax.set_xlabel("Slot s")
    ax.set_ylabel("Entropía (nats)")
    ax.legend(fontsize=10)
    _save(fig, outdir / "exp3_box_H_per_slot.png")

    # ── Exp 3 — H_s media por slot (barras) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(range(S), H_per_slot, color="steelblue", alpha=0.8)
    ax.axhline(H_max, color="r", linestyle="--", label=f"H_max={H_max:.2f}")
    ax.axhline(H_global, color="orange", linestyle="--",
               label=f"H={H_global:.3f}")
    ax.set_title("Exp 3 — H_s media por slot")
    ax.set_xlabel("Slot s")
    ax.set_ylabel("H_s (nats)")
    ax.legend(fontsize=9)
    _save(fig, outdir / "exp3_bar_H_per_slot.png")

    # ── Exp 3 — Histograma global de H^i_s ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(H_all.flatten(), bins=50, color="steelblue", edgecolor="white")
    ax.axvline(H_max, color="r", linestyle="--", label=f"H_max={H_max:.2f}")
    ax.axvline(H_global, color="orange", linestyle="--", label=f"H={H_global:.3f}")
    ax.set_title("Exp 3 — Histograma global de H^i_s")
    ax.set_xlabel("Entropía (nats)")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)
    _save(fig, outdir / "exp3_hist_H_global.png")

    # ── Exp 3 — Heatmap H^i_s (agrandado) ────────────────────────────────────
    # Ancho ∝ slots, alto ∝ misiones (con tope) para que se distinga celda a celda
    hm_w = max(12, S * 0.4)
    hm_h = min(20, max(8, N * 0.06))
    fig, ax = plt.subplots(figsize=(hm_w, hm_h), constrained_layout=True)
    im = ax.imshow(H_all, aspect="auto", cmap="hot", vmin=0, vmax=H_max,
                   interpolation="nearest")
    ax.set_title("Exp 3 — Heatmap H^i_s (misiones × slots)")
    ax.set_xlabel("Slot s")
    ax.set_ylabel("Misión i")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="H (nats)")
    _save(fig, outdir / "exp3_heatmap_H.png")

    # ── Exp 4 — Distribución π^i_s ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(pi_all.flatten(), bins=50, color="mediumseagreen", edgecolor="white")
    ax.axvline(pi_bar, color="darkgreen", linestyle="--",
               label=f"pi_bar={pi_bar:.3f}")
    ax.axvline(pi_chance, color="r", linestyle="--",
               label=f"1/K={pi_chance:.3f}")
    ax.set_title("Exp 4 — Distribución π^i_s (peak prob)")
    ax.set_xlabel("max_k p^i_sk")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)
    _save(fig, outdir / "exp4_hist_peak_prob.png")

    # ── Exp 4 — Heatmap π^i_s ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(pi_all, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_title("Exp 4 — Heatmap π^i_s (misiones × slots)")
    ax.set_xlabel("Slot s")
    ax.set_ylabel("Misión i")
    plt.colorbar(im, ax=ax, fraction=0.046, label="π^i_s")
    _save(fig, outdir / "exp4_heatmap_peak_prob.png")

    # ── Exp 5 — Heatmaps de probabilidad por misión ──────────────────────────
    n_show = prob_heatmaps.shape[0]
    for idx in range(n_show):
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        im = ax.imshow(prob_heatmaps[idx], aspect="auto", cmap="Blues",
                       vmin=0, vmax=1)
        ax.set_title(f"Exp 5 — P^{idx} (S×K) — misión {idx}")
        ax.set_xlabel("Clase k")
        ax.set_ylabel("Slot s")
        plt.colorbar(im, ax=ax, fraction=0.046, label="p^i_sk")
        _save(fig, outdir / f"exp5_prob_mission_{idx}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    uv run python -m experiments.stochasticity_state \
        --logdir logdir/text_goal_sample/01 \
        --exp text \
        --device cpu
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True,
                        help="Directorio del run (contiene latest.pt y .hydra/config.yaml)")
    parser.add_argument("--exp", default="text",
                        choices=["text", "wm", "frozen", "alignment"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--n_missions", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=50)
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)

    # ── 1. Cargar config guardado por Hydra ───────────────────────────────────
    config = OmegaConf.load(logdir / ".hydra" / "config.yaml")
    device = args.device or config.device
    config.device = device

    # ── 2. Reconstruir envs y agente ─────────────────────────────────────────
    _, eval_envs, obs_space, act_space = make_envs(config.env)
    reward_function = make_reward(config)

    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
        reward_function=reward_function,
    ).to(config.device)

    # ── 3. Cargar pesos del checkpoint ────────────────────────────────────────
    checkpoint = torch.load(logdir / "latest.pt", map_location=config.device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.eval()
    print(f"Checkpoint cargado desde {logdir / 'latest.pt'}")

    # ── 4. Tomar un env individual de eval_envs ───────────────────────────────
    env = eval_envs.envs[0]
    if not hasattr(env, "encoded_random_mission"):
        _fn = env.get_wrapper_attr("encoded_random_mission")
        env.encoded_random_mission = _fn

    # ── 5. Correr el experimento pedido ───────────────────────────────────────
    outdir = logdir / "experiments"

    if args.exp == "text":
        exp_text_encoder_stochasticity(
            agent=agent,
            env=env,
            n_missions=args.n_missions,
            n_samples_per_mission=args.n_samples,
            outdir=outdir / "text_stoch",
        )
