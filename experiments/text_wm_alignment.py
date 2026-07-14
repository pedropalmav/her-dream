"""Alineamiento text-encoder ↔ WM posterior (el experimento que faltaba).

Las visualizaciones existentes (`text_stochasticity_state.py` y
`WM_stochasticity_state.py`) miden la estocasticidad del text encoder y del
world model POR SEPARADO. Ninguna los compara sobre el MISMO condicionamiento.

Este experimento responde directamente: «¿qué tan similares son las
probabilidades del text encoder vs el WM real?» — emparejando, sobre rollouts
reales (la misma distribución de datos con la que se destiló), cada
``obs["mission"]`` con el posterior del WM ``q_wm(z|obs)`` del mismo paso, y
comparándolo contra ``q_text(z|mission)``.

Por qué importa para `post_train_from_distill`:
  - El reward `first_row_reward` solo da 0 cuando
        argmax(z_wm[:, 0, :]) == argmax(goal)
    es decir SOLO el slot 0, y por match exacto de muestras one-hot.
  - Con `goal_sample=text`, el goal es una MUESTRA del slot 0 del text encoder.
  - La destilación minimiza KL sobre los S grupos completos, no sobre el slot 0.

Entonces aunque la KL global sea baja, la política puede no aprender si el
slot 0 está desalineado o es de alta entropía. Este experimento mide eso.

Métrica central — «reward esperado»:
  Para un par (misión, estado), si el goal se muestrea de q_text(slot0) y el
  estado del WM se muestrea de q_wm(slot0), la probabilidad de que coincidan es
        P(match) = sum_k  p_text_sk * p_wm_sk     (probabilidad de colisión)
  promediada sobre pares. Si esto ≈ 1/K (azar) o ≈ 0, la política no recibe
  señal consistente y no puede aprender.

Uso:
    uv run python -m experiments.text_wm_alignment \
        --logdir logdir/distill_text_from_wm_only/01 \
        --device cpu \
        --n_traj 200 --traj_len 50 --stride 5
"""

import argparse
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.common import (
    add_common_args,
    dump_json,
    entropy,
    env_factory,
    experiment_outdir,
    load_agent,
    posterior_probs,
    posterior_rollout,
    random_policy,
    save_fig,
)

# ─────────────────────────────────────────────────────────────────────────────
# Recolección de pares emparejados (misión, posterior WM)
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def collect_pairs(
    agent,
    env_factory,
    n_trajectories: int,
    traj_len: int,
    stride: int,
    device: str,
) -> dict:
    """Corre `n_trajectories` rollouts con acciones aleatorias y, cada `stride`
    pasos, guarda el par emparejado:
      - post_logit del WM (S, K)   ← posterior temporal correcto (cadena obs_step)
      - mission ids (L,)           ← obs["mission"] de ESE mismo paso

    Devuelve
    --------
    wm_logits   : (N, S, K)
    mission_ids : (N, L)  int
    """
    wm_logits = []
    mission_ids = []

    for n in range(n_trajectories):
        env = env_factory()
        for step in posterior_rollout(agent, env, random_policy(np.random), device, traj_len - 1):
            # Emparejar misión ↔ posterior en el mismo paso.
            if step.t % stride == 0 and "mission" in step.obs:
                wm_logits.append(step.logit.squeeze(0))  # (S, K)
                mission_ids.append(
                    torch.as_tensor(step.obs["mission"], device=device).long()  # (L,)
                )

        if (n + 1) % 20 == 0:
            print(f"  Trayectoria {n + 1}/{n_trajectories}  (pares: {len(wm_logits)})")

    wm_logits = torch.stack(wm_logits, dim=0)  # (N, S, K)
    mission_ids = torch.stack(mission_ids, dim=0)  # (N, L)

    # q_text sobre las MISMAS misiones (forward espera (B, T, L) ids).
    text_logits = agent.text_encoder(mission_ids.unsqueeze(1))[:, 0]  # (N, S, K)

    return {
        "wm_logits": wm_logits,
        "text_logits": text_logits,
        "mission_ids": mission_ids,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Métricas de alineamiento
# ─────────────────────────────────────────────────────────────────────────────


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) por slot. p, q: (N, S, K) → (N, S)."""
    return (p * ((p + 1e-8).log() - (q + 1e-8).log())).sum(-1)


def compute_alignment(agent, wm_logits, text_logits) -> dict:
    p_wm = posterior_probs(agent, wm_logits)  # (N, S, K)
    p_tx = posterior_probs(agent, text_logits)  # (N, S, K)
    N, S, K = p_wm.shape

    # ── KL / JS por slot ──────────────────────────────────────────────────────
    kl_wm_tx = _kl(p_wm, p_tx).mean(0)  # (S,)  KL(wm||text)
    kl_tx_wm = _kl(p_tx, p_wm).mean(0)  # (S,)  KL(text||wm)
    m = 0.5 * (p_wm + p_tx)
    js = (0.5 * _kl(p_wm, m) + 0.5 * _kl(p_tx, m)).mean(0)  # (S,)

    # ── Acuerdo de argmax por slot ────────────────────────────────────────────
    am_wm = p_wm.argmax(-1)  # (N, S)
    am_tx = p_tx.argmax(-1)  # (N, S)
    argmax_agree = (am_wm == am_tx).float().mean(0)  # (S,)

    # ── Reward esperado = prob. de colisión ⟨p_text, p_wm⟩ por slot ───────────
    collision = (p_wm * p_tx).sum(-1).mean(0)  # (S,)

    # ── Foco en slot 0 (el único que usa el reward) ──────────────────────────
    p_wm0, p_tx0 = p_wm[:, 0], p_tx[:, 0]  # (N, K)
    H_wm0 = entropy(p_wm0)  # (N,)
    H_tx0 = entropy(p_tx0)
    am_wm0, am_tx0 = am_wm[:, 0], am_tx[:, 0]

    # Cobertura de clases del slot 0: ¿el text apunta a clases que el WM realiza?
    wm_support = torch.bincount(am_wm0, minlength=K).float()  # (K,)
    tx_support = torch.bincount(am_tx0, minlength=K).float()  # (K,)
    wm_reachable = wm_support > 0
    tx_in_wm_support = wm_reachable[am_tx0].float().mean().item()

    # Matriz de confusión slot 0: filas = clase WM, columnas = clase text.
    confusion = torch.zeros(K, K)
    for w, x in zip(am_wm0.tolist(), am_tx0.tolist()):
        confusion[w, x] += 1

    return {
        "N": N,
        "S": S,
        "K": K,
        "kl_wm_tx": kl_wm_tx.cpu().numpy(),
        "kl_tx_wm": kl_tx_wm.cpu().numpy(),
        "js": js.cpu().numpy(),
        "argmax_agree": argmax_agree.cpu().numpy(),
        "collision": collision.cpu().numpy(),
        # slot 0
        "slot0_argmax_agree": float((am_wm0 == am_tx0).float().mean()),
        "slot0_collision": float(collision[0]),
        "slot0_chance": 1.0 / K,
        "slot0_H_wm": H_wm0.cpu().numpy(),
        "slot0_H_tx": H_tx0.cpu().numpy(),
        "slot0_H_max": float(np.log(K)),
        "slot0_tx_in_wm_support": tx_in_wm_support,
        "slot0_n_wm_classes": int(wm_reachable.sum()),
        "slot0_n_tx_classes": int((tx_support > 0).sum()),
        "slot0_wm_support": wm_support.cpu().numpy(),
        "slot0_tx_support": tx_support.cpu().numpy(),
        "slot0_confusion": confusion.cpu().numpy(),
        # para heatmaps de pares
        "p_wm": p_wm.cpu().numpy(),
        "p_tx": p_tx.cpu().numpy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────


def _plot_all(r: dict, n_show_pairs: int, outdir: Path):
    S, K = r["S"], r["K"]

    # ── KL / JS por slot (slot 0 resaltado) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, S * 0.4), 5), constrained_layout=True)
    x = np.arange(S)
    w = 0.27
    ax.bar(x - w, r["kl_wm_tx"], w, label="KL(wm‖text)", color="steelblue")
    ax.bar(x, r["kl_tx_wm"], w, label="KL(text‖wm)", color="coral")
    ax.bar(x + w, r["js"], w, label="JS", color="mediumseagreen")
    ax.axvline(0, color="red", linestyle=":", alpha=0.6, label="slot 0 (reward)")
    ax.set_title("Alineamiento — divergencia por slot (slot 0 = el del reward)")
    ax.set_xlabel("Slot s")
    ax.set_ylabel("Divergencia (nats)")
    ax.legend(fontsize=9)
    save_fig(fig, outdir / "align_divergence_per_slot.png")

    # ── Acuerdo de argmax por slot + colisión ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, S * 0.4), 5), constrained_layout=True)
    ax.bar(x - 0.2, r["argmax_agree"], 0.4, label="acuerdo argmax", color="steelblue")
    ax.bar(x + 0.2, r["collision"], 0.4, label="reward esperado ⟨p_t,p_w⟩", color="coral")
    ax.axhline(r["slot0_chance"], color="gray", linestyle="--", label=f"azar = 1/K = {r['slot0_chance']:.3f}")
    ax.axvline(0, color="red", linestyle=":", alpha=0.6)
    ax.set_title("Alineamiento — acuerdo argmax y reward esperado por slot")
    ax.set_xlabel("Slot s")
    ax.set_ylabel("Fracción")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    save_fig(fig, outdir / "align_agreement_per_slot.png")

    # ── Slot 0 — matriz de confusión ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(r["slot0_confusion"], aspect="auto", cmap="magma")
    ax.set_title("Slot 0 — confusión argmax (filas=WM, cols=text)\ndiagonal = alineado")
    ax.set_xlabel("clase text encoder")
    ax.set_ylabel("clase WM posterior")
    plt.colorbar(im, ax=ax, fraction=0.046, label="conteo")
    save_fig(fig, outdir / "align_slot0_confusion.png")

    # ── Slot 0 — soporte de clases (¿el text apunta a clases que el WM realiza?)
    fig, ax = plt.subplots(figsize=(max(8, K * 0.4), 5), constrained_layout=True)
    xk = np.arange(K)
    ax.bar(xk - 0.2, r["slot0_wm_support"], 0.4, label="WM realiza", color="steelblue")
    ax.bar(xk + 0.2, r["slot0_tx_support"], 0.4, label="text propone", color="coral")
    ax.set_title(
        f"Slot 0 — soporte de clases\ntext dentro del soporte del WM: {r['slot0_tx_in_wm_support'] * 100:.1f}%"
    )
    ax.set_xlabel("clase k")
    ax.set_ylabel("conteo (argmax)")
    ax.legend(fontsize=9)
    save_fig(fig, outdir / "align_slot0_coverage.png")

    # ── Slot 0 — entropía WM vs text ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(r["slot0_H_wm"], bins=40, alpha=0.6, label="WM", color="steelblue")
    ax.hist(r["slot0_H_tx"], bins=40, alpha=0.6, label="text", color="coral")
    ax.axvline(r["slot0_H_max"], color="red", linestyle="--", label=f"H_max={r['slot0_H_max']:.2f}")
    ax.set_title("Slot 0 — entropía del posterior (WM vs text)")
    ax.set_xlabel("Entropía (nats)")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)
    save_fig(fig, outdir / "align_slot0_entropy.png")

    # ── Heatmaps de pares lado a lado (q_wm vs q_text) ────────────────────────
    n_show = min(n_show_pairs, r["p_wm"].shape[0])
    for idx in range(n_show):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        for ax, mat, name, cmap in (
            (axes[0], r["p_wm"][idx], "WM posterior", "Blues"),
            (axes[1], r["p_tx"][idx], "text encoder", "Oranges"),
        ):
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
            ax.axhline(0, color="red", linestyle="-", linewidth=2, alpha=0.4)
            ax.set_title(f"q_{name} (par {idx}) — slot 0 en rojo")
            ax.set_xlabel("clase k")
            ax.set_ylabel("slot s")
            plt.colorbar(im, ax=ax, fraction=0.046)
        save_fig(fig, outdir / f"align_pair_{idx}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_alignment(
    agent,
    env_factory,
    n_trajectories: int = 200,
    traj_len: int = 50,
    stride: int = 5,
    n_show_pairs: int = 6,
    outdir: Path = Path("results/text_wm_align"),
):
    outdir.mkdir(parents=True, exist_ok=True)
    device = agent.device

    print(
        f"\nRecolectando pares (misión, posterior) de {n_trajectories} "
        f"trayectorias de largo {traj_len}, stride {stride} …"
    )
    data = collect_pairs(agent, env_factory, n_trajectories, traj_len, stride, device)

    # Chequeo de misión estática (clave en fixed_goal): si casi todas las
    # misiones son idénticas, el text encoder no pudo aprender nada útil.
    n_unique = len(torch.unique(data["mission_ids"], dim=0))
    n_pairs = data["mission_ids"].shape[0]
    print(f"  Pares recolectados: {n_pairs}  |  misiones únicas: {n_unique}")
    if n_unique <= max(2, int(0.02 * n_pairs)):
        print(
            "  ⚠️  ADVERTENCIA: casi todas las misiones son idénticas. "
            "El text encoder se destiló sobre una misión esencialmente "
            "constante (típico de fixed_goal, que no varía obs['mission'])."
        )

    r = compute_alignment(agent, data["wm_logits"], data["text_logits"])

    # ── Consola ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Alineamiento Text Encoder ↔ WM Posterior")
    print("=" * 64)
    print(f"  Pares: {r['N']}   S={r['S']}  K={r['K']}")
    print(f"  Misiones únicas:                       {n_unique}")
    print("  ── Slot 0 (EL del reward) ──")
    print(f"  Acuerdo argmax slot 0:                 {r['slot0_argmax_agree']:.4f}")
    print(f"  Reward esperado ⟨p_t,p_w⟩ slot 0:      {r['slot0_collision']:.4f}  (azar={r['slot0_chance']:.4f})")
    print(f"  Text dentro del soporte del WM:        {r['slot0_tx_in_wm_support'] * 100:.1f}%")
    print(f"  Clases slot 0 que el WM realiza:       {r['slot0_n_wm_classes']}/{r['K']}")
    print(f"  Clases slot 0 que el text propone:     {r['slot0_n_tx_classes']}/{r['K']}")
    print(
        f"  H slot 0  WM:   {r['slot0_H_wm'].mean():.3f}   text: {r['slot0_H_tx'].mean():.3f}"
        f"   (H_max={r['slot0_H_max']:.2f})"
    )
    print("  ── Global (todos los slots) ──")
    print(f"  KL(wm‖text) media:                     {r['kl_wm_tx'].mean():.4f}")
    print(f"  KL(text‖wm) media:                     {r['kl_tx_wm'].mean():.4f}")
    print(f"  JS media:                              {r['js'].mean():.4f}")
    print(f"  Acuerdo argmax medio:                  {r['argmax_agree'].mean():.4f}")
    print("=" * 64)
    # Lectura del diagnóstico
    if r["slot0_collision"] < 2 * r["slot0_chance"]:
        print("  → DIAGNÓSTICO: el reward esperado en slot 0 está cerca del azar.")
        print("    La política recibe señal casi nula → no puede aprender.")
    print()

    _plot_all(r, n_show_pairs, outdir)

    # ── JSON (sin arrays gigantes) ────────────────────────────────────────────
    results = {
        "N": r["N"],
        "S": r["S"],
        "K": r["K"],
        "n_unique_missions": int(n_unique),
        "slot0": {
            "argmax_agree": r["slot0_argmax_agree"],
            "expected_reward_collision": r["slot0_collision"],
            "chance": r["slot0_chance"],
            "tx_in_wm_support": r["slot0_tx_in_wm_support"],
            "n_wm_classes": r["slot0_n_wm_classes"],
            "n_tx_classes": r["slot0_n_tx_classes"],
            "H_wm_mean": float(r["slot0_H_wm"].mean()),
            "H_tx_mean": float(r["slot0_H_tx"].mean()),
            "H_max": r["slot0_H_max"],
        },
        "global": {
            "kl_wm_tx_mean": float(r["kl_wm_tx"].mean()),
            "kl_tx_wm_mean": float(r["kl_tx_wm"].mean()),
            "js_mean": float(r["js"].mean()),
            "argmax_agree_mean": float(r["argmax_agree"].mean()),
            "kl_wm_tx_per_slot": r["kl_wm_tx"].tolist(),
            "argmax_agree_per_slot": r["argmax_agree"].tolist(),
            "collision_per_slot": r["collision"].tolist(),
        },
    }
    dump_json(results, outdir / "align_results.json")

    print(f"  Guardado en {outdir}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--n_traj", type=int, default=200)
    parser.add_argument("--traj_len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5, help="Guardar 1 par cada `stride` pasos (decorrelaciona).")
    parser.add_argument("--n_pairs", type=int, default=6, help="Pares de heatmaps q_wm vs q_text a graficar.")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)

    agent, config, _ = load_agent(logdir, args.device, require_mission_text=True)

    run_alignment(
        agent=agent,
        env_factory=env_factory(config),
        n_trajectories=args.n_traj,
        traj_len=args.traj_len,
        stride=args.stride,
        n_show_pairs=args.n_pairs,
        outdir=experiment_outdir(logdir, "text_wm_align"),
    )
