"""Does the z-goal reward carry usable signal in this world model?

Two questions, one pass over real rollouts, for any env (built for crafter,
whose z is far noisier than the goal-grid envs).

**A — dynamic range of the reward.** `row_by_row` compares the argmax of two
*samples* of z, so the reward has a ceiling that no policy can beat and a floor
that is not zero:

  ceiling: sample z twice from the SAME posterior logits -> matched rows. This
           is what a perfectly navigating policy gets, because the goal was one
           sample and the achieved state is another (the "sampling floor" lens
           of goal_reachability.py, here as the reward's upper bound).
  floor:   sample z from two UNRELATED states -> matched rows. Chance is S/K,
           but constant groups (ones that always pick the same category) match
           for free and push the floor up.

  ceiling - floor is the reward's usable range. A large ceiling with an equally
  large floor means the policy is optimizing Gumbel noise.

  For `max_cosine` (logits, no sampling) the ceiling is trivially 1.0, so what
  matters is the floor: if unrelated crafter states already sit at cos ~0.9,
  the reward is dense but has nowhere to move.

**B — non-triviality of the imagined goal.** `goal_sample=imagination` rolls the
prior `goal_imag_horizon` steps with uniformly random actions. In a grid that
moves you; in crafter most actions are context-dependent no-ops, so the goal can
end up indistinguishable from "stay where you are" — the policy then earns a top
reward by doing nothing, which reads as success on the training curve. For each
horizon and action source we report:

  goal vs z0:      how much the goal asks for (near the ceiling = trivial goal)
  goal vs real z_H: whether it is reachable, imagining along the SAME actions the
                    real rollout took (an oracle plan).

Usage:
    uv run python3 experiments/z_goal_signal.py \
        --logdir logdir/original_wm_crafter/02 --episodes 8 --steps 200
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from tensordict.base import TensorDictBase

# MPS pin_memory stub (harmless on CPU), kept for parity with the other scripts.
TensorDictBase.pin_memory = lambda self, *a, **k: self

# Repo root on the path: this script imports both `experiments.*` and the
# top-level `train` module, neither of which ships inside the her_dream package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import her_dream  # noqa: E402
import her_dream.goals as goals  # noqa: E402
import her_dream.networks as networks  # noqa: E402
import her_dream.tools as tools  # noqa: E402
import train as train_mod  # noqa: E402
from experiments.common.io import dump_json  # noqa: E402
from experiments.common.obs import onehot_action, preprocess_obs  # noqa: E402
from her_dream.dreamer import Dreamer  # noqa: E402
from her_dream.envs import make_env  # noqa: E402
from her_dream.rewards import make_reward  # noqa: E402

_CONFIG_DIR = str(pathlib.Path(her_dream.__file__).parent / "configs")
HORIZONS = (5, 15, 30, 50)


def build(logdir, env_name, goal_type, device, extra_overrides):
    """Build a goal-conditioned agent on `logdir`'s world model.

    The vanilla crafter runs predate goal-conditioning, so their `.hydra/config`
    has no goal keys at all and `common.loading.load_agent` cannot rebuild them.
    We compose a fresh config instead and load only the world model, exactly as
    a `reset_actor_critic=True` post-training run would.
    """
    overrides = [
        f"env={env_name}",
        f"goal_type={goal_type}",
        f"device={device}",
        "env.goal_sample=imagination",
        f"load_from={logdir}",
        "reset_actor_critic=True",
        "freeze_wm=True",
        *extra_overrides,
    ]
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        config = compose(config_name="configs", overrides=overrides)

    env = make_env(config.env, 0)
    agent = Dreamer(
        config.model,
        env.observation_space,
        env.action_space,
        reward_function=make_reward(config),
    ).to(device)
    train_mod.load_checkpoint(agent, config)
    agent.eval()
    for p in agent.parameters():
        p.requires_grad_(False)
    return agent, env, config


def load_pretrained_actor(agent, config, logdir, device):
    """Rebuild the checkpoint's goal-free actor, or None if it has none.

    The vanilla actor reads `feat` alone, so it is not loadable into the
    goal-conditioned agent — but it is exactly the policy the world model was
    trained under, which makes it the right action source both for collecting
    on-distribution rollouts and for proposing non-trivial imagined goals.
    """
    ckpt = torch.load(pathlib.Path(logdir) / "latest.pt", map_location=device)["agent_state_dict"]
    ckpt = tools.migrate_agent_state_dict(ckpt)
    actor_sd = {k[len("ac.actor.") :]: v for k, v in ckpt.items() if k.startswith("ac.actor.")}
    if not actor_sd:
        return None
    actor = networks.MLPHead(config.model.actor, agent.rssm.feat_size).to(device)
    expected = actor.state_dict()["mlp.layers.actor_linear0.weight"].shape
    if actor_sd["mlp.layers.actor_linear0.weight"].shape != expected:
        print("[warn] checkpoint actor is goal-conditioned; skipping the pretrained-policy arm.")
        return None
    actor.load_state_dict(actor_sd)
    actor.eval()
    return actor


@torch.no_grad()
def rollout(agent, env, actor, device, max_steps, seed):
    """One posterior rollout; returns per-step logits/stoch and the actions taken.

    `actor=None` acts uniformly at random.
    """
    rng = np.random.default_rng(seed)
    obs = env.reset()
    n_actions = env.action_space.shape[0]

    stoch, deter = agent.rssm.initial(1)
    prev_action = torch.zeros(1, n_actions, device=device)
    embed = agent.encoder(preprocess_obs(obs, device))
    is_first = torch.tensor([True], dtype=torch.bool, device=device)
    stoch, deter, logit = agent.rssm.obs_step(stoch, deter, prev_action, embed, is_first)

    obs0 = {k: np.array(v, copy=True) for k, v in obs.items()}
    logits, stochs, actions = [logit], [stoch], []
    for _ in range(max_steps):
        if actor is None:
            act_np = onehot_action(int(rng.integers(n_actions)), n_actions)
        else:
            feat = agent.rssm.get_feat(stoch, deter)
            act_np = actor(feat).mode[0].detach().cpu().numpy().astype(np.float32)
        obs, _r, done, _i = env.step(act_np)
        actions.append(act_np)

        prev_action = torch.as_tensor(act_np, device=device).unsqueeze(0)
        embed = agent.encoder(preprocess_obs(obs, device))
        is_first = torch.tensor([False], dtype=torch.bool, device=device)
        stoch, deter, logit = agent.rssm.obs_step(stoch, deter, prev_action, embed, is_first)
        logits.append(logit)
        stochs.append(stoch)
        if done:
            break
    return torch.cat(logits), torch.cat(stochs), actions, obs0


@torch.no_grad()
def imagine_along(agent, device, obs, actions, horizon):
    """Roll the prior `horizon` steps from `obs`, replaying `actions`.

    Replaying the rollout's own actions makes the comparison against the real
    z_H an oracle: the imagined and the real trajectory take the same decisions,
    so any gap is the world model's, not the policy's.
    """
    stoch, deter = agent.rssm.initial(1)
    prev_action = torch.zeros(1, agent.act_dim, device=device)
    embed = agent.encoder(preprocess_obs(obs, device))
    is_first = torch.tensor([True], dtype=torch.bool, device=device)
    stoch, deter, logit = agent.rssm.obs_step(stoch, deter, prev_action, embed, is_first)
    z0, logit0 = stoch, logit

    for t in range(horizon):
        action = torch.as_tensor(actions[t], device=device).unsqueeze(0)
        stoch, deter, logit = agent.rssm.img_step(stoch, deter, action)
    return z0, logit0, stoch, logit


def sample_z(agent, logit):
    """Draw a fresh one-hot sample from the posterior/prior logits."""
    return agent.rssm.get_dist(logit).rsample()


def reward_value(reward_fn, spec, agent, state_stoch, state_logit, goal_stoch, goal_logit):
    """Evaluate the configured reward for a (state, goal) pair -> float."""
    state = goals.reward_state(spec, stoch=state_stoch, logit=state_logit, rssm=agent.rssm)
    goal = goals.goal_from_latent(spec, stoch=goal_stoch, logit=goal_logit)
    return float(reward_fn(state, goal[0]).reshape(-1)[0])


def matched_rows(a_stoch, b_stoch):
    """Number of groups whose argmax agrees between two one-hot latents."""
    return int((a_stoch.argmax(-1) == b_stoch.argmax(-1)).sum())


def summarize(name, values):
    arr = np.asarray(values, dtype=np.float64)
    return {"name": name, "mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", required=True, help="run dir with latest.pt (world model to probe)")
    parser.add_argument("--env", default="crafter")
    parser.add_argument("--goal-type", default="row_by_row", help="reward whose dynamic range we measure")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200, help="max env steps per rollout")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="where to write the JSON (default: <logdir>/experiments/...)")
    parser.add_argument("overrides", nargs="*", help="extra Hydra overrides, e.g. model.rssm.obs_use_deter=False")
    args = parser.parse_args()

    agent, env, config = build(args.logdir, args.env, args.goal_type, args.device, args.overrides)
    spec = goals.make_goal_spec(config)
    reward_fn = make_reward(config)
    actor = load_pretrained_actor(agent, config, args.logdir, args.device)
    S, K = agent.rssm._stoch, agent.rssm._discrete

    print(f"\nProbing {args.logdir}  goal_type={args.goal_type}  S={S} K={K}")
    print(f"pretrained policy available: {actor is not None}\n")

    # --- collect rollouts (one per action source) -------------------------
    sources = {"random": None} | ({"policy": actor} if actor is not None else {})
    episodes = {}
    for src, pol in sources.items():
        episodes[src] = []
        for ep in range(args.episodes):
            seed = args.seed + ep
            torch.manual_seed(seed)
            logits, stochs, actions, obs0 = rollout(agent, env, pol, args.device, args.steps, seed)
            episodes[src].append({"logits": logits, "stochs": stochs, "actions": actions, "obs0": obs0})
            print(f"  [{src}] episode {ep}: {logits.shape[0]} steps")

    # --- A: dynamic range --------------------------------------------------
    results = {"logdir": args.logdir, "goal_type": args.goal_type, "S": S, "K": K}
    for src, eps in episodes.items():
        all_logits = torch.cat([e["logits"] for e in eps])  # (N, S, K)
        n = all_logits.shape[0]
        rng = np.random.default_rng(args.seed)

        ceil_rows, ceil_rew, floor_rows, floor_rew = [], [], [], []
        for _ in range(400):
            i = int(rng.integers(n))
            logit_i = all_logits[i : i + 1]
            a, b = sample_z(agent, logit_i), sample_z(agent, logit_i)
            ceil_rows.append(matched_rows(a, b))
            ceil_rew.append(reward_value(reward_fn, spec, agent, a, logit_i, b, logit_i))

            # An unrelated state: a different episode whenever possible.
            j = int(rng.integers(n))
            logit_j = all_logits[j : j + 1]
            floor_rows.append(matched_rows(sample_z(agent, logit_i), sample_z(agent, logit_j)))
            zi, zj = sample_z(agent, logit_i), sample_z(agent, logit_j)
            floor_rew.append(reward_value(reward_fn, spec, agent, zi, logit_i, zj, logit_j))

        modes = all_logits.argmax(-1)  # (N, S)
        constancy = [float((modes[:, g] == torch.mode(modes[:, g]).values).float().mean()) for g in range(S)]
        dead = int(sum(c > 0.9 for c in constancy))

        results[f"range_{src}"] = {
            "ceiling_rows": summarize("ceiling_rows", ceil_rows),
            "floor_rows": summarize("floor_rows", floor_rows),
            "chance_rows": S / K,
            "ceiling_reward": summarize("ceiling_reward", ceil_rew),
            "floor_reward": summarize("floor_reward", floor_rew),
            "constant_groups_over_90pct": dead,
            "constancy_per_group": constancy,
        }

        r = results[f"range_{src}"]
        print(f"\n=== A. Rango dinámico ({src} rollouts, {n} estados) ===")
        print(
            f"  techo  (mismo estado, 2 muestras): {r['ceiling_rows']['mean']:.1f}/{S} filas"
            f"   reward {r['ceiling_reward']['mean']:+.3f}"
        )
        print(
            f"  piso   (estados no relacionados):  {r['floor_rows']['mean']:.1f}/{S} filas"
            f"   reward {r['floor_reward']['mean']:+.3f}"
        )
        print(f"  azar puro:                         {S / K:.1f}/{S} filas")
        print(
            f"  rango útil: {r['ceiling_rows']['mean'] - r['floor_rows']['mean']:.1f} filas"
            f"  ({r['ceiling_reward']['mean'] - r['floor_reward']['mean']:+.3f} de reward)"
        )
        print(f"  grupos constantes (>90% la misma categoría): {dead}/{S}")

    # --- B: non-triviality of the imagined goal ---------------------------
    # The ceiling is recomputed *at step H* rather than reused from part A: z is
    # far sharper in the opening frames (flat grass) than in late ones (caves,
    # mobs, inventory), so a pooled ceiling would make short horizons look like
    # they beat the sampling floor. Every column below is measured at the same
    # step, so they are comparable to each other.
    print(f"\n=== B. Goal imaginado: ¿pide algo? (reward {args.goal_type}) ===")
    header = f"{'acciones':>10} {'H':>4} {'techo@H':>10} {'goal vs z0':>12} {'goal vs z_H real':>18}"
    print(header)
    results["imagined_goal"] = {}
    for src, eps in episodes.items():
        for horizon in HORIZONS:
            vs_z0, vs_real, ceil_h = [], [], []
            for ep_idx, ep in enumerate(eps):
                if len(ep["actions"]) <= horizon:
                    continue
                torch.manual_seed(args.seed + ep_idx)
                z0, logit0, zg, logitg = imagine_along(agent, args.device, ep["obs0"], ep["actions"], horizon)
                vs_z0.append(reward_value(reward_fn, spec, agent, z0, logit0, zg, logitg))
                real_logit = ep["logits"][horizon : horizon + 1]
                real_stoch = ep["stochs"][horizon : horizon + 1]
                vs_real.append(reward_value(reward_fn, spec, agent, real_stoch, real_logit, zg, logitg))
                a, b = sample_z(agent, real_logit), sample_z(agent, real_logit)
                ceil_h.append(reward_value(reward_fn, spec, agent, a, real_logit, b, real_logit))
            if not vs_z0:
                continue
            key = f"{src}_H{horizon}"
            results["imagined_goal"][key] = {
                "ceiling_at_H": summarize("ceiling_at_H", ceil_h),
                "vs_z0": summarize("vs_z0", vs_z0),
                "vs_real_zH": summarize("vs_real_zH", vs_real),
            }
            print(
                f"{src:>10} {horizon:>4} {np.mean(ceil_h):>+10.3f} {np.mean(vs_z0):>+12.3f} {np.mean(vs_real):>+18.3f}"
            )

    print("\nLectura:")
    print("  'goal vs z0' cerca del techo@H  -> goal trivial (la política cobra sin moverse).")
    print("  'goal vs z_H real' lejos del techo@H -> ni el plan oráculo alcanza el goal.")

    default_out = pathlib.Path(args.logdir) / "experiments" / f"z_goal_signal_{args.goal_type}.json"
    out = pathlib.Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(results, out)
    print(f"\nJSON -> {out}")


if __name__ == "__main__":
    main()
