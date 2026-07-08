#!/usr/bin/env python3
"""Interactive z-flow WebSocket server.

Loads an experiment directory (env config + checkpoint) and streams step data
to z-flow's browser frontend via WebSocket. The client sends action indices;
the server steps the environment, runs Dreamer inference, and returns the
observation image + stochastic embedding for each step.

Every step creates a brand new node and a new edge from the previous node,
even if the observation matches one seen earlier in the episode.

Also serves a separate cluster-explorer page: at startup, collects random-
policy episodes, clusters the visited states by their RSSM stoch, and exposes
the result over REST (/clusters/states, /clusters/states/{id},
/clusters/{cluster_id}/states, /clusters/row_stats) for z-flow's clustering
view.

Usage:
    uv run zflow/serve.py <experiment_dir>
    uv run zflow/serve.py <experiment_dir> --host 0.0.0.0 --port 8765 --time-limit 200
    uv run zflow/serve.py <experiment_dir> --cluster-episodes 30 --cluster-k 10
"""

import argparse
import pathlib
import sys
import warnings

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf
from tensordict import TensorDict

sys.path.append(str(pathlib.Path(__file__).parent.parent))  # repo root: tools, dreamer, envs, rewards
from clustering import ClusterStore, collect_random_states, compute_row_stats, run_clustering
from serve_utils import encode_image, make_trans

import goals
import tools
from dreamer import Dreamer
from envs import make_env
from rewards import make_reward

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


def _reward_input(agent_state: TensorDict, agent):
    """Return the correct first argument for reward_function given the goal type.

    Routes through the shared `goals.reward_state`: stoch sample / OneHotCategorical
    distribution / raw logit, depending on the goal type's `state_repr`.
    """
    return goals.reward_state(
        agent._goal_spec,
        stoch=agent_state["stoch"],
        logit=agent_state.get("logit"),
        rssm=agent._frozen_rssm,
    )


def _backfill_config(config, force_wm_only=False):
    """Fill in config keys added after older experiments were trained.

    Returns ``(config, wm_only)``. ``wm_only`` is True for pre-goal-conditioning
    checkpoints (e.g. the original-Dreamer WM on `feat/original-dreamer-wm`),
    detected by the absence of the top-level ``goal_type`` key. In that mode the
    goal-conditioned actor/critic and the reward/continue predictors in the
    checkpoint are not loaded; only the world model (encoder + RSSM) is used.
    """
    updates = {}
    if "goal_imag_horizon" not in config.model:
        updates["model"] = {"goal_imag_horizon": int(config.model.imag_horizon)}

    wm_only = force_wm_only or ("goal_type" not in config)
    if wm_only:
        # Dummy goal-conditioning keys so main's Dreamer can be *constructed*.
        # These only size the actor/value heads, which are neither loaded nor
        # run in WM-only mode; make_reward is skipped entirely.
        goal_defaults = {
            "goal_type": "full",
            "mission_text": False,
            "wm_only": False,
            "train_text_only": False,
            "freeze_wm": False,
            "prob_threshold": 0.5,
            "prob_threshold_step": 0.1,
        }
        top = {k: v for k, v in goal_defaults.items() if k not in config}
        model_missing = {k: v for k, v in goal_defaults.items() if k not in config.model}
        # Deep-merge: preserves the model.goal_imag_horizon backfill above.
        updates = OmegaConf.merge(updates, {**top, "model": model_missing})

    config = OmegaConf.merge(config, updates) if updates else config

    if "goal_type" in config.model and "state_repr" not in config.model:
        # Goal-type descriptors were added later; load them from the config group.
        config = OmegaConf.merge(config, {"model": goals.default_descriptors(config.model.goal_type)})
    return config, wm_only


# state_dict prefixes that make up the world model — identical in keys and
# shapes between the goal-conditioned Dreamer and the original-Dreamer runs.
_WM_PREFIXES = ("encoder.", "_frozen_encoder.", "rssm.", "_frozen_rssm.", "return_ema.")


def _load_agent_weights(agent, checkpoint, wm_only: bool):
    """Load agent weights, tolerating original-Dreamer checkpoints.

    Full (goal-conditioned) checkpoints load strictly. In ``wm_only`` mode only
    the world-model tensors (``_WM_PREFIXES``) are loaded; the checkpoint's
    reward/continue predictors and its non-goal-conditioned actor/value heads
    (which have incompatible shapes) are intentionally skipped. The randomly
    initialised actor/value on the agent are never exercised in WM-only mode.
    """
    if not wm_only:
        agent.load_state_dict(checkpoint)
        return

    filtered = {k: v for k, v in checkpoint.items() if k.startswith(_WM_PREFIXES)}
    missing, unexpected = agent.load_state_dict(filtered, strict=False)

    # Guard against a silent partial WM load: every WM tensor we filtered in
    # must have matched a parameter on the agent.
    wm_missing = [k for k in missing if k.startswith(_WM_PREFIXES)]
    if wm_missing or unexpected:
        raise RuntimeError(
            f"WM-only load failed. Unmatched WM keys: {wm_missing[:5]}... Unexpected keys: {list(unexpected)[:5]}..."
        )

    skipped = sorted({k.split(".")[0] for k in checkpoint if not k.startswith(_WM_PREFIXES)})
    print(f"  WM-only load: loaded {len(filtered)} world-model tensors (encoder + RSSM). Skipped modules: {skipped}.")
    print("  (actor/value use random init and are not run; reward = env reward.)")


def _wm_forward(agent, obs, state, prev_action):
    """World-model-only inference step for non-goal-conditioned checkpoints.

    Runs encoder + RSSM posterior (`obs_step`) to advance the latent state,
    threading the action that actually produced `obs` as `prev_action` (the
    real transition action, unlike the goal-mode path which reuses the actor's
    discarded action). No goal, no actor, no reward predictor.
    """
    trans = make_trans(obs, prev_action, agent.device)
    with torch.no_grad():
        p_obs = agent.preprocess(trans)
        embed = agent._frozen_encoder(p_obs)
        stoch, deter, _ = agent._frozen_rssm.obs_step(
            state["stoch"], state["deter"], prev_action, embed, trans["is_first"]
        )
    new_state = TensorDict(
        {"stoch": stoch, "deter": deter, "prev_action": prev_action},
        batch_size=state.batch_size,
    )
    return new_state, trans


def create_app(config, env, agent, reward_function, cluster_store: ClusterStore, wm_only: bool = False) -> FastAPI:
    n_actions = env.action_space.shape[0]
    S = config.model.rssm.stoch
    K = config.model.rssm.discrete
    D = config.model.rssm.deter
    metadata = {
        "model": "dreamer",
        "env": config.env.task,
        "stoch_size": [S, K],
        "deter_size": D,
    }

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def health():
        return {"status": "ok", "env": config.env.task}

    @app.get("/clusters/states")
    async def list_cluster_states():
        return {
            "k": cluster_store.k,
            "n_states": cluster_store.n_states,
            "n_trajectories": cluster_store.n_trajectories,
            "states": [
                {
                    "id": f"cz_{idx}",
                    "x": float(x),
                    "y": float(y),
                    "cluster_id": int(c),
                    "trajectory_id": int(t),
                    "timestep": int(ts),
                }
                for idx, ((x, y), c, t, ts) in enumerate(
                    zip(
                        cluster_store.coords,
                        cluster_store.cluster_ids,
                        cluster_store.trajectory_ids,
                        cluster_store.timesteps,
                    )
                )
            ],
        }

    @app.get("/clusters/row_stats")
    async def get_row_stats():
        return compute_row_stats(cluster_store.stochs)

    @app.get("/clusters/states/{state_id}")
    async def get_cluster_state(state_id: str):
        idx = cluster_store.parse_state_id(state_id)
        if idx is None:
            raise HTTPException(status_code=404, detail="state not found")
        return {
            "id": state_id,
            "cluster_id": int(cluster_store.cluster_ids[idx]),
            "x": float(cluster_store.coords[idx, 0]),
            "y": float(cluster_store.coords[idx, 1]),
            "image_b64": encode_image(cluster_store.images[idx]),
            "stoch": cluster_store.stochs[idx].tolist(),
            "trajectory_id": int(cluster_store.trajectory_ids[idx]),
            "timestep": int(cluster_store.timesteps[idx]),
            "prev_id": cluster_store.state_id(cluster_store.prev_ids[idx]),
            "next_id": cluster_store.state_id(cluster_store.next_ids[idx]),
        }

    @app.get("/clusters/{cluster_id}/states")
    async def get_cluster_states(cluster_id: int, limit: int = 200, offset: int = 0):
        if cluster_id < 0 or cluster_id >= cluster_store.k:
            raise HTTPException(status_code=404, detail="cluster not found")
        limit = max(0, min(limit, 500))
        members = cluster_store.cluster_members[cluster_id]
        page = members[offset : offset + limit]
        return {
            "cluster_id": cluster_id,
            "total": len(members),
            "offset": offset,
            "limit": limit,
            "states": [{"id": f"cz_{idx}", "image_b64": encode_image(cluster_store.images[idx])} for idx in page],
        }

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"type": "metadata", "metadata": metadata})

        # Per-connection state
        step_idx = 0
        prev_node_id = None
        done = False

        def _run_inference(obs, agent_state, action, env_reward=0.0):
            if wm_only:
                # No goal / actor / reward predictor; report the env's own reward.
                new_state, trans = _wm_forward(agent, obs, agent_state, action)
                return new_state, float(env_reward), trans
            trans = make_trans(obs, action, agent.device)
            with torch.no_grad():
                _, new_state, _ = agent.act(trans, agent_state, eval=True)
            reward = reward_function(_reward_input(new_state, agent), trans["goal"]).item()
            return new_state, reward, trans

        def _build_step_msg(obs, agent_state, reward, is_first):
            nonlocal step_idx, prev_node_id

            node_id = f"state_{step_idx}"

            stoch_entry = {
                "stoch": agent_state["stoch"][0].detach().cpu().numpy().tolist(),
                "reward": reward,
                "timestep": step_idx,
            }

            image_b64 = encode_image(np.asarray(obs["image"]))
            msg: dict = {
                "type": "step",
                "node_id": node_id,
                "is_new_node": True,
                "stoch_entry": stoch_entry,
                "image_b64": image_b64,  # always sent for the current-obs panel
                "node": {
                    "id": node_id,
                    "image_b64": image_b64,
                    "is_first": is_first,
                },
                "edge": None,
            }

            if prev_node_id is not None:
                msg["edge"] = {"source": prev_node_id, "target": node_id}

            prev_node_id = node_id
            step_idx += 1
            return msg

        # --- start episode ---
        obs = env.reset()
        agent_state = agent.get_initial_state(1)
        action = agent_state["prev_action"].clone()
        agent_state, reward, _ = _run_inference(obs, agent_state, action)
        await websocket.send_json(_build_step_msg(obs, agent_state, reward, is_first=True))

        try:
            while True:
                msg = await websocket.receive_json()

                if msg["type"] == "reset":
                    step_idx = 0
                    prev_node_id = None
                    done = False

                    obs = env.reset()
                    agent_state = agent.get_initial_state(1)
                    action = agent_state["prev_action"].clone()
                    agent_state, reward, _ = _run_inference(obs, agent_state, action)
                    await websocket.send_json(_build_step_msg(obs, agent_state, reward, is_first=True))

                elif msg["type"] == "action" and not done:
                    action_idx = int(msg["action_idx"])
                    action = torch.zeros(1, n_actions, device=agent.device)
                    action[0, action_idx] = 1.0

                    obs, env_reward, done, _ = env.step(action[0].cpu().numpy())
                    agent_state, reward, _ = _run_inference(obs, agent_state, action, env_reward=env_reward)
                    await websocket.send_json(_build_step_msg(obs, agent_state, reward, is_first=False))

                    if done:
                        await websocket.send_json({"type": "done"})

        except WebSocketDisconnect:
            pass

    return app


def main():
    parser = argparse.ArgumentParser(description="Interactive z-flow WebSocket server")
    parser.add_argument("experiment_dir", type=pathlib.Path, help="Path to Hydra experiment directory")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--time-limit", type=int, default=None, help="Override episode time limit")
    parser.add_argument("--device", default=None, help="Override device (e.g. cpu, cuda)")
    parser.add_argument(
        "--cluster-episodes", type=int, default=20, help="Random-policy episodes to collect for the cluster view"
    )
    parser.add_argument(
        "--cluster-max-steps", type=int, default=200, help="Max steps per episode collected for the cluster view"
    )
    parser.add_argument("--cluster-k", type=int, default=8, help="Number of KMeans clusters")
    parser.add_argument("--cluster-seed", type=int, default=0, help="Random seed for KMeans/PCA")
    parser.add_argument(
        "--wm-only",
        action="store_true",
        help="Force world-model-only load (encoder + RSSM). Auto-enabled for "
        "pre-goal-conditioning checkpoints, e.g. original-Dreamer WM runs.",
    )
    args = parser.parse_args()

    train_cfg = OmegaConf.load(args.experiment_dir / ".hydra/config.yaml")
    overrides = {}
    if args.time_limit is not None:
        overrides["env"] = {"time_limit": args.time_limit}
    if args.device is not None:
        overrides["device"] = args.device
    config = OmegaConf.merge(train_cfg, overrides)
    config, wm_only = _backfill_config(config, force_wm_only=args.wm_only)
    if wm_only:
        print("World-model-only mode: goal/actor/reward heads are skipped (WM latent stream only).")

    tools.set_seed_everywhere(config.seed)

    print(f"Loading env: {config.env.task}")
    env = make_env(config.env, 0)

    # In WM-only mode the goal-conditioned reward is undefined (no goal wrapper);
    # the interactive view uses the environment's own reward instead.
    reward_function = None if wm_only else make_reward(config)

    print(f"Loading agent from: {args.experiment_dir / 'latest.pt'}")
    agent = Dreamer(
        config.model,
        env.observation_space,
        env.action_space,
        reward_function=reward_function,
    ).to(config.device)
    checkpoint = torch.load(args.experiment_dir / "latest.pt", map_location=config.device)["agent_state_dict"]
    _load_agent_weights(agent, checkpoint, wm_only)
    agent.eval()

    print(
        f"Collecting {args.cluster_episodes} random episodes for clustering "
        f"(max {args.cluster_max_steps} steps each)..."
    )
    collected = collect_random_states(agent, env, args.cluster_episodes, args.cluster_max_steps, config.device)
    print(f"Collected {len(collected['stochs'])} states. Running KMeans (k={args.cluster_k}) + PCA...")
    clustered = run_clustering(collected["stochs"], args.cluster_k, args.cluster_seed)
    cluster_store = ClusterStore(
        coords=clustered["coords"],
        cluster_ids=clustered["cluster_ids"],
        images=collected["images"],
        stochs=collected["stochs"],
        trajectory_ids=collected["trajectory_ids"],
        timesteps=collected["timesteps"],
        k=args.cluster_k,
    )
    print(f"Clustering ready: {cluster_store.n_states} states, {args.cluster_k} clusters.")

    app = create_app(config, env, agent, reward_function, cluster_store, wm_only=wm_only)
    print(f"Serving at ws://{args.host}:{args.port}/ws")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
