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

import tools
from dreamer import Dreamer
from envs import make_env
from rewards import make_reward

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


def _reward_input(agent_state: TensorDict, agent):
    """Return the correct first argument for reward_function given the goal type.

    log_prob / prob  → Independent(OneHotCategorical) distribution built from logits
    argmax_full      → raw logit tensor
    others           → sampled stoch tensor
    """
    goal_type = agent.goal_type
    if goal_type in ("log_prob", "prob"):
        return agent._frozen_rssm.get_dist(agent_state["logit"])
    if goal_type == "argmax_full":
        return agent_state["logit"]
    return agent_state["stoch"]


def _backfill_config(config):
    """Fill in config keys added after older experiments were trained."""
    updates = {}
    if "goal_imag_horizon" not in config.model:
        updates["model"] = {"goal_imag_horizon": int(config.model.imag_horizon)}
    return OmegaConf.merge(config, updates) if updates else config


def create_app(config, env, agent, reward_function, cluster_store: ClusterStore) -> FastAPI:
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

        def _run_inference(obs, agent_state, action):
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

                    obs, _, done, _ = env.step(action[0].cpu().numpy())
                    agent_state, reward, _ = _run_inference(obs, agent_state, action)
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
    args = parser.parse_args()

    train_cfg = OmegaConf.load(args.experiment_dir / ".hydra/config.yaml")
    overrides = {}
    if args.time_limit is not None:
        overrides["env"] = {"time_limit": args.time_limit}
    if args.device is not None:
        overrides["device"] = args.device
    config = OmegaConf.merge(train_cfg, overrides)
    config = _backfill_config(config)

    tools.set_seed_everywhere(config.seed)

    print(f"Loading env: {config.env.task}")
    env = make_env(config.env, 0)

    reward_function = make_reward(config)

    print(f"Loading agent from: {args.experiment_dir / 'latest.pt'}")
    agent = Dreamer(
        config.model,
        env.observation_space,
        env.action_space,
        reward_function=reward_function,
    ).to(config.device)
    agent.load_state_dict(torch.load(args.experiment_dir / "latest.pt", map_location=config.device)["agent_state_dict"])
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

    app = create_app(config, env, agent, reward_function, cluster_store)
    print(f"Serving at ws://{args.host}:{args.port}/ws")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
