#!/usr/bin/env python3
"""Interactive z-flow WebSocket server.

Loads an experiment directory (env config + checkpoint) and streams step data
to z-flow's browser frontend via WebSocket. The client sends action indices;
the server steps the environment, runs Dreamer inference, and returns the
observation image + stochastic embedding for each step.

Every step creates a brand new node and a new edge from the previous node,
even if the observation matches one seen earlier in the episode.

Usage:
    uv run serve.py <experiment_dir>
    uv run serve.py <experiment_dir> --host 0.0.0.0 --port 8765 --time-limit 200
"""

import argparse
import base64
import io
import pathlib
import sys
import warnings

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf
from PIL import Image
from tensordict import TensorDict

sys.path.append(str(pathlib.Path(__file__).parent))
import tools
from dreamer import Dreamer
from envs import make_env
from rewards import make_reward

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


def _encode_image(img_np: np.ndarray) -> str:
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_np, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_trans(obs: dict, action: torch.Tensor, device) -> TensorDict:
    """Build a TensorDict from the raw obs dict + action, matching evaluate.py exactly."""
    trans = TensorDict(
        {k: torch.as_tensor(v, device="cpu").unsqueeze(0) for k, v in obs.items()},
        batch_size=(1,),
        device="cpu",
    ).pin_memory()
    trans = trans.to(device, non_blocking=True)
    trans["action"] = action
    return trans


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


def create_app(config, env, agent, reward_function) -> FastAPI:
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

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_json({"type": "metadata", "metadata": metadata})

        # Per-connection state
        step_idx = 0
        prev_node_id = None
        done = False

        def _run_inference(obs, agent_state, action):
            trans = _make_trans(obs, action, agent.device)
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

            image_b64 = _encode_image(np.asarray(obs["image"]))
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

    app = create_app(config, env, agent, reward_function)
    print(f"Serving at ws://{args.host}:{args.port}/ws")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
