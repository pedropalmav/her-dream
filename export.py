import base64
import io
import json
import pathlib

import numpy as np
from PIL import Image


class TrajectoryExporter:
    def __init__(self, stoch_size, deter_size, env_name):
        self._stoch_size = tuple(stoch_size)  # (S, K)
        self._deter_size = int(deter_size)
        self._env_name = str(env_name)
        self._steps = []

    def add_step(self, obs_image, rssm_output, reward, is_first):
        """Accumulate one step; all tensors are converted to Python types immediately."""
        # stoch: (B, S, K) → first env → (S, K) → list of lists
        stoch = rssm_output["stoch"][0].detach().cpu().numpy().tolist()

        # image: (H, W, C) tensor or ndarray, uint8 expected
        if hasattr(obs_image, "detach"):
            img_np = obs_image.detach().cpu().numpy()
        else:
            img_np = np.asarray(obs_image)
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img_np, mode="RGB").save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        reward_val = reward.item() if hasattr(reward, "item") else float(reward)

        self._steps.append({
            "id": f"step_{len(self._steps)}",
            "stoch": stoch,
            "image_b64": image_b64,
            "reward": reward_val,
            "is_first": bool(is_first),
        })

    def save(self, path):
        S, K = self._stoch_size
        nodes = self._steps
        edges = [
            {"source": f"step_{i}", "target": f"step_{i + 1}"}
            for i in range(len(nodes) - 1)
        ]
        payload = {
            "metadata": {
                "model": "dreamer",
                "env": self._env_name,
                "stoch_size": [S, K],
                "deter_size": self._deter_size,
            },
            "nodes": nodes,
            "edges": edges,
        }
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)
        self._steps = []
