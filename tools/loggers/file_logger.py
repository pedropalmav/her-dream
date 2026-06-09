import json

import numpy as np
import torch
import torchvision.io

from .base import BaseLogger


class FileLogger(BaseLogger):
    def __init__(self, logdir, filename="metrics.jsonl"):
        self._logdir = logdir
        self._filename = filename
        self._scalars = {}
        self._videos = {}

    def scalar(self, name: str, value: float) -> None:
        self._scalars[name] = float(value)

    def image(self, name: str, value) -> None:
        pass  # file logger does not persist images separately

    def video(self, name: str, value) -> None:
        self._videos[name] = np.array(value)

    def histogram(self, name: str, value) -> None:
        pass  # histograms are not written to file

    def write_step(self, name: str, value: float, step: int) -> None:
        pass  # per-step scalars are not written to the JSONL file

    def write(self, step: int, fps: bool = False) -> None:
        scalars = list(self._scalars.items())
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / self._filename).open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            video_tensor = torch.from_numpy(value[0])
            safe_name = name.replace("/", "_")
            torchvision.io.write_video(
                self._logdir / f"{safe_name}.mp4",
                video_tensor,
                fps=10,
                video_codec="libx264",
            )
        self._scalars = {}
        self._videos = {}

    def log_hydra_config(
        self,
        config,
        name: str = "config",
        step: int = 0,
        log_hparams: bool = False,
        hparams_run_name: str = ".",
    ) -> None:
        pass  # Hydra already saves .hydra/config.yaml; nothing extra needed here
