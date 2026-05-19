import contextlib

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .base import BaseLogger


class TensorboardLogger(BaseLogger):
    def __init__(self, logdir):
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._histograms = {}

    def scalar(self, name: str, value: float) -> None:
        self._scalars[name] = float(value)

    def image(self, name: str, value) -> None:
        self._images[name] = np.array(value)

    def video(self, name: str, value) -> None:
        self._videos[name] = np.array(value)

    def histogram(self, name: str, value) -> None:
        self._histograms[name] = np.array(value)

    def write_step(self, name: str, value: float, step: int) -> None:
        tag = name if "/" in name else "scalars/" + name
        self._writer.add_scalar(tag, float(value), step)

    def write(self, step: int, fps: bool = False) -> None:
        for name, value in self._scalars.items():
            tag = name if "/" in name else "scalars/" + name
            self._writer.add_scalar(tag, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, 16)
        for name, value in self._histograms.items():
            self._writer.add_histogram(name, value, step)
        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._histograms = {}

    def log_hydra_config(
        self, config, name: str = "config", step: int = 0,
        log_hparams: bool = False, hparams_run_name: str = ".",
    ) -> None:
        yaml_str = None
        try:
            from omegaconf import OmegaConf
            yaml_str = OmegaConf.to_yaml(config, resolve=True)
        except ImportError:
            yaml_str = str(config)
        self._writer.add_text(f"{name}/yaml", f"```yaml\n{yaml_str}\n```", step)

        if log_hparams:
            container = None
            try:
                from omegaconf import OmegaConf
                container = OmegaConf.to_container(config, resolve=True)
            except Exception:
                pass
            if container is not None:
                flat = {}

                def _flatten(prefix, obj):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            _flatten(f"{prefix}.{k}" if prefix else k, v)
                    elif isinstance(obj, (list, tuple)):
                        flat[prefix] = str(obj)
                    elif isinstance(obj, (int, float, bool, str)) or obj is None:
                        flat[prefix] = obj if obj is not None else "null"
                    else:
                        flat[prefix] = str(obj)

                _flatten("", container)
                with contextlib.suppress(TypeError):
                    self._writer.add_hparams(flat, {"_": 0}, run_name=hparams_run_name)
