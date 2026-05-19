import contextlib
import io
import json
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision.io


class Tee(io.TextIOBase):
    """A text stream that duplicates writes to multiple underlying streams.

    This is used to mirror stdout/stderr to a log file while keeping the
    original console output unchanged.
    """

    def __init__(self, *streams):
        super().__init__()
        # Filter out None and keep a stable order.
        self._streams = [s for s in streams if s is not None]

    def write(self, s):
        # io.TextIOBase requires returning number of characters written.
        # Some streams may return None; we still return len(s).
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self):
        for stream in self._streams:
            if not getattr(stream, "closed", False):
                stream.flush()

    def isatty(self):
        # Preserve tty detection for progress bars etc.
        return any(
            hasattr(stream, "isatty") and stream.isatty() for stream in self._streams
        )


class Logger:
    def __init__(self, logdir, filename="metrics.jsonl"):
        self._logdir = logdir
        self._filename = filename
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._histograms = {}

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def histogram(self, name, value):
        self._histograms[name] = np.array(value)

    def write_step(self, name, value, step):
        tag = name if "/" in name else "scalars/" + name
        self._writer.add_scalar(tag, float(value), step)

    def write(self, step, fps=False):
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps/fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / self._filename).open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)

            # Save video as mp4
            video_to_save = value[0]
            video_tensor = torch.from_numpy(video_to_save)
            safe_name = name.replace("/", "_")
            file_path = self._logdir / f"{safe_name}.mp4"
            torchvision.io.write_video(
                file_path, video_tensor, fps=10, video_codec="libx264"
            )

            # Save to tensorboard
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, 16)
        for name, value in self._histograms.items():
            self._writer.add_histogram(name, value, step)

        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def log_hydra_config(
        self, config, name="config", step=0, log_hparams=False, hparams_run_name="."
    ):
        """
        Log a Hydra/OmegaConf config to TensorBoard:
          - as YAML text under "{name}/yaml"
          - as flattened hparams to the HParams plugin
        """
        # 1) Log YAML to Text plugin
        yaml_str = None
        try:
            from omegaconf import (
                OmegaConf,  # local import to avoid hard dependency at module import
            )

            yaml_str = OmegaConf.to_yaml(config, resolve=True)
        except ImportError:
            # Fallback to string representation
            yaml_str = str(config)
        self._writer.add_text(f"{name}/yaml", f"```yaml\n{yaml_str}\n```", step)

        # 2) Log flattened hparams to HParams plugin
        flat = {}
        container = None
        try:
            from omegaconf import OmegaConf  # local import again

            container = OmegaConf.to_container(config, resolve=True)
        except Exception:
            container = None

        if log_hparams and container is not None:

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
            # add_hparams requires a non-empty metrics dict
            with contextlib.suppress(TypeError):
                # Avoid creating a timestamped subdirectory by specifying run_name (PyTorch >= 1.14)
                self._writer.add_hparams(flat, {"_": 0}, run_name=hparams_run_name)


def setup_console_log(logdir, filename="console.log"):
    """Mirror stdout/stderr to a file under logdir.

    After calling this, anything written to stdout/stderr (print, tracebacks,
    etc.) will be visible both in the terminal and in the log file.

    Returns
    -------
    file handle
        The opened file handle so that the caller can manage its lifetime.
    """
    import sys

    # Line-buffered text file for timely flushing.
    path = logdir / filename
    f = path.open("a", buffering=1)
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)
    return f
