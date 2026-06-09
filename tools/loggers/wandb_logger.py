import numpy as np

from .base import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(self, config, logdir):
        import wandb

        project = getattr(config, "project", "her-dream")
        run_name = "/".join(logdir.parts[-2:])
        self._run = wandb.init(
            project=project,
            name=run_name,
            dir=str(logdir),
            config={},
            sync_tensorboard=False,
        )
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._histograms = {}
        self._pending_step = None
        self._pending_payload = {}

    def scalar(self, name: str, value: float) -> None:
        self._scalars[name] = float(value)

    def image(self, name: str, value) -> None:
        self._images[name] = np.array(value)

    def video(self, name: str, value) -> None:
        self._videos[name] = np.array(value)

    def histogram(self, name: str, value) -> None:
        self._histograms[name] = np.array(value)

    def _flush_pending(self) -> None:
        import wandb

        if self._pending_payload and self._pending_step is not None:
            wandb.log(self._pending_payload, step=self._pending_step, commit=True)
        self._pending_payload = {}

    def write_step(self, name: str, value: float, step: int) -> None:
        if self._pending_step is not None and self._pending_step != step:
            self._flush_pending()
        self._pending_step = step
        self._pending_payload[name] = float(value)

    def write(self, step: int, fps: bool = False) -> None:
        import wandb

        if self._pending_step is not None and self._pending_step != step:
            self._flush_pending()
        self._pending_step = step

        for name, value in self._scalars.items():
            self._pending_payload[name] = value
        for name, value in self._images.items():
            self._pending_payload[name] = wandb.Image(value)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            video_tchw = np.transpose(value[0], (0, 3, 1, 2))  # (T,H,W,C) → (T,C,H,W)
            self._pending_payload[name] = wandb.Video(video_tchw, fps=10, format="gif")
        for name, value in self._histograms.items():
            self._pending_payload[name] = wandb.Histogram(value)

        self._scalars = {}
        self._images = {}
        self._videos = {}
        self._histograms = {}

    def log_hydra_config(
        self,
        config,
        name: str = "config",
        step: int = 0,
        log_hparams: bool = False,
        hparams_run_name: str = ".",
    ) -> None:
        try:
            from omegaconf import OmegaConf

            container = OmegaConf.to_container(config, resolve=True)
        except Exception:
            container = str(config)
        self._run.config.update(container, allow_val_change=True)

    def close(self) -> None:
        self._flush_pending()
        import wandb

        wandb.finish()
