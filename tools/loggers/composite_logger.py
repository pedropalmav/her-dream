import time

from .base import BaseLogger


class CompositeLogger(BaseLogger):
    def __init__(self, loggers: list):
        self._loggers = loggers
        self._last_step = None
        self._last_time = None

    def scalar(self, name: str, value: float) -> None:
        for lg in self._loggers:
            lg.scalar(name, value)

    def image(self, name: str, value) -> None:
        for lg in self._loggers:
            lg.image(name, value)

    def video(self, name: str, value) -> None:
        for lg in self._loggers:
            lg.video(name, value)

    def histogram(self, name: str, value) -> None:
        for lg in self._loggers:
            lg.histogram(name, value)

    def write_step(self, name: str, value: float, step: int) -> None:
        for lg in self._loggers:
            lg.write_step(name, value, step)

    def write(self, step: int, fps: bool = False) -> None:
        if fps:
            fps_value = self._compute_fps(step)
            for lg in self._loggers:
                lg.scalar("fps/fps", fps_value)
        for lg in self._loggers:
            lg.write(step, fps=False)

    def log_hydra_config(
        self,
        config,
        name: str = "config",
        step: int = 0,
        log_hparams: bool = False,
        hparams_run_name: str = ".",
    ) -> None:
        for lg in self._loggers:
            lg.log_hydra_config(config, name, step, log_hparams, hparams_run_name)

    def _compute_fps(self, step: int) -> float:
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0.0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration
