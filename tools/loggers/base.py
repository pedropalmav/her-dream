import abc


class BaseLogger(abc.ABC):

    @abc.abstractmethod
    def scalar(self, name: str, value: float) -> None: ...

    @abc.abstractmethod
    def image(self, name: str, value) -> None: ...

    @abc.abstractmethod
    def video(self, name: str, value) -> None: ...

    @abc.abstractmethod
    def histogram(self, name: str, value) -> None: ...

    @abc.abstractmethod
    def write_step(self, name: str, value: float, step: int) -> None: ...

    @abc.abstractmethod
    def write(self, step: int, fps: bool = False) -> None: ...

    @abc.abstractmethod
    def log_hydra_config(
        self, config, name: str = "config", step: int = 0,
        log_hparams: bool = False, hparams_run_name: str = ".",
    ) -> None: ...
