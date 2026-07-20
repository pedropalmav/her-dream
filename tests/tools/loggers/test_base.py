import pytest

from her_dream.tools.loggers.base import BaseLogger


class TestBaseLogger:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BaseLogger()

    def test_all_seven_methods_are_abstract(self):
        abstract = {name for name, val in vars(BaseLogger).items() if getattr(val, "__isabstractmethod__", False)}
        expected = {"scalar", "image", "video", "histogram", "write_step", "write", "log_hydra_config"}
        assert abstract == expected

    def test_partial_subclass_cannot_instantiate(self):
        class Partial(BaseLogger):
            def scalar(self, name, value):
                pass

        with pytest.raises(TypeError):
            Partial()

    def test_fully_implemented_subclass_can_instantiate(self):
        class Full(BaseLogger):
            def scalar(self, n, v):
                pass

            def image(self, n, v):
                pass

            def video(self, n, v):
                pass

            def histogram(self, n, v):
                pass

            def write_step(self, n, v, s):
                pass

            def write(self, s, fps=False):
                pass

            def log_hydra_config(self, c, name="config", step=0, log_hparams=False, hparams_run_name="."):
                pass

        f = Full()
        assert isinstance(f, BaseLogger)
