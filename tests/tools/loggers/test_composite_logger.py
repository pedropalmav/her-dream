from unittest.mock import MagicMock, patch

import pytest

from tools.loggers.composite_logger import CompositeLogger


class TestCompositeLoggerDelegation:
    def _make(self, n=2):
        loggers = [MagicMock() for _ in range(n)]
        cl = CompositeLogger(loggers)
        return cl, loggers

    def test_scalar_delegates_to_all(self):
        cl, lgs = self._make()
        cl.scalar("loss", 1.5)
        for lg in lgs:
            lg.scalar.assert_called_once_with("loss", 1.5)

    def test_image_delegates_to_all(self):
        cl, lgs = self._make()
        cl.image("img", "value")
        for lg in lgs:
            lg.image.assert_called_once_with("img", "value")

    def test_video_delegates_to_all(self):
        cl, lgs = self._make()
        cl.video("vid", "value")
        for lg in lgs:
            lg.video.assert_called_once_with("vid", "value")

    def test_histogram_delegates_to_all(self):
        cl, lgs = self._make()
        cl.histogram("h", "value")
        for lg in lgs:
            lg.histogram.assert_called_once_with("h", "value")

    def test_write_step_delegates_to_all(self):
        cl, lgs = self._make()
        cl.write_step("loss", 1.0, 5)
        for lg in lgs:
            lg.write_step.assert_called_once_with("loss", 1.0, 5)

    def test_log_hydra_config_delegates_to_all(self):
        cl, lgs = self._make()
        cl.log_hydra_config("cfg", name="c", step=0)
        for lg in lgs:
            lg.log_hydra_config.assert_called_once_with("cfg", "c", 0, False, ".")

    def test_zero_loggers_no_error(self):
        cl = CompositeLogger([])
        cl.scalar("x", 1.0)
        cl.write(1)


class TestCompositeLoggerWrite:
    def _make(self, n=2):
        loggers = [MagicMock() for _ in range(n)]
        cl = CompositeLogger(loggers)
        return cl, loggers

    def test_write_fps_false_calls_write_on_each(self):
        cl, lgs = self._make()
        cl.write(step=10, fps=False)
        for lg in lgs:
            lg.write.assert_called_once_with(10, fps=False)

    def test_write_fps_false_does_not_inject_scalar(self):
        cl, lgs = self._make()
        cl.write(step=10, fps=False)
        for lg in lgs:
            lg.scalar.assert_not_called()

    def test_write_fps_true_injects_fps_scalar(self):
        cl, lgs = self._make()
        # First call: _last_step is None → _compute_fps returns 0.0
        cl.write(step=0, fps=True)
        for lg in lgs:
            lg.scalar.assert_called_once_with("fps/fps", 0.0)

    def test_write_fps_true_calls_write_with_fps_false(self):
        cl, lgs = self._make()
        cl.write(step=10, fps=True)
        for lg in lgs:
            lg.write.assert_called_once_with(10, fps=False)

    def test_write_fps_true_computes_fps_once_shared_across_backends(self):
        cl, lgs = self._make(n=3)
        cl.write(step=0, fps=True)
        # All three backends receive the same fps value
        fps_vals = [lg.scalar.call_args[0][1] for lg in lgs]
        assert fps_vals[0] == fps_vals[1] == fps_vals[2]


class TestCompositeLoggerComputeFps:
    def test_first_call_returns_zero(self):
        cl = CompositeLogger([])
        assert cl._compute_fps(100) == 0.0

    def test_first_call_initializes_last_step(self):
        cl = CompositeLogger([])
        cl._compute_fps(42)
        assert cl._last_step == 42

    def test_subsequent_call_returns_rate(self):
        cl = CompositeLogger([])
        with patch("tools.loggers.composite_logger.time.time", side_effect=[0.0, 1.0]):
            cl._compute_fps(0)
            fps = cl._compute_fps(10)
        assert fps == pytest.approx(10.0)

    def test_subsequent_call_updates_last_step(self):
        cl = CompositeLogger([])
        with patch("tools.loggers.composite_logger.time.time", side_effect=[0.0, 1.0]):
            cl._compute_fps(0)
            cl._compute_fps(10)
        assert cl._last_step == 10
