import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tools.logging import Logger, Tee, make_logger, setup_console_log


class TestTee:
    def test_write_returns_length(self):
        stream = io.StringIO()
        tee = Tee(stream)
        assert tee.write("hello") == 5

    def test_write_propagates_to_all_streams(self):
        s1, s2 = io.StringIO(), io.StringIO()
        tee = Tee(s1, s2)
        tee.write("test")
        assert s1.getvalue() == "test"
        assert s2.getvalue() == "test"

    def test_none_filtered_from_streams(self):
        s = io.StringIO()
        tee = Tee(None, s, None)
        assert len(tee._streams) == 1

    def test_all_none_gives_empty_streams(self):
        tee = Tee(None, None)
        assert tee._streams == []

    def test_flush_skips_closed_stream(self):
        s = MagicMock()
        s.closed = True
        tee = Tee(s)
        tee.flush()
        s.flush.assert_not_called()

    def test_flush_calls_open_stream(self):
        s = MagicMock()
        s.closed = False
        tee = Tee(s)
        tee.flush()
        s.flush.assert_called_once()

    def test_flush_stream_without_closed_attr_is_flushed(self):
        # getattr(stream, "closed", False) → False → stream.flush() IS called
        class NoClosedAttr:
            def __init__(self):
                self.flushed = False

            def flush(self):
                self.flushed = True

        nc = NoClosedAttr()
        tee = Tee(nc)
        tee.flush()
        assert nc.flushed

    def test_isatty_true_if_any_stream_returns_true(self):
        s1 = MagicMock()
        s1.isatty.return_value = False
        s2 = MagicMock()
        s2.isatty.return_value = True
        tee = Tee(s1, s2)
        assert tee.isatty() is True

    def test_isatty_false_if_no_stream_returns_true(self):
        s = MagicMock()
        s.isatty.return_value = False
        tee = Tee(s)
        assert tee.isatty() is False

    def test_isatty_false_if_stream_has_no_isatty_attr(self):
        class NoIsatty:
            pass

        tee = Tee(NoIsatty())
        assert tee.isatty() is False

    def test_isatty_false_for_empty_streams(self):
        tee = Tee()
        assert tee.isatty() is False


class TestLogger:
    def test_scalar_stored_as_float(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        lg.scalar("loss", 1.5)
        assert lg._scalars["loss"] == 1.5

    def test_image_stored_as_array(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        arr = np.zeros((3, 4, 4))
        lg.image("img", arr)
        np.testing.assert_array_equal(lg._images["img"], arr)

    def test_video_stored_as_array(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        arr = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
        lg.video("vid", arr)
        assert lg._videos["vid"].shape == arr.shape

    def test_histogram_stored_as_array(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        arr = np.array([1.0, 2.0, 3.0])
        lg.histogram("hist", arr)
        np.testing.assert_array_equal(lg._histograms["hist"], arr)

    def test_write_step_with_slash_uses_name_as_tag(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        lg.write_step("train/loss", 1.0, step=5)
        mock_summary_writer_logging.return_value.add_scalar.assert_called_once_with("train/loss", 1.0, 5)

    def test_write_step_without_slash_prefixes_scalars(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        lg.write_step("loss", 1.0, step=5)
        mock_summary_writer_logging.return_value.add_scalar.assert_called_once_with("scalars/loss", 1.0, 5)

    def test_write_creates_jsonl_file(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        lg.scalar("loss", 1.5)
        lg.write(step=1)
        data = json.loads((tmp_path / "metrics.jsonl").read_text())
        assert data["step"] == 1
        assert data["loss"] == pytest.approx(1.5)

    def test_write_appends_multiple_steps(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        lg.scalar("loss", 1.0)
        lg.write(step=1)
        lg.scalar("loss", 0.5)
        lg.write(step=2)
        lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_write_fps_true_appends_fps_scalar(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        with patch("tools.logging.time.time", return_value=0.0):
            lg.write(step=0, fps=True)
        data = json.loads((tmp_path / "metrics.jsonl").read_text())
        assert "fps/fps" in data

    def test_write_fps_false_no_fps_metric(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        lg.write(step=1, fps=False)
        data = json.loads((tmp_path / "metrics.jsonl").read_text())
        assert "fps/fps" not in data

    def test_write_scalar_with_slash_uses_name(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        lg.scalar("train/loss", 1.0)
        lg.write(step=1)
        writer = mock_summary_writer_logging.return_value
        writer.add_scalar.assert_any_call("train/loss", 1.0, 1)

    def test_write_scalar_without_slash_prefixed(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        lg.scalar("loss", 1.0)
        lg.write(step=1)
        writer = mock_summary_writer_logging.return_value
        writer.add_scalar.assert_any_call("scalars/loss", 1.0, 1)

    def test_write_video_str_name(self, tmp_path, mock_summary_writer_logging, mock_write_video_logging):
        lg = Logger(tmp_path)
        vid = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
        lg.video("my_vid", vid)
        lg.write(step=1)
        assert mock_write_video_logging.called

    def test_write_video_bytes_name_decoded(self, tmp_path, mock_summary_writer_logging, mock_write_video_logging):
        lg = Logger(tmp_path)
        vid = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
        lg.video(b"my_vid", vid)
        lg.write(step=1)
        call_path = str(mock_write_video_logging.call_args[0][0])
        assert "my_vid" in call_path

    def test_write_video_float_clipped_to_uint8(self, tmp_path, mock_summary_writer_logging, mock_write_video_logging):
        lg = Logger(tmp_path)
        vid = np.ones((1, 2, 4, 4, 3), dtype=np.float32) * 0.5
        lg.video("vid", vid)
        lg.write(step=1)
        passed_tensor = mock_write_video_logging.call_args[0][1]
        assert passed_tensor.dtype == torch.uint8

    def test_write_video_uint8_passed_unchanged(self, tmp_path, mock_summary_writer_logging, mock_write_video_logging):
        lg = Logger(tmp_path)
        vid = np.full((1, 2, 4, 4, 3), 200, dtype=np.uint8)
        lg.video("vid", vid)
        lg.write(step=1)
        passed_tensor = mock_write_video_logging.call_args[0][1]
        assert passed_tensor.dtype == torch.uint8
        assert passed_tensor.max().item() == 200

    def test_write_resets_scalars(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        lg.scalar("loss", 1.0)
        lg.write(step=1)
        assert lg._scalars == {}

    def test_write_resets_images(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        lg.image("img", np.zeros((3, 4, 4)))
        lg.write(step=1)
        assert lg._images == {}

    def test_write_resets_videos(self, tmp_path, mock_summary_writer_logging, mock_write_video_logging):
        lg = Logger(tmp_path)
        lg.video("vid", np.zeros((1, 2, 4, 4, 3), dtype=np.uint8))
        lg.write(step=1)
        assert lg._videos == {}

    def test_write_does_not_reset_histograms(self, tmp_path, mock_summary_writer_logging):
        # Known divergence: Logger does NOT clear _histograms after write()
        lg = Logger(tmp_path)
        lg.histogram("h", np.array([1.0, 2.0]))
        lg.write(step=1)
        assert lg._histograms != {}

    def test_write_flushes_writer(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        lg.write(step=1)
        mock_summary_writer_logging.return_value.flush.assert_called()

    def test_compute_fps_first_call_returns_zero(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        with patch("tools.logging.time.time", return_value=0.0):
            fps = lg._compute_fps(0)
        assert fps == 0

    def test_compute_fps_subsequent_returns_rate(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        with patch("tools.logging.time.time", side_effect=[0.0, 1.0]):
            lg._compute_fps(0)  # first call: _last_step=0, _last_time=0.0
            fps = lg._compute_fps(10)  # steps=10, duration=1.0
        assert fps == pytest.approx(10.0)


class TestLoggerLogHydraConfig:
    def test_import_error_uses_str_fallback(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        with patch.dict("sys.modules", {"omegaconf": None}):
            lg.log_hydra_config({"key": "value"})
        call_arg = mock_summary_writer_logging.return_value.add_text.call_args[0][1]
        assert "key" in call_arg

    def test_add_text_called_with_yaml_content(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        mock_oc = MagicMock()
        mock_oc.OmegaConf.to_yaml.return_value = "lr: 0.001\n"
        mock_oc.OmegaConf.to_container.return_value = {"lr": 0.001}
        with patch.dict("sys.modules", {"omegaconf": mock_oc}):
            lg.log_hydra_config(MagicMock())
        call_arg = mock_summary_writer_logging.return_value.add_text.call_args[0][1]
        assert "lr: 0.001" in call_arg

    def test_log_hparams_false_skips_add_hparams(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        mock_oc = MagicMock()
        mock_oc.OmegaConf.to_yaml.return_value = ""
        mock_oc.OmegaConf.to_container.return_value = {"a": 1}
        with patch.dict("sys.modules", {"omegaconf": mock_oc}):
            lg.log_hydra_config(MagicMock(), log_hparams=False)
        mock_summary_writer_logging.return_value.add_hparams.assert_not_called()

    def test_log_hparams_true_container_none_skips_add_hparams(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        mock_oc = MagicMock()
        mock_oc.OmegaConf.to_yaml.return_value = ""
        mock_oc.OmegaConf.to_container.side_effect = RuntimeError("fail")
        with patch.dict("sys.modules", {"omegaconf": mock_oc}):
            lg.log_hydra_config(MagicMock(), log_hparams=True)
        mock_summary_writer_logging.return_value.add_hparams.assert_not_called()

    def test_log_hparams_true_calls_add_hparams(self, tmp_path, mock_summary_writer_logging):
        lg = Logger(tmp_path)
        mock_oc = MagicMock()
        mock_oc.OmegaConf.to_yaml.return_value = ""
        mock_oc.OmegaConf.to_container.return_value = {"lr": 0.001}
        with patch.dict("sys.modules", {"omegaconf": mock_oc}):
            lg.log_hydra_config(MagicMock(), log_hparams=True)
        mock_summary_writer_logging.return_value.add_hparams.assert_called_once()

    def _call_with_container(self, tmp_path, mock_summary_writer_logging, container):
        lg = Logger(tmp_path)
        mock_oc = MagicMock()
        mock_oc.OmegaConf.to_yaml.return_value = ""
        mock_oc.OmegaConf.to_container.return_value = container
        with patch.dict("sys.modules", {"omegaconf": mock_oc}):
            lg.log_hydra_config(MagicMock(), log_hparams=True)
        return mock_summary_writer_logging.return_value.add_hparams.call_args[0][0]

    def test_flatten_dict_recurses(self, tmp_path, mock_summary_writer_logging):
        flat = self._call_with_container(tmp_path, mock_summary_writer_logging, {"nested": {"lr": 0.001}})
        assert "nested.lr" in flat

    def test_flatten_list_str_coerced(self, tmp_path, mock_summary_writer_logging):
        flat = self._call_with_container(tmp_path, mock_summary_writer_logging, {"tags": [1, 2, 3]})
        assert flat["tags"] == str([1, 2, 3])

    def test_flatten_tuple_str_coerced(self, tmp_path, mock_summary_writer_logging):
        flat = self._call_with_container(tmp_path, mock_summary_writer_logging, {"pair": (1, 2)})
        assert flat["pair"] == str((1, 2))

    def test_flatten_int_stored_directly(self, tmp_path, mock_summary_writer_logging):
        flat = self._call_with_container(tmp_path, mock_summary_writer_logging, {"n": 42})
        assert flat["n"] == 42

    def test_flatten_float_stored_directly(self, tmp_path, mock_summary_writer_logging):
        flat = self._call_with_container(tmp_path, mock_summary_writer_logging, {"lr": 0.001})
        assert flat["lr"] == pytest.approx(0.001)

    def test_flatten_bool_stored_directly(self, tmp_path, mock_summary_writer_logging):
        flat = self._call_with_container(tmp_path, mock_summary_writer_logging, {"flag": True})
        assert flat["flag"] is True

    def test_flatten_str_stored_directly(self, tmp_path, mock_summary_writer_logging):
        flat = self._call_with_container(tmp_path, mock_summary_writer_logging, {"name": "foo"})
        assert flat["name"] == "foo"

    def test_flatten_none_becomes_null(self, tmp_path, mock_summary_writer_logging):
        flat = self._call_with_container(tmp_path, mock_summary_writer_logging, {"x": None})
        assert flat["x"] == "null"

    def test_flatten_other_type_str_coerced(self, tmp_path, mock_summary_writer_logging):
        class _Custom:
            def __str__(self):
                return "custom_repr"

        flat = self._call_with_container(tmp_path, mock_summary_writer_logging, {"obj": _Custom()})
        assert flat["obj"] == "custom_repr"


class TestMakeLogger:
    def test_file_backend_creates_file_logger(self, tmp_path):
        from tools.loggers import CompositeLogger, FileLogger

        cfg = SimpleNamespace(backends=["file"])
        result = make_logger(cfg, tmp_path)
        assert isinstance(result, CompositeLogger)
        assert isinstance(result._loggers[0], FileLogger)

    def test_tensorboard_backend_creates_tb_logger(self, tmp_path, mock_summary_writer_tb):
        from tools.loggers import CompositeLogger, TensorboardLogger

        cfg = SimpleNamespace(backends=["tensorboard"])
        result = make_logger(cfg, tmp_path)
        assert isinstance(result, CompositeLogger)
        assert isinstance(result._loggers[0], TensorboardLogger)

    def test_wandb_backend_creates_composite(self, tmp_path):
        from tools.loggers import CompositeLogger

        cfg = SimpleNamespace(backends=["wandb"], project="test")
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}), patch("tools.logging.atexit.register"):
            result = make_logger(cfg, tmp_path)
        assert isinstance(result, CompositeLogger)

    def test_wandb_backend_registers_atexit(self, tmp_path):
        cfg = SimpleNamespace(backends=["wandb"], project="test")
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}), patch("tools.logging.atexit.register") as mock_at:
            make_logger(cfg, tmp_path)
        mock_at.assert_called_once()

    def test_unknown_backend_raises_value_error(self, tmp_path):
        cfg = SimpleNamespace(backends=["unknown"])
        with pytest.raises(ValueError, match="Unknown logger backend"):
            make_logger(cfg, tmp_path)

    def test_multiple_backends_creates_composite_with_all(self, tmp_path, mock_summary_writer_tb):
        from tools.loggers import CompositeLogger

        cfg = SimpleNamespace(backends=["file", "tensorboard"])
        result = make_logger(cfg, tmp_path)
        assert isinstance(result, CompositeLogger)
        assert len(result._loggers) == 2


class TestSetupConsoleLog:
    def test_replaces_stdout_with_tee(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "stdout", sys.stdout)
        monkeypatch.setattr(sys, "stderr", sys.stderr)
        f = setup_console_log(tmp_path)
        assert isinstance(sys.stdout, Tee)
        f.close()

    def test_replaces_stderr_with_tee(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "stdout", sys.stdout)
        monkeypatch.setattr(sys, "stderr", sys.stderr)
        f = setup_console_log(tmp_path)
        assert isinstance(sys.stderr, Tee)
        f.close()

    def test_returns_open_file_handle(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "stdout", sys.stdout)
        monkeypatch.setattr(sys, "stderr", sys.stderr)
        f = setup_console_log(tmp_path)
        assert not f.closed
        f.close()

    def test_creates_log_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "stdout", sys.stdout)
        monkeypatch.setattr(sys, "stderr", sys.stderr)
        f = setup_console_log(tmp_path)
        f.close()
        assert (tmp_path / "console.log").exists()

    def test_custom_filename(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "stdout", sys.stdout)
        monkeypatch.setattr(sys, "stderr", sys.stderr)
        f = setup_console_log(tmp_path, filename="custom.log")
        f.close()
        assert (tmp_path / "custom.log").exists()
