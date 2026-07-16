import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from her_dream.tools.loggers.file_logger import FileLogger


class TestFileLoggerNoOps:
    def test_image_returns_none(self, tmp_path):
        lg = FileLogger(tmp_path)
        assert lg.image("img", np.zeros((3, 4, 4))) is None

    def test_histogram_returns_none(self, tmp_path):
        lg = FileLogger(tmp_path)
        assert lg.histogram("h", np.array([1.0])) is None

    def test_write_step_returns_none(self, tmp_path):
        lg = FileLogger(tmp_path)
        assert lg.write_step("loss", 1.0, 1) is None

    def test_log_hydra_config_returns_none(self, tmp_path):
        lg = FileLogger(tmp_path)
        assert lg.log_hydra_config({}) is None


class TestFileLoggerScalarAndVideo:
    def test_scalar_stored_as_float(self, tmp_path):
        lg = FileLogger(tmp_path)
        lg.scalar("loss", 2.5)
        assert lg._scalars["loss"] == 2.5

    def test_video_stored_as_array(self, tmp_path):
        lg = FileLogger(tmp_path)
        vid = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
        lg.video("v", vid)
        assert lg._videos["v"].shape == vid.shape


class TestFileLoggerWrite:
    def test_writes_json_to_file(self, tmp_path):
        lg = FileLogger(tmp_path)
        lg.scalar("loss", 1.5)
        with patch("her_dream.tools.loggers.file_logger.torchvision.io.write_video"):
            lg.write(step=1)
        data = json.loads((tmp_path / "metrics.jsonl").read_text())
        assert data == {"step": 1, "loss": pytest.approx(1.5)}

    def test_appends_multiple_steps(self, tmp_path):
        lg = FileLogger(tmp_path)
        with patch("her_dream.tools.loggers.file_logger.torchvision.io.write_video"):
            lg.scalar("loss", 1.0)
            lg.write(step=1)
            lg.scalar("loss", 0.5)
            lg.write(step=2)
        lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_resets_scalars_after_write(self, tmp_path):
        lg = FileLogger(tmp_path)
        lg.scalar("loss", 1.0)
        with patch("her_dream.tools.loggers.file_logger.torchvision.io.write_video"):
            lg.write(step=1)
        assert lg._scalars == {}

    def test_resets_videos_after_write(self, tmp_path):
        lg = FileLogger(tmp_path)
        lg.video("v", np.zeros((1, 2, 4, 4, 3), dtype=np.uint8))
        with patch("her_dream.tools.loggers.file_logger.torchvision.io.write_video"):
            lg.write(step=1)
        assert lg._videos == {}

    def test_video_bytes_name_decoded(self, tmp_path):
        lg = FileLogger(tmp_path)
        lg.video(b"my_vid", np.zeros((1, 2, 4, 4, 3), dtype=np.uint8))
        with patch("her_dream.tools.loggers.file_logger.torchvision.io.write_video") as mock_wv:
            lg.write(step=1)
        call_path = str(mock_wv.call_args[0][0])
        assert "my_vid" in call_path

    def test_video_float_clipped_to_uint8(self, tmp_path):
        lg = FileLogger(tmp_path)
        vid = np.ones((1, 2, 4, 4, 3), dtype=np.float32) * 0.5
        lg.video("v", vid)
        with patch("her_dream.tools.loggers.file_logger.torchvision.io.write_video") as mock_wv:
            lg.write(step=1)
        passed_tensor = mock_wv.call_args[0][1]
        assert passed_tensor.dtype == torch.uint8

    def test_video_uint8_passed_unchanged(self, tmp_path):
        lg = FileLogger(tmp_path)
        vid = np.full((1, 2, 4, 4, 3), 200, dtype=np.uint8)
        lg.video("v", vid)
        with patch("her_dream.tools.loggers.file_logger.torchvision.io.write_video") as mock_wv:
            lg.write(step=1)
        passed_tensor = mock_wv.call_args[0][1]
        assert passed_tensor.max().item() == 200

    def test_video_slash_in_name_replaced(self, tmp_path):
        lg = FileLogger(tmp_path)
        lg.video("a/b", np.zeros((1, 2, 4, 4, 3), dtype=np.uint8))
        with patch("her_dream.tools.loggers.file_logger.torchvision.io.write_video") as mock_wv:
            lg.write(step=1)
        call_path = str(mock_wv.call_args[0][0])
        assert "a_b" in call_path
        assert "a/b" not in call_path

    def test_custom_filename(self, tmp_path):
        lg = FileLogger(tmp_path, filename="custom.jsonl")
        lg.scalar("x", 1.0)
        with patch("her_dream.tools.loggers.file_logger.torchvision.io.write_video"):
            lg.write(step=1)
        assert (tmp_path / "custom.jsonl").exists()
