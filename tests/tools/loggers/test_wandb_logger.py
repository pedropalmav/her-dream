from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.loggers.wandb_logger import WandbLogger


def _make(mock_wandb, tmp_path, project="test"):
    return WandbLogger(SimpleNamespace(project=project), tmp_path)


class TestWandbLoggerInit:
    def test_calls_wandb_init(self, mock_wandb, tmp_path):
        WandbLogger(SimpleNamespace(project="p"), tmp_path)
        mock_wandb.init.assert_called_once()

    def test_project_name_from_config(self, mock_wandb, tmp_path):
        WandbLogger(SimpleNamespace(project="my-proj"), tmp_path)
        assert mock_wandb.init.call_args[1]["project"] == "my-proj"

    def test_project_defaults_to_her_dream_when_missing(self, mock_wandb, tmp_path):
        WandbLogger(SimpleNamespace(), tmp_path)
        assert mock_wandb.init.call_args[1]["project"] == "her-dream"

    def test_initial_state(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        assert lg._scalars == {}
        assert lg._images == {}
        assert lg._videos == {}
        assert lg._histograms == {}
        assert lg._pending_step is None
        assert lg._pending_payload == {}


class TestWandbLoggerBuffers:
    def test_scalar_stored(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.scalar("loss", 1.5)
        assert lg._scalars["loss"] == 1.5

    def test_image_stored(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        arr = np.zeros((4, 4, 3))
        lg.image("img", arr)
        assert lg._images["img"].shape == arr.shape

    def test_video_stored(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        vid = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
        lg.video("v", vid)
        assert lg._videos["v"].shape == vid.shape

    def test_histogram_stored(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        arr = np.array([1.0, 2.0])
        lg.histogram("h", arr)
        np.testing.assert_array_equal(lg._histograms["h"], arr)


class TestWandbLoggerFlushPending:
    def test_non_empty_payload_calls_wandb_log(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg._pending_step = 5
        lg._pending_payload = {"loss": 1.0}
        lg._flush_pending()
        mock_wandb.log.assert_called_once_with({"loss": 1.0}, step=5, commit=True)

    def test_empty_payload_no_wandb_log(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg._pending_step = 5
        lg._pending_payload = {}
        lg._flush_pending()
        mock_wandb.log.assert_not_called()

    def test_pending_step_none_no_wandb_log(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg._pending_step = None
        lg._pending_payload = {"loss": 1.0}
        lg._flush_pending()
        mock_wandb.log.assert_not_called()

    def test_flush_clears_payload(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg._pending_step = 5
        lg._pending_payload = {"loss": 1.0}
        lg._flush_pending()
        assert lg._pending_payload == {}


class TestWandbLoggerWriteStep:
    def test_first_write_step_no_flush(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.write_step("loss", 1.0, step=5)
        mock_wandb.log.assert_not_called()

    def test_same_step_accumulates(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.write_step("loss", 1.0, step=5)
        lg.write_step("acc", 0.9, step=5)
        assert "loss" in lg._pending_payload
        assert "acc" in lg._pending_payload
        mock_wandb.log.assert_not_called()

    def test_different_step_flushes_first(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.write_step("loss", 1.0, step=5)
        lg.write_step("acc", 0.9, step=10)
        mock_wandb.log.assert_called_once()

    def test_pending_step_updated(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.write_step("loss", 1.0, step=5)
        assert lg._pending_step == 5


class TestWandbLoggerWrite:
    def test_scalar_added_to_pending(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.scalar("loss", 1.5)
        lg.write(step=1)
        assert "loss" in lg._pending_payload

    def test_image_uses_wandb_image(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.image("img", np.zeros((4, 4, 3)))
        lg.write(step=1)
        mock_wandb.Image.assert_called_once()

    def test_histogram_uses_wandb_histogram(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.histogram("h", np.array([1.0, 2.0]))
        lg.write(step=1)
        mock_wandb.Histogram.assert_called_once()

    def test_video_uint8_transposed(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        vid = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
        lg.video("v", vid)
        lg.write(step=1)
        passed_arr = mock_wandb.Video.call_args[0][0]
        assert passed_arr.dtype == np.uint8
        assert passed_arr.shape == (2, 3, 4, 4)  # (T, C, H, W)

    def test_video_float_clipped_to_uint8_and_transposed(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        vid = np.ones((1, 2, 4, 4, 3), dtype=np.float32) * 0.5
        lg.video("v", vid)
        lg.write(step=1)
        passed_arr = mock_wandb.Video.call_args[0][0]
        assert passed_arr.dtype == np.uint8
        assert passed_arr.shape == (2, 3, 4, 4)

    def test_video_bytes_name_decoded(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.video(b"b_name", np.zeros((1, 2, 4, 4, 3), dtype=np.uint8))
        lg.write(step=1)
        assert "b_name" in lg._pending_payload

    def test_all_four_buffers_reset_after_write(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.scalar("s", 1.0)
        lg.image("i", np.zeros((4, 4, 3)))
        lg.video("v", np.zeros((1, 2, 4, 4, 3), dtype=np.uint8))
        lg.histogram("h", np.array([1.0]))
        lg.write(step=1)
        assert lg._scalars == {}
        assert lg._images == {}
        assert lg._videos == {}
        assert lg._histograms == {}

    def test_different_step_triggers_flush_before_write(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.write_step("loss", 1.0, step=5)
        lg.write(step=10)
        mock_wandb.log.assert_called_once()

    def test_same_step_no_flush(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.write_step("loss", 1.0, step=5)
        lg.write(step=5)
        mock_wandb.log.assert_not_called()

    def test_pending_step_updated(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.write(step=7)
        assert lg._pending_step == 7


class TestWandbLoggerLogHydraConfig:
    def test_omegaconf_success_passes_dict_to_run(self, mock_wandb, tmp_path):
        from omegaconf import OmegaConf

        lg = _make(mock_wandb, tmp_path)
        cfg = OmegaConf.create({"lr": 0.001})
        lg.log_hydra_config(cfg)
        call_arg = lg._run.config.update.call_args[0][0]
        assert isinstance(call_arg, dict)

    def test_exception_fallback_to_str(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        with patch.dict("sys.modules", {"omegaconf": None}):
            lg.log_hydra_config({"key": "val"})
        call_arg = lg._run.config.update.call_args[0][0]
        assert isinstance(call_arg, str)


class TestWandbLoggerClose:
    def test_close_flushes_pending(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg._pending_step = 1
        lg._pending_payload = {"loss": 0.5}
        lg.close()
        mock_wandb.log.assert_called_once()

    def test_close_calls_wandb_finish(self, mock_wandb, tmp_path):
        lg = _make(mock_wandb, tmp_path)
        lg.close()
        mock_wandb.finish.assert_called_once()
