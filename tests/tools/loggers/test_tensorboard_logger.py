from unittest.mock import MagicMock, patch

import numpy as np

from her_dream.tools.loggers.tensorboard_logger import TensorboardLogger


class TestTensorboardLoggerBuffer:
    def test_scalar_stored(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        lg.scalar("loss", 1.5)
        assert lg._scalars["loss"] == 1.5

    def test_image_stored(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        arr = np.zeros((3, 4, 4))
        lg.image("img", arr)
        np.testing.assert_array_equal(lg._images["img"], arr)

    def test_video_stored(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        vid = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
        lg.video("v", vid)
        assert lg._videos["v"].shape == vid.shape

    def test_histogram_stored(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        arr = np.array([1.0, 2.0])
        lg.histogram("h", arr)
        np.testing.assert_array_equal(lg._histograms["h"], arr)


class TestTensorboardLoggerWriteStep:
    def test_with_slash_uses_name_as_tag(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        lg.write_step("train/acc", 0.9, step=5)
        mock_summary_writer_tb.return_value.add_scalar.assert_called_once_with("train/acc", 0.9, 5)

    def test_without_slash_prefixes_scalars(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        lg.write_step("acc", 0.9, step=5)
        mock_summary_writer_tb.return_value.add_scalar.assert_called_once_with("scalars/acc", 0.9, 5)


class TestTensorboardLoggerWrite:
    def test_scalar_with_slash_uses_name(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        lg.scalar("train/loss", 1.0)
        lg.write(step=1)
        mock_summary_writer_tb.return_value.add_scalar.assert_any_call("train/loss", 1.0, 1)

    def test_scalar_without_slash_prefixed(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        lg.scalar("loss", 1.0)
        lg.write(step=1)
        mock_summary_writer_tb.return_value.add_scalar.assert_any_call("scalars/loss", 1.0, 1)

    def test_image_written_to_tb(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        lg.image("img", np.zeros((3, 4, 4)))
        lg.write(step=1)
        mock_summary_writer_tb.return_value.add_image.assert_called_once()

    def test_histogram_written_to_tb(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        lg.histogram("h", np.array([1.0, 2.0]))
        lg.write(step=1)
        mock_summary_writer_tb.return_value.add_histogram.assert_called_once()

    def test_video_uint8_written(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        vid = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
        lg.video("v", vid)
        lg.write(step=1)
        assert mock_summary_writer_tb.return_value.add_video.called

    def test_video_float_clipped(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        vid = np.ones((1, 2, 4, 4, 3), dtype=np.float32) * 0.5
        lg.video("v", vid)
        lg.write(step=1)
        assert mock_summary_writer_tb.return_value.add_video.called

    def test_video_bytes_name_decoded(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        vid = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)
        lg.video(b"my_vid", vid)
        lg.write(step=1)
        call_name = mock_summary_writer_tb.return_value.add_video.call_args[0][0]
        assert call_name == "my_vid"

    def test_flush_called(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        lg.write(step=1)
        mock_summary_writer_tb.return_value.flush.assert_called()

    def test_all_four_buffers_reset_after_write(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        lg.scalar("s", 1.0)
        lg.image("i", np.zeros((3, 4, 4)))
        lg.video("v", np.zeros((1, 2, 4, 4, 3), dtype=np.uint8))
        lg.histogram("h", np.array([1.0]))
        lg.write(step=1)
        assert lg._scalars == {}
        assert lg._images == {}
        assert lg._videos == {}
        assert lg._histograms == {}  # TensorboardLogger DOES reset histograms (unlike Logger)


class TestTensorboardLoggerLogHydraConfig:
    def test_import_error_fallback_to_str(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        with patch.dict("sys.modules", {"omegaconf": None}):
            lg.log_hydra_config({"key": "value"})
        call_arg = mock_summary_writer_tb.return_value.add_text.call_args[0][1]
        assert "key" in call_arg

    def test_yaml_written_to_add_text(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        mock_oc = MagicMock()
        mock_oc.OmegaConf.to_yaml.return_value = "lr: 0.001\n"
        mock_oc.OmegaConf.to_container.return_value = {"lr": 0.001}
        with patch.dict("sys.modules", {"omegaconf": mock_oc}):
            lg.log_hydra_config(MagicMock())
        call_arg = mock_summary_writer_tb.return_value.add_text.call_args[0][1]
        assert "lr: 0.001" in call_arg

    def test_log_hparams_false_skips_add_hparams(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        mock_oc = MagicMock()
        mock_oc.OmegaConf.to_yaml.return_value = ""
        mock_oc.OmegaConf.to_container.return_value = {"a": 1}
        with patch.dict("sys.modules", {"omegaconf": mock_oc}):
            lg.log_hydra_config(MagicMock(), log_hparams=False)
        mock_summary_writer_tb.return_value.add_hparams.assert_not_called()

    def test_log_hparams_true_container_none_skips_add_hparams(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        mock_oc = MagicMock()
        mock_oc.OmegaConf.to_yaml.return_value = ""
        mock_oc.OmegaConf.to_container.side_effect = RuntimeError("fail")
        with patch.dict("sys.modules", {"omegaconf": mock_oc}):
            lg.log_hydra_config(MagicMock(), log_hparams=True)
        mock_summary_writer_tb.return_value.add_hparams.assert_not_called()

    def test_log_hparams_true_calls_add_hparams(self, tmp_path, mock_summary_writer_tb):
        lg = TensorboardLogger(tmp_path)
        mock_oc = MagicMock()
        mock_oc.OmegaConf.to_yaml.return_value = ""
        mock_oc.OmegaConf.to_container.return_value = {"lr": 0.001}
        with patch.dict("sys.modules", {"omegaconf": mock_oc}):
            lg.log_hydra_config(MagicMock(), log_hparams=True)
        mock_summary_writer_tb.return_value.add_hparams.assert_called_once()

    def _call_with_container(self, tmp_path, mock_summary_writer_tb, container):
        lg = TensorboardLogger(tmp_path)
        mock_oc = MagicMock()
        mock_oc.OmegaConf.to_yaml.return_value = ""
        mock_oc.OmegaConf.to_container.return_value = container
        with patch.dict("sys.modules", {"omegaconf": mock_oc}):
            lg.log_hydra_config(MagicMock(), log_hparams=True)
        return mock_summary_writer_tb.return_value.add_hparams.call_args[0][0]

    def test_flatten_dict_recurses(self, tmp_path, mock_summary_writer_tb):
        flat = self._call_with_container(tmp_path, mock_summary_writer_tb, {"nested": {"lr": 0.001}})
        assert "nested.lr" in flat

    def test_flatten_list_str_coerced(self, tmp_path, mock_summary_writer_tb):
        flat = self._call_with_container(tmp_path, mock_summary_writer_tb, {"tags": [1, 2]})
        assert flat["tags"] == str([1, 2])

    def test_flatten_tuple_str_coerced(self, tmp_path, mock_summary_writer_tb):
        flat = self._call_with_container(tmp_path, mock_summary_writer_tb, {"pair": (1, 2)})
        assert flat["pair"] == str((1, 2))

    def test_flatten_none_becomes_null(self, tmp_path, mock_summary_writer_tb):
        flat = self._call_with_container(tmp_path, mock_summary_writer_tb, {"x": None})
        assert flat["x"] == "null"

    def test_flatten_other_type_str_coerced(self, tmp_path, mock_summary_writer_tb):
        class _Custom:
            def __str__(self):
                return "custom"

        flat = self._call_with_container(tmp_path, mock_summary_writer_tb, {"obj": _Custom()})
        assert flat["obj"] == "custom"

    def test_flatten_int_stored_directly(self, tmp_path, mock_summary_writer_tb):
        flat = self._call_with_container(tmp_path, mock_summary_writer_tb, {"n": 42})
        assert flat["n"] == 42

    def test_flatten_bool_stored_directly(self, tmp_path, mock_summary_writer_tb):
        flat = self._call_with_container(tmp_path, mock_summary_writer_tb, {"flag": True})
        assert flat["flag"] is True

    def test_flatten_str_stored_directly(self, tmp_path, mock_summary_writer_tb):
        flat = self._call_with_container(tmp_path, mock_summary_writer_tb, {"name": "foo"})
        assert flat["name"] == "foo"
