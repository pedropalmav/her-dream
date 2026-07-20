from unittest.mock import MagicMock, patch

from her_dream.tools.benchmark import CudaBenchmark


class TestCudaBenchmark:
    def _mock_cuda(self, elapsed_ms=500.0):
        mock_event_inst = MagicMock()
        mock_event_inst.elapsed_time.return_value = elapsed_ms
        mock_event_cls = MagicMock(return_value=mock_event_inst)
        return mock_event_cls, mock_event_inst

    def test_comment_stored_at_init(self):
        cb = CudaBenchmark("bench")
        assert cb._comment == "bench"

    def test_enter_returns_none(self):
        mock_event_cls, _ = self._mock_cuda()
        with (
            patch("her_dream.tools.benchmark.torch.cuda.Event", mock_event_cls),
            patch("her_dream.tools.benchmark.torch.cuda.synchronize"),
        ):
            cb = CudaBenchmark("bench")
            result = cb.__enter__()
        assert result is None

    def test_context_manager_as_clause_is_none(self):
        mock_event_cls, _ = self._mock_cuda()
        with (
            patch("her_dream.tools.benchmark.torch.cuda.Event", mock_event_cls),
            patch("her_dream.tools.benchmark.torch.cuda.synchronize"),
            CudaBenchmark("bench") as x,
        ):
            pass
        assert x is None

    def test_exit_calls_synchronize(self):
        mock_event_cls, _ = self._mock_cuda()
        with (
            patch("her_dream.tools.benchmark.torch.cuda.Event", mock_event_cls),
            patch("her_dream.tools.benchmark.torch.cuda.synchronize") as mock_sync,
            CudaBenchmark("bench"),
        ):
            pass
        mock_sync.assert_called_once()

    def test_exit_prints_comment_and_seconds(self, capsys):
        mock_event_cls, _ = self._mock_cuda(elapsed_ms=500.0)
        with (
            patch("her_dream.tools.benchmark.torch.cuda.Event", mock_event_cls),
            patch("her_dream.tools.benchmark.torch.cuda.synchronize"),
            CudaBenchmark("my_bench"),
        ):
            pass
        out = capsys.readouterr().out
        assert "my_bench" in out
        assert "0.5" in out  # 500ms / 1000

    def test_timing_divides_elapsed_by_1000(self, capsys):
        mock_event_cls, _ = self._mock_cuda(elapsed_ms=1000.0)
        with (
            patch("her_dream.tools.benchmark.torch.cuda.Event", mock_event_cls),
            patch("her_dream.tools.benchmark.torch.cuda.synchronize"),
            CudaBenchmark("t"),
        ):
            pass
        out = capsys.readouterr().out
        assert "1.0" in out
