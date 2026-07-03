import os
from unittest.mock import patch

import torch

from tools.training import Every, Once, enable_deterministic_run, set_seed_everywhere


class TestEvery:
    def test_zero_every_always_returns_zero(self):
        e = Every(0)
        assert e(5) == 0
        assert e(100) == 0

    def test_none_every_returns_zero(self):
        e = Every(None)
        assert e(5) == 0

    def test_false_every_returns_zero(self):
        e = Every(False)
        assert e(5) == 0

    def test_first_call_returns_one(self):
        e = Every(10)
        assert e(0) == 1

    def test_first_call_sets_last(self):
        e = Every(10)
        e(7)
        assert e._last == 7

    def test_within_interval_returns_zero(self):
        e = Every(10)
        e(0)
        assert e(5) == 0

    def test_at_interval_boundary_returns_one(self):
        e = Every(10)
        e(0)
        assert e(10) == 1

    def test_multiple_intervals_returns_count(self):
        e = Every(10)
        e(0)
        assert e(25) == 2  # int((25-0)/10) = 2

    def test_state_advances_after_multiple_intervals(self):
        e = Every(10)
        e(0)  # _last=0
        e(25)  # _last=20
        assert e(30) == 1  # int((30-20)/10) = 1

    def test_count_is_zero_when_just_under_interval(self):
        e = Every(10)
        e(0)
        assert e(9) == 0


class TestOnce:
    def test_first_call_returns_true(self):
        o = Once()
        assert o() is True

    def test_second_call_returns_false(self):
        o = Once()
        o()
        assert o() is False

    def test_all_subsequent_calls_return_false(self):
        o = Once()
        o()
        for _ in range(5):
            assert o() is False


class TestSetSeedEverywhere:
    def test_sets_torch_seed(self):
        with patch("tools.training.torch.manual_seed") as m:
            set_seed_everywhere(42)
        m.assert_called_once_with(42)

    def test_sets_numpy_seed(self):
        with patch("tools.training.np.random.seed") as m:
            set_seed_everywhere(42)
        m.assert_called_once_with(42)

    def test_sets_random_seed(self):
        with patch("tools.training.random.seed") as m:
            set_seed_everywhere(42)
        m.assert_called_once_with(42)

    def test_sets_cuda_seed_when_available(self):
        # Patch manual_seed too: torch.manual_seed internally calls manual_seed_all in PyTorch 2.x,
        # which would cause a double-call on the mock.
        with (
            patch("tools.training.torch.manual_seed"),
            patch("tools.training.torch.cuda.is_available", return_value=True),
            patch("tools.training.torch.cuda.manual_seed_all") as m,
        ):
            set_seed_everywhere(42)
        m.assert_called_once_with(42)

    def test_no_cuda_seed_when_unavailable(self):
        with (
            patch("tools.training.torch.manual_seed"),
            patch("tools.training.torch.cuda.is_available", return_value=False),
            patch("tools.training.torch.cuda.manual_seed_all") as m,
        ):
            set_seed_everywhere(42)
        m.assert_not_called()

    def test_produces_deterministic_output(self):
        set_seed_everywhere(0)
        a = torch.randn(3)
        set_seed_everywhere(0)
        b = torch.randn(3)
        assert torch.equal(a, b)


class TestEnableDeterministicRun:
    def test_sets_cublas_env_var(self, monkeypatch):
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        with patch("torch.use_deterministic_algorithms"):
            enable_deterministic_run()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    def test_disables_cudnn_benchmark(self):
        original = torch.backends.cudnn.benchmark
        try:
            with patch("torch.use_deterministic_algorithms"):
                enable_deterministic_run()
            assert torch.backends.cudnn.benchmark is False
        finally:
            torch.backends.cudnn.benchmark = original

    def test_calls_use_deterministic_algorithms(self):
        with patch("torch.use_deterministic_algorithms") as mock_det:
            enable_deterministic_run()
        mock_det.assert_called_once_with(True)
