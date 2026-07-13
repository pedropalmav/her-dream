from unittest.mock import patch

import pytest
import torch

from her_dream.optim import clip_grad_agc_

from .conftest import make_param


class TestClipGradAGC:
    def test_no_op_when_no_grad(self):
        p = torch.nn.Parameter(torch.randn(4))
        clip_grad_agc_([p], clip=0.1, pmin=1e-3)
        assert p.grad is None

    def test_single_tensor_input(self):
        # Passing a bare Tensor (not a list) should be handled by the [parameters] branch
        p = torch.nn.Parameter(torch.ones(4))
        p.grad = torch.full((4,), 100.0)
        original_gnorm = p.grad.norm(p=2).item()
        clip_grad_agc_(p, clip=0.1, pmin=1e-3)
        assert p.grad.norm(p=2).item() < original_gnorm

    def test_no_clipping_when_gnorm_below_upper(self):
        # pnorm = sqrt(2*100) ≈ 14.14, gnorm ≈ 0.00141
        # upper = 0.1 * 14.14 = 1.414, gnorm/upper ≈ 0.001 < 1 → no clipping
        p = torch.nn.Parameter(torch.full((2,), 10.0))
        p.grad = torch.tensor([0.001, 0.001])
        original_grad = p.grad.clone()
        clip_grad_agc_([p], clip=0.1, pmin=1e-3)
        assert torch.allclose(p.grad, original_grad)

    def test_clipping_occurs_when_gnorm_exceeds_upper(self):
        # pnorm = sqrt(2) ≈ 1.414, gnorm = sqrt(2*10000) ≈ 141.4
        # upper = 0.1 * 1.414 = 0.1414, scale = 0.1414/141.4 ≈ 0.001
        p = torch.nn.Parameter(torch.ones(2))
        p.grad = torch.tensor([100.0, 100.0])
        clip_grad_agc_([p], clip=0.1, pmin=1e-3)
        assert p.grad.norm(p=2).item() < 1.0

    def test_scale_formula_correctness(self):
        # pnorm = 2.0, gnorm = 10.0, clip = 0.1, pmin = 0.001
        # upper = 0.1 * max(2.0, 0.001) = 0.2
        # scale = 1 / max(1.0, 10.0 / 0.2) = 1 / 50 = 0.02
        # new grad = 10.0 * 0.02 = 0.2
        p = torch.nn.Parameter(torch.tensor([2.0]))
        p.grad = torch.tensor([10.0])
        clip_grad_agc_([p], clip=0.1, pmin=1e-3, foreach=False)
        assert torch.allclose(p.grad, torch.tensor([0.2]), atol=1e-5)

    def test_pmin_clamps_small_pnorm(self):
        # pnorm ≈ 0 << pmin = 1.0, so upper = clip * pmin = 0.1
        # gnorm = 10.0, scale = 1 / max(1, 10/0.1) = 0.01
        # new grad = 0.1
        p = torch.nn.Parameter(torch.tensor([1e-8]))
        p.grad = torch.tensor([10.0])
        clip_grad_agc_([p], clip=0.1, pmin=1.0, foreach=False)
        assert torch.allclose(p.grad, torch.tensor([0.1]), atol=1e-5)

    def test_foreach_false_same_result_as_auto(self):
        p_auto = torch.nn.Parameter(torch.ones(2))
        p_auto.grad = torch.tensor([50.0, 50.0])
        p_scalar = torch.nn.Parameter(torch.ones(2))
        p_scalar.grad = torch.tensor([50.0, 50.0])
        clip_grad_agc_([p_auto], clip=0.1, pmin=1e-3, foreach=None)
        clip_grad_agc_([p_scalar], clip=0.1, pmin=1e-3, foreach=False)
        assert torch.allclose(p_auto.grad, p_scalar.grad, atol=1e-6)

    def test_foreach_true_clips_correctly(self):
        # pnorm = sqrt(2), gnorm = 50*sqrt(2), upper = 0.1*sqrt(2) ≈ 0.1414
        # scale = 0.1414 / (50*sqrt(2)) ≈ 0.002
        p = torch.nn.Parameter(torch.ones(2))
        p.grad = torch.tensor([50.0, 50.0])
        clip_grad_agc_([p], clip=0.1, pmin=1e-3, foreach=True)
        assert p.grad.norm(p=2).item() < 1.0

    def test_foreach_true_raises_on_unsupported_device(self):
        p = make_param((4,), grad_val=1.0)
        with (
            patch("her_dream.optim.agc._device_has_foreach_support", return_value=False),
            pytest.raises(RuntimeError, match="foreach=True"),
        ):
            clip_grad_agc_([p], clip=0.1, pmin=1e-3, foreach=True)

    def test_mixed_grad_and_no_grad(self):
        p_with = torch.nn.Parameter(torch.ones(4))
        p_with.grad = torch.full((4,), 100.0)
        p_without = torch.nn.Parameter(torch.randn(4))
        assert p_without.grad is None
        original_with = p_with.grad.clone()
        clip_grad_agc_([p_with, p_without], clip=0.1, pmin=1e-3)
        assert p_without.grad is None
        assert not torch.equal(p_with.grad, original_with)

    def test_multiple_params_clipped_independently(self):
        # p1: pnorm ≈ sqrt(2), gnorm ≈ 141.4 → clipped
        p1 = torch.nn.Parameter(torch.ones(2))
        p1.grad = torch.tensor([100.0, 100.0])
        # p2: pnorm ≈ 14.14, gnorm ≈ 0.0014 → NOT clipped
        p2 = torch.nn.Parameter(torch.full((2,), 10.0))
        p2.grad = torch.tensor([0.001, 0.001])
        original_g2 = p2.grad.clone()
        clip_grad_agc_([p1, p2], clip=0.1, pmin=1e-3)
        assert p1.grad.norm(p=2).item() < 10.0
        assert torch.allclose(p2.grad, original_g2)
