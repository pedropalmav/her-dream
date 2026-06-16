import pytest
import torch
from torch import nn

from networks import BlockLinear

from .conftest import ACT_DIM, BLOCKS, DETER, DYNLAYERS, B, K, S, make_deter


class TestDeterInit:
    @pytest.fixture
    def model(self):
        return make_deter()

    def test_input_projections_are_sequential(self, model):
        assert isinstance(model._dyn_in0, nn.Sequential)
        assert isinstance(model._dyn_in1, nn.Sequential)
        assert isinstance(model._dyn_in2, nn.Sequential)

    def test_gru_output_dim(self, model):
        assert isinstance(model._dyn_gru, BlockLinear)
        # BlockLinear bias shape encodes output channels
        assert model._dyn_gru.bias.shape == (3 * DETER,)

    def test_hid_module_count(self, model):
        # Each dynlayer adds BlockLinear + RMSNorm + activation
        assert len(list(model._dyn_hid.children())) == 3 * DYNLAYERS

    def test_dynlayers_zero_empty_hid(self):
        model = make_deter(dynlayers=0)
        assert len(list(model._dyn_hid.children())) == 0

    def test_blocks_stored(self, model):
        assert model.blocks == BLOCKS


class TestDeterForward:
    @pytest.fixture
    def model(self):
        return make_deter()

    def test_output_shape(self, model, stoch_input, deter_input, action):
        out = model(stoch_input, deter_input, action)
        assert out.shape == (B, DETER)

    def test_output_shape_large_batch(self, model):
        stoch = torch.randn(8, S, K)
        deter = torch.randn(8, DETER)
        action = torch.randn(8, ACT_DIM)
        assert model(stoch, deter, action).shape == (8, DETER)

    def test_gradient_flows(self, model, stoch_input, deter_input, action):
        stoch_input.requires_grad_(True)
        out = model(stoch_input, deter_input, action)
        out.sum().backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_dynlayers_zero_forward_shape(self):
        model = make_deter(dynlayers=0)
        out = model(torch.randn(B, S, K), torch.randn(B, DETER), torch.randn(B, ACT_DIM))
        assert out.shape == (B, DETER)


class TestDeterActionNorm:
    @pytest.fixture
    def model(self):
        m = make_deter()
        for module in m.modules():
            if isinstance(module, BlockLinear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        return m

    def test_large_action_finite(self, model):
        # Large action magnitudes should not produce NaNs/Infs
        action = torch.full((B, ACT_DIM), 10.0)
        out = model(torch.randn(B, S, K), torch.randn(B, DETER), action)
        assert torch.isfinite(out).all()

    def test_small_action_finite(self, model):
        # Small action magnitudes should not produce NaNs/Infs
        action = torch.full((B, ACT_DIM), 0.5)
        out = model(torch.randn(B, S, K), torch.randn(B, DETER), action)
        assert torch.isfinite(out).all()
