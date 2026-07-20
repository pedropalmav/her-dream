import pytest
import torch

from her_dream.optim import LaProp


class TestLaPropValidation:
    def _p(self):
        return [torch.nn.Parameter(torch.randn(4))]

    def test_negative_lr_raises(self):
        with pytest.raises(ValueError, match="learning rate"):
            LaProp(self._p(), lr=-1e-4)

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            LaProp(self._p(), eps=-1.0)

    def test_beta0_too_high_raises(self):
        with pytest.raises(ValueError, match="beta parameter at index 0"):
            LaProp(self._p(), betas=(1.0, 0.999))

    def test_beta0_negative_raises(self):
        with pytest.raises(ValueError, match="beta parameter at index 0"):
            LaProp(self._p(), betas=(-0.1, 0.999))

    def test_beta1_too_high_raises(self):
        with pytest.raises(ValueError, match="beta parameter at index 1"):
            LaProp(self._p(), betas=(0.9, 1.0))

    def test_beta1_negative_raises(self):
        with pytest.raises(ValueError, match="beta parameter at index 1"):
            LaProp(self._p(), betas=(0.9, -0.1))

    def test_defaults_stored_in_group(self):
        opt = LaProp(self._p())
        g = opt.param_groups[0]
        assert g["lr"] == 4e-4
        assert g["betas"] == (0.9, 0.999)
        assert g["eps"] == 1e-15
        assert g["weight_decay"] == 0
        assert g["amsgrad"] is False
        assert g["centered"] is False

    def test_steps_before_using_centered(self):
        opt = LaProp(self._p())
        assert opt.steps_before_using_centered == 10


class TestLaPropStep:
    def _make(self, shape=(4,), lr=1e-3, init_val=1.0, **kwargs):
        p = torch.nn.Parameter(torch.full(shape, init_val))
        opt = LaProp([p], lr=lr, **kwargs)
        return p, opt

    def _set_grad(self, p, val):
        p.grad = torch.full_like(p.data, float(val))

    def test_state_initialized_on_first_step(self):
        p, opt = self._make()
        self._set_grad(p, 1.0)
        opt.step()
        state = opt.state[p]
        assert state["step"] == 1
        assert isinstance(state["exp_avg"], torch.Tensor)
        assert isinstance(state["exp_avg_sq"], torch.Tensor)
        assert state["exp_avg"].shape == p.shape
        assert state["exp_avg_sq"].shape == p.shape

    def test_amsgrad_state_not_created_by_default(self):
        p, opt = self._make()
        self._set_grad(p, 1.0)
        opt.step()
        assert "max_exp_avg_sq" not in opt.state[p]

    def test_centered_state_not_created_by_default(self):
        p, opt = self._make()
        self._set_grad(p, 1.0)
        opt.step()
        assert "exp_mean_avg_beta2" not in opt.state[p]

    def test_step_increments_each_call(self):
        p, opt = self._make()
        for i in range(1, 5):
            self._set_grad(p, 1.0)
            opt.step()
            assert opt.state[p]["step"] == i

    def test_params_move_toward_negative_gradient(self):
        p, opt = self._make()
        initial = p.data.clone()
        self._set_grad(p, 1.0)  # positive gradient → p should decrease
        opt.step()
        assert torch.all(p.data < initial)

    def test_params_move_toward_positive_for_negative_gradient(self):
        p, opt = self._make()
        initial = p.data.clone()
        self._set_grad(p, -1.0)  # negative gradient → p should increase
        opt.step()
        assert torch.all(p.data > initial)

    def test_sparse_grad_raises(self):
        p, opt = self._make(shape=(5,))
        indices = torch.tensor([[0, 2]])
        values = torch.tensor([1.0, 1.0])
        p.grad = torch.sparse_coo_tensor(indices, values, (5,))
        with pytest.raises(RuntimeError, match="sparse"):
            opt.step()

    def test_param_without_grad_not_updated(self):
        p, opt = self._make()
        p.grad = None
        initial = p.data.clone()
        opt.step()
        assert torch.equal(p.data, initial)

    def test_weight_decay_shrinks_params(self):
        # Both start at 5.0; weight_decay should multiply p by (1 - wd) after the step
        p_wd = torch.nn.Parameter(torch.full((4,), 5.0))
        p_no = torch.nn.Parameter(torch.full((4,), 5.0))
        opt_wd = LaProp([p_wd], lr=1e-3, weight_decay=0.1)
        opt_no = LaProp([p_no], lr=1e-3, weight_decay=0.0)
        for p in [p_wd, p_no]:
            p.grad = torch.ones(4)
        opt_wd.step()
        opt_no.step()
        assert torch.all(p_wd.data.abs() < p_no.data.abs())

    def test_amsgrad_creates_max_state(self):
        p, opt = self._make(amsgrad=True)
        self._set_grad(p, 1.0)
        opt.step()
        state = opt.state[p]
        assert "max_exp_avg_sq" in state
        assert state["max_exp_avg_sq"].shape == p.shape

    def test_amsgrad_max_is_monotone(self):
        p, opt = self._make(amsgrad=True)
        prev_max = None
        for grad_val in [1.0, 10.0, 5.0, 0.1, 8.0, 3.0]:
            self._set_grad(p, grad_val)
            opt.step()
            cur_max = opt.state[p]["max_exp_avg_sq"].clone()
            if prev_max is not None:
                assert torch.all(cur_max >= prev_max - 1e-8)
            prev_max = cur_max

    def test_centered_creates_mean_state(self):
        p, opt = self._make(centered=True)
        self._set_grad(p, 1.0)
        opt.step()
        state = opt.state[p]
        assert "exp_mean_avg_beta2" in state
        assert state["exp_mean_avg_beta2"].shape == p.shape

    def test_centered_not_applied_before_step_10(self):
        # For steps 1–10, denom = exp_avg_sq regardless of centered flag
        p_c = torch.nn.Parameter(torch.ones(4))
        p_b = torch.nn.Parameter(torch.ones(4))
        opt_c = LaProp([p_c], lr=1e-3, centered=True)
        opt_b = LaProp([p_b], lr=1e-3, centered=False)
        for _ in range(5):
            for p in [p_c, p_b]:
                p.grad = torch.full_like(p.data, 1.0)
            opt_c.step()
            opt_b.step()
        assert torch.allclose(p_c.data, p_b.data, atol=1e-6)

    def test_centered_applied_after_step_10(self):
        # After step 10, centered subtracts mean^2 from denom → updates diverge
        p_c = torch.nn.Parameter(torch.ones(4))
        p_b = torch.nn.Parameter(torch.ones(4))
        opt_c = LaProp([p_c], lr=1e-3, centered=True)
        opt_b = LaProp([p_b], lr=1e-3, centered=False)
        for _ in range(100):
            for p in [p_c, p_b]:
                p.grad = torch.full_like(p.data, 1.0)
            opt_c.step()
            opt_b.step()
        assert not torch.allclose(p_c.data, p_b.data, atol=1e-5)

    def test_lr_zero_no_div_zero(self):
        # lr=0 → bias_correction1 special-cased to 1.0; exp_avg stays 0 → p unchanged
        p, opt = self._make(lr=0.0)
        initial = p.data.clone()
        self._set_grad(p, 1.0)
        opt.step()
        assert torch.equal(p.data, initial)

    def test_exp_avg_lr_accumulates_correctly(self):
        # After N steps: exp_avg_lr_1 = lr * (1 - beta1^N)
        lr, beta1, N = 1e-3, 0.9, 5
        p, opt = self._make(lr=lr, betas=(beta1, 0.999))
        for _ in range(N):
            self._set_grad(p, 1.0)
            opt.step()
        expected = lr * (1 - beta1**N)
        assert abs(opt.state[p]["exp_avg_lr_1"] - expected) < 1e-12

    def test_convergence_on_quadratic(self):
        # Minimize f(x) = x^2 starting from x = 2.0; should reach |x| < 0.1
        p = torch.nn.Parameter(torch.tensor([2.0]))
        opt = LaProp([p], lr=1e-2)
        for _ in range(500):
            opt.zero_grad()
            (p**2).backward()
            opt.step()
        assert p.data.abs().item() < 0.1
