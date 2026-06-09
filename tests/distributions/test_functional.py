import math

import pytest
import torch

from distributions.functional import symexp, symlog

from .conftest import B, K


class TestSymlog:
    def test_zero(self):
        x = torch.tensor(0.0)
        assert symlog(x).item() == pytest.approx(0.0)

    def test_positive_known_value(self):
        # symlog(e - 1) = sign(e-1) * log1p(|e-1|) = log1p(e-1) = log(e) = 1
        x = torch.tensor(math.e - 1.0)
        assert symlog(x).item() == pytest.approx(1.0, abs=1e-6)

    def test_sign_preservation(self):
        pos = torch.tensor(3.0)
        neg = torch.tensor(-3.0)
        assert symlog(pos).item() > 0
        assert symlog(neg).item() < 0

    def test_sign_symmetry(self):
        x = torch.tensor(5.0)
        assert symlog(x).item() == pytest.approx(-symlog(-x).item(), abs=1e-6)

    def test_shape_preserved(self):
        x = torch.randn(B, K)
        assert symlog(x).shape == x.shape

    def test_large_values_no_inf(self):
        x = torch.tensor(1e6)
        out = symlog(x)
        assert not torch.isinf(out)
        assert not torch.isnan(out)

    def test_inverse_symexp_symlog(self):
        x = torch.randn(B, K) * 10
        assert torch.allclose(symexp(symlog(x)), x, atol=1e-5)

    def test_output_dtype_float32(self):
        x = torch.tensor(1.0, dtype=torch.float32)
        assert symlog(x).dtype == torch.float32


class TestSymexp:
    def test_zero(self):
        x = torch.tensor(0.0)
        assert symexp(x).item() == pytest.approx(0.0)

    def test_positive_known_value(self):
        # symexp(1) = sign(1) * expm1(|1|) = expm1(1) = e - 1
        x = torch.tensor(1.0)
        assert symexp(x).item() == pytest.approx(math.e - 1.0, abs=1e-6)

    def test_sign_preservation(self):
        pos = torch.tensor(2.0)
        neg = torch.tensor(-2.0)
        assert symexp(pos).item() > 0
        assert symexp(neg).item() < 0

    def test_sign_symmetry(self):
        x = torch.tensor(4.0)
        assert symexp(x).item() == pytest.approx(-symexp(-x).item(), abs=1e-6)

    def test_shape_preserved(self):
        x = torch.randn(B, K)
        assert symexp(x).shape == x.shape

    def test_large_values_no_inf(self):
        # symexp of moderate values should be finite
        x = torch.tensor(20.0)
        out = symexp(x)
        assert not torch.isinf(out)
        assert not torch.isnan(out)

    def test_inverse_symlog_symexp(self):
        x = torch.randn(B, K) * 5
        assert torch.allclose(symlog(symexp(x)), x, atol=1e-5)

    def test_output_dtype_float32(self):
        x = torch.tensor(1.0, dtype=torch.float32)
        assert symexp(x).dtype == torch.float32
