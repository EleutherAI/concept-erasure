import pytest
import torch

from concept_erasure.psd_sqrt import (
    newton_schulz_sqrt_rsqrt,
    psd_sqrt,
)


@pytest.mark.parametrize("d", [2, 4, 8, 16, 32, 64, 128, 256, 512])
def test_newton_schulz(d: int):
    torch.manual_seed(42)

    # Generate random positive definite matrix A
    A = torch.randn(d, d, dtype=torch.float64)
    A = A @ A.T / d

    # Compute the p.s.d. sqrt and pinv sqrt
    sqrt, rsqrt = newton_schulz_sqrt_rsqrt(A)
    sqrt, rsqrt = sqrt.float(), rsqrt.float()

    # Check that the two implementations agree
    torch.testing.assert_close(psd_sqrt(A).float(), sqrt)

    # Check that the sqrt and pinv sqrt are correct
    torch.testing.assert_close(sqrt @ sqrt, A.float())
    torch.testing.assert_close(sqrt @ rsqrt, torch.eye(d))
