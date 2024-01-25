import pytest
import torch

from concept_erasure.psd_sqrt import (
    newton_schulz_sqrt_rsqrt,
    psd_sqrt_rsqrt,
)


@pytest.mark.parametrize("d", [2, 4, 8, 16])
def test_newton_schulz(d: int):
    torch.manual_seed(42)

    # Generate random positive definite matrix A
    A = torch.randn(d, d, device="cuda")
    A = A @ A.T / d

    # Compute the p.s.d. sqrt and pinv sqrt
    gt_sqrt, gt_rsqrt = psd_sqrt_rsqrt(A)
    sqrt, rsqrt = newton_schulz_sqrt_rsqrt(A)

    # Check that the two implementations agree
    tol = torch.finfo(A.dtype).eps ** 0.5
    torch.testing.assert_close(gt_sqrt, sqrt, atol=tol, rtol=0)
    torch.testing.assert_close(gt_rsqrt, rsqrt, atol=tol, rtol=0)

    # Check that the sqrt and pinv sqrt are correct
    I = torch.eye(d, device="cuda")
    torch.testing.assert_close(sqrt @ sqrt, A)
    torch.testing.assert_close(sqrt @ rsqrt, I)
