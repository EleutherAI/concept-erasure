import pytest
import torch
from torch.distributions import Dirichlet

from concept_erasure.optimal_transport import (
    ot_barycenter,
    ot_distance,
    ot_map,
    ot_midpoint,
)
from concept_erasure.psd_sqrt import is_positive_definite


@pytest.mark.parametrize("d", [1, 2, 4, 8, 16, 32])
def test_gaussian_ot_map(d: int):
    torch.manual_seed(42)

    # Generate random positive definite matrices A and B
    A = torch.randn(d, d, dtype=torch.float64)
    A = A @ A.T / d

    B = torch.randn_like(A)
    B = B @ B.T / d

    C = torch.randn_like(A)
    C = C @ C.T / d

    # Compute the optimal transport map
    M = ot_map(A, B)

    # Analytically compute the covariance after transport
    torch.testing.assert_close(M @ A @ M, B)
    assert is_positive_definite(M)

    # Midpoint of two Gaussians, with different weights
    for w1 in [0.25, 0.5, 0.75]:
        w2 = 1 - w1

        mu = ot_midpoint(A, B, w1, w2)
        assert is_positive_definite(mu)

        # Check first order optimality condition
        mu.requires_grad_(True)
        loss = w1 * ot_distance(mu, A).square() + w2 * ot_distance(mu, B).square()
        loss.backward()

        torch.testing.assert_close(mu.grad, torch.zeros_like(mu), rtol=1e-6, atol=2e-6)

    # Barycenter of three Gaussians, with a few random weights
    weight_prior = Dirichlet(A.new_ones(3))
    for weights in weight_prior.sample(torch.Size([3])):
        Ks = torch.stack([A, B, C])
        mu = ot_barycenter(Ks, weights)
        assert is_positive_definite(mu)

        # Check first order optimality condition
        mu.requires_grad_(True)
        loss = ot_distance(mu, Ks).square().mul(weights).sum()
        loss.backward()
        torch.testing.assert_close(mu.grad, torch.zeros_like(mu), rtol=1e-6, atol=2e-6)
