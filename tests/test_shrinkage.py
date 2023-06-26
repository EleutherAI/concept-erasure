import pytest
import torch
from torch.distributions import MultivariateNormal

from concept_erasure import oracle_shrinkage


@pytest.mark.parametrize(
    "p,n",
    [
        # Test the n < p case
        (32, 25),
        (64, 50),
        (128, 100),
        # And the n > p case
        (32, 64),
        (64, 128),
        (128, 256),
    ],
)
def test_oracle_shrinkage(p: int, n: int):
    torch.manual_seed(42)

    # Number of matrices
    N = 1000

    # Generate a random covariance matrix
    A = torch.randn(N, p, p)
    S_true = A @ A.mT / p
    torch.linalg.diagonal(S_true).add_(1e-3)

    # Generate data with this covariance
    mean = torch.zeros(N, p)
    dist = MultivariateNormal(mean, S_true)
    X = dist.sample([n]).movedim(1, 0)
    assert X.shape == (N, n, p)

    # Compute the sample covariance matrix
    X_centered = X - X.mean(dim=0, keepdim=True)
    S_hat = (X_centered.mT @ X_centered) / n

    # Apply shrinkage
    S_shrunk = oracle_shrinkage(S_hat, n)

    # Check that the Frobenius norm of the difference has decreased
    norm_before = torch.norm(S_hat - S_true, dim=(-1, -2)).mean()
    norm_after = torch.norm(S_shrunk - S_true, dim=(-1, -2)).mean()

    assert norm_after <= norm_before