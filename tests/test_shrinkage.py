import pytest
import torch
from torch.distributions import MultivariateNormal

from concept_erasure import optimal_linear_shrinkage


@pytest.mark.parametrize(
    "p,n",
    [
        # Test the n < p case
        (32, 16),
        (64, 32),
        (128, 64),
        # And the n > p case
        (4, 64),
        (8, 128),
        (16, 256),
    ],
)
def test_olse_shrinkage(p: int, n: int):
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
    S_olse = optimal_linear_shrinkage(S_hat, n)

    # Check that the Frobenius norm of the difference has decreased
    norm_naive = torch.norm(S_hat - S_true, dim=(-1, -2)).mean()
    norm_olse = torch.norm(S_olse - S_true, dim=(-1, -2)).mean()

    assert norm_olse <= norm_naive
