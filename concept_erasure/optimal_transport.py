import torch
from torch import Tensor

from .shrinkage import trace


def is_positive_definite(A: Tensor) -> Tensor:
    """Efficiently check if `A` is p.d. by attempting Cholesky decomposition."""
    return torch.linalg.cholesky_ex(A).info.eq(0)


@torch.jit.script
def psd_sqrt(A: Tensor) -> Tensor:
    """Compute the unique p.s.d. square root of a positive semidefinite matrix."""
    L, U = torch.linalg.eigh(A)
    L = L[..., None, :].clamp_min(0.0)
    return U * L.sqrt() @ U.mT


@torch.jit.script
def psd_sqrt_rsqrt(A: Tensor) -> tuple[Tensor, Tensor]:
    """Efficiently compute both the p.s.d. sqrt & inverse sqrt of p.s.d. matrix `A`."""
    L, U = torch.linalg.eigh(A)
    L = L[..., None, :].clamp_min(0.0)
    return U * L.sqrt() @ U.mT, U * L.rsqrt() @ U.mT


def ot_barycenter(
    Ks: Tensor, weights: Tensor | None = None, *, max_iter: int = 100
) -> Tensor:
    """Fixed-point iteration for the 2-Wasserstein barycenter of a set of Gaussians.

    Algorithm derived in "A fixed-point approach to barycenters in Wasserstein space"
    by Álvarez-Esteban et al. (2016) <https://arxiv.org/abs/1511.05355>.

    Args:
        Ks: `[n, d, d]` batch of covariance matrices, one for each centered Gaussian
            in the set whose barycenter we want to compute.
        weights: `[n]` batch of weights for each Gaussian.

    Returns:
        Covariance matrix of the barycenter.
    """
    n = len(Ks)
    assert n > 1, "Need at least two Gaussians to compute a barycenter"

    # Uniform weights by default
    if weights is None:
        weights = Ks.new_ones(n) / n
    else:
        assert len(weights) == n, "Need one weight per Gaussian"
        weights = weights / weights.sum()

    # Bookkeeping variables
    loss = torch.inf
    tol = torch.finfo(Ks.dtype).eps
    weights = weights.view(-1, 1, 1)  # Broadcastable to Ks

    # Initialize with arithmetic mean of covariance matrices
    mu = Ks.mul(weights).sum(dim=0)
    trace_avg = mu.trace()

    # Begin Álvarez-Esteban et al. fixed-point iteration
    for _ in range(max_iter):
        sqrt_mu, rsqrt_mu = psd_sqrt_rsqrt(mu)
        inner = psd_sqrt(sqrt_mu @ Ks @ sqrt_mu)

        # Equation 15 from Álvarez-Esteban et al. (2016)
        new_loss = mu.trace() + trace_avg - 2 * inner.mul(weights).sum(dim=0).trace()

        # Break if the loss is not decreasing
        if loss - new_loss < tol:
            break
        else:
            loss = new_loss

        # Equation 7 from Álvarez-Esteban et al. (2016)
        T = torch.sum(weights * rsqrt_mu @ inner @ rsqrt_mu, dim=0)
        mu = T @ mu @ T.mT

    return mu


def ot_distance(K1: Tensor, K2: Tensor) -> Tensor:
    """2-Wasserstein distance between N(0, K1) and N(0, K2)."""
    sqrt_K1 = psd_sqrt(K1)
    inner = psd_sqrt(sqrt_K1 @ K2 @ sqrt_K1)

    # Compute the 2-Wasserstein distance
    dist = torch.sqrt(trace(K1) + trace(K2) - 2 * trace(inner))
    return dist.squeeze(-1).squeeze(-1)


def ot_map(K1: Tensor, K2: Tensor) -> Tensor:
    """Optimal transport map from N(0, K1) to N(0, K2) in matrix form.

    Args:
        K1: Covariance matrix of the first Gaussian.
        K2: Covariance matrix of the second Gaussian.

    Returns:
        Unique p.s.d. matrix A such that N(0, A @ K1 @ A.T) = N(0, K2).
    """
    sqrt_K1, rsqrt_K1 = psd_sqrt_rsqrt(K1)
    return rsqrt_K1 @ psd_sqrt(sqrt_K1 @ K2 @ sqrt_K1) @ rsqrt_K1


def ot_midpoint(K1: Tensor, K2: Tensor, w1: float = 0.5, w2: float = 0.5) -> Tensor:
    """Covariance matrix of the 2-Wasserstein barycenter of N(0, K1) and N(0, K2).

    The barycenter of a set of distributions S is the unique distribution mu which
    minimizes the mean squared Wasserstein distance from each distribution in S to mu.

    Derived in "On Gaussian Wasserstein Barycenters" (Wessel Bruinsma & Gabriel Arpino)
    <https://gabrielarpino.github.io/files/wasserstein.pdf>.

    Args:
        K1: Covariance matrix of the first Gaussian.
        K2: Covariance matrix of the second Gaussian.
        w1: Weight of the first Gaussian.
        w2: Weight of the second Gaussian.

    Returns:
        Covariance matrix of the barycenter.
    """
    sqrt_K1, rsqrt_K1 = psd_sqrt_rsqrt(K1)
    product = sqrt_K1 @ psd_sqrt(sqrt_K1 @ K2 @ sqrt_K1) @ rsqrt_K1

    return w1 * w1 * K1 + w2 * w2 * K2 + w1 * w2 * (product + product.T)
