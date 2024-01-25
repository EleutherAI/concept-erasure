import torch
from torch import Tensor


def is_positive_definite(A: Tensor) -> Tensor:
    """Efficiently check if `A` is p.d. by attempting Cholesky decomposition."""
    return torch.linalg.cholesky_ex(A).info.eq(0)


@torch.jit.script
def psd_sqrt(A: Tensor) -> Tensor:
    """Compute the unique p.s.d. square root of a positive semidefinite matrix."""
    L, U = torch.linalg.eigh(A)
    L = L[..., None, :].clamp_min(0.0)
    return U * L.sqrt() @ U.mH


def psd_sqrt_rsqrt(A: Tensor) -> tuple[Tensor, Tensor]:
    """Efficiently compute both the p.s.d. sqrt & pinv sqrt of p.s.d. matrix `A`."""
    L, U = torch.linalg.eigh(A)
    L = L[..., None, :].clamp_min(0.0)

    # Square root is easy
    sqrt = U * L.sqrt() @ U.mH

    # We actually compute the pseudo-inverse here for numerical stability.
    # Use the same heuristic as `torch.linalg.pinv` to determine the tolerance.
    thresh = L[..., None, -1] * A.shape[-1] * torch.finfo(A.dtype).eps
    rsqrt = U * torch.where(L > thresh, L.rsqrt(), 0.0) @ U.mH

    return sqrt, rsqrt


def newton_schulz_sqrt_rsqrt(A: Tensor, max_iter: int = 100, rsqrt_iter: int = 6):
    """Compute the p.s.d sqrt & pinv sqrt of p.s.d. matrix `A` using Newton-Schulz.

    Args:
        A: A positive semidefinite matrix.
        max_iter: The maximum number of Newton-Schulz iterations to perform.
        rsqrt_iter: The number of Razavi iterations for error reduction for the rsqrt.
    """
    *_, d, d2 = A.shape
    assert d == d2, "A must be a square matrix"

    # The error floor is the *square root* of machine epsilon because numerical errors
    # get squared when reconstructing A from sqrt A.
    tol = torch.finfo(A.dtype).eps ** 0.5

    # Pre-normalize A to help with convergence
    scale = torch.norm(A, p="fro", keepdim=True)
    Y = A_scaled = A / scale
    Z = I = torch.eye(len(A), device=A.device, dtype=A.dtype)

    # Perform the Newton-Schulz iterations
    for _ in range(max_iter):
        # How close is Y to being a sqrt of A?
        err = torch.norm(A_scaled - Y @ Y, p="fro").max()
        if err < tol:
            break

        T = 3 * I - Z @ Y
        Y = 0.5 * Y @ T
        Z = 0.5 * T @ Z

    # Post-compensate the resultant approximation
    sqrt_norm = scale.sqrt()
    Y *= sqrt_norm
    Z /= sqrt_norm

    # The error for the inverse sqrt tends to be ~300x larger than that of the sqrt,
    # which is unacceptable in many cases. Improve the approximation using the fixed
    # point iteration from Razavi et al. (2014), used by Nyströmformer.
    for _ in range(rsqrt_iter):
        prod = Y @ Z
        Z = 0.25 * Z @ (13 * I - prod @ (15 * I - prod @ ((7 * I) - prod)))

    return Y, Z
