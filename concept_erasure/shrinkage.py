import torch
from torch import Tensor


def oracle_shrinkage(S_hat: Tensor, n: int | Tensor) -> Tensor:
    """Oracle-approximating shrinkage for a sample covariance matrix or batch thereof.

    Formula from https://arxiv.org/abs/0907.4698. Derivation assumes that the
    datapoints are Gaussian, but is a consistent estimator under any distribution.

    Args:
        S_hat: Sample covariance matrices of shape (*, p, p).
        n: Sample size.
    """
    p = S_hat.shape[-1]
    assert n > 1 and S_hat.shape[-2:] == (p, p)

    trace_S = trace(S_hat)
    trace_S_sq = trace(S_hat**2)
    trace_sq_S = trace_S**2

    phi_hat = (trace_S_sq + trace_sq_S / p) / (trace_S_sq + trace_sq_S)
    rho_inv = (n + 1 - 2 / p) * phi_hat
    rho = torch.clamp(1 / rho_inv, 0, 1)

    eye = torch.eye(p, dtype=S_hat.dtype, device=S_hat.device).expand_as(S_hat)
    F_hat = eye * trace_S / p

    return (1 - rho) * S_hat + rho * F_hat


def trace(matrices: Tensor) -> Tensor:
    """Version of `torch.trace` that works for batches of matrices."""
    diag = torch.linalg.diagonal(matrices)
    return diag.sum(dim=-1, keepdim=True).unsqueeze(-1)
