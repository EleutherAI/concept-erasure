import torch
from torch import Tensor


def gaussian_shrinkage(S_hat: Tensor, n: int | Tensor) -> Tensor:
    """Applies Rao-Blackwell LW shrinkage to a sample covariance matrix.

    Formula from https://arxiv.org/abs/0907.4698. Derivation assumes that the
    datapoints are Gaussian, but is a consistent estimator under any distribution.

    Args:
        S_hat: Sample covariance matrix of shape (p, p).
        n: Sample size.
    """
    p = S_hat.shape[-1]
    assert n > 1 and S_hat.shape == (p, p)

    trace_S = torch.trace(S_hat)
    trace_S_sq = torch.trace(S_hat**2)
    trace_sq_S = trace_S**2

    top = (n - 2) / n * trace_S_sq + trace_sq_S
    bottom = (n + 2) * (trace_S_sq + trace_sq_S / p)
    rho = torch.clamp(top / bottom, 0, 1)

    eye = torch.eye(p, dtype=S_hat.dtype, device=S_hat.device)
    F_hat = eye * trace_S / p

    return (1 - rho) * S_hat + rho * F_hat
