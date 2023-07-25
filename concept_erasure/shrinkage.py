import torch
from torch import Tensor


def optimal_linear_shrinkage(S_n: Tensor, n: int | Tensor) -> Tensor:
    """Optimal linear shrinkage for a sample covariance matrix or batch thereof.

    Given a sample covariance matrix `S_n` of shape (*, p, p) and a sample size `n`,
    this function computes the optimal shrinkage coefficients `alpha` and `beta`, then
    returns the covariance estimate `alpha * S_n + beta * Sigma0`, where ``Sigma0` is
    an isotropic covariance matrix with the same trace as `S_n`.

    The formula is distribution-free and asymptotically optimal in the Frobenius norm
    among all linear shrinkage estimators as the dimensionality `p` and sample size `n`
    jointly tend to infinity, with the ratio `p / n` converging to a finite positive
    constant `c`. The derivation is based on Random Matrix Theory and assumes that the
    underlying distribution has finite moments up to 4 + eps, for some eps > 0.

    See "On the Strong Convergence of the Optimal Linear Shrinkage Estimator for Large
    Dimensional Covariance Matrix" <https://arxiv.org/abs/1308.2608> for details.

    Args:
        S_n: Sample covariance matrices of shape (*, p, p).
        n: Sample size.
    """
    p = S_n.shape[-1]
    assert n > 1 and S_n.shape[-2:] == (p, p)

    # TODO: Make this configurable, try using diag(S_n) or something
    eye = torch.eye(p, dtype=S_n.dtype, device=S_n.device).expand_as(S_n)
    trace_S = trace(S_n)
    sigma0 = eye * trace_S / p

    sigma0_norm_sq = sigma0.pow(2).sum(dim=(-2, -1), keepdim=True)
    S_norm_sq = S_n.pow(2).sum(dim=(-2, -1), keepdim=True)

    prod_trace = trace(S_n @ sigma0)
    top = trace_S.pow(2) * sigma0_norm_sq / n
    bottom = S_norm_sq * sigma0_norm_sq - prod_trace**2

    alpha = 1 - top / bottom
    beta = (1 - alpha) * prod_trace / sigma0_norm_sq

    return alpha * S_n + beta * sigma0


def trace(matrices: Tensor) -> Tensor:
    """Version of `torch.trace` that works for batches of matrices."""
    diag = torch.linalg.diagonal(matrices)
    return diag.sum(dim=-1, keepdim=True).unsqueeze(-1)
