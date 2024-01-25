import torch
from torch import Tensor


def optimal_linear_shrinkage(
    S_n: Tensor, n: int | Tensor, *, inplace: bool = False
) -> Tensor:
    """Optimal linear shrinkage for a sample covariance matrix or batch thereof.

    Given a sample covariance matrix `S_n` of shape (*, p, p) and a sample size `n`,
    this function computes the optimal shrinkage coefficients `alpha` and `beta`, then
    returns the covariance estimate `alpha * S_n + beta * Sigma0`, where `Sigma0` is
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
    assert S_n.shape[-2:] == (p, p)

    trace_S = trace(S_n)

    # Since sigma0 is I * tr(S_n) / p, its squared Frobenius norm is just tr(S_n) ** 2 / p.
    sigma0_norm_sq = trace_S ** 2 / p
    S_norm_sq = S_n.norm(dim=(-2, -1), keepdim=True) ** 2

    prod_trace = sigma0_norm_sq #torch.linalg.diagonal(S_n) # trace(S_n @ sigma0)
    top = trace_S * trace_S.conj() * sigma0_norm_sq / n
    bottom = S_norm_sq * sigma0_norm_sq - prod_trace * prod_trace.conj()

    # Epsilon prevents dividing by zero for the zero matrix. In that case we end up
    # setting alpha = 0, beta = 1, but it doesn't matter since we're shrinking toward
    # tr(0)*I = 0, so it's a no-op.
    eps = torch.finfo(S_n.dtype).eps
    alpha = 1 - (top + eps) / (bottom + eps)
    beta = (1 - alpha) * (prod_trace + eps) / (sigma0_norm_sq + eps)

    ret = S_n.mul_(alpha) if inplace else alpha * S_n
    diag = beta * trace_S / p
    torch.linalg.diagonal(ret).add_(diag.squeeze(-1))
    return ret


def trace(matrices: Tensor) -> Tensor:
    """Version of `torch.trace` that works for batches of matrices."""
    diag = torch.linalg.diagonal(matrices)
    return diag.sum(dim=-1, keepdim=True).unsqueeze(-1)
