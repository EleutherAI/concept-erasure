import torch
from torch import Tensor


def cdf(x: float | Tensor, q: Tensor) -> Tensor:
    """Evaluate empirical CDF defined by quantiles `q` on `x`.

    Args:
        x: `[...]` Scalar or tensor of data points of arbitrary shape.
        q: `[..., num_quantiles]` batch of quantiles. Must be sorted, and
            should broadcast with `x` except for the last dimension.

    Returns:
        `[...]` Empirical CDF evaluated for each element of `x`.
    """
    n = q.shape[-1]
    assert n > 2, "Must have at least two quantiles to interpolate."

    # Approach used by SciPy interp1d with kind='previous'
    # Shift x toward +inf by epsilon to appropriately handle ties
    x = torch.nextafter(torch.as_tensor(x), q.new_tensor(torch.inf))
    return torch.searchsorted(q, x, out_int32=True) / n


def icdf(p: Tensor, q: Tensor) -> Tensor:
    """(Pseudo-)inverse of the ECDF defined by quantiles `q`.

    Returns the *smallest* `x` such that the ECDF of `x` is greater than or
    equal to `p`. If `interpolate` is true, then the ECDF is linearly

    Args:
        x: `[...]` Tensor of data points of arbitrary shape.
        q: `[..., num_quantiles]` batch of quantiles. Must be sorted, and
            should broadcast with `x` except for the last dimension.

    Returns:
        `[...]` Empirical CDF evaluated for each element of `x`.
    """
    n = q.shape[-1]
    assert n > 2, "Must have at least two quantiles to interpolate."

    # Silently handle the case where p is outside [0, 1]
    soft_ranks = torch.nextafter(p * n, p.new_tensor(0.0))
    left_q = q[..., soft_ranks.int()]

    # The ICDF of zero is always -inf because there is no finite smallest `x`
    # such that the CDF of `x` is greater than or equal to zero.
    return left_q.where(p > 0, -torch.inf)
