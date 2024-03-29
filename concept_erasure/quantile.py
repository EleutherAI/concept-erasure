import torch
from torch import Tensor

from .groupby import groupby


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
    equal to `p`.

    NOTE: Strictly speaking, this function should return `-inf` when `p` is exactly
    zero, because there is no smallest `x` such that `p(x) = 0`. But in practice we
    want this function to always return a finite value, so we clip to the minimum
    value in `q`.

    Args:
        x: `[...]` Tensor of data points of arbitrary shape.
        q: `[..., num_quantiles]` batch of quantiles. Must be sorted, and
            should broadcast with `x` except for the last dimension.

    Returns:
        `[...]` Empirical CDF evaluated for each element of `x`.
    """
    n = q.shape[-1]
    assert n > 2, "Must have at least two quantiles to interpolate."

    soft_ranks = torch.nextafter(p * n, p.new_tensor(0.0))
    return q.gather(-1, soft_ranks.long())


class QuantileNormalizer:
    """Componentwise quantile normalization."""

    lut: Tensor
    """`[k, ..., num_bins]` batch of lookup tables."""

    dim: int
    """Dimension along which to group the data."""

    def __init__(
        self,
        x: Tensor,
        z: Tensor,
        num_bins: int = 256,
        dim: int = 0,
    ):
        # Efficiently get a view onto each class
        grouped = groupby(x, z, dim=dim)
        self.dim = dim

        k = len(grouped.labels)
        self.lut = x.new_empty([k, *x.shape[1:], num_bins])

        grid = torch.linspace(0, 1, num_bins, device=x.device)
        for i, grp in grouped:
            self.lut[i] = grp.quantile(grid, dim=dim).movedim(0, -1)

    @property
    def num_bins(self) -> int:
        return self.lut.shape[-1]

    def cdf(self, z: int, x: Tensor) -> Tensor:
        return cdf(x.movedim(0, -1), self.lut[z]).movedim(-1, 0)

    def sample(self, z: int, n: int) -> Tensor:
        lut = self.lut[z]

        # Sample p from uniform distribution, then apply inverse CDF
        p = torch.rand(*lut[..., 0].shape, n, device=lut.device)
        return icdf(p, lut).movedim(-1, 0)

    def transport(self, x: Tensor, source_z: Tensor, target_z: int) -> Tensor:
        """Transport `x` from class `source_z` to class `target_z`"""
        return (
            groupby(x, source_z, dim=self.dim)
            .map(
                # Probability integral transform, followed by inverse for target class
                lambda z, x: icdf(
                    cdf(x.movedim(0, -1), self.lut[z]), self.lut[target_z]
                ).movedim(-1, 0)
            )
            .coalesce()
        )
