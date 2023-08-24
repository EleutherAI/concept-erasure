from dataclasses import dataclass

import torch
from torch import Tensor

from .caching import cached_property, invalidates_cache
from .groupby import groupby
from .optimal_transport import ot_barycenter, ot_map, ot_midpoint
from .shrinkage import optimal_linear_shrinkage


@dataclass(frozen=True)
class QuadraticEraser:
    """Performs surgical quadratic concept erasure given oracle concept labels."""

    class_means: Tensor
    """`[k, d]` batch of class centroids."""

    global_mean: Tensor
    """`[d]` global centroid of the dataset."""

    ot_maps: Tensor
    """`[k, d, d]` batch of optimal transport matrices to the concept barycenter."""

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "QuadraticEraser":
        """Convenience method to fit a QuadraticEraser on data and return it."""
        return QuadraticFitter.fit(x, z, **kwargs).eraser

    def optimal_transport(self, z: int, x: Tensor) -> Tensor:
        """Transport `x` to the barycenter, assuming it was sampled from class `z`"""
        return (x - self.class_means[z]) @ self.ot_maps[z].mT + self.global_mean

    def __call__(self, x: Tensor, z: Tensor) -> Tensor:
        """Apply erasure to `x` with oracle labels `z`."""

        # Efficiently group `x` by `z`, optimally transport each group, then coalesce
        return groupby(x, z).map(self.optimal_transport).coalesce()


class QuadraticFitter:
    """Compute barycenter & optimal transport maps for a quadratic concept eraser."""

    mean_x: Tensor
    """Running mean of X."""

    mean_z: Tensor
    """Running mean of Z."""

    sigma_xx_: Tensor
    """Unnormalized class-conditional covariance matrices X^T X."""

    n: Tensor
    """Number of samples seen so far in each class."""

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "QuadraticFitter":
        """Convenience method to fit a OracleFitter on data and return it."""
        d = x.shape[-1]
        k = int(z.max()) + 1  # Number of classes

        fitter = QuadraticFitter(d, k, device=x.device, dtype=x.dtype, **kwargs)
        return fitter.update(x, z)

    def __init__(
        self,
        x_dim: int,
        num_classes: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        shrinkage: bool = True,
    ):
        """Initialize a `QuadraticFitter`.

        Args:
            x_dim: Dimensionality of the representation.
            num_classes: Number of distinct classes in the dataset.
            device: Device to put the statistics on.
            dtype: Data type to use for the statistics.
            shrinkage: Whether to use shrinkage to estimate the covariance matrix of X.
        """
        super().__init__()

        self.num_classes = num_classes
        self.shrinkage = shrinkage

        self.mean_x = torch.zeros(num_classes, x_dim, device=device, dtype=dtype)
        self.n = torch.zeros(num_classes, device=device, dtype=torch.long)
        self.sigma_xx_ = torch.zeros(
            num_classes, x_dim, x_dim, device=device, dtype=dtype
        )

    def update(self, x: Tensor, z: Tensor) -> "QuadraticFitter":
        """Update the running statistics with a new batch of data."""
        x = x.flatten(0, -2).type_as(self.mean_x)

        for label, group in groupby(x, z, dim=0):
            self.update_single(group, label)

        return self

    @torch.no_grad()
    @invalidates_cache("eraser")
    def update_single(self, x: Tensor, z: int) -> "QuadraticFitter":
        """Update the running statistics with `x`, all sampled from class `z`."""
        x = x.flatten(0, -2).type_as(self.mean_x)

        self.n[z] += len(x)

        # Welford's online algorithm
        delta_x = x - self.mean_x[z]
        self.mean_x[z] += delta_x.sum(dim=0) / self.n[z]
        delta_x2 = x - self.mean_x[z]

        self.sigma_xx_[z].addmm_(delta_x.mT, delta_x2)

        return self

    @cached_property
    def eraser(self) -> QuadraticEraser:
        """Erasure function lazily computed given the current statistics."""

        # Compute Wasserstein barycenter of the classes
        if self.num_classes == 2:
            # Use closed form solution for the binary case
            sigmas = self.sigma_xx
            weights = self.n / self.n.sum()
            center = ot_midpoint(sigmas[0], sigmas[1], *weights.tolist())
        else:
            # Use fixed point iteration for the general case
            center = ot_barycenter(self.sigma_xx, self.n)

        # Then compute the optimal ransport maps from each class to the barycenter
        ot_maps = ot_map(self.sigma_xx, center)
        return QuadraticEraser(self.mean_x, self.mean_x.mean(dim=0), ot_maps)

    @property
    def sigma_xx(self) -> Tensor:
        """Class-conditional covariance matrices of X."""
        assert torch.all(self.n > 1), "Some classes have < 2 samples"
        assert (
            self.sigma_xx_ is not None
        ), "Covariance statistics are not being tracked for X"

        # Accumulated numerical error may cause this to be slightly non-symmetric
        S_hat = (self.sigma_xx_ + self.sigma_xx_.mT) / 2

        # Apply Random Matrix Theory-based shrinkage
        n = self.n.view(-1, 1, 1)
        if self.shrinkage:
            return optimal_linear_shrinkage(S_hat / n, n)

        # Just apply Bessel's correction
        else:
            return S_hat / (n - 1)
