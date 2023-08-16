from dataclasses import dataclass

import torch
from torch import Tensor

from .caching import cached_property, invalidates_cache
from .optimal_transport import ot_barycenter, ot_map
from .shrinkage import optimal_linear_shrinkage


@dataclass(frozen=True)
class QuadraticEraser:
    """Surgically erases a concept from a representation, given concept labels."""

    class_means: Tensor

    global_mean: Tensor

    ot_maps: Tensor
    """`[k, d, d]` batch of optimal transport matrices to the concept barycenter."""

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "QuadraticEraser":
        """Convenience method to fit an QuadraticEraser on data and return it."""
        return QuadraticFitter.fit(x, z, **kwargs).eraser

    def __call__(self, x: Tensor, z: int) -> Tensor:
        """Replace `x` with the OLS residual given `z`."""

        # Subtract E[X | Z = z] for each (x, z) and add E[X]
        # x = x.index_add(0, z, self.ot_biases)
        return (x - self.class_means[z]) @ self.ot_maps[z].mT + self.global_mean


class QuadraticFitter:
    """Compute stats needed for surgically erasing a concept Z from a random vector X.

    Unlike `LeaceFitter`, the resulting erasure function requires oracle concept labels
    at inference time. In exchange, it achieves more surgical edits.
    """

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
        n, d = x.shape
        _, k = z.reshape(n, -1).shape

        fitter = QuadraticFitter(d, k, device=x.device, dtype=x.dtype, **kwargs)

        # TODO: Split up the data into chunks, each with a different concept label
        return fitter.update(x, z)

    def __init__(
        self,
        x_dim: int,
        num_classes: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        shrinkage: bool = False,
    ):
        """Initialize a `QuadraticFitter`.

        Args:
            x_dim: Dimensionality of the representation.
            z_dim: Dimensionality of the concept.
            method: Type of projection matrix to use.
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

    @torch.no_grad()
    @invalidates_cache("eraser")
    def update(self, x: Tensor, z: int) -> "QuadraticFitter":
        """Update the running statistics with a new batch of data."""
        _, d = x.shape
        x = x.reshape(-1, d).type_as(self.mean_x)

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
        assert torch.all(self.n > 1), "Some classes have < 2 samples"

        # Compute Wasserstein barycenter of the classes, then compute the optimal
        # transport maps from each class to the barycenter
        center = ot_barycenter(self.sigma_xx, self.n)
        ot_maps = ot_map(self.sigma_xx, center)

        return QuadraticEraser(self.mean_x, self.mean_x.mean(dim=0), ot_maps)

    @property
    def sigma_xx(self) -> Tensor:
        """Class-conditional covariance matrices of X."""
        assert self.n > 1, "Call update() before accessing sigma_xx"
        assert (
            self.sigma_xx_ is not None
        ), "Covariance statistics are not being tracked for X"

        # Accumulated numerical error may cause this to be slightly non-symmetric
        S_hat = (self.sigma_xx_ + self.sigma_xx_.mT) / 2

        # Apply Random Matrix Theory-based shrinkage
        if self.shrinkage:
            return optimal_linear_shrinkage(S_hat / self.n, self.n)

        # Just apply Bessel's correction
        else:
            return S_hat / (self.n - 1)
