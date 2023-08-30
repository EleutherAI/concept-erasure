from dataclasses import dataclass

import torch
from torch import Tensor

from .caching import cached_property, invalidates_cache
from .online_stats import OnlineStats


@dataclass(frozen=True)
class OracleEraser:
    """Surgically erases a concept from a representation, given concept labels."""

    coef: Tensor
    mean_z: Tensor

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "OracleEraser":
        """Convenience method to fit an OracleEraser on data and return it."""
        return OracleFitter.fit(x, z, **kwargs).eraser

    def __call__(self, x: Tensor, z: Tensor) -> Tensor:
        """Replace `x` with the OLS residual given `z`."""
        # Ensure Z is at least 2D
        z = z.reshape(len(z), -1).type_as(x)
        expected_x = (z - self.mean_z) @ self.coef.T

        return x.sub(expected_x).type_as(x)


class OracleFitter(OnlineStats):
    """Compute stats needed for surgically erasing a concept Z from a random vector X.

    Unlike `LeaceFitter`, the resulting erasure function requires oracle concept labels
    at inference time. In exchange, it achieves more surgical edits.
    """

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "OracleFitter":
        """Convenience method to fit a OracleFitter on data and return it."""
        n, d = x.shape
        _, k = z.reshape(n, -1).shape

        fitter = OracleFitter(d, k, device=x.device, dtype=x.dtype, **kwargs)
        return fitter.update(x, z)

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        shrinkage: bool = False,
        svd_tol: float = 0.01,
    ):
        """Initialize a `OracleFitter`.

        Args:
            x_dim: Dimensionality of the representation.
            z_dim: Dimensionality of the concept.
            method: Type of projection matrix to use.
            device: Device to put the statistics on.
            dtype: Data type to use for the statistics.
            shrinkage: Whether to use shrinkage to estimate the covariance matrix of X.
            svd_tol: Threshold for singular values of the covariance matrix of Z.
        """
        super().__init__(
            x_dim,
            z_dim,
            device=device,
            dtype=dtype,
            shrinkage=shrinkage,
            sigma_xx=False,
        )

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.shrinkage = shrinkage

        assert svd_tol > 0.0, "`svd_tol` must be positive for numerical stability."
        self.svd_tol = svd_tol

    @invalidates_cache("eraser")
    def update(self, x: Tensor, z: Tensor) -> "OracleFitter":
        """Update the running statistics with a new batch of data."""
        super().update(x, z)
        return self

    @cached_property
    def eraser(self) -> OracleEraser:
        """Erasure function lazily computed given the current statistics."""
        assert self.mean_z is not None, "Z must be present to compute erasure function."

        return OracleEraser(
            self.sigma_xz @ torch.linalg.pinv(self.sigma_zz, atol=self.svd_tol),
            self.mean_z,
        )
