from dataclasses import dataclass

import torch
from torch import Tensor

from .caching import cached_property, invalidates_cache
from .shrinkage import optimal_linear_shrinkage


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


class OracleFitter:
    """Compute stats needed for surgically erasing a concept Z from a random vector X.

    Unlike `LeaceFitter`, the resulting erasure function requires oracle concept labels
    at inference time. In exchange, it achieves more surgical edits.
    """

    mean_x: Tensor
    """Running mean of X."""

    mean_z: Tensor
    """Running mean of Z."""

    sigma_xz_: Tensor
    """Unnormalized cross-covariance matrix X^T Z."""

    sigma_zz_: Tensor
    """Unnormalized covariance matrix Z^T Z."""

    n: Tensor
    """Number of X samples seen so far."""

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
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.shrinkage = shrinkage

        assert svd_tol > 0.0, "`svd_tol` must be positive for numerical stability."
        self.svd_tol = svd_tol

        self.mean_x = torch.zeros(x_dim, device=device, dtype=dtype)
        self.mean_z = torch.zeros(z_dim, device=device, dtype=dtype)

        self.n = torch.tensor(0, device=device, dtype=dtype)
        self.sigma_xz_ = torch.zeros(x_dim, z_dim, device=device, dtype=dtype)
        self.sigma_zz_ = torch.zeros(z_dim, z_dim, device=device, dtype=dtype)

    @torch.no_grad()
    @invalidates_cache("eraser")
    def update(self, x: Tensor, z: Tensor) -> "OracleFitter":
        """Update the running statistics with a new batch of data."""
        d, c = self.sigma_xz_.shape
        x = x.reshape(-1, d).type_as(self.mean_x)
        n, d2 = x.shape

        assert d == d2, f"Unexpected number of features {d2}"
        self.n += n

        # Welford's online algorithm
        delta_x = x - self.mean_x
        self.mean_x += delta_x.sum(dim=0) / self.n

        z = z.reshape(n, -1).type_as(x)
        assert z.shape[-1] == c, f"Unexpected number of classes {z.shape[-1]}"

        delta_z = z - self.mean_z
        self.mean_z += delta_z.sum(dim=0) / self.n
        delta_z2 = z - self.mean_z

        self.sigma_xz_.addmm_(delta_x.mT, delta_z2)
        self.sigma_zz_.addmm_(delta_z.mT, delta_z2)

        return self

    @cached_property
    def eraser(self) -> OracleEraser:
        """Erasure function lazily computed given the current statistics."""
        return OracleEraser(
            self.sigma_xz @ torch.linalg.pinv(self.sigma_zz, atol=self.svd_tol),
            self.mean_z,
        )

    @property
    def sigma_zz(self) -> Tensor:
        """The covariance matrix of Z."""
        assert self.n > 1, "Call update() before accessing sigma_xx"
        assert (
            self.sigma_zz_ is not None
        ), "Covariance statistics are not being tracked for X"

        # Accumulated numerical error may cause this to be slightly non-symmetric
        S_hat = (self.sigma_zz_ + self.sigma_zz_.mT) / 2

        # Apply Random Matrix Theory-based shrinkage
        if self.shrinkage:
            return optimal_linear_shrinkage(S_hat / self.n, self.n)

        # Just apply Bessel's correction
        else:
            return S_hat / (self.n - 1)

    @property
    def sigma_xz(self) -> Tensor:
        """The cross-covariance matrix."""
        assert self.n > 1, "Call update() with labels before accessing sigma_xz"
        return self.sigma_xz_ / (self.n - 1)
