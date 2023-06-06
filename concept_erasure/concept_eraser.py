from typing import Literal

import torch
from torch import Tensor, nn


class ConceptEraser(nn.Module):
    """Minimally edit features to make specified concepts linearly undetectable."""

    mean_x: Tensor
    """Running mean of X."""

    mean_z: Tensor
    """Running mean of Z."""

    sigma_xz_M2: Tensor
    """Unnormalized cross-covariance matrix X^T Z."""

    x_M2: Tensor | None
    """Unnormalized covariance matrix X^T X."""

    n_x: Tensor
    """Number of X samples seen so far."""

    n_z: Tensor
    """Number of Z samples seen so far."""

    _P: Tensor | None

    @classmethod
    def fit(cls, x: Tensor, y: Tensor, **kwargs) -> "ConceptEraser":
        """Convenience method to fit a ConceptEraser on data and return it."""
        n, d = x.shape
        _, k = y.reshape(n, -1).shape

        return cls(d, k, device=x.device, dtype=x.dtype, **kwargs).update(x, y)

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        proj_type: Literal["leace", "orth", "relaxed"] = "leace",
        *,
        affine: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        svd_tol: float = 0.01,
    ):
        """Initialize a ConceptEraser.

        Args:
            x_dim: Dimensionality of the input.
            z_dim: Dimensionality of the labels.
            proj_type: Type of projection matrix to use.
            affine: Whether to use a bias term to ensure the unconditional mean of the
                features remains the same after erasure.
            device: Device to put the statistics on.
            dtype: Data type to use for the statistics.
            svd_tol: Singular values under this threshold are truncated, both during
                the phase where we do SVD on the cross-covariance matrix, and at the
                phase where we compute the pseudoinverse of the projected covariance
                matrix. Higher values are more numerically stable and result in less
                damage to the representation, but may leave trace correlations intact.
        """
        super().__init__()

        self.z_dim = z_dim
        self.x_dim = x_dim

        self.affine = affine
        self.proj_rank = z_dim
        self.proj_type = proj_type
        self.z_dim = z_dim

        assert svd_tol > 0.0, "`svd_tol` must be positive for numerical stability."
        self.svd_tol = svd_tol

        self.register_buffer("mean_x", torch.zeros(x_dim, device=device, dtype=dtype))
        self.register_buffer("mean_z", self.mean_x.new_zeros(z_dim))
        self.register_buffer(
            "sigma_xz_M2",
            self.mean_x.new_zeros(x_dim, z_dim),
        )
        self.register_buffer("n_x", torch.tensor(0, device=device, dtype=dtype))
        self.register_buffer("n_z", torch.tensor(0, device=device, dtype=dtype))
        self.register_buffer("_P", None)

        if self.proj_type == "leace":
            M2 = self.mean_x.new_zeros(x_dim, x_dim)
        elif self.proj_type in ("orth", "relaxed"):
            M2 = None
        else:
            raise ValueError(f"Unknown projection type {self.proj_type}")

        self.register_buffer("x_M2", M2)

    def forward(self, x: Tensor) -> Tensor:
        """Minimally edit `x` to remove correlations with the target concepts.

        Args:
            x: Representations of shape (..., x_dim).

        Returns:
            The edited representations of shape (..., x_dim).
        """
        d, _ = self.sigma_xz_M2.shape
        assert self.n_x > 0, "Call update() before forward()"
        assert x.shape[-1] == d

        if self.affine:
            x_ = (x - self.mean_x) @ self.P.T + self.mean_x
            return x_.type_as(x)
        else:
            return (x.type_as(self.P) @ self.P.T).type_as(x)

    @torch.no_grad()
    def update(self, x: Tensor, z: Tensor | None = None) -> "ConceptEraser":
        """Update the running statistics with a new batch of data.

        It's possible to call this method without `z` if you only want to update the
        statistics of X. This is useful if you don't have labels but want to adjust the
        mean and covariance of X to match a new dataset.
        """
        d, c = self.sigma_xz_M2.shape
        x = x.reshape(-1, d).type_as(self.mean_x)

        if not x.isfinite().all():
            raise RuntimeError("Non-finite values in input")

        n, d2 = x.shape
        assert d == d2, f"Unexpected number of features {d2}"

        # We always have an X, we might not have a Z
        self.n_x += n

        # Welford's online algorithm
        delta_x = x - self.mean_x
        self.mean_x += delta_x.sum(dim=0) / self.n_x
        delta_x2 = x - self.mean_x

        # Update the covariance matrix of X if needed (for LEACE)
        if self.proj_type == "leace":
            assert self.x_M2 is not None
            self.x_M2.addmm_(delta_x.mT, delta_x2)

        # Invalidate the cached projection matrix
        self._P = None

        # We do have labels, so we can update the Z statistics
        if z is not None:
            # y might start out 1D, but we want to treat it as 2D
            z = z.reshape(n, -1).type_as(x)
            assert z.shape[-1] == c, f"Unexpected number of classes {z.shape[-1]}"

            self.n_z += n

            delta_z = z - self.mean_z
            self.mean_z += delta_z.sum(dim=0) / self.n_x
            delta_z2 = z - self.mean_z

            # Update the cross-covariance matrix
            self.sigma_xz_M2.addmm_(delta_x.mT, delta_z2)

        return self

    @property
    def P(self) -> Tensor:
        """Projection matrix for removing the subspace."""
        if self._P is not None:
            return self._P

        eye = torch.eye(self.x_dim, device=self.mean_x.device, dtype=self.mean_x.dtype)
        u, s, _ = torch.linalg.svd(self.sigma_xz, full_matrices=False)

        if self.proj_type == "relaxed":
            # Treat `svd_tol` as a constraint on the spectral norm of sigma_xz after
            # erasure. In this case Q is not actually a projection matrix, it's a
            # PSD non-expansive map (spectral norm <= 1).
            Q = eye - u * torch.clamp(1 - self.svd_tol / s, 0, 1) @ u.T
        else:
            # Throw away singular values that are too small
            mask = s > self.svd_tol
            if not mask.any():
                return eye

            u = u[:, mask]
            Q = eye - u @ u.T if mask.any() else eye

            # Save this for debugging
            self.proj_rank = mask.sum().item()

        if self.proj_type != "leace":
            self._P = Q
            return self._P

        self._P = self.proj_for_subspace(u)
        return self._P

    def proj_for_subspace(self, u: Tensor) -> Tensor:
        """Projection matrix for removing the subspace."""
        eye = torch.eye(self.x_dim, device=self.mean_x.device, dtype=self.mean_x.dtype)
        Q = eye - u @ u.T

        # LEACE and orthogonal projection matrix computation
        # Adjust Q to account for the covariance of X
        sigma = self.cov_x
        A = Q @ sigma @ Q
        try:
            L, V = torch.linalg.eigh(A)
        except torch.linalg.LinAlgError as e:
            # Better error messages in the common case where A is non-finite
            if not A.isfinite().all():
                raise RuntimeError("Non-finite values in covariance matrix") from e
            else:
                raise e

        # Manual pseudoinverse
        L = L.reciprocal().where(L > self.svd_tol, 0.0)
        P = sigma @ V @ torch.diag_embed(L) @ V.mT

        # Prevent the covariance trace from increasing
        old_trace = torch.trace(sigma)
        new_trace = torch.trace(P @ sigma @ P.mT)

        # If applying the projection matrix increases the variance, this might
        # cause instability, especially when erasure is applied multiple times.
        # We regularize toward the orthogonal projection matrix to avoid this.
        if new_trace > old_trace:
            # Set up the variables for the quadratic equation
            x = new_trace
            y = 2 * torch.trace(P @ sigma @ Q.mT)
            z = torch.trace(Q @ sigma @ Q.mT)
            w = old_trace

            # Solve for the mixture of P and Q that makes the trace equal to the
            # trace of the original covariance matrix
            discr = torch.sqrt(4 * w * x - 4 * w * y + 4 * w * z - 4 * x * z + y**2)
            alpha1 = (-y / 2 + z - discr / 2) / (x - y + z)
            alpha2 = (-y / 2 + z + discr / 2) / (x - y + z)

            # Choose the positive root
            alpha = torch.where(alpha1 > 0, alpha1, alpha2).clamp(0, 1)
            P = alpha * P + (1 - alpha) * Q

        return P

    @property
    def cov_x(self) -> Tensor:
        """The covariance matrix of X."""
        assert self.n_x > 1, "Call update() before accessing cov_x"
        assert (
            self.x_M2 is not None
        ), "Covariance statistics are not being tracked for X"

        cov = self.x_M2 / (self.n_x - 1)

        # Accumulated numerical error may cause this to be slightly non-symmetric
        return (cov + cov.mT) / 2

    @property
    def sigma_xz(self) -> Tensor:
        """The cross-covariance matrix."""
        assert self.n_z > 1, "Call update() with labels before accessing sigma_xz"
        return self.sigma_xz_M2 / (self.n_z - 1)

    def finalize(self) -> "ConceptEraser":
        """Compute the projection matrix and drop covariance matrices."""
        self.P
        self.x_M2 = None
        return self
