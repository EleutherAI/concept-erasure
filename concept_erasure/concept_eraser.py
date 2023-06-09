from typing import Literal

import torch
from torch import Tensor, nn

ErasureMethod = Literal["leace", "orth", "relaxed"]


class ConceptEraser(nn.Module):
    """Minimally edit features to make specified concepts linearly undetectable.

    There are three erasure methods currently supported:
    - `"leace"`: Least-squares Concept Erasure from https://arxiv.org/abs/2306.03819.
    - `"orth"`: Orthogonal projection onto colsp(Sigma_xz)^perp.
    - `"relaxed"`: Applies a PSD map with spectral norm <= 1 that ensures the resulting
    cross-covariance matrix Cov(PX, Z) has spectral norm no greater than `svd_tol`.
    Importantly, this method **does not** ensure linear guardedness, but may be useful
    for concept scrubbing purposes. It has the benefit of being less sensitive to noise
    and the choice of `svd_tol` than other methods.
    """

    mean_x: Tensor
    """Running mean of X."""

    mean_z: Tensor
    """Running mean of Z."""

    sigma_xz_: Tensor
    """Unnormalized cross-covariance matrix X^T Z."""

    sigma_: Tensor | None
    """Unnormalized covariance matrix X^T X."""

    n: Tensor
    """Number of X samples seen so far."""

    _P: Tensor | None

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "ConceptEraser":
        """Convenience method to fit a ConceptEraser on data and return it."""
        n, d = x.shape
        _, k = z.reshape(n, -1).shape

        return cls(d, k, device=x.device, dtype=x.dtype, **kwargs).update(x, z)

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        method: ErasureMethod = "leace",
        *,
        affine: bool = True,
        constrain_cov_trace: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        svd_tol: float = 0.01,
    ):
        """Initialize a ConceptEraser.

        Args:
            x_dim: Dimensionality of the input.
            z_dim: Dimensionality of the labels.
            method: Type of projection matrix to use.
            affine: Whether to use a bias term to ensure the unconditional mean of the
                features remains the same after erasure.
            constrain_cov_trace: Whether to constrain the trace of the covariance of X
                after erasure to be no greater than before erasure. This is especially
                useful when injecting the scrubbed features back into a model. Without
                this constraint, the norm of the model's hidden states may diverge in
                some cases.
            device: Device to put the statistics on.
            dtype: Data type to use for the statistics.
            svd_tol: Singular values under this threshold are truncated, both during
                the phase where we do SVD on the cross-covariance matrix, and at the
                phase where we compute the pseudoinverse of the projected covariance
                matrix. Higher values are more numerically stable and result in less
                damage to the representation, but may leave trace correlations intact.
        """
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.affine = affine
        self.constrain_cov_trace = constrain_cov_trace
        self.proj_rank = z_dim
        self.method = method
        self.z_dim = z_dim

        assert svd_tol > 0.0, "`svd_tol` must be positive for numerical stability."
        self.svd_tol = svd_tol

        self.register_buffer("mean_x", torch.zeros(x_dim, device=device, dtype=dtype))
        self.register_buffer("mean_z", self.mean_x.new_zeros(z_dim))
        self.register_buffer(
            "sigma_xz_",
            self.mean_x.new_zeros(x_dim, z_dim),
        )
        self.register_buffer("n", torch.tensor(0, device=device, dtype=dtype))
        self.register_buffer("_P", None)

        if self.method == "leace":
            M2 = self.mean_x.new_zeros(x_dim, x_dim)
        elif self.method in ("orth", "relaxed"):
            M2 = None
        else:
            raise ValueError(f"Unknown projection type {self.method}")

        self.register_buffer("sigma_", M2)

    def forward(self, x: Tensor) -> Tensor:
        """Minimally edit `x` to remove correlations with the target concepts.

        Args:
            x: Representations of shape (..., x_dim).

        Returns:
            The edited representations of shape (..., x_dim).
        """
        d, _ = self.sigma_xz_.shape
        assert self.n > 0, "Call update() before forward()"
        assert x.shape[-1] == d

        if self.affine:
            x_ = (x - self.mean_x) @ self.P.T + self.mean_x
            return x_.type_as(x)
        else:
            return (x.type_as(self.P) @ self.P.T).type_as(x)

    @torch.no_grad()
    def update(self, x: Tensor, z: Tensor) -> "ConceptEraser":
        """Update the running statistics with a new batch of data."""
        d, c = self.sigma_xz_.shape
        x = x.reshape(-1, d).type_as(self.mean_x)
        n, d2 = x.shape

        assert d == d2, f"Unexpected number of features {d2}"
        self.n += n

        # Welford's online algorithm
        delta_x = x - self.mean_x
        self.mean_x += delta_x.sum(dim=0) / self.n
        delta_x2 = x - self.mean_x

        # Update the covariance matrix of X if needed (for LEACE)
        if self.method == "leace":
            assert self.sigma_ is not None
            self.sigma_.addmm_(delta_x.mT, delta_x2)

        # Invalidate the cached projection matrix
        self._P = None

        z = z.reshape(n, -1).type_as(x)
        assert z.shape[-1] == c, f"Unexpected number of classes {z.shape[-1]}"

        delta_z = z - self.mean_z
        self.mean_z += delta_z.sum(dim=0) / self.n
        delta_z2 = z - self.mean_z

        # Update the cross-covariance matrix
        self.sigma_xz_.addmm_(delta_x.mT, delta_z2)

        return self

    @property
    def P(self) -> Tensor:
        """Projection matrix for removing the subspace."""
        if self._P is not None:
            return self._P

        eye = torch.eye(self.x_dim, device=self.mean_x.device, dtype=self.mean_x.dtype)
        u, s, _ = torch.linalg.svd(self.sigma_xz, full_matrices=False)

        if self.method == "relaxed":
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

        if self.method != "leace":
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
        sigma = self.sigma_xx
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
        if self.constrain_cov_trace and new_trace > old_trace:
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
    def sigma(self) -> Tensor:
        """The covariance matrix of X."""
        assert self.n > 1, "Call update() before accessing sigma_xx"
        assert (
            self.sigma_ is not None
        ), "Covariance statistics are not being tracked for X"

        cov = self.sigma_ / (self.n - 1)

        # Accumulated numerical error may cause this to be slightly non-symmetric
        return (cov + cov.mT) / 2

    # Support multiple naming conventions
    cov_x = sigma
    sigma_xx = sigma

    @property
    def sigma_xz(self) -> Tensor:
        """The cross-covariance matrix."""
        assert self.n > 1, "Call update() with labels before accessing sigma_xz"
        return self.sigma_xz_ / (self.n - 1)

    def finalize(self) -> "ConceptEraser":
        """Compute the projection matrix and drop covariance matrices."""
        self.P
        self.sigma_ = None
        return self
