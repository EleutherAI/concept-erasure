from typing import Literal

import torch
from torch import Tensor, nn

from .shrinkage import gaussian_shrinkage

ErasureMethod = Literal["leace", "orth"]


class ConceptEraser(nn.Module):
    """Minimally edit features to make specified concepts linearly undetectable.

    This class implements Least-squares Concept Erasure (LEACE) from
    https://arxiv.org/abs/2306.03819. You can also use a slightly simpler orthogonal
    projection-based method by setting `method="orth"`.

    This class stores both the covariance statistics needed to compute the LEACE
    transformation, and the fitted parameters themselves. This allows the statistics to
    be updated incrementally. The downside is that the covariance matrix of X takes
    O(d^2) memory, which can become a problem when using many erasers at once for a
    deep neural network. To mitigate this issue, we allow you to drop the covariance
    matrix after fitting using the `finalize` method. After calling `finalize` this
    class only takes O(dk) memory, where k is the dimensionality of Z.

    Since the LEACE projection matrix is guaranteed to be a rank k - 1 perturbation of
    the identity, we store it implicitly in the d x k matrices `proj_left` and
    `proj_right`. The full matrix is given by `torch.eye(d) - proj_left @ proj_right`.
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

    proj_left: Tensor
    """d x k matrix used for constructing the projection."""

    proj_right: Tensor
    """k x d matrix used for constructing the projection."""

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
        shrinkage: bool = True,
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
            shrinkage: Whether to use shrinkage to estimate the covariance matrix of X.
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
        self.dirty = True
        self.method = method
        self.shrinkage = shrinkage

        assert svd_tol > 0.0, "`svd_tol` must be positive for numerical stability."
        self.svd_tol = svd_tol

        self.register_buffer("mean_x", torch.zeros(x_dim, device=device, dtype=dtype))
        self.register_buffer("mean_z", self.mean_x.new_zeros(z_dim))
        self.register_buffer(
            "sigma_xz_",
            self.mean_x.new_zeros(x_dim, z_dim),
        )
        self.register_buffer("n", torch.tensor(0, device=device, dtype=dtype))

        rank = min(x_dim, z_dim)
        self.register_buffer("proj_left", self.mean_x.new_zeros(x_dim, rank))
        self.register_buffer("proj_right", self.mean_x.new_zeros(rank, x_dim))

        if self.method == "leace":
            M2 = self.mean_x.new_zeros(x_dim, x_dim)
        elif self.method == "orth":
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
        self._compute_proj()  # Ensure the projection matrix is up to date

        bias = self.mean_x if self.affine else 0.0
        x_ = x - ((x - bias) @ self.proj_right.T) @ self.proj_left.T
        return x_.type_as(x)

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
        self.dirty = True

        z = z.reshape(n, -1).type_as(x)
        assert z.shape[-1] == c, f"Unexpected number of classes {z.shape[-1]}"

        delta_z = z - self.mean_z
        self.mean_z += delta_z.sum(dim=0) / self.n
        delta_z2 = z - self.mean_z

        # Update the cross-covariance matrix
        self.sigma_xz_.addmm_(delta_x.mT, delta_z2)

        return self

    def _compute_proj(self):
        # Only fit if the statistics are dirty
        if not self.dirty:
            return

        eye = torch.eye(self.x_dim, device=self.mean_x.device, dtype=self.mean_x.dtype)

        # Compute the whitening and unwhitening matrices
        if self.method == "leace":
            sigma = self.sigma_xx
            L, V = torch.linalg.eigh(sigma)

            # Threshold used by torch.linalg.pinv
            mask = L > (L[-1] * sigma.shape[-1] * torch.finfo(L.dtype).eps)

            # Assuming PSD; account for numerical error
            L.clamp_min_(0.0)

            W = V * L.rsqrt().where(mask, 0.0) @ V.mT
            W_inv = V * L.sqrt().where(mask, 0.0) @ V.mT
        else:
            W, W_inv = eye, eye

        u, s, _ = torch.linalg.svd(W @ self.sigma_xz, full_matrices=False)

        # Throw away singular values that are too small
        u *= s > self.svd_tol

        self.proj_left = W_inv @ u
        self.proj_right = u.T @ W

        if self.constrain_cov_trace and self.method == "leace":
            P = eye - self.proj_left @ self.proj_right

            # Prevent the covariance trace from increasing
            sigma = self.sigma_xx
            old_trace = torch.trace(sigma)
            new_trace = torch.trace(P @ sigma @ P.mT)

            # If applying the projection matrix increases the variance, this might
            # cause instability, especially when erasure is applied multiple times.
            # We regularize toward the orthogonal projection matrix to avoid this.
            if new_trace > old_trace:
                Q = eye - u @ u.T

                # Set up the variables for the quadratic equation
                x = new_trace
                y = 2 * torch.trace(P @ sigma @ Q.mT)
                z = torch.trace(Q @ sigma @ Q.mT)
                w = old_trace

                # Solve for the mixture of P and Q that makes the trace equal to the
                # trace of the original covariance matrix
                discr = torch.sqrt(
                    4 * w * x - 4 * w * y + 4 * w * z - 4 * x * z + y**2
                )
                alpha1 = (-y / 2 + z - discr / 2) / (x - y + z)
                alpha2 = (-y / 2 + z + discr / 2) / (x - y + z)

                # Choose the positive root
                alpha = torch.where(alpha1 > 0, alpha1, alpha2).clamp(0, 1)
                P = alpha * P + (1 - alpha) * Q

                u, s, vh = torch.linalg.svd(eye - P)
                self.proj_left = u * s.sqrt()
                self.proj_right = vh * s.sqrt()

        # Don't duplicate work
        self.dirty = False

    @property
    def P(self) -> Tensor:
        """Projection matrix used to erase information about Z from X."""
        eye = torch.eye(self.x_dim, device=self.mean_x.device, dtype=self.mean_x.dtype)
        self._compute_proj()
        return eye - self.proj_left @ self.proj_right

    @property
    def sigma(self) -> Tensor:
        """The covariance matrix of X."""
        assert self.n > 1, "Call update() before accessing sigma_xx"
        assert (
            self.sigma_ is not None
        ), "Covariance statistics are not being tracked for X"

        # Accumulated numerical error may cause this to be slightly non-symmetric
        S_hat = (self.sigma_ + self.sigma_.mT) / 2

        # Apply Rao-Blackwell Ledoit-Wolf shrinkage
        if self.shrinkage:
            return gaussian_shrinkage(S_hat / self.n, self.n)

        # Just apply Bessel's correction
        else:
            return S_hat / (self.n - 1)

    # Support naming conventions from both v1 and v2 of the paper
    sigma_xx = sigma

    @property
    def sigma_xz(self) -> Tensor:
        """The cross-covariance matrix."""
        assert self.n > 1, "Call update() with labels before accessing sigma_xz"
        return self.sigma_xz_ / (self.n - 1)

    def finalize(self) -> "ConceptEraser":
        """Compute the projection matrix and drop covariance matrices."""
        self._compute_proj()
        self.sigma_ = None
        return self
