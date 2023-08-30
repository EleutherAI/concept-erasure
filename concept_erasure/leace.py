from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .caching import cached_property, invalidates_cache
from .online_stats import OnlineStats

ErasureMethod = Literal["leace", "orth"]


@dataclass(frozen=True)
class LeaceEraser:
    """LEACE eraser that surgically erases a concept from a representation.

    Since the LEACE projection matrix is guaranteed to be a rank k - 1 perturbation of
    the identity, we store it implicitly in the d x k matrices `proj_left` and
    `proj_right`. The full matrix is given by `torch.eye(d) - proj_left @ proj_right`.
    """

    proj_left: Tensor
    proj_right: Tensor
    bias: Tensor | None

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "LeaceEraser":
        """Convenience method to fit a LeaceEraser on data and return it."""
        return LeaceFitter.fit(x, z, **kwargs).eraser

    @property
    def P(self) -> Tensor:
        """The projection matrix."""
        eye = torch.eye(
            self.proj_left.shape[0],
            device=self.proj_left.device,
            dtype=self.proj_left.dtype,
        )
        return eye - self.proj_left @ self.proj_right

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the projection to the input tensor."""
        delta = x - self.bias if self.bias is not None else x

        # Ensure we do the matmul in the most efficient order.
        x_ = x - (delta @ self.proj_right.T) @ self.proj_left.T
        return x_.type_as(x)


class LeaceFitter(OnlineStats):
    """Fits an affine transform that surgically erases a concept from a representation.

    This class implements Least-squares Concept Erasure (LEACE) from
    https://arxiv.org/abs/2306.03819. You can also use a slightly simpler orthogonal
    projection-based method by setting `method="orth"`.

    This class stores all the covariance statistics needed to compute the LEACE eraser.
    This allows the statistics to be updated incrementally with `update()`.
    """

    stats: OnlineStats
    """Online statistics for X and Z."""

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "LeaceFitter":
        """Convenience method to fit a LeaceFitter on data and return it."""
        n, d = x.shape
        _, k = z.reshape(n, -1).shape

        fitter = LeaceFitter(d, k, device=x.device, dtype=x.dtype, **kwargs)
        return fitter.update(x, z)

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
        """Initialize a `LeaceFitter`.

        Args:
            x_dim: Dimensionality of the representation.
            z_dim: Dimensionality of the concept.
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
        super().__init__(
            x_dim,
            z_dim,
            device=device,
            dtype=dtype,
            shrinkage=shrinkage,
            sigma_xx=method == "leace",
        )
        self.affine = affine
        self.constrain_cov_trace = constrain_cov_trace
        self.method = method

        assert svd_tol > 0.0, "`svd_tol` must be positive for numerical stability."
        self.svd_tol = svd_tol

    @invalidates_cache("eraser")
    def update(self, x: Tensor, z: Tensor) -> "LeaceFitter":
        """Update the running statistics with a new batch of data."""
        super().update(x, z)
        return self

    @cached_property
    def eraser(self) -> LeaceEraser:
        """Erasure function lazily computed given the current statistics."""
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

        proj_left = W_inv @ u
        proj_right = u.T @ W

        if self.constrain_cov_trace and self.method == "leace":
            P = eye - proj_left @ proj_right

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

                # TODO: Avoid using SVD here
                u, s, vh = torch.linalg.svd(eye - P)
                proj_left = u * s.sqrt()
                proj_right = vh * s.sqrt()

        return LeaceEraser(
            proj_left, proj_right, bias=self.mean_x if self.affine else None
        )
