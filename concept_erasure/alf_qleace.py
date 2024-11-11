from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .caching import cached_property, invalidates_cache
from .groupby import groupby
from .shrinkage import optimal_linear_shrinkage

ErasureMethod = Literal["leace", "orth"]


@dataclass(frozen=True)
class AlfQLeaceEraser:
    """QLEACE eraser that erases concepts from a representation. First applies LEACE,
    then applies pair-wise QLEACE using a projection matrix optimized to the class with
    the covariance most divergent from the mean covariance.

    Since the LEACE projection matrix is guaranteed to be a rank k - 1 perturbation of
    the identity, we store it implicitly in the d x k matrices `proj_left` and
    `proj_right`. The full matrix is given by `torch.eye(d) - proj_left @ proj_right`.

    The ALF-QLEACE projection matrix is guaranteed to be a rank 1 perturbation of the
    identity, given by torch.eye(d) - alf_qleace_vec @ alf_qleace_vec.
    """

    proj_left: Tensor
    proj_right: Tensor
    bias: Tensor | None
    alf_qleace_vec: Tensor

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "AlfQLeaceEraser":
        """Convenience method to fit a LeaceEraser on data and return it."""
        return AlfQLeaceFitter.fit(x, z, **kwargs).eraser

    @property
    def P(self) -> Tensor:
        """The LEACE projection matrix."""
        eye = torch.eye(
            self.proj_left.shape[0],
            device=self.proj_left.device,
            dtype=self.proj_left.dtype,
        )
        return eye - self.proj_left @ self.proj_right

    @property
    def Q(self) -> Tensor:
        """The ALF-QLEACE projection matrix."""
        eye = torch.eye(
            self.alf_qleace_vec.shape[0],
            device=self.alf_qleace_vec.device,
            dtype=self.alf_qleace_vec.dtype,
        )
        return eye - torch.outer(self.alf_qleace_vec, self.alf_qleace_vec)

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the projection to the input tensor."""
        delta = x - self.bias if self.bias is not None else x

        # Ensure we do the matmul in the most efficient order.
        x_ = x - (delta @ self.proj_right.mH) @ self.proj_left.mH

        # Apply the ALF-QLEACE projection
        v = self.alf_qleace_vec
        x_ = x_ - torch.einsum("i,bi->bi", v, (v @ x_.mH).unsqueeze(1))

        return x_.type_as(x)

    def to(self, device: torch.device | str) -> "AlfQLeaceEraser":
        """Move eraser to a new device."""
        return AlfQLeaceEraser(
            self.proj_left.to(device),
            self.proj_right.to(device),
            self.bias.to(device) if self.bias is not None else None,
            self.alf_qleace_vec.to(device),
        )


class AlfQLeaceFitter:
    """Fits LEACE plus a linear transform that surgically erases the direction of
    maximum covariance from a representation.

    This class implements Least-squares Concept Erasure (LEACE) from
    https://arxiv.org/abs/2306.03819. You can also use a slightly simpler orthogonal
    projection-based method by setting `method="orth"`.

    This class stores all the covariance statistics needed to compute the QLEACE eraser.
    This allows the statistics to be updated incrementally with `update()`.
    """

    global_mean_x: Tensor
    """Running mean of X."""

    global_mean_z: Tensor
    """Running mean of Z."""

    sigma_xz_: Tensor
    """Unnormalized cross-covariance matrix X^T Z."""

    sigma_xx_: Tensor | None
    """Unnormalized covariance matrix X^T X."""

    sigma_xx_z_: Tensor
    """Unnormalized cross-covariance matrix X^T X for each class Z"""

    global_n: Tensor
    """Number of X samples seen so far."""

    @classmethod
    def fit(cls, x: Tensor, z: Tensor, **kwargs) -> "AlfQLeaceFitter":
        """Convenience method to fit a LeaceFitter on data and return it."""
        n, d = x.shape
        _, k = z.reshape(n, -1).shape

        fitter = AlfQLeaceFitter(d, k, device=x.device, dtype=x.dtype, **kwargs)
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
        self.method = method
        self.shrinkage = shrinkage

        assert svd_tol > 0.0, "`svd_tol` must be positive for numerical stability."
        self.svd_tol = svd_tol

        self.global_mean_x = torch.zeros(x_dim, device=device, dtype=dtype)
        self.global_mean_z = torch.zeros(z_dim, device=device, dtype=dtype)

        self.global_n = torch.tensor(0, device=device)
        self.sigma_xz_ = torch.zeros(x_dim, z_dim, device=device, dtype=dtype)

        self.sigma_xx_ = torch.zeros(x_dim, x_dim, device=device, dtype=dtype)

        self.mean_x = torch.zeros(z_dim, x_dim, device=device, dtype=dtype)
        self.n = torch.zeros(z_dim, device=device)
        self.sigma_xx_z_ = torch.zeros(z_dim, x_dim, x_dim, device=device, dtype=dtype)

    @torch.no_grad()
    @invalidates_cache("eraser")
    def update(self, x: Tensor, z: Tensor) -> "AlfQLeaceFitter":
        """Update the running statistics with a new batch of data."""

        # Update the QLEACE-specific statistics
        x_for_quadratic = x.flatten(0, -2).type_as(self.mean_x)
        label_encoded_z = torch.argmax(z, dim=1)
        for label, group in groupby(x_for_quadratic, label_encoded_z, dim=0):
            self.update_single(group, label)

        # Update the LEACE statistics
        d, c = self.sigma_xz_.shape

        x = x.reshape(-1, d).type_as(self.global_mean_x)

        n, d2 = x.shape

        assert d == d2, f"Unexpected number of features {d2}"
        self.global_n += n

        # Welford's online algorithm
        delta_x = x - self.global_mean_x
        self.global_mean_x += delta_x.sum(dim=0) / self.global_n
        delta_x2 = x - self.global_mean_x

        # Update the covariance matrix of X if needed (for LEACE)
        if self.method == "leace":
            assert self.sigma_xx_ is not None
            self.sigma_xx_.addmm_(delta_x.mH, delta_x2)

        z = z.reshape(n, -1).type_as(x)
        assert z.shape[-1] == c, f"Unexpected number of classes {z.shape[-1]}"

        delta_z = z - self.global_mean_z
        self.global_mean_z += delta_z.sum(dim=0) / self.global_n
        delta_z2 = z - self.global_mean_z

        # Update the cross-covariance matrix
        self.sigma_xz_.addmm_(delta_x.mH, delta_z2)

        return self

    @torch.no_grad()
    @invalidates_cache("eraser")
    def update_single(self, x: Tensor, z: int) -> "AlfQLeaceFitter":
        """Update the running statistics with `x`, all sampled from class `z`."""
        x = x.flatten(0, -2).type_as(self.mean_x)

        self.n[z] += len(x)

        # Welford's online algorithm
        delta_x = x - self.mean_x[z]
        self.mean_x[z] += delta_x.sum(dim=0) / self.n[z]
        delta_x2 = x - self.mean_x[z]

        self.sigma_xx_z_[z].addmm_(delta_x.mH, delta_x2)

        return self

    @cached_property
    def eraser(self) -> AlfQLeaceEraser:
        """Erasure function lazily computed given the current statistics."""
        eye = torch.eye(
            self.x_dim, device=self.global_mean_x.device, dtype=self.global_mean_x.dtype
        )

        # Compute QLEACE component
        # Compute the (covariance - mean covariance) matrix difference for each class
        self.sigma_xx_z_.shape
        mean_sigma_xx_z = self.sigma_xx_z_.mean(dim=0)
        sigma_xx_z_diffs = self.sigma_xx_z_ - mean_sigma_xx_z

        # Find the class that has the difference with the largest singular
        # value (spectral norm)
        svds: list[tuple[Tensor, Tensor, Tensor]] = [
            torch.svd_lowrank(sigma_xx_z_diffs[i], q=1) for i in range(self.z_dim)
        ]
        spectral_norms = torch.stack([svd[1][0] for svd in svds])
        z_idx = spectral_norms.argmax()

        # Select the principal direction associated with the singular value
        U, S, Vh = svds[z_idx]
        principal_direction = U[:, 0]

        # Projection collapses the principal direction
        proj_qleace = eye - torch.outer(principal_direction, principal_direction)

        assert torch.isclose(
            principal_direction.norm(p=2), torch.tensor(1.0), rtol=1e-5
        )
        assert torch.allclose(proj_qleace @ proj_qleace, proj_qleace, rtol=1e-5)
        del proj_qleace

        # Compute LEACE component
        # Compute the whitening and unwhitening matrices
        sigma = self.sigma_xx

        # Find the transformation that minimizes
        L, V = torch.linalg.eigh(sigma)

        # Threshold used by torch.linalg.pinv
        mask = L > (L[-1] * sigma.shape[-1] * torch.finfo(L.dtype).eps)

        # Assuming PSD; account for numerical error
        L.clamp_min_(0.0)

        W = V * torch.where(mask, L.rsqrt(), 0.0) @ V.mH
        W_inv = V * torch.where(mask, L.sqrt(), 0.0) @ V.mH

        u, s, _ = torch.linalg.svd(W @ self.sigma_xz, full_matrices=False)

        # Throw away singular values that are too small
        u *= s > self.svd_tol

        proj_left = W_inv @ u
        proj_right = u.mH @ W

        if self.constrain_cov_trace:
            P = eye - proj_left @ proj_right

            # Prevent the covariance trace from increasing
            sigma = self.sigma_xx
            old_trace = torch.trace(sigma)
            new_trace = torch.trace(P @ sigma @ P.mH)

            # If applying the projection matrix increases the variance, this might
            # cause instability, especially when erasure is applied multiple times.
            # We regularize toward the orthogonal projection matrix to avoid this.
            if new_trace.real > old_trace.real:
                Q = eye - u @ u.mH

                # Set up the variables for the quadratic equation
                x = new_trace
                y = 2 * torch.trace(P @ sigma @ Q.mH)
                z = torch.trace(Q @ sigma @ Q.mH)
                w = old_trace

                # Solve for the mixture of P and Q that makes the trace equal to the
                # trace of the original covariance matrix
                discr = torch.sqrt(
                    4 * w * x - 4 * w * y + 4 * w * z - 4 * x * z + y**2
                )
                alpha1 = (-y / 2 + z - discr / 2) / (x - y + z)
                alpha2 = (-y / 2 + z + discr / 2) / (x - y + z)

                # Choose the positive root
                alpha = torch.where(alpha1.real > 0, alpha1, alpha2).clamp(0, 1)
                P = alpha * P + (1 - alpha) * Q

                # TODO: Avoid using SVD here
                u, s, vh = torch.linalg.svd(eye - P)
                proj_left = u * s.sqrt()
                proj_right = vh * s.sqrt()

        return AlfQLeaceEraser(
            proj_left,
            proj_right,
            bias=self.global_mean_x if self.affine else None,
            alf_qleace_vec=principal_direction,
        )

    @property
    def sigma_xx(self) -> Tensor:
        """The covariance matrix of X."""
        assert self.global_n > 1, "Call update() before accessing sigma_xx"
        assert (
            self.sigma_xx_ is not None
        ), "Covariance statistics are not being tracked for X"

        # Accumulated numerical error may cause this to be slightly non-symmetric
        S_hat = (self.sigma_xx_ + self.sigma_xx_.mH) / 2

        # Apply Random Matrix Theory-based shrinkage
        if self.shrinkage:
            return optimal_linear_shrinkage(
                S_hat / self.global_n, self.global_n, inplace=True
            )

        # Just apply Bessel's correction
        else:
            return S_hat / (self.global_n - 1)

    @property
    def sigma_xz(self) -> Tensor:
        """The cross-covariance matrix."""
        assert self.global_n > 1, "Call update() with labels before accessing sigma_xz"
        return self.sigma_xz_ / (self.global_n - 1)
