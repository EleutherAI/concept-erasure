from typing import Literal

import torch
from torch import Tensor, nn


class ConceptEraser(nn.Module):
    """Minimally edit features to make specified concepts linearly undetectable."""

    mean_x: Tensor
    """Running mean of X."""

    mean_y: Tensor
    """Running mean of Y."""

    xcov_M2: Tensor
    """Unnormalized cross-covariance matrix X^T Y."""

    x_M2: Tensor | None
    """Unnormalized covariance matrix X^T X."""

    n_x: Tensor
    """Number of X samples seen so far."""

    n_y: Tensor
    """Number of Y samples seen so far."""

    _P: Tensor | None
    """Cached projection matrix, computed lazily."""

    @classmethod
    def fit(cls, x: Tensor, y: Tensor, **kwargs) -> "ConceptEraser":
        """Convenience method to fit a ConceptEraser on data and return it."""
        n, d = x.shape
        _, k = y.reshape(n, -1).shape

        return cls(d, k, device=x.device, dtype=x.dtype, **kwargs).update(x, y)

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        cov_type: Literal["eye", "diag", "full"] = "full",
        *,
        affine: bool = True,
        clip_variances: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        rank: int | None = None,
        shrinkage: float = 0.0,
    ):
        super().__init__()

        self.y_dim = y_dim
        self.x_dim = x_dim

        self.affine = affine
        self.clip_variances = clip_variances
        self.cov_type = cov_type
        self.rank = rank or y_dim
        self.shrinkage = shrinkage

        self.register_buffer("mean_x", torch.zeros(x_dim, device=device, dtype=dtype))
        self.register_buffer("mean_y", self.mean_x.new_zeros(y_dim))
        self.register_buffer(
            "xcov_M2",
            self.mean_x.new_zeros(x_dim, y_dim),
        )
        self.register_buffer("n_x", torch.tensor(0, device=device, dtype=dtype))
        self.register_buffer("n_y", torch.tensor(0, device=device, dtype=dtype))

        if self.cov_type == "full":
            M2 = self.mean_x.new_zeros(x_dim, x_dim)
        elif self.cov_type == "diag":
            M2 = self.mean_x.new_zeros(x_dim)
        elif self.cov_type == "eye":
            M2 = None
        else:
            raise ValueError(f"Unknown covariance type {self.cov_type}")

        self.register_buffer("x_M2", M2)

    def clear_x(self):
        """Clear the running statistics of X."""
        self.n_x.zero_()
        self.mean_x.zero_()

        if self.x_M2 is not None:
            self.x_M2.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Minimally edit `x` to remove correlations with the target concepts.

        Args:
            x: Representations of shape (..., x_dim).

        Returns:
            The edited representations of shape (..., x_dim).
        """
        d, _ = self.xcov_M2.shape
        assert self.n_x > 0, "Call update() before forward()"
        assert x.shape[-1] == d

        if self.affine:
            return (x - self.mean_x) @ self.P.T + self.mean_x
        else:
            return x @ self.P.T

    def proj_for_subspace(self, u: Tensor) -> Tensor:
        """Compute MSE-optimal projection matrix given orthonormal basis `u`."""
        # Compute the orthogonal projection matrix w.r.t. the Euclidean inner product
        eye = torch.eye(self.x_dim, device=u.device, dtype=u.dtype)
        Q = eye - u @ u.mT

        # We're not keeping track of covariance statistics, so we just use Q directly
        if self.cov_type == "eye":
            return Q

        # Adjust Q to account for the covariance of X
        else:
            sigma = self.cov_x.diag_embed() if self.cov_type == "diag" else self.cov_x

            return sigma @ torch.linalg.pinv(Q @ sigma @ Q)

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor | None = None) -> "ConceptEraser":
        """Update the running statistics with a new batch of data.

        It's possible to call this method without `y` if you only want to update the
        statistics of X. This is useful if you don't have labels but want to adjust the
        mean and covariance of X to match a new dataset.
        """
        d, c = self.xcov_M2.shape
        x = x.reshape(-1, d).type_as(self.mean_x)

        n, d2 = x.shape
        assert d == d2, f"Unexpected number of features {d2}"

        # We always have an X, we might not have a Y
        self.n_x += n

        # Welford's online algorithm
        delta_x = x - self.mean_x
        self.mean_x += delta_x.sum(dim=0) / self.n_x
        delta_x2 = x - self.mean_x

        # Update the covariance matrix of X if needed
        if self.cov_type != "eye":
            assert self.x_M2 is not None

            # Keep track of the whole covariance matrix
            if self.cov_type == "full":
                self.x_M2.addmm_(delta_x.mT, delta_x2)
            # Only keep track of the diagonal to save memory
            elif self.cov_type == "diag":
                self.x_M2.add_(delta_x2.pow(2).sum(dim=0))

        # Invalidate the cached projection matrix
        self._P = None

        # We do have labels, so we can update the Y statistics
        if y is not None:
            # y might start out 1D, but we want to treat it as 2D
            y = y.reshape(n, -1).type_as(x)
            assert y.shape[-1] == c, f"Unexpected number of classes {y.shape[-1]}"

            self.n_y += n

            delta_y = y - self.mean_y
            self.mean_y += delta_y.sum(dim=0) / self.n_x
            delta_y2 = y - self.mean_y

            # Update the cross-covariance matrix
            self.xcov_M2.addmm_(delta_x.mT, delta_y2)

        return self

    @property
    def P(self) -> Tensor:
        """Projection matrix for removing the subspace."""
        # Compute the projection matrix if we haven't already
        if self._P is None:
            if self.y_dim == self.rank:
                # Get an orthonormal basis for the column space of the xcov matrix
                u, _ = torch.linalg.qr(self.xcov)
            else:
                # We only want to erase the highest energy part of the subspace
                u, _, _ = torch.svd_lowrank(self.xcov, q=self.rank)

            self._P = self.proj_for_subspace(u)

        return self._P

    @property
    def cov_x(self) -> Tensor:
        """The covariance matrix of X, or its diagonal if `cov_type == 'diag'`.

        Shrinkage and eigenvalue clipping may be applied to the covariance matrix
        in order to make it more well-behaved and robust to noise.
        """
        assert self.n_x > 1, "Call update() before accessing cov_x"
        assert (
            self.x_M2 is not None
        ), "Can't compute covariance matrix for cov_type='eye'"

        cov = self.x_M2 / (self.n_x - 1)

        # Apply "All But the Top" style pre-processing. We want to limit the influence
        # of the highest variance directions on the projection matrix, since this often
        # leads to poor performance in practice.
        if self.clip_variances:
            d = cov.shape[-1]
            thresh = d // 100  # Heuristic from the paper

            # Same as below but without the PCA
            if self.cov_type == "diag":
                sorted_vars, indices = cov.sort(descending=True)
                cov[indices[thresh:]] = sorted_vars[thresh]

            # We have to do PCA and then clip
            elif self.cov_type == "full":
                L, Q = torch.linalg.eigh(cov)

                # We don't want to literally zero out the highest eigenvalues
                # but we do want to clip how big they can be
                max_val = L[-thresh]
                L[-thresh:] = max_val

                cov = Q @ torch.diag_embed(L) @ Q.mT

        # Apply shrinkage toward the identity
        if self.shrinkage > 0.0:
            cov *= 1 - self.shrinkage

            if self.cov_type == "diag":
                cov.add_(self.shrinkage)
            elif self.cov_type == "full":
                torch.linalg.diagonal(cov).add_(self.shrinkage)

        return cov

    @property
    def xcov(self) -> Tensor:
        """The cross-covariance matrix."""
        assert self.n_y > 1, "Call update() with labels before accessing xcov"
        return self.xcov_M2 / (self.n_y - 1)
