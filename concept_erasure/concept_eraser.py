from typing import Literal

import torch
from torch import Tensor, nn


class ConceptEraser(nn.Module):
    """Minimally edit features to make specified concepts linearly undetectable."""

    mean_x: Tensor
    """Running mean of X."""

    mean_y: Tensor
    """Running mean of Y."""

    u: Tensor
    """Orthonormal basis of the subspace to remove."""

    xcov_M2: Tensor
    """Unnormalized cross-covariance matrix X^T Y."""

    x_M2: Tensor | None
    """Unnormalized covariance matrix X^T X."""

    n: Tensor
    """Number of samples seen so far."""

    _P: Tensor | None
    """Cached projection matrix, computed lazily."""

    @classmethod
    def fit(
        cls,
        x: Tensor,
        y: Tensor,
        cov_type: Literal["eye", "diag", "full"] = "full",
        rank: int | None = None,
    ) -> "ConceptEraser":
        """Convenience method to fit a ConceptEraser on data and return it."""
        n, d = x.shape
        _, k = y.reshape(n, -1).shape

        return cls(
            d, k, cov_type=cov_type, device=x.device, dtype=x.dtype, rank=rank
        ).update(x, y)

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        cov_type: Literal["eye", "diag", "full"] = "full",
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        rank: int | None = None,
    ):
        super().__init__()

        self.y_dim = y_dim
        self.x_dim = x_dim
        self.cov_type = cov_type
        self.rank = rank or y_dim

        self.register_buffer("mean_x", torch.zeros(x_dim, device=device, dtype=dtype))
        self.register_buffer("mean_y", self.mean_x.new_zeros(y_dim))
        self.register_buffer("u", self.mean_x.new_zeros(x_dim, self.rank))
        self.register_buffer(
            "xcov_M2",
            self.mean_x.new_zeros(x_dim, y_dim),
        )
        self.register_buffer("n", torch.tensor(0, device=device, dtype=dtype))
        if self.cov_type == "full":
            M2 = self.mean_x.new_zeros(x_dim, x_dim)
        elif self.cov_type == "diag":
            M2 = self.mean_x.new_zeros(x_dim)
        elif self.cov_type == "eye":
            M2 = None
        else:
            raise ValueError(f"Unknown covariance type {self.cov_type}")

        self.register_buffer("x_M2", M2)

    def forward(self, x: Tensor) -> Tensor:
        """Minimally edit `x` to remove correlations with the target concepts.

        Args:
            x: Representations of shape (..., x_dim).

        Returns:
            The edited representations of shape (..., x_dim).
        """
        d, _ = self.xcov_M2.shape
        assert self.n > 0, "Call update() before forward()"
        assert x.shape[-1] == d

        if self.cov_type == "eye":
            # Remove the subspace. We want to make sure we do this in a way that keeps
            # the unconditional mean of the data exactly the same.
            delta = (x - self.mean_x) @ self.u @ self.u.mT
            return x - delta
        else:
            return x @ self.P.T + (self.mean_x - self.mean_x @ self.P.T)

    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor) -> "ConceptEraser":
        """Update the running statistics with a new batch of data."""
        d, c = self.xcov_M2.shape
        x = x.reshape(-1, d).type_as(self.mean_x)

        n, d2 = x.shape
        assert d == d2, f"Unexpected number of features {d2}"

        # y might start out 1D, but we want to treat it as 2D
        y = y.reshape(n, -1).type_as(x)
        assert y.shape[-1] == c, f"Unexpected number of classes {y.shape[-1]}"

        self.n += n

        # Welford's online algorithm
        delta_x = x - self.mean_x
        self.mean_x += delta_x.sum(dim=0) / self.n
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

        delta_y = y - self.mean_y
        self.mean_y += delta_y.sum(dim=0) / self.n
        delta_y2 = y - self.mean_y

        self.xcov_M2.addmm_(delta_x.mT, delta_y2)

        # Invalidate the cached projection matrix
        self._P = None
        if self.y_dim == self.rank:
            # When we're entirely erasing the subspace, we can use QR instead of SVD to
            # get an orthonormal basis for the column space of the xcov matrix
            self.u, _ = torch.linalg.qr(self.xcov)
        else:
            # We only want to erase the highest energy part of the subspace
            self.u, _, _ = torch.svd_lowrank(self.xcov, q=self.rank)

        return self

    @property
    def P(self) -> Tensor:
        """Projection matrix for removing the subspace."""
        # Check if we've cached this result before
        if self._P is not None:
            return self._P

        # Compute the orthogonal projection matrix w.r.t. the Euclidean inner product
        eye = torch.eye(self.x_dim, device=self.u.device, dtype=self.u.dtype)
        Q = eye - self.u @ self.u.mT

        # We're not keeping track of covariance statistics, so we just use Q directly
        if self.cov_type == "eye":
            self._P = Q

        # Adjust Q to account for the covariance of X
        else:
            # Full formula: P = Σ (Q Σ Q)^+
            sigma = self.cov_x.diag_embed() if self.cov_type == "diag" else self.cov_x
            self._P = sigma @ torch.linalg.pinv(Q @ sigma @ Q, hermitian=True)

        return self._P

    @property
    def cov_x(self) -> Tensor:
        """The covariance matrix of X, or its diagonal if `cov_type == 'diag'`."""
        assert self.n > 0, "Call update() before accessing cov_x"
        assert (
            self.x_M2 is not None
        ), "Can't compute covariance matrix for cov_type='eye'"

        return self.x_M2 / self.n

    @property
    def xcov(self) -> Tensor:
        """The cross-covariance matrix."""
        assert self.n > 0, "Call update() before accessing cov_x"
        return self.xcov_M2 / self.n
