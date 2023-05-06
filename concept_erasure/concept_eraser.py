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

    n: Tensor
    """Number of samples seen so far."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        rank: int | None = None,
    ):
        super().__init__()

        self.y_dim = y_dim
        self.x_dim = x_dim
        self.rank = rank or y_dim

        self.register_buffer("mean_x", torch.zeros(x_dim, device=device, dtype=dtype))
        self.register_buffer("mean_y", self.mean_x.new_zeros(y_dim))
        self.register_buffer("u", self.mean_x.new_zeros(x_dim, self.rank))
        self.register_buffer(
            "xcov_M2",
            self.mean_x.new_zeros(x_dim, y_dim),
        )
        self.register_buffer("n", torch.tensor(0, device=device, dtype=dtype))

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

        # First center the input
        x_ = x - self.mean_x
        # Then remove the subspace
        x_ -= x_ @ self.u @ self.u.mT

        return x_

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

        delta_y = y - self.mean_y
        self.mean_y += delta_y.sum(dim=0) / self.n
        delta_y2 = y - self.mean_y

        self.xcov_M2 += torch.einsum("b...m,b...n->...mn", delta_x, delta_y2)
        if self.y_dim == self.rank:
            # When we're entirely erasing the subspace, we can use QR instead of SVD to
            # get an orthonormal basis for the column space of the xcov matrix
            self.u, _ = torch.linalg.qr(self.xcov)
        else:
            # We only want to erase the highest energy part of the subspace
            self.u, _, _ = torch.svd_lowrank(self.xcov, q=self.rank)

        return self

    # When there's only one batch of data, "fit" is a synonym for "update"
    fit = update

    @property
    def P(self) -> Tensor:
        """Projection matrix for removing the subspace."""
        eye = torch.eye(self.x_dim, device=self.u.device, dtype=self.u.dtype)
        return eye - self.u @ self.u.mT

    @property
    def xcov(self) -> Tensor:
        """The cross-covariance matrix."""
        return self.xcov_M2 / self.n
