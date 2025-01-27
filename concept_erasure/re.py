import torch
from torch import Tensor


class RandomEraser:
    """Random eraser that projects a random subspace to the nullspace."""

    basis: Tensor

    def __init__(self, ndims: int, erase_dims: int, **kwargs):
        """Create a RandomEraser that projects erase_dims dimensions to zero.

        Args:
            ndims: Total dimensions of the input space
            erase_dims: Number of dimensions to project to zero
        """
        # Create a random orthonormal basis
        rand_basis = torch.randn(ndims, ndims)
        Q, R = torch.linalg.qr(rand_basis)

        # Take the first erase_dims columns to get basis for subspace to nullify
        Q = Q[:, :erase_dims]

        self.basis = Q

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the projection to the input tensor."""
        result = x - (x @ self.basis) @ self.basis.T
        torch.testing.assert_close(
            result, result - (result @ self.basis) @ self.basis.T
        )
        return result

    def to(self, device: torch.device | str) -> "RandomEraser":
        """Move eraser to a new device."""
        self.basis = self.basis.to(device)
        return self
