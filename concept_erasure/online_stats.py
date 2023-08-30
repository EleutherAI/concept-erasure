from copy import deepcopy
from typing import TypeVar

import torch
import torch.distributed as dist
from torch import Tensor

from .shrinkage import optimal_linear_shrinkage

# Do this manually until we bump the minimum Python version to 3.11
# https://docs.python.org/3/library/typing.html#typing.Self
Self = TypeVar("Self", bound="OnlineStats")


class OnlineStats:
    """Numerically stable, online mean & (cross-)covariance matrix estimation."""

    mean_x: Tensor
    """Running mean of X."""

    sigma_xx_: Tensor | None
    """Unnormalized covariance matrix of X."""

    mean_z: Tensor | None
    """Running mean of Z."""

    sigma_zz_: Tensor | None
    """Unnormalized covariance matrix of Z."""

    sigma_xz_: Tensor | None
    """Unnormalized cross-covariance matrix of X and Z."""

    n: Tensor
    """Number of samples seen so far."""

    def __init__(
        self,
        x_dim: int,
        z_dim: int | None = None,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        shrinkage: bool = False,
        sigma_xx: bool = True,
    ):
        """Initialize an `OnlineStats` object.

        Args:
            x_dim: Dimensionality of the X variable.
            z_dim: Dimensionality of the Z variable, if applicable.
            device: Device to put the statistics on.
            dtype: Data type to use for the statistics.
            shrinkage: Whether to use shrinkage to estimate the covariance matrix of X.
            sigma_xx: Whether to compute the covariance matrix of X.
        """
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.shrinkage = shrinkage

        self.n = torch.tensor(0, device=device, dtype=dtype)
        self.mean_x = torch.zeros(x_dim, device=device, dtype=dtype)
        if z_dim is not None:
            self.mean_z = torch.zeros(z_dim, device=device, dtype=dtype)
        else:
            self.mean_z = None

        if sigma_xx:
            self.sigma_xx_ = torch.zeros(x_dim, x_dim, device=device, dtype=dtype)
        else:
            self.sigma_xx_ = None

        # Only allocate memory for Z statistics if Z is present.
        if z_dim is not None:
            self.sigma_zz_ = torch.zeros(z_dim, z_dim, device=device, dtype=dtype)
            self.sigma_xz_ = torch.zeros(x_dim, z_dim, device=device, dtype=dtype)
        else:
            self.sigma_zz_ = None
            self.sigma_xz_ = None

    @property
    def sigma_xx(self) -> Tensor:
        """Normalized covariance matrix of X."""
        if self.sigma_xx_ is None:
            raise RuntimeError("sigma_xx is not available.")

        if self.shrinkage:
            return optimal_linear_shrinkage(self.sigma_xx_ / self.n, self.n)
        else:
            return self.sigma_xx_ / (self.n - 1)

    @property
    def sigma_xz(self) -> Tensor | None:
        """Normalized cross-covariance matrix of X and Z."""
        return self.sigma_xz_ / (self.n - 1) if self.sigma_xz_ is not None else None

    @property
    def sigma_zz(self) -> Tensor:
        """Normalized covariance matrix of Z."""
        if self.sigma_zz_ is None:
            raise RuntimeError("sigma_zz is not available.")

        if self.shrinkage:
            return optimal_linear_shrinkage(self.sigma_zz_ / self.n, self.n)
        else:
            return self.sigma_zz_ / (self.n - 1)

    def all_reduce(self: Self) -> "Self":
        """Reduce the statistics across all ranks."""

        # We don't do this in-place because we don't want to accidentally inflate the
        # sample size if this is called multiple times.
        pooled = deepcopy(self)

        # Don't assume that all ranks have the same number of samples
        dist.all_reduce(pooled.n)

        # Fast path: only computing means, not (cross-)covariances. Here we can
        # do everything with all_reduce and don't need the overhead of all_gather.
        if pooled.sigma_xx_ is None:
            pooled.mean_x *= self.n
            dist.all_reduce(pooled.mean_x)
            pooled.mean_x /= pooled.n

            if pooled.mean_z is not None:
                pooled.mean_z *= self.n
                dist.all_reduce(pooled.mean_z)
                pooled.mean_z /= pooled.n

            return pooled

        # Parallel computation of the (cross-)covariance matrix is tricky. Wikipedia
        # cites Schubert & Gertz (2018), who propose a pairwise method. But extending
        # this to arbitrary world sizes would require implementing a complex
        # tree-reduction scheme. We instead use the law of total covariance:
        #   cov(X, Z) = E[cov(X, Z) | rank] + cov(E[X | rank], E[Z | rank])
        # Since we store the unnormalized covariance, both sides are multiplied by n.
        dist.all_reduce(pooled.sigma_xx_)
        num_P = dist.get_world_size()  # Number of partitions
        N = pooled.n.item()  # Total number of samples

        # Collect all local means
        local_x_means = all_gather(pooled.mean_x)

        # Compute global mean
        pooled.mean_x = local_x_means.mean(dim=0)

        # Compute the deviation of each local mean from the global mean
        local_x_means -= pooled.mean_x
        pooled.sigma_xx_.addmm_(
            # Divide by world size to normalize, then multiply by the global sample
            # size because sigma_xx_ is unnormalized.
            local_x_means.mT,
            local_x_means,
            alpha=N / num_P,
        )

        if pooled.mean_z is not None:
            assert pooled.sigma_xz_ is not None
            assert pooled.sigma_zz_ is not None
            dist.all_reduce(pooled.sigma_zz_)

            local_z_means = all_gather(pooled.mean_z)
            pooled.mean_z = local_z_means.mean(dim=0)

            local_z_means -= pooled.mean_z
            pooled.sigma_zz_.addmm_(local_z_means.mT, local_z_means, alpha=N / num_P)

            # Now cross-covariance
            dist.all_reduce(pooled.sigma_xz_)
            pooled.sigma_xz_.addmm_(local_x_means.mT, local_z_means, alpha=N / num_P)

        return pooled

    @torch.no_grad()
    def update(self: Self, x: Tensor, z: Tensor | None = None) -> "Self":
        """Update statistics with a batch of data.

        Args:
            x: Batch of data to update the covariance matrix estimate with.
            y: Batch of labels to update the covariance matrix estimate with.

        Returns:
            The updated covariance matrix estimate.
        """
        assert (z is None) == (self.mean_z is None), "y must be present iff mean_z is."

        n, *_, d = x.shape
        assert d == self.x_dim, f"x should have dimension {self.x_dim}, got {d}."

        # Update sample size
        self.n += n

        # Welford's online algorithm for (co-)variance and mean
        delta_x = x - self.mean_x
        self.mean_x += delta_x.sum(dim=0) / self.n

        # Update X covariance if needed
        if self.sigma_xx_ is not None:
            delta_x2 = x - self.mean_x

            # Fuse the outer product and addition to save memory
            self.sigma_xx_.addmm_(delta_x.mT, delta_x2)

        # Update Z statistics if Z is present
        if z is not None:
            assert self.z_dim is not None
            z = z.reshape(-1, self.z_dim)

            # Sanity checks
            n2 = z.shape[0]
            assert n == n2, f"x and y should have same batch size, got {n} and {n2}."
            assert self.sigma_zz_ is not None
            assert self.sigma_xz_ is not None

            delta_z = z - self.mean_z
            self.mean_z += delta_z.sum(dim=0) / self.n
            delta_z2 = z - self.mean_z

            self.sigma_zz_.addmm_(delta_z.mT, delta_z2)
            self.sigma_xz_.addmm_(delta_x.mT, delta_z2)

        return self


def all_gather(x: Tensor) -> Tensor:
    """Gather a tensor from all ranks, and return the concatenated result."""
    world_size = dist.get_world_size()

    # Gloo doesn't support all_gather_into_tensor
    if dist.get_backend() == "gloo":
        tensor_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(tensor_list, x)
        return torch.stack(tensor_list, dim=0)

    buffer = x.new_empty(world_size, *x.shape)
    dist.all_gather_into_tensor(buffer, x)
    return buffer
