from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from torch import LongTensor, Tensor


@dataclass(frozen=True)
class GroupedTensor:
    """A tensor split into groups along a given dimension.

    This class contains all the information needed to reconstruct the original tensor,
    or to take a list of tensors derived from the groups and coalesce them in such a
    way that the original order is restored.
    """

    dim: int
    """Dimension along which the tensor was split."""

    groups: list[Tensor]
    """List of tensors such that `groups[i]` contains all elements of `x` whose group
    label is `labels[i]`."""

    indices: LongTensor
    """Indices used to sort the original tensor."""

    labels: list[int]
    """Unique label for each element of `groups`."""

    def coalesce(self, groups: list[Tensor] | None = None) -> Tensor:
        """Fuse `groups or self.groups` back together, restoring the original order.

        This method is most useful when you want to group a tensor, perform an operation
        on each group, then combine the results back together.
        """
        if groups is None:
            groups = self.groups

        # First concatenate the groups back together
        fused = torch.cat(groups, dim=self.dim)

        # Invert the permutation to restore the original order
        return fused.index_select(self.dim, invert_indices(self.indices))

    def map(self, fn: Callable[[int, Tensor], Tensor]) -> "GroupedTensor":
        """Apply `fn` to each group & return a new `GroupedTensor` with the results."""
        results = [fn(label, group) for label, group in zip(self.labels, self.groups)]
        return GroupedTensor(self.dim, results, self.indices, self.labels)

    def __iter__(self) -> Iterator[tuple[int, Tensor]]:
        """Iterate over the groups and their labels."""
        for label, group in zip(self.labels, self.groups):
            yield label, group


def groupby(
    x: Tensor, key: Tensor, dim: int = 0, *, stable: bool = False
) -> GroupedTensor:
    """Efficiently split `x` into groups along `dim` according to `key`.

    This function is intended to mimic the behavior of `itertools.groupby`, but for
    PyTorch tensors. Under the hood, we sort `x` by `key` once, then return views
    onto the sorted tensor in order to minimize the number of memcpy and equality
    checking operations performed.

    By necessity this operation performs a host-device sync since we need to know
    the number of groups and their sizes in order to create a view for each.

    Args:
        x: Tensor to split into groups.
        key: Tensor of group labels.
        dim: Dimension along which to split `x`.
        stable: If `True`, use a stable sorting algorithm. This is slower but ensures
            that the order of elements within each group is preserved.

    Returns:
        A `GroupedTensor` containing the groups, sorting indices, and labels.
    """
    assert key.dtype == torch.int64, "`key` must be int64"
    assert key.ndim == 1, "`key` must be 1D"

    key, indices = key.sort(stable=stable)
    labels, counts = key.unique_consecutive(return_counts=True)

    # Sort `x` by `key` along `dim`
    x = x.index_select(dim, indices)
    groups = x.split(counts.tolist(), dim=dim)

    return GroupedTensor(dim, groups, indices, labels.tolist())


@torch.jit.script
def invert_indices(indices: Tensor) -> Tensor:
    """Efficiently invert the permutation represented by `indices`.

    Example:
        >>> indices = torch.tensor([2, 0, 1])
        >>> invert_indices(indices)
        tensor([1, 2, 0])
    """
    # Create an empty tensor to hold the reverse permutation
    reverse_indices = torch.empty_like(indices)

    # Scatter the indices to reverse the permutation
    reverse_indices.scatter_(0, indices, torch.arange(len(indices)))

    return reverse_indices
