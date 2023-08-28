import pytest
import torch

from concept_erasure import groupby


@pytest.mark.parametrize("stable", [True, False])
def test_groupby(stable: bool):
    n, k, d = 1000, 3, 10

    x = torch.randn(n, d)
    z = torch.randint(0, k, (n,))
    grouped = groupby(x, z, stable=stable)

    # Check that we can coalesce the groups back together
    torch.testing.assert_allclose(x, grouped.coalesce())
    torch.testing.assert_allclose(
        x + 1,
        grouped.map(lambda _, g: g + 1).coalesce(),
    )

    # We only expect these to be the same when stable=True
    if stable:
        for label in torch.unique(z):
            torch.testing.assert_allclose(
                x[z == label],
                grouped.groups[label],
            )
