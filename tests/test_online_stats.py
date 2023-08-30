import multiprocessing
import os

import torch
import torch.distributed as dist

from concept_erasure.online_stats import OnlineStats


def run_all_reduce_test(rank, world_size):
    # Initialize the distributed environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42)

    d, k, n = 5, 3, 10
    X = torch.randn(world_size, n, d)
    Y = torch.randn(world_size, n, k)

    stats = OnlineStats(d, k)
    for x, y in zip(X[rank], Y[rank]):
        stats.update(x[None], y[None])

    # Test serial streaming update
    torch.testing.assert_close(X[rank].mT.cov(), stats.sigma_xx)
    torch.testing.assert_close(Y[rank].mT.cov(), stats.sigma_zz)
    torch.testing.assert_close(X[rank].mean(dim=0), stats.mean_x)
    torch.testing.assert_close(Y[rank].mean(dim=0), stats.mean_z)

    # cross-covariance
    torch.testing.assert_close(
        (X[rank] - stats.mean_x).mT @ (Y[rank] - stats.mean_z) / (n - 1), stats.sigma_xz
    )

    # Test parallel all-reduce
    pooled = stats.all_reduce()
    X, Y = X.flatten(0, 1), Y.flatten(0, 1)
    torch.testing.assert_close(X.mT.cov(), pooled.sigma_xx)
    torch.testing.assert_close(Y.mT.cov(), pooled.sigma_zz)
    torch.testing.assert_close(X.mean(dim=0), pooled.mean_x)
    torch.testing.assert_close(Y.mean(dim=0), pooled.mean_z)

    # cross-covariance
    torch.testing.assert_close(
        (X - pooled.mean_x).mT @ (Y - pooled.mean_z) / (n * world_size - 1),
        pooled.sigma_xz,
    )

    # Finalize the distributed environment
    dist.destroy_process_group()


def test_all_reduce():
    world_size = 4  # Number of processes for distributed testing

    # Create a pool of processes and execute each one
    mp = multiprocessing.get_context("spawn")
    with mp.Pool(processes=world_size) as pool:
        pool.starmap(run_all_reduce_test, [(i, world_size) for i in range(world_size)])
