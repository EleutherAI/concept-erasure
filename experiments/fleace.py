import torch
import torch.nn.functional as F


def v_transform(K: torch.Tensor):
    """ReLU(Kernel) V-transform as described in Tensor Programs I: Wide Feedforward or
    Recurrent Neural Networks of Any Architecture are Gaussian Processes https://arxiv.org/abs/1910.12478.
    """
    diag = torch.diagonal(K)
    scale = torch.sqrt(diag.unsqueeze(1) * diag.unsqueeze(0))
    c = K / scale

    return (
        (1 / (2 * torch.pi))
        * ((1 - c.pow(2)).sqrt() + (torch.pi - torch.acos(c)) * c)
        * scale
    )


def g(x: torch.Tensor, k: int, device="cpu"):
    """Apply `ReLU(x) + ε` to x with ε ~ N(0, 1) and return the first k dimensions."""
    e = torch.randn(x.shape, device=device)

    return (F.relu(x) + e)[..., :k]


def relu_eraser(x: torch.Tensor, n: int, k: int):
    """Use closed-form solution for free-form LEACE for ReLU to remove linearly accessible
    information about ReLU(x) from x as described in Non-Linear Least-Squares Concept Erasure.
    """
    # E[Z | X] where Z = ReLU(X) + ε and ε ~ N(0, 1)
    f = F.relu(x)[:, :k]

    # Closed form solution for E[X ReLU(X).T] in R^(n k)
    # Note: cross_cov is not centered
    cross_cov = torch.eye(n)[:, :k] * 0.5

    # Closed form solution for E[ReLU(X) ReLU(X).T] in R^(k k)
    V = v_transform(torch.eye(k))

    A = cross_cov @ torch.linalg.pinv(V)

    return x - (A @ f.T).T


def test_v_transform_monte_carlo():
    num_samples = 100_000
    dim = 10
    X = torch.randn((num_samples, dim))

    cov = (F.relu(X).T @ F.relu(X)) / (num_samples - 1)

    torch.testing.assert_close(cov, v_transform(torch.eye(10)), rtol=0.01, atol=0.01)


def test_relu_linear_erasure():
    batch_size = 2_000_000
    n, k = 16, 8

    x = torch.randn((batch_size, n))
    z = g(x, k)
    r_x = relu_eraser(x, n, k)

    assert torch.norm((r_x.T @ z) / (batch_size - 1)) < 0.01


if __name__ == "__main__":
    test_v_transform_monte_carlo()
    test_relu_linear_erasure()
