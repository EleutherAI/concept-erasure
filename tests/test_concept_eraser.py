import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from concept_erasure import ConceptEraser, to_one_hot


def test_stats():
    num_features = 3
    num_classes = 2
    batch_size = 10
    num_batches = 5

    # Initialize the ConceptEraser
    eraser = ConceptEraser(num_features, num_classes)

    # Generate random data
    torch.manual_seed(42)
    x_data = [torch.randn(batch_size, num_features) for _ in range(num_batches)]
    y_data = [
        torch.randint(0, num_classes, (batch_size, num_classes))
        for _ in range(num_batches)
    ]

    # Compute cross-covariance matrix using batched updates
    for x, y in zip(x_data, y_data):
        eraser.update(x, y)

    # Compute the expected cross-covariance matrix using the whole dataset
    x_all = torch.cat(x_data)
    y_all = torch.cat(y_data)
    mean_x = x_all.mean(dim=0)
    mean_y = y_all.type_as(x_all).mean(dim=0)
    x_centered = x_all - mean_x
    y_centered = y_all - mean_y
    expected_xcov = torch.einsum("b...m,b...n->...mn", x_centered, y_centered)
    expected_xcov /= batch_size * num_batches

    # Compare the computed cross-covariance matrix with the expected one
    torch.testing.assert_close(eraser.xcov, expected_xcov)


# Both `1` and `2` are binary classification problems, but `1` means the labels are
# encoded in a 1D one-hot vector, while `2` means the labels are encoded in an
# n x 2 one-hot matrix.
@pytest.mark.parametrize("num_classes", [1, 2, 3, 5, 10, 20])
def test_projection(num_classes: int):
    n, d = 2048, 128
    num_distinct = max(num_classes, 2)

    X, Y = make_classification(
        n_samples=n,
        n_features=d,
        n_classes=num_distinct,
        n_informative=num_distinct,
        random_state=42,
    )
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)
    if num_classes > 1:
        Y_t = to_one_hot(Y_t, num_classes)

    eraser = ConceptEraser(d, num_classes, dtype=torch.float64).update(X_t, Y_t)
    X_ = eraser(X_t)

    # Heuristic threshold for singular values taken from torch.linalg.pinv
    eps = max(n, d) * torch.finfo(X_.dtype).eps

    # Check that the rank of the update is num_classes + 1
    # The +1 comes from subtracting the mean before projection
    rank = torch.linalg.svdvals(X_t - X_).gt(eps).sum().float()
    torch.testing.assert_close(rank, torch.tensor(num_classes + 1.0))

    # Compute class means and check that they are equal after the projection
    class_means_ = [X_.numpy()[Y == c].mean(axis=0) for c in range(num_distinct)]
    np.testing.assert_almost_equal(class_means_[1:], class_means_[:-1])

    # Sanity check that class means are NOT equal before the projection
    class_means = [X[Y == c].mean(axis=0) for c in range(num_distinct)]
    assert not np.allclose(class_means[1:], class_means[:-1])

    # Logistic regression should not be able to learn anything
    null_lr = LogisticRegression(max_iter=1000, tol=0.0).fit(X_.numpy(), Y)
    beta = torch.from_numpy(null_lr.coef_)
    assert beta.norm(p=torch.inf) < eps

    # Sanity check that it DOES learn something before the projection
    real_lr = LogisticRegression(max_iter=1000).fit(X, Y)
    beta = torch.from_numpy(real_lr.coef_)
    assert beta.norm(p=torch.inf) > 0.1
