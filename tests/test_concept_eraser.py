from itertools import product

import pytest
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from concept_erasure import ConceptEraser, gaussian_shrinkage


@pytest.mark.parametrize("shrinkage", [False, True])
def test_stats(shrinkage: bool):
    batch_size = 10
    num_batches = 5
    num_classes = 2
    num_features = 3
    N = batch_size * num_batches

    eraser = ConceptEraser(num_features, num_classes, shrinkage=shrinkage)

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

    expected_cov = torch.einsum("b...m,b...n->...mn", x_centered, x_centered)
    if shrinkage:
        expected_cov = gaussian_shrinkage(expected_cov / N, batch_size * num_batches)
    else:
        expected_cov /= N - 1

    expected_sigma_xz = torch.einsum("b...m,b...n->...mn", x_centered, y_centered)
    expected_sigma_xz /= N - 1

    torch.testing.assert_close(eraser.sigma_xx, expected_cov)
    torch.testing.assert_close(eraser.sigma_xz, expected_sigma_xz)


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
        Y_1h = torch.nn.functional.one_hot(Y_t, num_classes)
    else:
        Y_1h = Y_t

    bools = [False, True]
    eps = 2e-9
    for affine, method, shrink in product(bools, ["leace", "orth"], bools):
        # Shrinkage not applicable to orthogonal projection
        if method == "orth" and shrink:
            continue

        eraser = ConceptEraser.fit(
            X_t, Y_1h, affine=affine, method=method, shrinkage=shrink
        )
        X_ = eraser(X_t)

        # Check first-order optimality condition. To account for the nullspace
        # constraint, we use the canonical form for oblique projection matrices,
        # fixing the nullspace and only optimizing over the range space.
        A, s, B_t = torch.linalg.svd(eraser.P)
        A, B = A[:, s > 0.5].requires_grad_(True), B_t[s > 0.5].T

        # See "A matrix representation formula for a nonzero projection operator" in
        # https://en.wikipedia.org/wiki/Projection_(linear_algebra)
        P = A @ torch.inverse(B.T @ A) @ B.T
        x_ = (X_t - eraser.mean_x) @ P.T + eraser.mean_x if affine else X_t @ P.T

        # Define a random positive definite inner product
        L = torch.randn(d, d, dtype=X_t.dtype) / d**0.5
        loss = F.mse_loss(x_ @ L, X_t @ L)
        loss.backward()

        # Should be optimal iff we're using LEACE with the bias term and no shrinkage
        assert (A.grad.norm() < eps) == (affine and method == "leace" and not shrink)

        # Check idempotence
        torch.testing.assert_close(eraser(X_), X_)

        # Check that the rank of the update <= num_classes
        rank = torch.linalg.svdvals(X_t - X_).gt(eps).sum()
        assert rank <= num_classes

        # Check that the unconditional mean is unchanged
        if affine:
            torch.testing.assert_close(X_t.mean(dim=0), X_.mean(dim=0))

        # Compute class means and check that they are equal after the projection
        class_means_ = torch.stack(
            [X_[Y_t == c].mean(dim=0) for c in range(num_distinct)]
        )
        torch.testing.assert_close(class_means_[1:], class_means_[:-1])

        # Sanity check that class means are NOT equal before the projection
        class_means = torch.stack(
            [X_t[Y_t == c].mean(dim=0) for c in range(num_distinct)]
        )
        assert not torch.allclose(class_means[1:], class_means[:-1])

        # Logistic regression should not be able to learn anything.
        null_lr = LogisticRegression(penalty=None, tol=0.0).fit(
            # Weirdly, in order for this to work consistently with solver='lbfgs', we
            # need to center the design matrix first; otherwise the coefficients aren't
            # quite small enough. Other solvers don't have this problem but they're
            # slow. In theory, the centroid of X shouldn't matter to the solution.
            X_.numpy() - X_.numpy().mean(axis=0),
            Y,
        )
        assert abs(null_lr.coef_).max() < eps

        # Sanity check that it DOES learn something before the projection
        real_lr = LogisticRegression(penalty=None, tol=0.0).fit(
            # Do the same centering operation here to be consistent
            X - X.mean(axis=0),
            Y,
        )
        assert abs(real_lr.coef_).max() > 0.1

        # Linear SVM shouldn't be able to learn anything either
        null_svm = LinearSVC(
            # The dual formulation injects random noise into the solution
            dual=False,
            # Unfortunately the intercept is subject to L2 regularization; setting this
            # to a large value largely cancels out the effect. Regularizing the
            # intercept can cause the coefficients to get larger than they should be.
            intercept_scaling=1e6,
            tol=eps,
        ).fit(X_.numpy(), Y)
        assert abs(null_svm.coef_).max() < eps

        # But it should learn something before the projection
        real_svm = LinearSVC(dual=False, intercept_scaling=1e6, tol=eps).fit(X, Y)
        assert abs(real_svm.coef_).max() > 0.1
