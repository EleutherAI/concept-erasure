from itertools import pairwise, product

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.svm import LinearSVC

from concept_erasure import (
    ErasureMethod,
    SqrtMethod,
    LeaceFitter,
    OracleEraser,
    OracleFitter,
    QuadraticEraser,
    optimal_linear_shrinkage,
)
from concept_erasure.psd_sqrt import is_positive_definite


@pytest.mark.parametrize("sqrt_method", ["legacy", "psd", "newton"])
@pytest.mark.parametrize("shrinkage", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_stats(sqrt_method: SqrtMethod, shrinkage: bool, dtype: torch.dtype):
    batch_size = 10
    num_batches = 5
    num_classes = 2
    num_features = 3
    N = batch_size * num_batches

    fitter = LeaceFitter(num_features, num_classes, dtype=dtype, shrinkage=shrinkage, sqrt_method=sqrt_method)
    oracle = OracleFitter(num_features, num_classes, dtype=dtype, shrinkage=shrinkage)

    # Generate random data
    torch.manual_seed(42)
    x_data = [
        torch.randn(batch_size, num_features, dtype=dtype) for _ in range(num_batches)
    ]
    z_data = [
        torch.randint(0, num_classes, (batch_size, num_classes))
        for _ in range(num_batches)
    ]
    projections = []

    # Compute cross-covariance matrix using batched updates
    for x, z in zip(x_data, z_data):
        fitter.update(x, z)
        oracle.update(x, z)
        projections.append(fitter.eraser.P)

    # Make sure the cached eraser is getting invalidated on update() correctly
    for p1, p2 in pairwise(projections):
        assert not torch.allclose(p1, p2)

    # Compute the expected cross-covariance matrix using the whole dataset
    x_all = torch.cat(x_data)
    z_all = torch.cat(z_data)
    mean_x = x_all.mean(dim=0)
    mean_z = z_all.type_as(x_all).mean(dim=0)
    x_centered = x_all - mean_x
    z_centered = z_all - mean_z

    expected_sigma_xx = torch.einsum(
        "b...m,b...n->...mn", x_centered.conj(), x_centered
    )
    expected_sigma_zz = torch.einsum(
        "b...m,b...n->...mn", z_centered.conj(), z_centered
    )
    if shrinkage:
        expected_sigma_xx = optimal_linear_shrinkage(
            expected_sigma_xx / N, batch_size * num_batches
        )
        expected_sigma_zz = optimal_linear_shrinkage(
            expected_sigma_zz / N, batch_size * num_batches
        )
    else:
        expected_sigma_xx /= N - 1
        expected_sigma_zz /= N - 1

    expected_sigma_xz = torch.einsum(
        "b...m,b...n->...mn", x_centered.conj(), z_centered
    )
    expected_sigma_xz /= N - 1

    torch.testing.assert_close(fitter.sigma_xx, expected_sigma_xx)
    torch.testing.assert_close(fitter.sigma_xz, expected_sigma_xz)

    torch.testing.assert_close(oracle.sigma_xz, expected_sigma_xz)
    torch.testing.assert_close(oracle.sigma_zz, expected_sigma_zz)


def check_linear_guardedness(
    dirty_x: np.ndarray | None, scrubbed_x: np.ndarray, y: np.ndarray, eps: float
):
    # Logistic regression should not be able to learn anything.
    null_lr = LogisticRegression(penalty=None, tol=0.0).fit(
        # Weirdly, in order for this to work consistently with solver='lbfgs', we
        # need to center the design matrix first; otherwise the coefficients aren't
        # quite small enough. Other solvers don't have this problem but they're
        # slow. In theory, the centroid of X shouldn't matter to the solution.
        scrubbed_x - scrubbed_x.mean(axis=0),
        y,
    )
    assert abs(null_lr.coef_).max() < eps

    # Linear SVM shouldn't be able to learn anything either
    null_svm = LinearSVC(
        # The dual formulation injects random noise into the solution
        dual=False,
        # Unfortunately the intercept is subject to L2 regularization; setting this
        # to a large value largely cancels out the effect. Regularizing the
        # intercept can cause the coefficients to get larger than they should be.
        intercept_scaling=1e6,
        tol=eps,
    ).fit(scrubbed_x, y)
    assert abs(null_svm.coef_).max() < eps

    if dirty_x is not None:
        # Sanity check that it DOES learn something before erasure
        real_lr = LogisticRegression(penalty=None, tol=0.0).fit(
            # Do the same centering operation here to be consistent
            dirty_x - dirty_x.mean(axis=0),
            y,
        )
        assert abs(real_lr.coef_).max() > 0.05

        real_svm = LinearSVC(dual=False, intercept_scaling=1e6, tol=eps).fit(dirty_x, y)
        assert abs(real_svm.coef_).max() > 0.05


# Both `1` and `2` are binary classification problems, but `1` means the labels are
# encoded in a 1D one-hot vector, while `2` means the labels are encoded in an
# n x 2 one-hot matrix.
@pytest.mark.parametrize("num_classes", [1, 2, 3, 5, 10, 20])
def test_linear_erasure(num_classes: int):
    # Error tolerance for numerical checks. This value is basically "the smallest I can
    # empirically make it while still passing the tests". Ideally I'd like to set it
    # using the machine epsilon or something, but sklearn convergence isn't quite
    # predictable enough to do that.
    eps = 4e-9

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

    # Benchmark against the theoretical optimal eraser
    oracle = OracleEraser.fit(X_t, Y_1h)
    X_oracle = oracle(X_t, Y_1h)

    check_linear_guardedness(None, X_oracle.numpy(), Y.reshape(n, -1), eps)
    oracle_loss = F.mse_loss(X_oracle, X_t)  # Optimal surgicality

    bools = [False, True]
    methods: list[ErasureMethod] = ["leace", "orth"]
    for affine, method, shrink in product(bools, methods, bools):
        # Shrinkage not applicable to orthogonal projection
        if method == "orth" and shrink:
            continue

        fitter = LeaceFitter(
            d,
            num_classes,
            affine=affine,
            dtype=X_t.dtype,
            method=method,
            shrinkage=shrink,
        ).update(X_t, Y_1h)

        eraser = fitter.eraser
        X_ = eraser(X_t)

        # Check first-order optimality condition. To account for the nullspace
        # constraint, we use the canonical form for oblique projection matrices,
        # fixing the nullspace and only optimizing over the range space.
        A, s, B_t = torch.linalg.svd(eraser.P)
        A, B = A[:, s > 0.5].requires_grad_(True), B_t[s > 0.5].T

        # See "A matrix representation formula for a nonzero projection operator" in
        # https://en.wikipedia.org/wiki/Projection_(linear_algebra)
        P = A @ torch.inverse(B.T @ A) @ B.T
        x_ = (X_t - fitter.mean_x) @ P.T + fitter.mean_x if affine else X_t @ P.T

        # Define a random positive definite inner product
        L = torch.randn(d, d, dtype=X_t.dtype) / d**0.5
        loss = F.mse_loss(x_ @ L, X_t @ L)
        loss.backward()

        # Should be optimal iff we're using LEACE with the bias term and no shrinkage
        assert (A.grad.norm() < eps) == (affine and method == "leace" and not shrink)

        # Sanity check that we're not beating the oracle
        assert oracle_loss <= loss

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

        check_linear_guardedness(X, X_.numpy(), Y, eps)


@pytest.mark.parametrize("num_classes", [2, 3, 5, 10, 20])
def test_quadratic_erasure(num_classes: int):
    tol = 2e-9
    n, d = 1024, 32

    X, Y = make_classification(
        n_samples=n,
        n_features=d,
        n_classes=num_classes,
        n_informative=num_classes,
        # QDA goes berserk if the covariance matrices are singular
        n_redundant=0,
        random_state=42,
    )
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    eraser = QuadraticEraser.fit(X_t, Y_t, shrinkage=False)
    X_scrubbed = eraser(X_t, Y_t).numpy()

    # Quadratic LEACE should ensure both linear & quadratic guardedness
    check_linear_guardedness(X, X_scrubbed, Y.reshape(n, -1), tol)

    # Now check quadratic guardedness using QDA
    qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X_scrubbed, Y)
    loss = log_loss(Y, qda.predict_proba(X_scrubbed))
    trivial_loss = log_loss(Y, np.tile(qda.priors_, [n, 1]))
    np.testing.assert_allclose(loss, trivial_loss)

    # Sanity check that it DOES learn something before erasure
    real_qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, Y)
    real_loss = log_loss(Y, real_qda.predict_proba(X))
    assert real_loss < trivial_loss

    # Check that the covariance matrices and means are all the same
    cov, mu = qda.covariance_, np.asarray(qda.means_)
    np.testing.assert_allclose(cov[1:], cov[:-1])
    np.testing.assert_allclose(mu[1:], mu[:-1])

    # Optimal transport maps should all be positive definite
    assert is_positive_definite(eraser.ot_maps).all()
