import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.stats import ecdf as scipy_cdf

from concept_erasure import cdf, icdf

floats = st.floats(allow_nan=False, allow_infinity=False)

# Quantiles should be finite, unique, and sorted
quantiles = (
    arrays(
        float,
        (5,),
        elements=floats,
        unique=True,
    )
    .map(np.sort)
    .map(torch.from_numpy)
)


@given(
    arrays(float, (10,), elements=floats).map(torch.from_numpy),
    quantiles,
)
def test_cdf(x, q):
    # Pseudo-inverse property
    p = cdf(x, q)

    # Strictly speaking, icdf ought to return -inf when p = 0, but in practice we
    # want to clip to the minimum value, so we've implemented it that way.
    p_inv = icdf(p, q).where(p > 0, -torch.inf)
    assert torch.all(p_inv <= x)
    assert torch.all(cdf(p_inv, q) >= p)

    # Should match SciPy
    torch.testing.assert_close(
        cdf(x, q),
        torch.from_numpy(scipy_cdf(q).cdf.evaluate(x)).float(),
    )
