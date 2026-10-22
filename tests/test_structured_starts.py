"""Unit contract for the deterministic structured EM start bank.

Catches a start count or layout that disagrees with the CUDA kernel input
(datasets, starts, 5, 2), or starts that violate the M-step constraints the
solver assumes from its first iteration. The bank's search quality is an
experiment result, not a unit property.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surface_computation import structured_starts
from surface_computation import wnm_simulation as simulation


@pytest.mark.parametrize('n', [4, 10, 20, 100])
def test_bank_layout_counts_and_constraints(n):
    design = jnp.array([[10., 40., 20., 60.], [3., 120., 80., 175.]])
    x, _ = simulation.generate_block(simulation.simulation_keys(jax.random.key(5), 2), design, 3, n)
    theta = np.asarray(structured_starts.batch(x))
    assert theta.shape == (len(x), structured_starts.count(n), 5, 2)
    assert structured_starts.count(n) == min(n, 20)+34
    assert np.isfinite(theta).all()
    weights = theta[:, :, 0]
    assert (weights >= .1-1e-7).all() and (weights <= .9+1e-7).all()
    np.testing.assert_allclose(weights.sum(-1), 1., atol=1e-6)
    assert (theta[:, :, 1] >= -180.).all() and (theta[:, :, 1] <= 180.).all()
    assert (theta[:, :, 3:] >= 1.).all()
    # The last start is the even split shared by both components.
    np.testing.assert_allclose(theta[:, -1, 1, 0], theta[:, -1, 1, 1], atol=1e-4)
    # A dataset's starts depend on batch shape only through float rounding; a
    # refit therefore indexes its batch's starts (see wnm_simulation.repair_failed).
    np.testing.assert_allclose(theta[1:2], np.asarray(structured_starts.batch(x[1:2])), rtol=1e-5, atol=1e-4)
