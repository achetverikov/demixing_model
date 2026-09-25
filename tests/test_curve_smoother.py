"""The model-side curve smoother's contract.

Width in grid steps, edge padding, unit gain. These are part of the density
target's definition rather than implementation details -- both surrogate families
smooth their curves with this, so a change here changes what every density fit
means.

The extraction that created this helper -- it previously existed twice, once in
the mu1 asymmetry path and once inside the mu2 simulator -- was verified against
verbatim copies of both predecessors at the time. That check has been removed:
it was a migration proof, and keeping it would pin the helper to the shape of the
code it replaced rather than to the behaviour that matters.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.utils import compute_single_density_asymmetry, gaussian_curve_smoother  # noqa: E402

# The production sigma is weights_sd / feat_diff_step = 20 / 2; the others span
# the range the surface and mu2 paths have used.
# The production sigma is weights_sd / feat_diff_step = 20 / 2. The others are
# one below and one above it; more values of the same kind add instances, not
# coverage. Lengths cover even and odd, which the kernel padding distinguishes.
SIGMAS = [2.0, 10.0, 20.0]
LENGTHS = [90, 91]


@pytest.mark.parametrize("length", LENGTHS)
def test_length_is_preserved(length):
    curve = jnp.asarray(np.random.default_rng(1).normal(size=length))
    assert gaussian_curve_smoother(curve, 10.0).shape == (length,)


def test_a_constant_curve_is_unchanged():
    """Normalized kernel plus edge padding: no gain, and no edge droop.

    A curve that droops at its ends under smoothing would bias every fitted
    asymmetry toward zero at the extreme feature differences.
    """
    curve = jnp.full(90, 0.37)
    np.testing.assert_allclose(np.asarray(gaussian_curve_smoother(curve, 10.0)),
                               0.37, rtol=0, atol=1e-6)


def test_smoothing_is_a_weighted_average_not_a_gain():
    """Output stays inside the input's range: the kernel sums to one."""
    curve = jnp.asarray(np.random.default_rng(2).normal(size=90))
    out = np.asarray(gaussian_curve_smoother(curve, 10.0))
    assert out.min() >= float(curve.min()) - 1e-6
    assert out.max() <= float(curve.max()) + 1e-6


def test_kernel_width_follows_sigma():
    """A spike spreads over ~4*sigma, so sigma is in grid steps, not degrees.

    The production value comes from ``weights_sd / feat_diff_step``; reading it
    as degrees would smooth by half as much on the standard 2-degree grid.
    """
    spike = jnp.asarray(np.eye(1, 91, 45).ravel())
    widths = []
    for sigma in (2.0, 5.0, 10.0):
        out = np.asarray(gaussian_curve_smoother(spike, sigma))
        widths.append(int(np.sum(out > out.max() * 0.05)))
    assert widths[0] < widths[1] < widths[2]


def test_asymmetry_path_still_routes_through_the_helper():
    """The caller must use the helper, not re-inline an equivalent smoother."""
    from shared.mu1_axis import mu1_grid

    grid = mu1_grid()
    rng = np.random.default_rng(11)
    surface = jnp.asarray(np.log(np.abs(rng.normal(size=(len(grid), 90))) + 1e-3))
    indices = jnp.arange(0, 90, 5)

    smoothed = compute_single_density_asymmetry(surface, indices, grid,
                                                apply_smoothing=True, smoothing_sigma=10.0)
    raw = compute_single_density_asymmetry(surface, indices, grid, apply_smoothing=False)
    np.testing.assert_array_equal(np.asarray(smoothed),
                                  np.asarray(gaussian_curve_smoother(raw, 10.0)))
