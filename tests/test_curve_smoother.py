"""The model-side curve smoother has exactly one implementation.

It used to have two: the mu1 density-asymmetry path in ``shared/utils.py`` and a
verbatim copy inside ``surface_simulator._generate_mu2_density_asymmetry_batch``
that differed only in calling ``jnp.correlate`` where the other called
``jnp.convolve``.  Both are now :func:`shared.utils.gaussian_curve_smoother`.

These tests exist for two different reasons.  The parity tests pin the extraction
itself: the shared helper must reproduce both pre-extraction copies bit for bit,
because the surface path's numbers are the reference the whole density target is
calibrated against and an extraction that changed them would be a silent
re-definition of the estimator.  The property tests pin the smoother's *contract*
-- kernel width, edge padding, normalization -- which is part of the density
target's definition rather than an implementation detail, and which the WNM path
is about to depend on as well.
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
SIGMAS = [2.0, 5.0, 7.5, 10.0, 20.0]
LENGTHS = [90, 91, 120]


def _convolve_copy(curve, sigma):
    """``shared/utils.py`` before the extraction, transcribed verbatim."""
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = jnp.arange(kernel_size) - kernel_size // 2
    kernel = jnp.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / jnp.sum(kernel)
    pad = kernel_size // 2
    return jnp.convolve(jnp.pad(curve, pad, mode='edge'), kernel, mode='valid')


def _correlate_copy(curve, sigma):
    """``surface_simulator.py``'s mu2 copy before the extraction, verbatim."""
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = jnp.arange(kernel_size) - kernel_size // 2
    kernel = jnp.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / jnp.sum(kernel)
    pad = kernel_size // 2
    return jnp.correlate(jnp.pad(curve, pad, mode='edge'), kernel, mode='valid')


@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("copy", [_convolve_copy, _correlate_copy],
                         ids=["mu1_convolve", "mu2_correlate"])
def test_extraction_is_bit_identical_to_both_previous_copies(sigma, length, copy):
    curve = jnp.asarray(np.random.default_rng(length).normal(size=length))
    np.testing.assert_array_equal(
        np.asarray(gaussian_curve_smoother(curve, sigma)),
        np.asarray(copy(curve, sigma)))


def test_the_two_previous_copies_agreed_with_each_other():
    """Why collapsing them was safe: a symmetric kernel makes the two identical.

    If this ever fails, the kernel has stopped being symmetric and the mu1 and
    mu2 paths were never the same operation after all.
    """
    curve = jnp.asarray(np.random.default_rng(7).normal(size=90))
    np.testing.assert_array_equal(np.asarray(_convolve_copy(curve, 10.0)),
                                  np.asarray(_correlate_copy(curve, 10.0)))


@pytest.mark.parametrize("sigma", SIGMAS)
def test_length_is_preserved(sigma):
    curve = jnp.asarray(np.random.default_rng(1).normal(size=90))
    assert gaussian_curve_smoother(curve, sigma).shape == (90,)


@pytest.mark.parametrize("sigma", SIGMAS)
def test_a_constant_curve_is_unchanged(sigma):
    """Normalized kernel plus edge padding: no gain, and no edge droop.

    A curve that droops at its ends under smoothing would bias every fitted
    asymmetry toward zero at the extreme feature differences.
    """
    curve = jnp.full(90, 0.37)
    np.testing.assert_allclose(np.asarray(gaussian_curve_smoother(curve, sigma)),
                               0.37, rtol=0, atol=1e-6)


@pytest.mark.parametrize("sigma", SIGMAS)
def test_smoothing_is_a_weighted_average_not_a_gain(sigma):
    """Output stays inside the input's range: the kernel sums to one."""
    curve = jnp.asarray(np.random.default_rng(2).normal(size=90))
    out = np.asarray(gaussian_curve_smoother(curve, sigma))
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
