"""Properties of the corrected empirical SD and the bin-pooled model SD.

These checks existed before as module-level assertions that ran during
collection. They did protect the estimators -- a failure broke collection -- but
pytest reported "0 tests collected" and exited 5, so the file appeared in no pass
total and a reader could not tell which properties were covered. A check that
does not report itself is one revision away from being deleted as dead code.

One gap the conversion exposed: the file's headline claim, that the small-sample
correction removes the downward bias in circular SD, was **printed and never
asserted**. The only assertion in that block checked a trial count. It is
asserted here.

The bias it corrects is real and large. The sample resultant is biased upward
(``E[r̄²] = ρ² + (1−ρ²)/n`` -- a handful of unit vectors cannot cancel), and
because SD is ``sqrt(-2 ln r)`` that propagates to an *under*-estimate: roughly
−27% at n=3. The model side is an integral with no sampling error, so uncorrected
this reads on the plot as "the model overestimates variability".
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_fit_to_data.create_unified_subject_plots import (  # noqa: E402
    SD_N_BINS, compute_empirical_sd_curve, compute_feat_bin_weights,
    compute_predicted_sd_curves_batch, compute_predicted_sd_curves_batch_pooled)
from shared.config import config  # noqa: E402
from shared.mu1_axis import mu1_cell_width  # noqa: E402

FEAT_VALS = np.arange(config.feat_diff_range[0], config.feat_diff_range[1] + 1, 2)
TRUE_SD = 30.0


def wrapped_normal_surface(sds_per_column, means_per_column):
    """Log densities: one wrapped-normal-ish column per feature difference."""
    grid = np.asarray(config.create_grid('mu1_bias'), dtype=float)
    offsets = np.radians((grid[:, None] - means_per_column[None, :] + 180) % 360 - 180)
    sd_rad = np.radians(sds_per_column)[None, :]
    density = np.exp(-0.5 * (offsets / sd_rad) ** 2)
    density /= density.sum(axis=0, keepdims=True) * mu1_cell_width()
    return np.log(density)[None, ...]


def _uncorrected_sd(bias):
    """The naive estimator, for comparison: SD from the raw sample resultant."""
    angles = np.radians(bias)
    resultant = np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles)))
    return np.degrees(np.sqrt(-2 * np.log(max(resultant, 1e-10))))


# ---------------------------------------------------------------------------
# The small-sample correction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_trials,worst_uncorrected_bias", [(3, -0.15), (10, -0.05)])
def test_the_correction_removes_the_small_sample_bias(n_trials, worst_uncorrected_bias):
    """The claim this file exists for, now asserted rather than printed.

    The uncorrected estimator must come out materially low, and the corrected one
    must be much closer to the truth. Both halves matter: without the first, the
    test would pass on an estimator that needed no correction.
    """
    rng = np.random.default_rng(0)
    replicates = 1500
    uncorrected, corrected = [], []
    for _ in range(replicates):
        bias = rng.normal(0, TRUE_SD, size=n_trials)
        feat_diff = np.full(n_trials, 50.0)
        uncorrected.append(_uncorrected_sd(bias))
        _, sds, counts = compute_empirical_sd_curve(feat_diff, bias)
        assert counts.sum() == n_trials
        corrected.append(np.nanmean(sds))

    naive_bias = (np.mean(uncorrected) - TRUE_SD) / TRUE_SD
    fixed_bias = (np.nanmean(corrected) - TRUE_SD) / TRUE_SD

    assert naive_bias < worst_uncorrected_bias, (
        f"the uncorrected estimator was only {naive_bias:+.1%} off at n={n_trials}; "
        "if it no longer under-estimates, this test is not measuring the correction")
    assert abs(fixed_bias) < abs(naive_bias) / 2, (
        f"correction left {fixed_bias:+.1%} bias against the naive {naive_bias:+.1%}")


def test_the_correction_leaves_large_samples_alone():
    """It must not distort where there was nothing to fix."""
    rng = np.random.default_rng(1)
    values = []
    for _ in range(400):
        bias = rng.normal(0, TRUE_SD, size=200)
        _, sds, _ = compute_empirical_sd_curve(np.full(200, 50.0), bias)
        values.append(np.nanmean(sds))
    assert abs(np.nanmean(values) - TRUE_SD) / TRUE_SD < 0.02


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------

def test_an_antipodal_pair_is_missing_not_a_clamped_spike():
    """Two opposite angles have a zero resultant, so the SD is undefined. The
    clamp would otherwise turn it into ~389 degrees, which plots as a real and
    enormous measurement."""
    _, sds, _ = compute_empirical_sd_curve(np.array([50.0, 50.0]),
                                           np.array([0.0, 180.0]))
    assert np.all(np.isnan(sds)), sds


# ---------------------------------------------------------------------------
# Bin weights
# ---------------------------------------------------------------------------

def test_a_discrete_design_puts_one_column_in_each_occupied_bin():
    """The Moors case. If a bin mixed columns no trial visited, the pooled model
    SD would be inflated by feature differences the data never sampled."""
    discrete = np.repeat([30.0, 70.0, 110.0], 200)
    weights = compute_feat_bin_weights(discrete, FEAT_VALS)
    occupied = np.where(weights.sum(axis=1) > 0)[0]

    assert len(occupied) == 3, occupied
    for index in occupied:
        assert np.count_nonzero(weights[index]) == 1


def test_a_continuous_design_spreads_across_columns_and_sums_to_one():
    rng = np.random.default_rng(0)
    weights = compute_feat_bin_weights(rng.uniform(2, 180, size=5000), FEAT_VALS)
    occupied = weights.sum(axis=1) > 0

    np.testing.assert_allclose(weights[occupied].sum(axis=1), 1.0, atol=1e-12)
    assert np.count_nonzero(weights[5]) > 1


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def test_pooling_reduces_to_the_single_column_when_a_bin_has_one():
    discrete = np.repeat([30.0, 70.0, 110.0], 200)
    weights = compute_feat_bin_weights(discrete, FEAT_VALS)
    occupied = np.where(weights.sum(axis=1) > 0)[0]

    surface = wrapped_normal_surface(np.full(len(FEAT_VALS), 25.0),
                                     np.linspace(-40, 40, len(FEAT_VALS)))
    fine = np.asarray(compute_predicted_sd_curves_batch(jnp.asarray(surface), FEAT_VALS))[0]
    pooled = np.asarray(compute_predicted_sd_curves_batch_pooled(
        jnp.asarray(surface), jnp.asarray(weights[None, ...])))[0]

    for index in occupied:
        column = int(np.argmax(weights[index]))
        assert abs(pooled[index] - fine[column]) < 1e-3

    empty = [b for b in range(SD_N_BINS) if b not in occupied]
    assert np.all(np.isnan(pooled[empty]))


def _broadening(mean_span, weights):
    """How much pooling widens the model SD, for a given mean gradient."""
    surface = wrapped_normal_surface(
        np.full(len(FEAT_VALS), 25.0),
        np.linspace(-mean_span / 2, mean_span / 2, len(FEAT_VALS)))
    fine = np.asarray(compute_predicted_sd_curves_batch(jnp.asarray(surface), FEAT_VALS))[0]
    pooled = np.asarray(compute_predicted_sd_curves_batch_pooled(
        jnp.asarray(surface), jnp.asarray(weights[None, ...])))[0]

    edges = np.linspace(2, 180, SD_N_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    nearest = np.array([fine[int(np.argmin(np.abs(FEAT_VALS - c)))] for c in centres])
    return pooled, nearest


def test_pooling_a_moving_mean_broadens_and_never_narrows():
    """Pooling columns whose means differ adds a between-column term, so it can
    only widen the model SD. Narrowing would be wrong in the direction that
    flatters the model against the data."""
    rng = np.random.default_rng(0)
    weights = compute_feat_bin_weights(rng.uniform(2, 180, size=5000), FEAT_VALS)
    pooled, nearest = _broadening(80.0, weights)
    assert np.all(pooled > nearest - 1e-6)


def test_a_steeper_bias_curve_broadens_more():
    """The mechanism, rather than a threshold I would have to justify.

    The between-column term grows with the spread of column means inside a bin,
    so a steeper mean gradient must widen the pooled SD further. On this fixture
    the effect is small in absolute terms -- 0.037 degrees across an 80-degree
    mean span -- which is precisely why asserting a magnitude would be arbitrary
    while asserting the ordering is not.
    """
    rng = np.random.default_rng(0)
    weights = compute_feat_bin_weights(rng.uniform(2, 180, size=5000), FEAT_VALS)

    gentle_pooled, gentle_fine = _broadening(20.0, weights)
    steep_pooled, steep_fine = _broadening(160.0, weights)

    gentle = np.nanmax(gentle_pooled - gentle_fine)
    steep = np.nanmax(steep_pooled - steep_fine)

    assert steep > gentle > 0.0, (
        f"broadening did not grow with the mean gradient: {gentle:.4f} -> {steep:.4f}")


def test_a_constant_mean_surface_is_unaffected_by_pooling():
    """No between-column term to add, so pooling must be a no-op. This is the
    control for the test above."""
    rng = np.random.default_rng(0)
    weights = compute_feat_bin_weights(rng.uniform(2, 180, size=5000), FEAT_VALS)
    flat = wrapped_normal_surface(np.full(len(FEAT_VALS), 25.0),
                                  np.zeros(len(FEAT_VALS)))

    fine = np.asarray(compute_predicted_sd_curves_batch(jnp.asarray(flat), FEAT_VALS))[0]
    pooled = np.asarray(compute_predicted_sd_curves_batch_pooled(
        jnp.asarray(flat), jnp.asarray(weights[None, ...])))[0]

    assert np.nanmax(np.abs(pooled - fine.mean())) < 1e-3
