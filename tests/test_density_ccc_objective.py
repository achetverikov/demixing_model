"""The density objective is 1 - CCC, and it refuses to run against a constant target.

Two failures are pinned here.

The first is what the switch fixes: the old objective scaled its MSE term by
``range`` instead of ``range**2``, so the term is not scale-free and a curve an
order of magnitude too small can score better than the right one. Over 204
condition-fits it left 115 with density curves >=5x too small. CCC cannot do
that, and `test_legacy_objective_prefers_a_flat_curve_and_ccc_does_not` shows the
two objectives disagreeing on exactly that comparison.

The second is what the constant-target refusal guards against: against a constant
target the CCC formula returns 1 -- the WORST possible loss -- for a perfect
match, because its numerator and denominator collapse together. The fit raises
there rather than dropping the condition; see
test_density_degenerate_target_refusal.py. This is defensive: no empirical target
in any current dataset is within six orders of magnitude of degenerate (measured
minimum variance 2.2e-04 over 681 real target curves).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax.numpy as jnp

from grid_based_multi_condition_optimizer_jax_loops import (
    DEGENERATE_TARGET_EPS,
    _compute_curve_losses,
)
from create_unified_subject_plots import _ccc_components


def ccc_loss(pred, target):
    return float(_compute_curve_losses(
        jnp.asarray(pred)[None, :], jnp.asarray(target)[None, :], loss_type="ccc")[0])


def legacy_loss(pred, target, corr_weight=0.25):
    return float(_compute_curve_losses(
        jnp.asarray(pred)[None, :], jnp.asarray(target)[None, :],
        loss_type="combined", corr_weight=corr_weight)[0])


@pytest.fixture
def curve():
    grid = np.linspace(2, 180, 90)
    return 0.05 * np.sin(np.pi * grid / 180.0) - 0.01


# --- the objective itself ---------------------------------------------------

def test_ccc_loss_is_zero_for_an_exact_match(curve):
    assert ccc_loss(curve, curve) == pytest.approx(0.0, abs=1e-12)


def test_ccc_loss_equals_mse_over_D_computed_independently(curve):
    """The stated identity 1 - CCC = MSE/D, checked against numpy rather than
    against the implementation under test."""
    rng = np.random.default_rng(3)
    pred = 0.7 * curve + rng.normal(0, 0.004, curve.size)
    mse = np.mean((pred - curve) ** 2)
    D = pred.var() + curve.var() + (pred.mean() - curve.mean()) ** 2
    assert ccc_loss(pred, curve) == pytest.approx(mse / D, rel=1e-6)


def test_ccc_is_scale_invariant_but_the_legacy_objective_is_not(curve):
    """Multiplying BOTH curves by a constant is a change of units, and must not
    change the score. The legacy objective's MSE term divides by `range` rather
    than `range**2`, so it does change -- which is the dimensional error that
    made it too weak to constrain amplitude."""
    rng = np.random.default_rng(5)
    pred = 0.8 * curve + rng.normal(0, 0.003, curve.size)
    assert ccc_loss(10 * pred, 10 * curve) == pytest.approx(ccc_loss(pred, curve), rel=1e-6)
    assert legacy_loss(10 * pred, 10 * curve) != pytest.approx(legacy_loss(pred, curve), rel=1e-3)


def test_legacy_objective_prefers_a_flat_curve_and_ccc_does_not(curve):
    """The concrete production failure, as a head-to-head.

    A candidate 10x too small but perfectly shaped is scored against one with the
    right amplitude and a noisier shape. The old objective picks the flat one
    (0.0087 vs 0.0212) because shrinking a curve shrinks its MSE faster than the
    range-scaling compensates; CCC picks the right-amplitude one by an order of
    magnitude (0.088 vs 0.924). This is why 115/204 production fits came out >=5x
    too small under the old objective and 0/204 do under CCC.
    """
    rng = np.random.default_rng(7)
    too_small = 0.1 * curve
    right_scale_noisy = curve + rng.normal(0, 0.5 * curve.std(), curve.size)

    assert legacy_loss(too_small, curve) < legacy_loss(right_scale_noisy, curve)
    assert ccc_loss(too_small, curve) > ccc_loss(right_scale_noisy, curve)


def test_ccc_penalises_amplitude_error_through_C_b(curve):
    """A perfectly correlated but mis-scaled curve keeps r = 1 and loses only
    through the accuracy factor, which is the term the old objective lacked."""
    components = _ccc_components(0.2 * curve, curve)
    assert components['r'] == pytest.approx(1.0, abs=1e-9)
    assert components['C_b'] < 0.5
    assert components['ccc'] == pytest.approx(1 - ccc_loss(0.2 * curve, curve), rel=1e-6)


# --- degenerate targets -----------------------------------------------------

def test_identical_constants_would_score_worst_possible_if_ever_scored():
    """Why the objective refuses instead of scoring: the formula's answer for a
    perfect match against a constant target is 1, the worst loss there is."""
    constant = np.full(90, 0.03)
    assert ccc_loss(constant, constant) == pytest.approx(1.0)


def test_a_constant_target_falls_under_the_refusal_threshold():
    constant = np.full(90, 0.03)
    assert np.var(constant) < DEGENERATE_TARGET_EPS


def test_the_refusal_threshold_matches_the_scorer_guard():
    """If these two ever diverge, targets in the gap between them pass the refusal
    and then silently get their denominator altered by the guard -- the exact
    silent distortion the policy removes."""
    import grid_based_multi_condition_optimizer_jax_loops as opt
    source = Path(opt.__file__).read_text()
    assert "jnp.where(D < DEGENERATE_TARGET_EPS, 1.0, D)" in source, (
        "the ccc branch must guard on the same eps the refusal uses")


def test_a_real_low_variance_target_is_not_refused(curve):
    """Real target variances start at 2.2e-04 (1% quantile 4.6e-04), six orders
    above the threshold, so an ordinary small curve must survive untouched."""
    small_but_real = 1e-3 * curve
    assert np.var(small_but_real) > DEGENERATE_TARGET_EPS
    assert ccc_loss(small_but_real, small_but_real) == pytest.approx(0.0, abs=1e-12)
