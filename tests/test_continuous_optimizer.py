"""Contracts for the bounded gradient search.

Two things have to hold before this search can be compared against the lattice
backends on equal terms. First, the gradients must be the gradients of the thing
being minimised -- a wrong derivative produces a confident, converged, wrong
answer, and no amount of multistart detects it. Second, the search must report
its own failures: a railed solution, a wide spread across starts, a start that
did not converge. A search that quietly returns the best of several bad answers
would win a benchmark it should lose.

The finite-difference checks deliberately cross the wrapped-normal's internal
branch switch at sigma = 60 degrees, where the implementation changes from a
spatial wrap sum to a Fourier series. A derivative that is right on both sides of
a branch but wrong at the switch is exactly what a smooth-looking optimisation
would hide.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from continuous_density import wrapped_mixture_model as wm  # noqa: E402
from continuous_optimizer import (  # noqa: E402
    ContinuousFit, build_bounds, condition_parameter_layout, dispersed_starts,
    minimize_continuous)
from shared import surrogate  # noqa: E402
from shared.prediction import predictor_from_surrogate  # noqa: E402

ARTIFACT = surrogate.WNM_DEFAULTS[20]
needs_artifact = pytest.mark.skipif(not ARTIFACT.exists(),
                                    reason="no packaged WNM artifact installed")


# ---------------------------------------------------------------------------
# Starts
# ---------------------------------------------------------------------------

def test_starts_are_reproducible_from_the_seed():
    """The seed is part of a run's identity; two runs of the same fit must match."""
    bounds = [(2.5, 200.0), (5.0, 200.0)]
    np.testing.assert_array_equal(dispersed_starts(bounds, 8, seed=4),
                                  dispersed_starts(bounds, 8, seed=4))
    assert not np.array_equal(dispersed_starts(bounds, 8, seed=4),
                              dispersed_starts(bounds, 8, seed=5))


def test_starts_are_dispersed_in_log_space_and_inside_the_bounds():
    """Spread multiplicatively: a linear spread wastes most starts above 100."""
    bounds = [(2.5, 200.0)]
    starts = dispersed_starts(bounds, 200, seed=0)
    assert starts.min() >= 2.5 and starts.max() <= 200.0

    # Half the starts should fall below the geometric middle, not the linear one.
    geometric_middle = np.sqrt(2.5 * 200.0)
    below = float(np.mean(starts < geometric_middle))
    assert 0.4 < below < 0.6, below


def test_every_stratum_is_visited():
    """Stratified, so 'the search missed the basin' reports the search and not
    an unlucky draw that left a region unsampled."""
    starts = dispersed_starts([(2.5, 200.0)], 10, seed=1).ravel()
    edges = np.exp(np.linspace(np.log(2.5), np.log(200.0), 11))
    counts, _ = np.histogram(starts, bins=edges)
    assert np.all(counts == 1), counts


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

def _directional_check(function, point, direction, step=0.1):
    """Central difference along one direction, against the analytic gradient.

    The step is 0.1 model degrees, not something smaller, and that is measured
    rather than guessed. These objectives are evaluated in float32, so the
    central difference has two competing errors: truncation, which grows with the
    step, and cancellation, which grows as the step shrinks because the numerator
    is a difference of two nearly equal float32 numbers. Sweeping the step on the
    asymmetry objective gives a clean U -- relative error 0.8% at 1e-3, 0.09% at
    0.05, 0.002% at 0.1, back to 1% at 1.0 -- so 0.1 sits in the flat region.

    A single step in the noise-dominated regime would fail against a correct
    gradient, which is how this test first read.
    """
    point = jnp.asarray(point, dtype=jnp.float32)
    direction = jnp.asarray(direction, dtype=jnp.float32)
    direction = direction / jnp.linalg.norm(direction)
    analytic = float(jnp.dot(jax.grad(function)(point), direction))
    plus = float(function(point + step * direction))
    minus = float(function(point - step * direction))
    return analytic, (plus - minus) / (2 * step)


@needs_artifact
@pytest.mark.parametrize("sigma_region,params", [
    ("below the branch switch", [12.0, 18.0, 20.0, 30.0]),
    ("across the branch switch", [55.0, 65.0, 30.0, 45.0]),
    ("above the branch switch", [120.0, 150.0, 60.0, 90.0]),
    ("narrow, below the old floor", [2.6, 3.0, 8.0, 10.0]),
])
def test_asymmetry_gradients_match_finite_differences(sigma_region, params):
    """The wrapped normal switches implementation at sigma = 60 degrees.

    A gradient that is right on both sides but wrong at the switch would still
    produce a smooth-looking optimisation, converging confidently to the wrong
    parameters.
    """
    predictor = predictor_from_surrogate(surrogate.load_surrogate(checkpoint_path=ARTIFACT))

    def objective(p):
        return jnp.sum(predictor.signed_arc_asymmetry(p[None, :], validate=False))

    # Fixed directions, not a hash-seeded draw: Python salts string hashes per
    # process, so seeding from one made this test sample different directions on
    # every run -- it failed intermittently against correct gradients and would
    # equally have passed intermittently against wrong ones.
    directions = np.array([[0.3, -0.5, 0.7, 0.4],
                           [-0.8, 0.2, 0.1, -0.6],
                           [0.5, 0.5, -0.5, 0.5]])
    for direction in directions:
        analytic, numeric = _directional_check(objective, params, direction)
        assert np.isclose(analytic, numeric, rtol=2e-2, atol=1e-7), (
            f"{sigma_region}: analytic {analytic:.6e} vs finite difference {numeric:.6e}")


@needs_artifact
def test_gradients_survive_motor_noise():
    """Motor noise widens sigma inside the objective, so it is on the gradient path."""
    predictor = predictor_from_surrogate(
        surrogate.load_surrogate(checkpoint_path=ARTIFACT)).with_motor_noise(25.0)

    def objective(p):
        return jnp.sum(predictor.signed_arc_asymmetry(p[None, :], validate=False))

    for direction in ([0.3, -0.5, 0.7, 0.4], [-0.8, 0.2, 0.1, -0.6], [0.5, 0.5, -0.5, 0.5]):
        analytic, numeric = _directional_check(objective, [30.0, 50.0, 25.0, 40.0],
                                               np.asarray(direction))
        assert np.isclose(analytic, numeric, rtol=2e-2, atol=1e-7)


@needs_artifact
def test_the_finite_difference_step_sits_in_its_flat_region():
    """Guards the guard: if the step ever leaves the flat region, the gradient
    tests start failing on correct gradients, or passing on wrong ones."""
    predictor = predictor_from_surrogate(surrogate.load_surrogate(checkpoint_path=ARTIFACT))

    def objective(p):
        return jnp.sum(predictor.signed_arc_asymmetry(p[None, :], validate=False))

    point = [12.0, 18.0, 20.0, 30.0]
    direction = np.array([0.3, -0.5, 0.7, 0.4])
    errors = {}
    for step in (1e-3, 1e-2, 0.1, 1.0):
        analytic, numeric = _directional_check(objective, point, direction, step=step)
        errors[step] = abs(numeric / analytic - 1)

    assert errors[0.1] < errors[1e-3], "cancellation should dominate at tiny steps"
    assert errors[0.1] < errors[1.0], "truncation should dominate at large steps"
    assert errors[0.1] < 1e-3


@needs_artifact
def test_circular_moment_gradients_match_finite_differences():
    """The moment path feeds expectation and smoothed_exp, not just density."""
    predictor = predictor_from_surrogate(surrogate.load_surrogate(checkpoint_path=ARTIFACT))

    def objective(p):
        mean, resultant = predictor.mean_and_resultant(p[None, :], validate=False)
        return jnp.sum(resultant) + jnp.sum(jnp.cos(jnp.radians(mean)))

    for point, direction in (([15.0, 40.0, 20.0, 25.0], [0.3, -0.5, 0.7, 0.4]),
                             ([70.0, 80.0, 40.0, 120.0], [-0.8, 0.2, 0.1, -0.6])):
        analytic, numeric = _directional_check(objective, point, np.asarray(direction))
        assert np.isclose(analytic, numeric, rtol=2e-2, atol=1e-7)


# ---------------------------------------------------------------------------
# The search itself
# ---------------------------------------------------------------------------

def _log_distance_objective(truth):
    truth = jnp.asarray(truth, dtype=jnp.float32)

    def objective(p):
        return jnp.sum((jnp.log(p) - jnp.log(truth)) ** 2)

    return objective


def test_a_known_optimum_is_recovered_to_the_precision_floor():
    """float32 gradients cap this at roughly 1e-7 relative; see the module docstring."""
    truth = np.array([12.0, 45.0, 30.0])
    fit = minimize_continuous(_log_distance_objective(truth), [(2.5, 200.0)] * 3,
                              ["a", "b", "c"], n_starts=6, seed=1)
    np.testing.assert_allclose(fit.parameters, truth, rtol=1e-5)
    assert fit.loss_spread < 1e-6
    assert fit.at_bound == ()
    assert all(start.success for start in fit.starts)


def test_runtime_objective_arrays_are_not_captured_in_the_solver():
    def objective(parameters, truth):
        return jnp.sum((jnp.log(parameters) - jnp.log(truth)) ** 2)

    cache = {}
    kwargs = dict(bounds=[(2.5, 200.0)] * 3, names=["a", "b", "c"],
                  n_starts=6, seed=1, solver_cache=cache, solver_key=(3,))
    first = minimize_continuous(
        objective, objective_args=(jnp.asarray([12.0, 45.0, 30.0]),), **kwargs)
    second = minimize_continuous(
        objective, objective_args=(jnp.asarray([25.0, 60.0, 80.0]),), **kwargs)
    np.testing.assert_allclose(first.parameters, [12.0, 45.0, 30.0], rtol=1e-5)
    np.testing.assert_allclose(second.parameters, [25.0, 60.0, 80.0], rtol=1e-5)
    assert len(cache) == 1


def test_a_railed_solution_is_reported_rather_than_hidden():
    """Bounds are real, not a squashing function.

    A sigmoid reparameterisation would make every solution interior and lose the
    fact that the fit wanted to leave the box -- which is information about the
    model or the data, not a nuisance.
    """
    fit = minimize_continuous(_log_distance_objective([1.0, 1.0]), [(2.5, 200.0)] * 2,
                              ["x", "y"], n_starts=3, seed=2)
    assert fit.at_bound == ("x@low", "y@low")
    np.testing.assert_allclose(fit.parameters, [2.5, 2.5], rtol=1e-6)
    assert "x@low" in fit.summary()["at_bound"]


def test_every_start_is_kept_not_just_the_winner():
    """The spread across starts is what says whether the search or the objective
    chose the answer, and it is what the search comparison needs."""
    fit = minimize_continuous(_log_distance_objective([12.0, 45.0]), [(2.5, 200.0)] * 2,
                              ["a", "b"], n_starts=5, seed=3)
    assert len(fit.starts) == 5
    assert fit.summary()["n_starts"] == 5
    assert fit.summary()["total_evaluations"] > 0
    for start in fit.starts:
        assert start.n_evaluations > 0
        assert np.isfinite(start.loss)


def test_the_selected_production_configuration_is_recorded():
    fit = minimize_continuous(_log_distance_objective([12.0]), [(2.5, 200.0)], ["a"])
    assert fit.n_starts == 64
    assert fit.settings["optimizer_version"] == "jax-lbfgsb@0350da1"
    assert fit.settings["batch_size"] == 32
    assert fit.settings["dtype"] == "float32"
    assert fit.settings["matmul_precision"] == "highest"


def test_a_multimodal_objective_shows_a_loss_spread():
    """If every start agreed here the multistart would be pointless, and a
    single-start search would be reported as equally good."""
    def bumpy(p):
        return jnp.sum(jnp.sin(3.0 * jnp.log(p)) + 0.05 * jnp.log(p) ** 2)

    fit = minimize_continuous(bumpy, [(2.5, 200.0)] * 2, ["a", "b"], n_starts=12, seed=5)
    assert fit.loss_spread > 1e-3, "a multimodal objective should not agree across starts"


def test_non_finite_objective_regions_do_not_masquerade_as_convergence():
    """Handing L-BFGS-B a NaN makes it wander and then report success, so the run
    looks converged while the answer is the last finite point it happened to see.
    """
    def with_a_hole(p):
        return jnp.where(p[0] > 100.0, jnp.nan, jnp.sum((jnp.log(p) - np.log(20.0)) ** 2))

    fit = minimize_continuous(with_a_hole, [(2.5, 200.0)] * 2, ["a", "b"],
                              n_starts=6, seed=6)
    assert np.isfinite(fit.loss)
    np.testing.assert_allclose(fit.parameters, [20.0, 20.0], rtol=1e-4)


def test_an_everywhere_non_finite_objective_raises():
    """Returning the least-bad NaN would be a fit no one could interpret."""
    with pytest.raises(RuntimeError, match="not evaluable"):
        minimize_continuous(lambda p: jnp.asarray(np.nan), [(2.5, 200.0)],
                            ["a"], n_starts=2, seed=0)


# ---------------------------------------------------------------------------
# Parameter layout
# ---------------------------------------------------------------------------

def test_layout_is_two_feature_sds_per_condition_plus_shared_terms():
    assert condition_parameter_layout(1, fit_motor=False) == (
        "sd_feat1_c0", "sd_feat2_c0", "sd_spat")
    assert condition_parameter_layout(2, fit_motor=True) == (
        "sd_feat1_c0", "sd_feat2_c0", "sd_feat1_c1", "sd_feat2_c1", "sd_spat", "sd_motor")


def test_bounds_follow_the_layout_and_come_from_the_surrogate():
    """The feature and spatial axes have different domains, and the search must
    respect that rather than applying one interval to both."""
    bounds = build_bounds(2, (2.5, 200.0), (5.0, 200.0), motor_bounds=(0.1, 50.0))
    assert bounds.shape == (6, 2)
    np.testing.assert_array_equal(bounds[:4], np.array([[2.5, 200.0]] * 4))
    np.testing.assert_array_equal(bounds[4], [5.0, 200.0])
    np.testing.assert_array_equal(bounds[5], [0.1, 50.0])


def test_no_motor_case_has_no_motor_parameter():
    """A fixed zero motor SD is a separate case, not a log parameter: log space
    has no zero, and a bound at 1e-9 would be a different model."""
    assert "sd_motor" not in condition_parameter_layout(3, fit_motor=False)
    assert len(build_bounds(3, (2.5, 200.0), (5.0, 200.0))) == 7


def test_non_positive_bounds_are_refused():
    with pytest.raises(ValueError, match="strictly positive"):
        minimize_continuous(lambda p: jnp.sum(p), [(0.0, 200.0)], ["a"], n_starts=1)


def test_empty_and_mismatched_bounds_are_refused():
    with pytest.raises(ValueError, match="empty bounds"):
        minimize_continuous(lambda p: jnp.sum(p), [(200.0, 2.5)], ["a"], n_starts=1)
    with pytest.raises(ValueError, match="names for"):
        minimize_continuous(lambda p: jnp.sum(p), [(2.5, 200.0)], ["a", "b"], n_starts=1)
