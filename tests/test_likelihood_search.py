"""Contracts for the WNM likelihood optimizer-comparison search arms."""
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import model_fit_to_data.likelihood_search as likelihood_search
from model_fit_to_data.continuous_optimizer import dispersed_starts
from model_fit_to_data.likelihood_search import (
    JaxBads,
    JaxoptLbfgsb,
    LikelihoodEvaluator,
    hierarchical_search,
    polish_search,
    scipy_lbfgsb,
)
from model_fit_to_data.wnm_scoring import trial_log_density
from shared import surrogate
from shared.prediction import predictor_from_surrogate


ARTIFACT = surrogate.WNM_DEFAULTS[100]
needs_artifact = pytest.mark.skipif(not ARTIFACT.exists(),
                                    reason="no packaged n=100 WNM artifact installed")


@needs_artifact
def test_reusable_likelihood_is_the_canonical_wnm_point_likelihood():
    predictor = predictor_from_surrogate(
        surrogate.load_surrogate(checkpoint_path=ARTIFACT))
    evaluator = LikelihoodEvaluator(predictor)
    trials = np.asarray([[2.0, -1.5], [40.0, 3.0], [180.0, 0.25]], np.float32)
    parameters = np.asarray([12.0, 35.0, 28.0], np.float32)

    expected = -jnp.sum(trial_log_density(
        predictor, *parameters, jnp.asarray(trials[:, 0]), jnp.asarray(trials[:, 1])))
    assert evaluator.score(parameters, trials) == pytest.approx(float(expected), rel=1e-6)

    direct = jax.value_and_grad(lambda log_parameters: -jnp.sum(trial_log_density(
        predictor, *jnp.exp(log_parameters), jnp.asarray(trials[:, 0]),
        jnp.asarray(trials[:, 1]))))
    actual = evaluator.value_and_grad(
        jnp.log(parameters), jnp.asarray(trials[:, 0]), jnp.asarray(trials[:, 1]))
    expected_value, expected_gradient = direct(jnp.log(parameters))
    assert float(actual[0]) == pytest.approx(float(expected_value), rel=1e-6)
    np.testing.assert_allclose(actual[1], expected_gradient, rtol=1e-6, atol=1e-6)


class QuadraticEvaluator:
    """The optimizer interface with a known optimum in natural units."""

    def __init__(self, truth):
        truth = jnp.asarray(truth)

        def loss(log_parameters, _feature, _bias):
            return jnp.sum((log_parameters - jnp.log(truth)) ** 2)

        self.loss_fn = loss
        self.value = jax.jit(loss)
        self.value_and_grad = jax.jit(jax.value_and_grad(loss))
        self.batch_value = jax.jit(jax.vmap(loss, in_axes=(0, None, None)))

    @staticmethod
    def trials(values):
        values = np.asarray(values, np.float32)
        return jnp.asarray(values[:, 0]), jnp.asarray(values[:, 1])

    def score(self, parameters, trials):
        feature, bias = self.trials(trials)
        return float(self.value(jnp.log(jnp.asarray(parameters)), feature, bias))


TRIALS = np.zeros((3, 2), np.float32)
BOUNDS = [(2.5, 200.0), (2.5, 200.0), (5.0, 200.0)]
TRUTH = np.asarray([11.0, 47.0, 23.0])


def test_threaded_scipy_keeps_start_order_and_matches_serial():
    evaluator = QuadraticEvaluator(TRUTH)
    starts = dispersed_starts(BOUNDS, 4, seed=2)
    serial = scipy_lbfgsb(evaluator, TRIALS, BOUNDS, starts=starts, workers=1)
    threaded = scipy_lbfgsb(evaluator, TRIALS, BOUNDS, starts=starts, workers=2)

    np.testing.assert_allclose(serial.parameters, TRUTH, rtol=1e-5)
    np.testing.assert_allclose(threaded.parameters, serial.parameters, rtol=1e-6)
    assert [row.start_index for row in threaded.candidates] == list(range(4))
    assert threaded.loss == pytest.approx(serial.loss, abs=1e-10)
    assert threaded.n_evaluations > 0


def test_scipy_does_not_report_a_failed_start_as_a_fit(monkeypatch):
    def failed_minimize(_objective, x0, **_kwargs):
        return SimpleNamespace(
            x=np.asarray(x0), fun=1.0, success=False, message="iteration limit",
            nit=0, nfev=1)

    monkeypatch.setattr(likelihood_search, "minimize", failed_minimize)
    with pytest.raises(RuntimeError, match="none of 2 .* starts converged"):
        scipy_lbfgsb(
            QuadraticEvaluator(TRUTH), TRIALS, BOUNDS, n_starts=2)


@pytest.mark.parametrize("coordinates", ["linear", "log"])
def test_hierarchical_search_refines_toward_a_known_optimum(coordinates):
    result = hierarchical_search(
        QuadraticEvaluator(TRUTH), TRIALS, BOUNDS, points_per_axis=7,
        n_stages=6, batch_size=128, coordinates=coordinates)

    np.testing.assert_allclose(result.parameters, TRUTH, rtol=2e-2, atol=0.1)
    assert result.loss < 1e-3
    assert len(result.candidates) == 6


def test_jaxopt_batched_starts_recover_a_known_optimum():
    evaluator = QuadraticEvaluator(TRUTH)
    result = JaxoptLbfgsb(
        evaluator, BOUNDS, n_starts=4, max_iterations=100,
        tolerance=1e-6).run(TRIALS, seed=3)

    np.testing.assert_allclose(result.parameters, TRUTH, rtol=1e-4)
    assert result.loss < 1e-8
    assert len(result.candidates) == 4
    assert result.n_evaluations > 0


class BadsStub:
    @staticmethod
    def _make_config(_dimensions, budget, **overrides):
        return {"budget": budget, **overrides}

    @staticmethod
    def _run_bads(objective, z0, _key, _dimensions, _config):
        values = jax.vmap(objective)(z0)
        return z0, values, jnp.ones(len(z0), jnp.int32), jnp.int32(1)


def test_jax_bads_adapter_preserves_common_scoring_and_traces():
    evaluator = QuadraticEvaluator(TRUTH)
    starts = dispersed_starts(BOUNDS, 3, seed=4)
    search = JaxBads(
        evaluator, BOUNDS, BadsStub, n_starts=3, budget=10)
    search.compile(TRIALS, starts=starts)
    result = search.run(TRIALS, starts=starts)

    assert len(result.candidates) == 3
    assert result.n_evaluations == 3
    assert result.loss == min(candidate.loss for candidate in result.candidates)
    for candidate in result.candidates:
        assert candidate.reported_loss == pytest.approx(candidate.loss, abs=5e-6)


def test_polish_keeps_global_trace_and_improves_coarse_candidate():
    evaluator = QuadraticEvaluator(TRUTH)
    coarse = hierarchical_search(
        evaluator, TRIALS, BOUNDS, points_per_axis=3, n_stages=1,
        batch_size=32, coordinates="log")
    result = polish_search(evaluator, TRIALS, BOUNDS, coarse)

    assert result.method == "hierarchical-log+scipy-lbfgsb"
    assert result.loss < coarse.loss
    np.testing.assert_allclose(result.parameters, TRUTH, rtol=1e-5)
    assert len(result.candidates) == len(coarse.candidates) + 1
