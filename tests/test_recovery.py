"""The recovery runner, and the attribution it exists to make.

Recovery separates three failures that a single error number conflates: the
search missed the basin, the objective genuinely prefers other parameters, or the
target's construction moved the optimum. They call for opposite responses -- more
starts, a different objective, or a different target -- so a runner that reports
only "the parameters were 12% out" is not useful.

The load-bearing test here is the noise-free one. Fitting the model's own curve
at the generating parameters removes sampling noise, the KDE, and the empirical
feature weighting; if recovery is exact there, then any error against a real
empirical target came from the target's construction rather than from the
objective being unidentifiable or the search being weak.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import recovery as R  # noqa: E402
import wnm_scoring as S  # noqa: E402
from continuous_fit import fit_continuous  # noqa: E402
from continuous_optimizer import (  # noqa: E402
    build_bounds, condition_parameter_layout, minimize_continuous)
from density_objective import degenerate_targets  # noqa: E402
from fitting_targets import build_fitting_targets  # noqa: E402
from grid_based_multi_condition_optimizer_jax_loops import (  # noqa: E402
    _compute_curve_losses, bwcrps_energy_score, compute_bwcrps_condition_targets,
    compute_target_bias_curve_core)
from shared import surrogate  # noqa: E402
from shared.config import config  # noqa: E402
from shared.prediction import predictor_from_surrogate  # noqa: E402

ARTIFACT = surrogate.WNM_DEFAULTS[20]
pytestmark = pytest.mark.skipif(not ARTIFACT.exists(),
                                reason="no packaged WNM artifact")


@pytest.fixture(scope="module")
def predictor():
    return predictor_from_surrogate(surrogate.load_surrogate(checkpoint_path=ARTIFACT))


@pytest.fixture(scope="module")
def grids():
    feat_grid = config.create_grid('feat_diff')
    bias_grid = config.create_grid('mu1_bias')
    difference = jnp.abs(bias_grid[:, None] - bias_grid[None, :])
    return feat_grid, jnp.minimum(difference, 360.0 - difference)


def _targets_builder(grids):
    feat_grid, d_circ = grids

    def build(datasets):
        return build_fitting_targets(
            {name: jnp.asarray(values) for name, values in datasets.items()},
            feat_diff_grid=feat_grid, d_circ_matrix=d_circ,
            n_mu1_bias=len(config.create_grid('mu1_bias')),
            emp_density_weights_sd=20.0, density_bandwidth_rule="sj",
            density_bandwidth_mode="pooled", degenerate_targets=degenerate_targets,
            bwcrps_condition_targets=compute_bwcrps_condition_targets,
            target_bias_curve_core=compute_target_bias_curve_core)

    return build


# ---------------------------------------------------------------------------
# Generating data
# ---------------------------------------------------------------------------

def test_sampled_responses_follow_the_mixture(predictor):
    """The sampler must draw from the distribution being fitted, or the closed
    loop is not closed and every later conclusion is about the wrong model."""
    rng = np.random.default_rng(0)
    feat_diff = np.full(20000, 40.0)
    bias = R.sample_mixture_responses(predictor, 20.0, 40.0, 25.0, feat_diff, rng)

    assert bias.shape == (20000,)
    assert np.all(bias >= -180.0) and np.all(bias < 180.0)

    # Compare the sample's circular mean and resultant against the analytic ones.
    rows = jnp.asarray([[20.0, 40.0, 25.0, 40.0]], jnp.float32)
    mean, resultant = predictor.mean_and_resultant(rows, validate=False)
    angles = np.radians(bias)
    sample_resultant = np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles)))
    sample_mean = np.degrees(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))

    assert abs(sample_resultant - float(resultant[0])) < 0.02
    assert abs(((sample_mean - float(mean[0])) + 180) % 360 - 180) < 2.0


def test_motor_noise_widens_the_sample(predictor):
    rng = np.random.default_rng(1)
    feat_diff = np.full(20000, 40.0)
    plain = R.sample_mixture_responses(predictor, 20.0, 40.0, 25.0, feat_diff, rng)
    noisy = R.sample_mixture_responses(predictor, 20.0, 40.0, 25.0, feat_diff, rng,
                                       sd_motor=30.0)

    def resultant(values):
        angles = np.radians(values)
        return np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles)))

    assert resultant(noisy) < resultant(plain)


def test_a_case_generates_one_dataset_per_condition(predictor):
    case = R.RecoveryCase(name="two", condition_feature_sds=[(15.0, 45.0), (60.0, 20.0)],
                          sd_spat=25.0, n_trials_per_condition=50)
    datasets = R.generate_case_data(predictor, case, np.random.default_rng(0))

    assert list(datasets) == ["c0", "c1"]
    for values in datasets.values():
        assert values.shape == (50, 2)
    np.testing.assert_array_equal(case.truth_vector(), [15.0, 45.0, 60.0, 20.0, 25.0])
    np.testing.assert_array_equal(case.truth_vector(fit_motor=True),
                                  [15.0, 45.0, 60.0, 20.0, 25.0, 0.0])


def test_a_discrete_design_only_visits_its_own_feature_values(predictor):
    """Moors-shaped designs sample a handful of feature differences, not a range."""
    case = R.RecoveryCase(name="discrete", condition_feature_sds=[(20.0, 20.0)],
                          sd_spat=25.0, n_trials_per_condition=200,
                          feat_diff_values=[10.0, 50.0, 90.0])
    datasets = R.generate_case_data(predictor, case, np.random.default_rng(0))
    assert set(np.unique(datasets["c0"][:, 0])) <= {10.0, 50.0, 90.0}


# ---------------------------------------------------------------------------
# The attribution
# ---------------------------------------------------------------------------

def test_a_noise_free_target_recovers_the_generating_parameters(predictor, grids):
    """The reference point for every other recovery number.

    Fitting the model's own smoothed curve at the truth removes sampling noise,
    the KDE and the empirical feature weighting. Recovery is exact here -- to
    about 1e-4 in log ratio -- which establishes that the density objective is
    identifiable and the search finds its optimum. Any error against a real
    empirical target is therefore attributable to that target's construction,
    not to the objective or the search.
    """
    feat_grid, _ = grids
    truth = [15.0, 45.0, 60.0, 20.0, 25.0]

    ideal = jnp.stack([
        S.predicted_asymmetry_curve(predictor, truth[0], truth[1], truth[4], feat_grid, 20.0),
        S.predicted_asymmetry_curve(predictor, truth[2], truth[3], truth[4], feat_grid, 20.0)])

    def objective(params):
        total = 0.0
        for index in range(2):
            predicted = S.predicted_asymmetry_curve(
                predictor, params[2 * index], params[2 * index + 1], params[4],
                feat_grid, 20.0)
            total = total + _compute_curve_losses(
                predicted[None, :], ideal[index][None, :], loss_type="ccc",
                is_angular=False)[0]
        return total

    bounds = surrogate.search_bounds(predictor.domain)
    fit = minimize_continuous(objective,
                              build_bounds(2, bounds["sd_feat"], bounds["sd_spat"]),
                              condition_parameter_layout(2, fit_motor=False),
                              n_starts=16, seed=0)

    for true_value, fitted in zip(truth, fit.parameters):
        assert abs(np.log(fitted / true_value)) < 5e-3, (truth, fit.parameters)


def test_the_diagnosis_distinguishes_the_three_failure_modes():
    """The classification is what makes a recovery number actionable."""
    def result(loss_at_fit, loss_at_truth):
        return R.RecoveryResult(
            case="c", replicate=0, truth=np.array([10.0]), recovered=np.array([12.0]),
            names=("sd_feat1_c0",), loss_at_truth=loss_at_truth, loss_at_fit=loss_at_fit,
            loss_spread=0.0, n_starts=4, n_converged=4, at_bound=(), runtime_seconds=1.0)

    assert result(0.5, 0.3).diagnosis == "search_failed"
    assert result(0.3, 0.5).diagnosis == "objective_prefers_other_parameters"
    assert result(0.4, 0.4).diagnosis == "loss_tied_with_truth"


def test_errors_are_reported_as_log_ratios_as_well_as_signed():
    """These are multiplicative scales: 5 degrees out at 200 is not the error
    that 5 degrees out at 5 is."""
    outcome = R.RecoveryResult(
        case="c", replicate=0, truth=np.array([10.0, 100.0]),
        recovered=np.array([20.0, 200.0]), names=("a", "b"), loss_at_truth=1.0,
        loss_at_fit=1.0, loss_spread=0.0, n_starts=4, n_converged=4, at_bound=(),
        runtime_seconds=1.0)
    errors = outcome.errors()

    assert errors["signed_error_a"] == 10.0 and errors["signed_error_b"] == 100.0
    # The same relative error, reported as the same number.
    assert errors["log_ratio_a"] == pytest.approx(errors["log_ratio_b"])


def test_the_summary_reports_bias_and_spread_but_not_correlation():
    """Correlation between fitted and true parameters is high whenever the design
    spans a range, regardless of whether any single estimate is any good."""
    results = [
        R.RecoveryResult(case="c", replicate=index, truth=np.array([10.0]),
                         recovered=np.array([10.0 * np.exp(0.1)]), names=("a",),
                         loss_at_truth=1.0, loss_at_fit=1.0, loss_spread=0.0,
                         n_starts=4, n_converged=4, at_bound=(), runtime_seconds=1.0)
        for index in range(4)]
    summary = R.summarise(results)

    assert summary["n_replicates"] == 4
    assert summary["a_median_log_ratio"] == pytest.approx(0.1, rel=1e-6)
    assert summary["a_rmse_log_ratio"] == pytest.approx(0.1, rel=1e-6)
    assert not any("correl" in key for key in summary)


def test_summarising_nothing_raises():
    with pytest.raises(ValueError, match="no replicates"):
        R.summarise([])


# ---------------------------------------------------------------------------
# End to end, small
# ---------------------------------------------------------------------------

def test_a_replicate_runs_and_records_what_it_needs(predictor, grids):
    feat_grid, d_circ = grids
    case = R.RecoveryCase(name="small", condition_feature_sds=[(20.0, 45.0)],
                          sd_spat=25.0, n_trials_per_condition=300)

    outcome = R.run_replicate(
        predictor, case, 0, objective="density",
        build_targets=_targets_builder(grids), fit_continuous=fit_continuous,
        score_all_conditions=S.score_all_conditions, curve_losses=_compute_curve_losses,
        energy_score=bwcrps_energy_score, d_circ_matrix=d_circ, feat_diff_grid=feat_grid,
        emp_density_weights_sd=20.0, n_starts=3, seed=0)

    assert outcome.names == ("sd_feat1_c0", "sd_feat2_c0", "sd_spat")
    assert np.all(np.isfinite(outcome.recovered))
    assert np.isfinite(outcome.loss_at_truth) and np.isfinite(outcome.loss_at_fit)
    assert outcome.diagnosis in ("search_failed", "objective_prefers_other_parameters",
                                 "loss_tied_with_truth")

    row = outcome.row()
    for column in ("case", "replicate", "loss_at_truth", "loss_at_fit", "loss_gap",
                   "diagnosis", "true_sd_spat", "fit_sd_spat", "log_ratio_sd_spat"):
        assert column in row
