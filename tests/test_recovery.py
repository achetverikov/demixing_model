"""The recovery runner, and the attribution it exists to make.

Recovery separates three failures that a single error number conflates: the
search missed the basin, the objective genuinely prefers other parameters, or the
target's construction moved the optimum. They call for opposite responses -- more
starts, a different objective, or a different target -- so a runner that reports
only "the parameters were 12% out" is not useful.

The load-bearing test here is the noise-free one. Fitting the model's own curve
at the generating parameters removes sampling noise, the KDE, and the empirical
feature weighting, so it isolates the objective and the search from the target's
construction at that one point.

It is worth being precise about what that buys, because the first write-up of
this panel was not. Exact recovery there shows the objective's optimum sits at
the truth *for this generating vector* and that the best of many starts finds it.
It does not make every later error attributable to the target's construction:
finite-trial sampling variation and a residual search gap both survive it, and
separating those needs replicates and a start budget, not this test.
"""
import csv
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
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
    feat_diff = np.full(8000, 40.0)
    bias = R.sample_mixture_responses(predictor, 20.0, 40.0, 25.0, feat_diff, rng)

    assert bias.shape == (8000,)
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
    feat_diff = np.full(8000, 40.0)
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

@pytest.mark.development
@pytest.mark.slow
def test_a_noise_free_target_recovers_the_generating_parameters(predictor, grids):
    """The reference point for every other recovery number.

    Fitting the model's own smoothed curve at the truth removes sampling noise,
    the KDE and the empirical feature weighting. Recovery is exact here, to
    about 1.5e-4 in log ratio at worst.

    What that licenses is narrow, and the tolerance is set to the achieved
    precision so a regression to 0.5% cannot pass quietly. It shows that *at this
    one generating vector* the objective has its optimum at the truth and the
    best of 16 starts reaches it. It does not show the objective is identifiable
    across the parameter space -- one interior point cannot -- and it does not
    show the search is reliable: most starts stop elsewhere, which is why
    ``loss_spread`` is recorded. See RECOVERY_FINDINGS.md.
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
        assert abs(np.log(fitted / true_value)) < 5e-4, (truth, fit.parameters)


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


def test_float32_rounding_is_a_tie_and_not_a_scientific_claim():
    """The losses are float32. One ULP near a loss of 1 is ~1.2e-7, so a fixed
    1e-9 band turned rounding into two opposite conclusions -- "the search
    failed" on one side, "the objective prefers other parameters" on the other.
    """
    def result(loss_at_fit, loss_at_truth):
        return R.RecoveryResult(
            case="c", replicate=0, truth=np.array([10.0]), recovered=np.array([10.0]),
            names=("sd_feat1_c0",), loss_at_truth=loss_at_truth, loss_at_fit=loss_at_fit,
            loss_spread=0.0, n_starts=4, n_converged=4, at_bound=(), runtime_seconds=1.0)

    truth = np.float32(1.0)
    for gap in (np.float32(1.19e-7), np.float32(-1.19e-7), np.float32(-2.4e-7)):
        assert result(float(truth + gap), float(truth)).diagnosis == "loss_tied_with_truth"

    # Still sensitive to a difference that means something.
    assert result(1.001, 1.0).diagnosis == "search_failed"
    assert result(1.0, 1.001).diagnosis == "objective_prefers_other_parameters"

    # The band scales, so a loss of 1e4 is not diagnosed on its own rounding.
    assert result(10000.0 + 0.002, 10000.0).diagnosis == "loss_tied_with_truth"


def test_a_failed_loss_evaluation_raises_rather_than_reporting_a_tie():
    """Every comparison against NaN is false, so an unguarded chain falls
    through to the tie branch and a failed evaluation is recorded as agreement
    with the truth."""
    bad = R.RecoveryResult(
        case="c", replicate=3, truth=np.array([10.0]), recovered=np.array([12.0]),
        names=("sd_feat1_c0",), loss_at_truth=float("nan"), loss_at_fit=0.4,
        loss_spread=0.0, n_starts=4, n_converged=4, at_bound=(), runtime_seconds=1.0)
    with pytest.raises(ValueError, match="not a tie"):
        bad.diagnosis
def test_the_fixed_truth_summary_reports_bias_and_spread_but_not_correlation():
    """For a fixed generating vector, correlation is meaningless: every replicate
    sits at the same truth, so it measures scatter against a constant.

    Correlation belongs to the range panel instead, where the truths are drawn
    across the parameter space and "does the estimate track the parameter" is the
    actual question. That is `range_summary`, tested below."""
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
def test_a_negative_motor_sd_is_refused_rather_than_squared_away(predictor):
    """Motor noise enters as a variance, so -30 draws exactly what +30 draws
    while the case record says -30."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="non-negative"):
        R.sample_mixture_responses(predictor, 20.0, 40.0, 25.0, np.full(8, 40.0), rng,
                                   sd_motor=-30.0)


def test_a_zero_motor_override_beats_the_predictors_own_noise(predictor):
    """`0.0` means no motor noise even when the predictor carries some. Collapsing
    it to "unspecified" would have made the request silently mean its opposite."""
    noisy = predictor.with_motor_noise(30.0)
    feat_diff = np.full(8000, 40.0)

    plain = R.sample_mixture_responses(predictor, 20.0, 40.0, 25.0, feat_diff,
                                       np.random.default_rng(0))
    overridden = R.sample_mixture_responses(noisy, 20.0, 40.0, 25.0, feat_diff,
                                            np.random.default_rng(0), sd_motor=0.0)
    np.testing.assert_array_equal(plain, overridden)


@pytest.mark.development
@pytest.mark.slow
def test_a_case_with_motor_noise_is_fitted_at_that_motor_noise(predictor, grids):
    """The loop is only closed if the motor SD the data was drawn at reaches the
    fit. Generating with motor noise and fitting without it measures that
    mismatch, not recovery, and every diagnosis below inherits the error."""
    feat_grid, d_circ = grids
    case = R.RecoveryCase(name="motor", condition_feature_sds=[(20.0, 45.0)],
                          sd_spat=25.0, sd_motor=20.0, n_trials_per_condition=200)

    outcome = R.run_replicate(
        predictor, case, 0, objective="density",
        build_targets=_targets_builder(grids), fit_continuous=fit_continuous,
        score_all_conditions=S.score_all_conditions, curve_losses=_compute_curve_losses,
        energy_score=bwcrps_energy_score, d_circ_matrix=d_circ, feat_diff_grid=feat_grid,
        emp_density_weights_sd=20.0, n_starts=2, seed=0)

    # The claim under test is the wiring, not the quality of the recovery: the
    # loss reported at the truth must be the loss of the motor-carrying model.
    # Rebuild both and check which one the replicate reported.
    datasets = R.generate_case_data(predictor, case, np.random.default_rng([0, 0]))
    targets = _targets_builder(grids)(datasets)
    trials = [(jnp.asarray(v[:, 0]), jnp.asarray(v[:, 1])) for v in datasets.values()]

    def loss_through(model):
        return float(S.score_all_conditions(
            "density", model, targets, jnp.asarray(case.truth_vector()),
            curve_losses=_compute_curve_losses, energy_score=bwcrps_energy_score,
            d_circ_matrix=d_circ, feat_diff_grid=feat_grid,
            emp_density_weights_sd=20.0, condition_trials=trials, fit_motor=False))

    with_motor = loss_through(predictor.with_motor_noise(case.sd_motor))
    without_motor = loss_through(predictor)

    assert with_motor != without_motor, "the fixture cannot distinguish the two models"
    assert outcome.loss_at_truth == pytest.approx(with_motor, rel=1e-6)


@pytest.mark.development
@pytest.mark.slow
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


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------

@pytest.mark.development
@pytest.mark.slow
def test_the_panel_runner_writes_rows_and_reproducible_settings(tmp_path):
    """The runner exists because the first panel was run interactively and only
    its conclusions were kept: the truth vector, seed and start count had to be
    reverse-engineered afterwards, and the write-up could not be regenerated.

    So what this pins is that a run leaves behind enough to repeat it -- the
    settings block, the checkpoint digest, and a row per replicate -- not the
    values, which are the artifact's business.
    """
    import run_recovery_panel as panel

    out = tmp_path / "recovery"
    assert panel.main(["--panel", "noise_free", "--out", str(out), "--n-starts", "2"]) == 0

    rows = list(csv.DictReader((out / "noise_free_rows.csv").open()))
    assert len(rows) == 1
    for column in ("case", "diagnosis", "loss_at_truth", "loss_at_fit", "start_losses",
                   "true_sd_spat", "fit_sd_spat"):
        assert column in rows[0], column

    summary = json.loads((out / "noise_free_summary.json").read_text())
    settings = summary["settings"]
    assert settings["n_starts"] == 2
    assert settings["truth_conditions"] == [list(pair) for pair in panel.TRUTH_CONDITIONS]
    # The digest is what ties the numbers to a surrogate; a panel recorded
    # without it cannot be told apart from one run on a different artifact.
    assert len(settings["checkpoint_sha256"]) == 64
    assert summary["by_case"]["noise_free"]["n_replicates"] == 1


def test_the_worst_parameter_travels_with_its_name():
    """The first write-up quoted a sequence it called "the worst parameter"
    across sample sizes that in fact tracked one fixed parameter. Which one is
    worst changes with the sample size, so the name has to be reported with it.
    """
    import run_recovery_panel as panel

    def result(case, recovered):
        return R.RecoveryResult(
            case=case, replicate=0, truth=np.array([10.0, 10.0]),
            recovered=np.array(recovered), names=("a", "b"), loss_at_truth=1.0,
            loss_at_fit=1.0, loss_spread=0.0, n_starts=2, n_converged=2,
            at_bound=(), runtime_seconds=1.0)

    record = panel.worst_parameter([result("small", [30.0, 11.0]),
                                    result("large", [11.0, 3.0])])
    assert record["small_r0"]["parameter"] == "a"
    assert record["large_r0"]["parameter"] == "b"


# ---------------------------------------------------------------------------
# The range panel
# ---------------------------------------------------------------------------

def _range_result(case, truth, recovered, at_bound=()):
    return R.RecoveryResult(
        case=case, replicate=0, truth=np.asarray(truth, dtype=float),
        recovered=np.asarray(recovered, dtype=float),
        names=("sd_feat1_c0", "sd_feat2_c0", "sd_spat"), loss_at_truth=1.0,
        loss_at_fit=0.9, loss_spread=0.0, n_starts=4, n_converged=4,
        at_bound=at_bound, runtime_seconds=1.0)


def test_the_range_summary_pools_by_parameter_family():
    """`sd_feat1_c0` and `sd_feat2_c7` are the same quantity in different slots;
    a correlation computed per slot would be a dozen tiny samples instead of one
    usable one."""
    rng = np.random.default_rng(0)
    results = []
    for index in range(30):
        truth = np.exp(rng.uniform(np.log(3), np.log(180), 3))
        results.append(_range_result(f"r{index}", truth, truth * 1.05))

    summary = R.range_summary(results)
    assert set(summary) >= {"sd_feat", "sd_spat"}
    assert summary["sd_feat"]["n"] == 60      # two slots per replicate
    assert summary["sd_spat"]["n"] == 30


def test_slope_catches_compression_that_correlation_hides():
    """The reason correlation is not reported alone. An estimator that returns
    the square root of the truth tracks it almost perfectly by correlation while
    being badly wrong, and only the slope says so."""
    rng = np.random.default_rng(1)
    results = []
    for index in range(40):
        truth = np.exp(rng.uniform(np.log(3), np.log(180), 3))
        # Compressed toward the middle of the range: a slope of 0.5 in logs.
        results.append(_range_result(f"r{index}", truth, np.sqrt(truth * 20.0)))

    summary = R.range_summary(results)
    assert summary["sd_feat"]["pearson_r_log"] > 0.99
    assert summary["sd_feat"]["slope_log"] == pytest.approx(0.5, abs=0.02)
    assert summary["sd_feat"]["rmse_log_ratio"] > 0.3


def test_railed_parameters_are_counted_and_excluded_by_default():
    """A fit sitting on a bound reports the bound's position, not an estimate.
    Averaging those in measures where the bounds were put."""
    results = [_range_result("a", [10.0, 20.0, 30.0], [10.5, 2.5, 31.0],
                             at_bound=("sd_feat2_c0@low",)),
               _range_result("b", [15.0, 25.0, 35.0], [15.5, 26.0, 36.0])]

    kept = R.range_summary(results, drop_railed=False)
    dropped = R.range_summary(results, drop_railed=True)

    assert dropped["sd_feat"]["n"] == 3 and kept["sd_feat"]["n"] == 4
    assert dropped["sd_feat"]["n_railed"] == 1
    assert dropped["sd_feat"]["railed_fraction"] == pytest.approx(0.25)
    # The railed value is far from its truth, so keeping it inflates the error.
    assert kept["sd_feat"]["rmse_log_ratio"] > dropped["sd_feat"]["rmse_log_ratio"]


def test_random_cases_span_the_range_and_stay_inside_the_bounds():
    """Log-uniform, because the bounds span two decades on a multiplicative
    scale: a uniform draw would leave the narrow corner -- the regime the mixture
    exists to represent -- almost unsampled."""
    import run_recovery_panel as panel

    bounds = {"sd_feat": (2.5, 200.0), "sd_spat": (5.0, 200.0)}
    cases = panel.random_cases(400, 2, 500, bounds, np.random.default_rng(0))

    feats = np.array([sd for case in cases for pair in case.condition_feature_sds
                      for sd in pair])
    spats = np.array([case.sd_spat for case in cases])

    assert np.all(feats > 2.5) and np.all(feats < 200.0)
    assert np.all(spats > 5.0) and np.all(spats < 200.0)
    # Both decades populated, which a uniform draw would not manage.
    assert (feats < 10).mean() > 0.15, "narrow corner undersampled"
    assert (feats > 100).mean() > 0.10
    # Each case has its own truth, and conditions differ within a case.
    assert len({case.sd_spat for case in cases}) == len(cases)


def test_the_same_seed_gives_the_same_truths_at_every_trial_count():
    """What makes the range panel a paired design: the truths depend on the seed
    and the case count, never on the trial count. So cells that match on the
    recorded `truth_digest` differ in the trial count and nothing else.

    Correlation depends on how the design happened to span the range and on how
    many points estimated it, so cells drawn from different truths -- or from
    different numbers of them -- are not comparable, however similar the settings
    otherwise look.
    """
    import run_recovery_panel as panel

    bounds = {"sd_feat": (2.5, 200.0), "sd_spat": (5.0, 200.0)}

    def truths(n_cases, n_trials):
        cases = panel.random_cases(n_cases, 2, n_trials, bounds,
                                   np.random.default_rng(0))
        return np.concatenate([case.truth_vector() for case in cases])

    np.testing.assert_array_equal(truths(50, 100), truths(50, 10000))
    # A different case count is a different design, not a shorter one to compare
    # against: the shared prefix is the same but the correlation is over a
    # different sample.
    assert len(truths(25, 100)) != len(truths(50, 100))


@pytest.mark.development
@pytest.mark.slow
def test_running_cases_in_parallel_gives_the_same_numbers(tmp_path):
    """Parallelism here must be a wall-clock change and nothing else.

    Each case is an independent deterministic L-BFGS-B search over data seeded
    from its own case, so nothing crosses between workers -- but that is the kind
    of claim that is easy to assert and easy to have wrong, and a panel whose
    numbers depended on the worker count would be worthless. Checked on the
    fitted parameters and both losses, not just the diagnosis. The diagnosis must
    match exactly. The numbers need not be bit-identical: workers run XLA
    single-threaded (see ``main``) while the serial run uses multithreaded Eigen,
    so float32 reductions are ordered differently, and on this small panel
    L-BFGS-B carries that last-bit difference into the fitted point at ~0.1%.
    Losses stay within float32 rounding; a case crossing between workers would
    move any of these by far more than the tolerances.
    """
    import run_recovery_panel as panel

    common = ["--panel", "random", "--n-cases", "4", "--n-replicates", "1",
              "--n-trials", "300", "--n-starts", "3", "--seed", "0"]
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    assert panel.main(common + ["--out", str(serial)]) == 0
    assert panel.main(common + ["--out", str(parallel), "--n-workers", "4"]) == 0

    def rows(directory):
        return sorted(csv.DictReader((directory / "random_rows.csv").open()),
                      key=lambda row: row["case"])

    left, right = rows(serial), rows(parallel)
    assert len(left) == len(right) == 4
    for one, other in zip(left, right):
        assert one["case"] == other["case"]
        for column, value in one.items():
            if column == "diagnosis":
                assert value == other[column], (one["case"], column)
            elif column.startswith("true_") or column in ("loss_at_fit", "loss_at_truth"):
                assert float(value) == pytest.approx(
                    float(other[column]), rel=1e-5, abs=1e-6), (one["case"], column)
            elif column.startswith(("fit_", "log_ratio_")):
                assert float(value) == pytest.approx(
                    float(other[column]), rel=1e-2, abs=1e-3), (one["case"], column)


def test_the_start_sweep_uses_the_worst_cases_and_optimizer_truth_layout(
        tmp_path, monkeypatch):
    """Condition-major, and parsed rather than sorted.

    The optimizer's layout is [feat1_c0, feat2_c0, feat1_c1, feat2_c1, sd_spat].
    Sorting the column names alphabetically gives [feat1_c0, feat1_c1, feat2_c0,
    feat2_c1, sd_spat] -- condition-minor -- which hands condition 0 a pair of
    SDs drawn from two different conditions. Every loss stays finite and the
    summary looks plausible, so nothing downstream would have caught it; the
    only reason it surfaced was disagreeing with an earlier ad-hoc run.

    Exercise the runner itself: the first regression test reconstructed the
    parser in test code and merely checked that this function existed, which
    could not catch either the production ordering or selection error.
    """
    import run_recovery_panel as panel

    rows = pd.DataFrame({
        "case": ["moderate", "worst", "below", "second"],
        "log_ratio_sd_feat1_c0": [1.1, 4.0, 0.5, -3.0],
        "log_ratio_sd_feat2_c1": [0.2, 0.1, 0.3, 0.4],
        "true_sd_feat1_c0": [11.0, 14.0, 10.5, 13.0],
        "true_sd_feat2_c0": [21.0, 24.0, 20.5, 23.0],
        "true_sd_feat1_c1": [31.0, 34.0, 30.5, 33.0],
        "true_sd_feat2_c1": [41.0, 44.0, 40.5, 43.0],
        "true_sd_spat": [51.0, 54.0, 50.5, 53.0],
    })
    source = tmp_path / "rows.csv"
    rows.to_csv(source, index=False)

    class Fit:
        loss = 0.0
        starts = []

    truths = []

    def fake_fit(predictor, feat_grid, truth, n_starts, seed):
        truths.append(truth)
        fit = Fit()
        fit.parameters = np.asarray(truth)
        return fit, lambda parameters: 0.0

    monkeypatch.setattr(panel, "_noise_free_fit", fake_fit)

    result = panel.run_start_sweep(
        None, None, source, [1], seed=0, worst_above=1.0, limit=2)

    assert list(result["case"]) == ["worst", "second"]
    assert truths == [[14.0, 24.0, 34.0, 44.0, 54.0],
                      [13.0, 23.0, 33.0, 43.0, 53.0]]
