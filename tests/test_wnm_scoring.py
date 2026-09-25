"""Every objective must score against the mixture, through the shared definitions.

The public fitting command writes every objective's score at each fitted method's
parameters, not just the one it was asked to fit. So a surrogate that can be fit
under ``density`` but not scored under ``crps`` would produce a results table
mixing two families column by column, with nothing in the file saying so. These
tests exist to keep that from being possible.

What they check is mostly *wiring*, deliberately: that each method reads its own
target, from the right condition, through the same loss helper the surface branch
calls. The numerical content of those losses is already pinned by the surface
suites, and re-asserting it here against a second implementation would defeat the
point of there being only one.
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

import wnm_scoring as S  # noqa: E402
from density_objective import degenerate_targets  # noqa: E402
from fitting_targets import build_fitting_targets  # noqa: E402
from grid_based_multi_condition_optimizer_jax_loops import (  # noqa: E402
    _compute_curve_losses, bwcrps_energy_score, compute_bwcrps_condition_targets,
    compute_target_bias_curve_core, generate_nn_matched_bias_curve_batch,
    generate_nn_matched_density_asymmetry_batch)
from shared import surrogate  # noqa: E402
from shared.config import config  # noqa: E402
from shared.prediction import predictor_from_surrogate  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "data" / "fitting_targets_golden.npz"
ARTIFACT = surrogate.WNM_DEFAULTS[20]
pytestmark = pytest.mark.skipif(
    not (GOLDEN.exists() and ARTIFACT.exists()),
    reason="needs the target reference and a packaged WNM artifact")

CURVE_METHODS = ("density", "density_legacy", "expectation", "smoothed_exp",
                 "balanced_crps", "bias_weighted_crps")
TRIAL_METHODS = ("likelihood", "crps")


@pytest.fixture(scope="module")
def d_circ():
    grid = config.create_grid('mu1_bias')
    diff = jnp.abs(grid[:, None] - grid[None, :])
    return jnp.minimum(diff, 360.0 - diff)


@pytest.fixture(scope="module")
def datasets():
    golden = np.load(GOLDEN)
    prefix = "unequal_180/input/"
    return {key[len(prefix):]: golden[key]
            for key in golden.files if key.startswith(prefix)}


def _targets(datasets, d_circ):
    return build_fitting_targets(
        {name: jnp.asarray(values) for name, values in datasets.items()},
        feat_diff_grid=config.create_grid('feat_diff'), d_circ_matrix=d_circ,
        n_mu1_bias=len(config.create_grid('mu1_bias')), emp_density_weights_sd=20.0,
        density_bandwidth_rule="sj", density_bandwidth_mode="pooled",
        degenerate_targets=degenerate_targets,
        bwcrps_condition_targets=compute_bwcrps_condition_targets,
        target_bias_curve_core=compute_target_bias_curve_core)


@pytest.fixture(scope="module")
def targets(datasets, d_circ):
    return _targets(datasets, d_circ)


@pytest.fixture(scope="module")
def predictor():
    return predictor_from_surrogate(surrogate.load_surrogate(checkpoint_path=ARTIFACT))


@pytest.fixture(scope="module")
def trials(datasets):
    return [(jnp.asarray(v[:, 0]), jnp.asarray(v[:, 1])) for v in datasets.values()]


def _score(method, predictor, targets, d_circ, params, trials=None):
    return S.score_all_conditions(
        method, predictor, targets, jnp.asarray(params), curve_losses=_compute_curve_losses,
        energy_score=bwcrps_energy_score, d_circ_matrix=d_circ,
        feat_diff_grid=config.create_grid('feat_diff'),
        emp_density_weights_sd=20.0, condition_trials=trials)


PARAMS = [20.0, 35.0, 25.0, 40.0, 30.0, 30.0, 22.0]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_every_method_the_command_writes_can_be_scored():
    """Anything missing here would leave a results row half-populated."""
    import fit_model_to_data as F

    assert set(F.LOSS_EVALUATION_METHODS) <= set(S.SUPPORTED_METHODS), (
        set(F.LOSS_EVALUATION_METHODS) - set(S.SUPPORTED_METHODS))


@pytest.mark.parametrize("method", CURVE_METHODS + TRIAL_METHODS)
def test_each_method_produces_one_finite_scalar(method, predictor, targets, d_circ, trials):
    loss = _score(method, predictor, targets, d_circ, PARAMS,
                  trials if method in TRIAL_METHODS else None)
    assert np.ndim(loss) == 0
    assert np.isfinite(float(loss))


def test_an_unknown_method_raises_rather_than_scoring_something_else(predictor, targets, d_circ):
    with pytest.raises(ValueError, match="unknown fitting method"):
        _score("mse", predictor, targets, d_circ, PARAMS)


def test_trial_summed_methods_refuse_to_run_without_trials(predictor, targets, d_circ):
    """Silently scoring them on curves would put a different quantity in the column."""
    for method in TRIAL_METHODS:
        with pytest.raises(ValueError, match="trial-summed"):
            _score(method, predictor, targets, d_circ, PARAMS, trials=None)


def test_the_parameter_vector_length_must_be_exact(predictor, targets, d_circ):
    """Not "at least": a trailing sd_motor would be silently dropped, and the fit
    scored without the motor noise its own record claims it was fitted with."""
    with pytest.raises(ValueError, match="expected exactly"):
        _score("density", predictor, targets, d_circ, [20.0, 35.0, 22.0])
    with pytest.raises(ValueError, match="Motor noise is carried by the predictor"):
        _score("density", predictor, targets, d_circ,
               [20.0, 35.0, 25.0, 40.0, 30.0, 30.0, 22.0, 15.0])


def test_a_trial_list_of_the_wrong_length_is_refused(predictor, targets, d_circ, trials):
    with pytest.raises(ValueError, match="trial arrays for"):
        _score("likelihood", predictor, targets, d_circ, PARAMS, trials[:-1])


def test_the_scorer_uses_the_grid_it_is_given(predictor, targets, d_circ):
    """It used to rebuild the configured grid, so a shifted grid of the same
    length was validated, accepted, and then ignored."""
    on_grid = float(S.score_all_conditions(
        "density", predictor, targets, jnp.asarray(PARAMS), curve_losses=_compute_curve_losses,
        energy_score=bwcrps_energy_score, d_circ_matrix=d_circ,
        feat_diff_grid=config.create_grid('feat_diff'), emp_density_weights_sd=20.0))
    off_grid = float(S.score_all_conditions(
        "density", predictor, targets, jnp.asarray(PARAMS), curve_losses=_compute_curve_losses,
        energy_score=bwcrps_energy_score, d_circ_matrix=d_circ,
        feat_diff_grid=config.create_grid('feat_diff') - 1.0, emp_density_weights_sd=20.0))
    assert not np.isclose(on_grid, off_grid, rtol=1e-6)


# ---------------------------------------------------------------------------
# Wiring: the right target, for the right condition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", CURVE_METHODS)
def test_each_condition_is_scored_against_its_own_target(method, predictor, datasets, d_circ):
    """Reversing the conditions must reverse which target each is scored against.

    An off-by-one or a fixed index here gives every condition a plausible loss
    computed against another condition's data.
    """
    forward = _targets(datasets, d_circ)
    reversed_datasets = {k: datasets[k] for k in reversed(list(datasets))}
    backward = _targets(reversed_datasets, d_circ)

    params = [20.0, 35.0, 25.0, 40.0, 30.0, 30.0, 22.0]
    reversed_params = [30.0, 30.0, 25.0, 40.0, 20.0, 35.0, 22.0]

    assert float(_score(method, predictor, forward, d_circ, params)) == pytest.approx(
        float(_score(method, predictor, backward, d_circ, reversed_params)), rel=1e-5)


@pytest.mark.parametrize("method", CURVE_METHODS)
def test_the_score_responds_to_the_parameters(method, predictor, targets, d_circ):
    """A score that ignored its parameters would fit anything equally well."""
    base = float(_score(method, predictor, targets, d_circ, PARAMS))
    moved = float(_score(method, predictor, targets, d_circ,
                         [60.0, 70.0, 55.0, 80.0, 65.0, 75.0, 40.0]))
    assert not np.isclose(base, moved, rtol=1e-4)


def test_the_two_density_objectives_are_different_objectives(predictor, targets, d_circ):
    """density is CCC; density_legacy is the pre-2026-08 weighted MSE plus
    1 - correlation, kept only so published numbers stay explicable."""
    assert not np.isclose(float(_score("density", predictor, targets, d_circ, PARAMS)),
                          float(_score("density_legacy", predictor, targets, d_circ, PARAMS)),
                          rtol=1e-3)


def test_density_uses_the_matched_kde_and_observed_design_operator(
        predictor, d_circ, datasets):
    one = _targets({next(iter(datasets)): next(iter(datasets.values()))}, d_circ)
    grid = config.create_grid("feat_diff")
    rows = S.condition_rows(PARAMS[0], PARAMS[1], PARAMS[-1], grid)
    raw = predictor.signed_arc_asymmetry(
        rows, validate=False, sd_motor=one.density_bandwidth[0])
    predicted = one.feature_operator[0] @ raw
    expected = _compute_curve_losses(
        predicted[None, :], one.matched_density_target[0][None, :],
        loss_type="ccc", is_angular=False)[0]
    got = _score("density", predictor, one, d_circ,
                 [PARAMS[0], PARAMS[1], PARAMS[-1]])
    assert float(got) == pytest.approx(float(expected), rel=1e-6)


def test_smoothed_exp_pools_complex_moments_on_the_observed_design(
        predictor, d_circ, datasets):
    one = _targets({next(iter(datasets)): next(iter(datasets.values()))}, d_circ)
    grid = config.create_grid("feat_diff")
    rows = S.condition_rows(PARAMS[0], PARAMS[1], PARAMS[-1], grid)
    mean, resultant = predictor.mean_and_resultant(rows, validate=False)
    radians = jnp.radians(mean)
    operator = one.feature_operator[0]
    predicted = jnp.degrees(jnp.arctan2(
        operator @ (resultant * jnp.sin(radians)),
        operator @ (resultant * jnp.cos(radians))))
    expected = _compute_curve_losses(
        predicted[None, :], one.target_bias_curve[0][None, :],
        loss_type="mse", is_angular=True,
        weights=one.smoothed_support[0][None, :])[0]
    got = _score("smoothed_exp", predictor, one, d_circ,
                 [PARAMS[0], PARAMS[1], PARAMS[-1]])
    assert float(got) == pytest.approx(float(expected), rel=1e-6)


@pytest.mark.parametrize("method", ["density", "smoothed_exp"])
def test_packed_exact_loss_matches_condition_loop_without_padding(
        method, predictor, datasets, d_circ):
    exact = build_fitting_targets(
        {name: jnp.asarray(values) for name, values in datasets.items()},
        feat_diff_grid=config.create_grid('feat_diff'), d_circ_matrix=d_circ,
        n_mu1_bias=len(config.create_grid('mu1_bias')), emp_density_weights_sd=20.0,
        density_bandwidth_rule="sj", density_bandwidth_mode="pooled",
        degenerate_targets=degenerate_targets,
        bwcrps_condition_targets=compute_bwcrps_condition_targets,
        target_bias_curve_core=compute_target_bias_curve_core,
        feature_coordinate_mode="exact")
    target = (exact.matched_density_target if method == "density"
              else exact.target_bias_curve)
    packed = S.packed_curve_loss(
        method, predictor, jnp.asarray(PARAMS), exact.prediction_coordinates,
        exact.prediction_condition_index, exact.feature_operator, target,
        exact.smoothed_support, jnp.asarray(exact.density_bandwidth),
        curve_losses=_compute_curve_losses)
    loop = _score(method, predictor, exact, d_circ, PARAMS)
    assert float(packed) == pytest.approx(float(loop), rel=2e-6, abs=2e-6)


def test_exact_padding_has_zero_effect(predictor, datasets, d_circ):
    kwargs = dict(
        condition_datasets={name: jnp.asarray(values) for name, values in datasets.items()},
        feat_diff_grid=config.create_grid('feat_diff'), d_circ_matrix=d_circ,
        n_mu1_bias=len(config.create_grid('mu1_bias')), emp_density_weights_sd=20.0,
        density_bandwidth_rule="sj", density_bandwidth_mode="pooled",
        degenerate_targets=degenerate_targets,
        bwcrps_condition_targets=compute_bwcrps_condition_targets,
        target_bias_curve_core=compute_target_bias_curve_core,
        feature_coordinate_mode="exact")
    exact = build_fitting_targets(**kwargs)
    padded = build_fitting_targets(
        **kwargs, prediction_capacity=exact.prediction_coordinate_count + 64)
    for targets_ in (exact, padded):
        value = S.packed_curve_loss(
            "smoothed_exp", predictor, jnp.asarray(PARAMS),
            targets_.prediction_coordinates, targets_.prediction_condition_index,
            targets_.feature_operator, targets_.target_bias_curve,
            targets_.smoothed_support, jnp.asarray(targets_.density_bandwidth),
            curve_losses=_compute_curve_losses)
        if targets_ is exact:
            reference = value
    assert float(value) == pytest.approx(float(reference), rel=1e-7, abs=1e-7)


def test_exact_wnm_evaluates_nonzero_coordinates_below_artifact_support(
        predictor, datasets, d_circ):
    name, values = next(iter(datasets.items()))
    values = values.copy()
    values[:5, 0] = 0.25
    exact = build_fitting_targets(
        {name: jnp.asarray(values)}, feat_diff_grid=config.create_grid('feat_diff'),
        d_circ_matrix=d_circ, n_mu1_bias=len(config.create_grid('mu1_bias')),
        emp_density_weights_sd=20.0, density_bandwidth_rule="sj",
        density_bandwidth_mode="pooled", degenerate_targets=degenerate_targets,
        bwcrps_condition_targets=compute_bwcrps_condition_targets,
        target_bias_curve_core=compute_target_bias_curve_core,
        feature_coordinate_mode="exact")
    assert np.any(np.asarray(exact.prediction_coordinates) == np.float32(0.25))
    loss = S.packed_curve_loss(
        "density", predictor, jnp.asarray([20.0, 35.0, 22.0]),
        exact.prediction_coordinates, exact.prediction_condition_index,
        exact.feature_operator, exact.matched_density_target,
        exact.smoothed_support, jnp.asarray(exact.density_bandwidth),
        curve_losses=_compute_curve_losses)
    assert np.isfinite(float(loss))


def test_surface_curve_operators_match_the_current_wnm_semantics(
        predictor, d_circ, datasets):
    """A sampled WNM surface and the analytic WNM must feed the same operators."""
    one = _targets({next(iter(datasets)): next(iter(datasets.values()))}, d_circ)
    feat_grid = config.create_grid("feat_diff")
    rows = S.condition_rows(PARAMS[0], PARAMS[1], PARAMS[-1], feat_grid)
    log_surface = predictor.grid_log_density(
        rows, grid=config.create_grid("mu1_bias"), validate=False).T[None, :, :]
    operator = one.feature_operator[:1]

    surface_bias = generate_nn_matched_bias_curve_batch(log_surface, operator)[0]
    analytic_bias = S.predicted_matched_mean_bias(
        predictor, PARAMS[0], PARAMS[1], PARAMS[-1], feat_grid, operator[0])
    np.testing.assert_allclose(np.asarray(surface_bias), np.asarray(analytic_bias),
                               atol=1e-5, rtol=1e-5)

    bandwidth = one.density_bandwidth[0]
    surface_density = generate_nn_matched_density_asymmetry_batch(
        log_surface, operator, jnp.asarray([bandwidth]))[0]
    analytic_density = S.predicted_matched_density_curve(
        predictor, PARAMS[0], PARAMS[1], PARAMS[-1], feat_grid,
        operator[0], bandwidth)
    # The surface path integrates a 2-degree sampled density; the WNM path uses
    # the corresponding analytic wrapped-normal arc probability.
    np.testing.assert_allclose(np.asarray(surface_density), np.asarray(analytic_density),
                               atol=3e-4, rtol=3e-4)


# ---------------------------------------------------------------------------
# Motor noise reaches every score
# ---------------------------------------------------------------------------

DISTRIBUTIONAL_METHODS = tuple(m for m in CURVE_METHODS + TRIAL_METHODS
                               if m not in S.MEAN_ONLY_METHODS)


@pytest.mark.parametrize("method", DISTRIBUTIONAL_METHODS)
def test_motor_noise_changes_every_distributional_score(method, predictor, targets,
                                                        d_circ, trials):
    """The failure this prevents: a fit smoothed by motor noise, cross-scored
    without it, so the comparison across objectives is not at one model."""
    use_trials = trials if method in TRIAL_METHODS else None
    without = float(_score(method, predictor, targets, d_circ, PARAMS, use_trials))
    with_motor = float(_score(method, predictor.with_motor_noise(25.0), targets, d_circ,
                              PARAMS, use_trials))
    assert not np.isclose(without, with_motor, rtol=1e-4)


@pytest.mark.parametrize("method", S.MEAN_ONLY_METHODS)
def test_mean_only_objectives_cannot_see_motor_noise_at_all(method, predictor, targets,
                                                            d_circ):
    """Not a gap in the wiring -- a property of the model, and the reason the
    fitter skips these objectives when the motor SD is free.

    Motor noise is a zero-mean symmetric wrapped normal, and convolving with one
    leaves the circular mean exactly where it was while shrinking the resultant.
    Verified directly: at sd_motor 0, 10, 40 and 120 the mean is identical to
    float32 while the resultant falls from 0.94 to 0.10. An objective built on
    the mean therefore carries no information about the motor SD, so fitting it
    free would return whatever the optimiser happened to start from and report it
    as an estimate.
    """
    without = float(_score(method, predictor, targets, d_circ, PARAMS))
    for sd_motor in (10.0, 40.0, 120.0):
        with_motor = float(_score(method, predictor.with_motor_noise(sd_motor), targets,
                                  d_circ, PARAMS))
        assert np.isclose(without, with_motor, rtol=1e-5), (
            f"{method} moved with sd_motor={sd_motor}, which would mean the mean-only "
            "objectives had become sensitive to it")


# ---------------------------------------------------------------------------
# Differentiability, which is what makes the continuous search possible
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", CURVE_METHODS)
def test_curve_objectives_are_differentiable(method, predictor, targets, d_circ):
    gradient = jax.grad(lambda p: _score(method, predictor, targets, d_circ, p))(
        jnp.asarray(PARAMS))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_likelihood_is_differentiable(predictor, targets, d_circ, trials):
    gradient = jax.grad(lambda p: _score("likelihood", predictor, targets, d_circ, p, trials))(
        jnp.asarray(PARAMS))
    assert np.all(np.isfinite(np.asarray(gradient)))


# ---------------------------------------------------------------------------
# Refusals and the fixed grid
# ---------------------------------------------------------------------------

def test_a_degenerate_density_target_is_refused_but_only_for_density(predictor, d_circ):
    """A flat density target says nothing about whether the condition can be fit
    by likelihood or CRPS, so the refusal must not reach those."""
    rng = np.random.default_rng(3)
    # Bias independent of feature difference gives a near-constant signed-mass curve.
    flat = np.stack([rng.uniform(2.0, 180.0, 400), rng.normal(0.0, 40.0, 400)], axis=-1)
    targets = _targets({"flat": flat.astype(np.float32)}, d_circ)
    if not bool(np.asarray(targets.matched_density_degenerate)[0]):
        pytest.skip("fixture did not produce a degenerate target on this build")

    with pytest.raises(ValueError, match="constant density target"):
        _score("density", predictor, targets, d_circ, [20.0, 35.0, 22.0])
    assert np.isfinite(float(_score("balanced_crps", predictor, targets, d_circ,
                                    [20.0, 35.0, 22.0])))


def test_the_feature_grid_is_validated_once_against_the_surrogate(predictor):
    """The per-evaluation path skips validation so it stays differentiable, so the
    grid it will be evaluated on has to be checked before fitting starts."""
    S.validate_feature_grid(config.create_grid('feat_diff'), predictor)

    with pytest.raises(ValueError, match="feat_diff"):
        S.validate_feature_grid(jnp.arange(0.0, 180.0, 2.0), predictor)  # starts below 0.5
    with pytest.raises(ValueError, match="evenly spaced"):
        S.validate_feature_grid(jnp.asarray([2.0, 4.0, 10.0, 12.0]), predictor)


def test_the_model_smoother_width_follows_the_empirical_feature_weights():
    """Both curves must be smoothed by the same operation at the same width, or
    the objective compares differently blurred things."""
    assert S._smoothing_sigma(20.0, None) == 20.0 / config.feat_diff_step
    assert S._smoothing_sigma(20.0, 7.0) == 7.0
