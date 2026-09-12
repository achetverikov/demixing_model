"""The curve families the subject plots draw, computed for a mixture fit.

The surface backend derives these by integrating its 180-row grid. The mixture
has a closed form for each, so it computes them directly. That is not an
optimisation: a component narrower than the 2-degree reporting cell is mis-massed
by the grid, and those fits are the reason for the surrogate.

These tests check the bundle against the mixture's own grid route where a grid
route exists, and pin the properties that would let a plotted curve be silently
wrong -- a motor SD paired with the wrong row, a curve that ignores its
parameters, an asymmetry that skipped the density target's smoother.

The surface path is deliberately untouched by the code under test here, so no
number on that side can move; its own behaviour is pinned in
``test_plot_estimators.py``.
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

from shared import surrogate  # noqa: E402
from shared.config import config  # noqa: E402
from shared.prediction import (  # noqa: E402
    mixture_plot_curves, pooled_bias_weighted_crps, predictor_from_surrogate)
from shared.utils import gaussian_curve_smoother  # noqa: E402
import create_unified_subject_plots as subject_plots  # noqa: E402

ARTIFACT = surrogate.WNM_DEFAULTS[20]
pytestmark = pytest.mark.skipif(not ARTIFACT.exists(),
                                reason="no packaged WNM artifact")

PARAMS = np.array([[25.0, 40.0, 30.0], [10.0, 60.0, 20.0], [3.0, 3.0, 8.0]])


@pytest.fixture(scope="module")
def predictor():
    return predictor_from_surrogate(surrogate.load_surrogate(checkpoint_path=ARTIFACT))


@pytest.fixture(scope="module")
def feat_grid():
    return np.asarray(config.create_grid('feat_diff'), dtype=np.float32)


def _weights(n_rows, n_feat, n_bins=18):
    """One populated column per bin, so pooled SD reduces to the unpooled value."""
    weights = np.zeros((n_rows, n_bins, n_feat))
    for row in range(n_rows):
        for index in range(n_bins):
            weights[row, index, min(index * 5, n_feat - 1)] = 1.0
    return weights


def _stored_result(feat_grid, params):
    operator = np.eye(len(feat_grid), dtype=np.float32)
    data = np.array([[2.0, -3.0], [10.0, 4.0], [20.0, 1.0]], dtype=np.float32)
    return {
        "condition": "S#exp#condition",
        "data_df": data,
        "density_fitted_params": np.asarray(params, dtype=np.float32),
        "density_loss": 1.0,
        "density_evaluation_losses": {},
        "empirical_curves": {
            "target_bias": np.zeros(45),
            "bias_weights": np.ones(45),
            "target_density": np.zeros(len(feat_grid)),
            "matched_density_target": np.zeros(len(feat_grid)),
            "target_bias_curve": np.zeros(len(feat_grid)),
            "bias_feat_indices": np.arange(0, len(feat_grid), 2),
            "density_feat_grid": feat_grid,
            "feature_operator": operator,
            "density_bandwidth": 7.5,
        },
    }


def test_every_curve_the_plots_draw_is_produced(predictor, feat_grid):
    bundle = mixture_plot_curves(predictor, PARAMS, feat_grid,
                                 bin_weights=_weights(len(PARAMS), len(feat_grid)))
    assert set(bundle) == {"bias", "asymmetry", "sd", "pooled_sd"}
    for name in ("bias", "asymmetry", "sd"):
        assert bundle[name].shape == (len(PARAMS), len(feat_grid)), name
        assert np.all(np.isfinite(bundle[name])), name
    assert bundle["pooled_sd"].shape == (len(PARAMS), 18)


def test_each_row_uses_its_own_parameters(predictor, feat_grid):
    """A bundle that ignored its rows would draw one fit's curve for every fit."""
    bundle = mixture_plot_curves(predictor, PARAMS, feat_grid)
    for name in ("bias", "asymmetry", "sd"):
        assert not np.allclose(bundle[name][0], bundle[name][1]), name
        assert not np.allclose(bundle[name][1], bundle[name][2]), name


def test_motor_noise_is_paired_with_its_own_row(predictor, feat_grid):
    """Motor noise is a fitted parameter, so it belongs to a row. Pairing it
    positionally is easy to get wrong and impossible to see in a plot."""
    without = mixture_plot_curves(predictor, PARAMS, feat_grid)
    with_motor = mixture_plot_curves(predictor, PARAMS, feat_grid,
                                     sd_motor_by_row=[0.0, 25.0, 0.0])

    # Only the row that was given motor noise may move.
    np.testing.assert_array_equal(without["sd"][0], with_motor["sd"][0])
    np.testing.assert_array_equal(without["sd"][2], with_motor["sd"][2])
    assert np.all(with_motor["sd"][1] > without["sd"][1] - 1e-6)
    assert np.max(with_motor["sd"][1] - without["sd"][1]) > 1.0


def test_a_zero_row_overrides_a_motor_carrying_predictor(predictor, feat_grid):
    """A row of 0 means no motor noise. Previously 0 was collapsed to
    "unspecified", which means "use the predictor's own SD" -- so on a predictor
    built at motor SD 30 the request for no motor noise silently returned the
    30-degree curve, differing from the true one by over 20 degrees of SD.

    Every other motor test here uses a zero-motor base predictor, where the two
    readings coincide, so none of them can see this.
    """
    noisy = predictor.with_motor_noise(30.0)
    overridden = mixture_plot_curves(noisy, PARAMS, feat_grid, sd_motor_by_row=[0.0] * 3)
    clean = mixture_plot_curves(predictor, PARAMS, feat_grid)

    np.testing.assert_allclose(overridden["sd"], clean["sd"], rtol=1e-6)
    # And the two models are far enough apart that the check is not vacuous.
    kept = mixture_plot_curves(noisy, PARAMS, feat_grid, sd_motor_by_row=[30.0] * 3)
    assert np.max(kept["sd"] - clean["sd"]) > 5.0


def test_an_out_of_domain_parameter_raises_instead_of_plotting_nan(predictor, feat_grid):
    """The helper ran every predictor call with validation off, so a negative SD
    produced a NaN curve that a plot renders as a gap rather than an error."""
    with pytest.raises(ValueError):
        mixture_plot_curves(predictor, np.array([[-1.0, 40.0, 30.0]]), feat_grid)


def test_a_negative_motor_sd_is_refused(predictor, feat_grid):
    with pytest.raises(ValueError, match="non-negative"):
        mixture_plot_curves(predictor, PARAMS, feat_grid, sd_motor_by_row=[0.0, -5.0, 0.0])


def test_a_motor_length_mismatch_is_refused(predictor, feat_grid):
    with pytest.raises(ValueError, match="paired positionally"):
        mixture_plot_curves(predictor, PARAMS, feat_grid, sd_motor_by_row=[0.0, 5.0])


def test_motor_noise_leaves_the_bias_curve_alone(predictor, feat_grid):
    """A zero-mean symmetric convolution cannot move a circular mean. If this
    ever fails, the motor model has stopped being symmetric."""
    without = mixture_plot_curves(predictor, PARAMS, feat_grid)
    with_motor = mixture_plot_curves(predictor, PARAMS, feat_grid,
                                     sd_motor_by_row=[30.0, 30.0, 30.0])
    np.testing.assert_allclose(without["bias"], with_motor["bias"], atol=1e-4)


def test_the_asymmetry_curve_carries_the_density_target_smoother(predictor, feat_grid):
    """The plotted asymmetry must be the quantity the density objective fits, or
    the panel shows a curve the fit never optimised."""
    bundle = mixture_plot_curves(predictor, PARAMS, feat_grid)

    rows = jnp.stack([
        jnp.full(feat_grid.shape, 25.0, jnp.float32),
        jnp.full(feat_grid.shape, 40.0, jnp.float32),
        jnp.full(feat_grid.shape, 30.0, jnp.float32),
        jnp.asarray(feat_grid)], axis=-1)
    raw = predictor.signed_arc_asymmetry(rows, validate=False)
    expected = gaussian_curve_smoother(raw, 20.0 / config.feat_diff_step)

    np.testing.assert_allclose(bundle["asymmetry"][0], np.asarray(expected), rtol=1e-6)
    assert not np.allclose(bundle["asymmetry"][0], np.asarray(raw))


def test_selected_curve_operators_are_applied_directly(predictor, feat_grid):
    operators = np.stack([np.eye(len(feat_grid), dtype=np.float32)] * len(PARAMS))
    bandwidths = np.asarray([5.0, 7.0, 9.0])
    bundle = mixture_plot_curves(
        predictor, PARAMS, feat_grid, feature_operators=operators,
        density_bandwidths=bandwidths)
    rows = jnp.column_stack([
        jnp.full(len(feat_grid), PARAMS[0, 0]),
        jnp.full(len(feat_grid), PARAMS[0, 1]),
        jnp.full(len(feat_grid), PARAMS[0, 2]),
        jnp.asarray(feat_grid)])
    expected = predictor.signed_arc_asymmetry(
        rows, validate=False, sd_motor=bandwidths[0])
    np.testing.assert_allclose(bundle["asymmetry"][0], expected, rtol=1e-6)


def test_selected_curve_inputs_must_be_paired(predictor, feat_grid):
    operators = np.stack([np.eye(len(feat_grid), dtype=np.float32)] * len(PARAMS))
    with pytest.raises(ValueError, match="supplied together"):
        mixture_plot_curves(predictor, PARAMS, feat_grid,
                            feature_operators=operators)


def test_observed_design_operator_can_use_exact_coordinates(predictor, feat_grid):
    coordinates = np.array([3.25, 17.5, 44.75], dtype=np.float32)
    operator = np.zeros((1, len(feat_grid), len(coordinates)), dtype=np.float32)
    operator[0, 0, 0] = 1
    operator[0, 1, 1] = 1
    operator[0, 2, 2] = 1
    bundle = mixture_plot_curves(
        predictor, PARAMS[:1], feat_grid, feature_operators=operator,
        density_bandwidths=[5.0], operator_feature_coordinates=coordinates)
    assert bundle["bias"].shape == (1, len(feat_grid))
    assert bundle["asymmetry"].shape == (1, len(feat_grid))
    assert np.isfinite(bundle["bias"][0, :3]).all()


def test_the_smoother_width_follows_the_empirical_weights(predictor, feat_grid):
    default = mixture_plot_curves(predictor, PARAMS, feat_grid)
    wider = mixture_plot_curves(predictor, PARAMS, feat_grid,
                                emp_density_weights_sd=40.0)
    assert not np.allclose(default["asymmetry"], wider["asymmetry"])


def test_pooled_sd_reduces_to_the_unpooled_curve_on_delta_weights(predictor, feat_grid):
    """The same degeneracy the surface path has: a bin drawing on one column must
    give that column's value, not a mixture over columns no trial visited."""
    weights = _weights(len(PARAMS), len(feat_grid))
    bundle = mixture_plot_curves(predictor, PARAMS, feat_grid, bin_weights=weights)

    for row in range(len(PARAMS)):
        for index in range(weights.shape[1]):
            column = int(np.argmax(weights[row, index]))
            assert bundle["pooled_sd"][row, index] == pytest.approx(
                bundle["sd"][row, column], rel=1e-4)


def test_the_narrow_corner_is_representable(predictor, feat_grid):
    """The region the surrogate exists for: components far below the 2-degree
    reporting cell, where a grid route would mis-mass them."""
    narrow = np.array([[2.5, 2.5, 5.0]])
    bundle = mixture_plot_curves(predictor, narrow, feat_grid)
    for name in ("bias", "asymmetry", "sd"):
        assert np.all(np.isfinite(bundle[name])), name
    assert np.min(bundle["sd"]) < 25.0


def test_standard_subject_pipeline_uses_direct_matched_wnm_curves(predictor, feat_grid):
    params = np.array([25.0, 40.0, 30.0, 12.0])
    result = _stored_result(feat_grid, params)
    subjects = {"S": {"exp": [{
        "noise_condition": "condition", "result": result,
    }]}}
    prepared = subject_plots.prepare_all_subjects_data(
        subjects, predictor,
        {"emp_density_weights_sd": 20.0, "density_smoothing_sigma": None})
    got = prepared["S"]["experiments"]["exp"]["optimizer_curves"][
        "condition"]["density"]

    bin_weights = subject_plots.compute_feat_bin_weights(
        result["data_df"][:, 0], feat_grid)[None, :, :]
    expected = mixture_plot_curves(
        predictor, params[None, :3], feat_grid,
        bin_weights=bin_weights, sd_motor_by_row=[params[3]],
        feature_operators=result["empirical_curves"]["feature_operator"][None, :, :],
        density_bandwidths=[7.5])
    for actual_name, expected_name in (
            ("bias", "bias"), ("asymmetry", "asymmetry"),
            ("predicted_sd", "sd"), ("predicted_sd_pooled", "pooled_sd")):
        np.testing.assert_array_equal(np.asarray(got[actual_name]), expected[expected_name][0])
    assert prepared["S"]["experiments"]["exp"]["surrogate_identity"][
        "surrogate_family"] == "wnm"
    assert prepared["S"]["experiments"]["exp"]["surrogate_identity"][
        "dm_version"] == "wnm_k12_20samples"


def test_standard_pipeline_pools_report_orders_from_exact_wnm_cell_masses(
        predictor, feat_grid):
    first_params = np.array([25.0, 40.0, 30.0, 12.0])
    second_params = np.array([40.0, 25.0, 30.0, 0.0])
    first = _stored_result(feat_grid, first_params)
    second = _stored_result(feat_grid, second_params)
    second["data_df"] = np.array(
        [[4.0, 2.0], [12.0, -5.0], [22.0, 3.0]], dtype=np.float32)
    subjects = {"S": {
        "color_2_first": [{"noise_condition": "low", "result": first}],
        "color_2_second": [{"noise_condition": "low", "result": second}],
    }}
    prepared = subject_plots.prepare_all_subjects_data(
        subjects, predictor,
        {"emp_density_weights_sd": 20.0, "density_smoothing_sigma": None})
    got_first = prepared["S"]["experiments"]["color_2_first"]["parameters"][
        "low"]["pooled_bwcrps"]["density"]
    got_second = prepared["S"]["experiments"]["color_2_second"]["parameters"][
        "low"]["pooled_bwcrps"]["density"]

    probabilities = subject_plots._mixture_probability_surfaces(
        predictor, np.stack([first_params[:3], second_params[:3]]),
        [first_params[3], second_params[3]], feat_grid)
    bias_grid = np.asarray(config.create_grid("mu1_bias"))
    difference = np.abs(bias_grid[:, None] - bias_grid[None, :])
    expected = pooled_bias_weighted_crps(
        probabilities, [first["data_df"], second["data_df"]], feat_grid,
        np.minimum(difference, 360.0 - difference), 20.0)
    assert got_first == pytest.approx(expected)
    assert got_second == pytest.approx(expected)
