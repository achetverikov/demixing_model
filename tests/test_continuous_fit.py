"""The continuous backend must be interchangeable with the other two.

``process_subject`` consumes one result shape and does not know which search
produced it, which is what keeps result handling from forking three ways. So the
contract this file pins is mostly structural: the same keys, the same
per-condition entries, the same aggregation -- plus the extra fields only a
continuous search can report, which the benchmark needs and a lattice search has
no analogue for.

It also pins two refusals that would otherwise produce a confident wrong number:
fitting a mean-only objective at a non-zero motor SD, which reports a parameter
the objective cannot see, and fitting against targets built in a different
condition order, which scores every condition against another one's data.
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

from continuous_fit import fit_continuous  # noqa: E402
from density_objective import degenerate_targets  # noqa: E402
from fitting_targets import build_fitting_targets  # noqa: E402
from grid_based_multi_condition_optimizer_jax_loops import (  # noqa: E402
    _compute_curve_losses, bwcrps_energy_score, compute_bwcrps_condition_targets,
    compute_target_bias_curve_core)
from shared import surrogate  # noqa: E402
from shared.config import config  # noqa: E402
from shared.prediction import predictor_from_surrogate  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "data" / "fitting_targets_golden.npz"
ARTIFACT = surrogate.WNM_DEFAULTS[20]
pytestmark = pytest.mark.skipif(
    not (GOLDEN.exists() and ARTIFACT.exists()),
    reason="needs the target reference and a packaged WNM artifact")

#: Keys `process_subject` reads from whichever backend ran.
SHARED_RESULT_KEYS = ("best_loss", "shared_params", "condition_results", "total_time",
                      "stage_times", "n_conditions", "condition_names", "search_backend")


@pytest.fixture(scope="module")
def setup():
    golden = np.load(GOLDEN)
    prefix = "unequal_180/input/"
    datasets = {key[len(prefix):]: golden[key]
                for key in golden.files if key.startswith(prefix)}

    feat_diff_grid = config.create_grid('feat_diff')
    bias_grid = config.create_grid('mu1_bias')
    diff = jnp.abs(bias_grid[:, None] - bias_grid[None, :])
    d_circ = jnp.minimum(diff, 360.0 - diff)

    targets = build_fitting_targets(
        {k: jnp.asarray(v) for k, v in datasets.items()}, feat_diff_grid=feat_diff_grid,
        d_circ_matrix=d_circ, n_mu1_bias=len(bias_grid), emp_density_weights_sd=20.0,
        density_bandwidth_rule="sj", density_bandwidth_mode="pooled",
        degenerate_targets=degenerate_targets,
        bwcrps_condition_targets=compute_bwcrps_condition_targets,
        target_bias_curve_core=compute_target_bias_curve_core)

    predictor = predictor_from_surrogate(surrogate.load_surrogate(checkpoint_path=ARTIFACT))
    return dict(datasets=datasets, targets=targets, predictor=predictor,
                d_circ=d_circ, feat_diff_grid=feat_diff_grid)


def _fit(setup, objective="density", n_starts=4, **kwargs):
    return fit_continuous(
        setup["predictor"], setup["targets"], list(setup["datasets"]),
        objective=objective, curve_losses=_compute_curve_losses,
        energy_score=bwcrps_energy_score, d_circ_matrix=setup["d_circ"],
        feat_diff_grid=setup["feat_diff_grid"], emp_density_weights_sd=20.0,
        n_starts=n_starts, seed=0, verbosity=0, **kwargs)


def test_the_result_shape_matches_the_other_backends(setup):
    """Result handling must not fork three ways."""
    from exhaustive_density import fit_exhaustive_density  # noqa: F401  (contract source)

    result = _fit(setup)
    for key in SHARED_RESULT_KEYS:
        assert key in result, key
    assert result["search_backend"] == "continuous"
    assert result["n_conditions"] == len(setup["datasets"])
    assert result["condition_names"] == list(setup["datasets"])
    assert set(result["shared_params"]) == {"sd_spat", "sd_motor"}
    assert len(result["stage_times"]) >= 1


def test_each_condition_entry_has_the_fields_downstream_reads(setup):
    result = _fit(setup)
    for name, entry in result["condition_results"].items():
        assert entry["condition_name"] == name
        for key in ("sd_feat1", "sd_feat2", "loss", "surface_idx"):
            assert key in entry
        assert np.isfinite(entry["sd_feat1"]) and np.isfinite(entry["sd_feat2"])


def test_per_condition_losses_sum_to_the_reported_total(setup):
    """The fit minimises the sum, so the decomposition must reconstruct it.

    If it does not, a per-condition loss is being computed at parameters other
    than the ones reported, and the diagnosis it supports is fiction.
    """
    result = _fit(setup)
    total = sum(entry["loss"] for entry in result["condition_results"].values())
    assert total == pytest.approx(result["best_loss"], rel=1e-5)


def test_the_extra_fields_a_lattice_search_cannot_report_are_present(setup):
    """Comparing backends on one number each would hide how the answer was found."""
    result = _fit(setup, n_starts=4)
    assert result["n_starts"] == 4
    assert len(result["start_losses"]) == 4
    assert 0 <= result["n_converged"] <= 4
    assert np.isfinite(result["loss_spread"])
    assert isinstance(result["at_bound"], list)
    assert result["search_settings"]["parameterisation"] == "log"
    assert result["search_settings"]["method"] == "L-BFGS-B"


def test_the_winning_loss_is_the_best_of_the_starts(setup):
    result = _fit(setup, n_starts=5)
    assert result["best_loss"] == pytest.approx(min(result["start_losses"]), rel=1e-6)


def test_bounds_come_from_the_surrogate_so_the_narrow_region_is_reachable(setup):
    """The mixture's feature coverage goes to 2.5; a search floored at 5 would
    make the coverage that motivates this transition unreachable."""
    result = _fit(setup, n_starts=8)
    reached = min(min(e["sd_feat1"], e["sd_feat2"])
                  for e in result["condition_results"].values())
    assert reached < 5.0, (
        f"lowest fitted sd_feat was {reached:.3f}; the WNM search should be able to go "
        "below the surface backend's floor of 5")


def test_a_mean_only_objective_at_non_zero_motor_noise_is_refused(setup):
    """It would report a motor SD the objective is mathematically blind to."""
    with pytest.raises(ValueError, match="invariant to motor noise"):
        _fit(setup, objective="expectation", sd_motor=15.0)
    # And it is fine at zero, which is the no-motor case rather than a fitted one.
    assert np.isfinite(_fit(setup, objective="expectation", sd_motor=0.0)["best_loss"])


def test_a_condition_order_mismatch_is_refused(setup):
    """Targets are positional; a reordered name list silently pairs each
    condition with another one's data and every loss still looks plausible."""
    with pytest.raises(ValueError, match="condition order mismatch"):
        fit_continuous(
            setup["predictor"], setup["targets"], list(reversed(list(setup["datasets"]))),
            objective="density", curve_losses=_compute_curve_losses,
            energy_score=bwcrps_energy_score, d_circ_matrix=setup["d_circ"],
            feat_diff_grid=setup["feat_diff_grid"], emp_density_weights_sd=20.0,
            n_starts=2, seed=0, verbosity=0)


def test_an_unknown_objective_is_refused(setup):
    with pytest.raises(ValueError, match="unknown objective"):
        _fit(setup, objective="mse")


def test_motor_noise_is_applied_and_recorded(setup):
    """The fitted parameters are conditional on it, so it has to appear in the
    result rather than only in the caller's memory."""
    without = _fit(setup, objective="density", sd_motor=0.0)
    with_motor = _fit(setup, objective="density", sd_motor=20.0)
    assert with_motor["shared_params"]["sd_motor"] == 20.0
    assert without["shared_params"]["sd_motor"] == 0.0
    assert not np.isclose(without["best_loss"], with_motor["best_loss"], rtol=1e-4)


def test_the_run_is_reproducible_from_its_seed(setup):
    """Two runs of the same fit must agree, or a comparison between backends is
    comparing noise."""
    first = _fit(setup, n_starts=4)
    second = _fit(setup, n_starts=4)
    assert first["best_loss"] == pytest.approx(second["best_loss"], rel=1e-9)
    np.testing.assert_allclose(first["start_losses"], second["start_losses"], rtol=1e-9)


# ---------------------------------------------------------------------------
# Regressions from the 2026-09-06 audit of step 3
# ---------------------------------------------------------------------------

def test_the_feature_grid_reaches_the_scorer(setup):
    """It was accepted, validated, and then ignored while the scorer rebuilt the
    configured grid. A shifted grid of the same length -- 1, 3, ... 179 against
    2, 4, ... 180 -- lines up shape for shape while every feature location is
    wrong, so the losses stay finite and plausible and the fit is wrong.
    """
    shifted = setup["feat_diff_grid"] - 1.0
    on_grid = _fit(setup, n_starts=2)
    off_grid = fit_continuous(
        setup["predictor"], setup["targets"], list(setup["datasets"]),
        objective="density", curve_losses=_compute_curve_losses,
        energy_score=bwcrps_energy_score, d_circ_matrix=setup["d_circ"],
        feat_diff_grid=shifted, emp_density_weights_sd=20.0,
        n_starts=2, seed=0, verbosity=0)
    assert not np.isclose(on_grid["best_loss"], off_grid["best_loss"], rtol=1e-6), (
        "a shifted feature grid produced an identical loss, so the grid is still "
        "being rebuilt inside the scorer rather than used as given")


def test_a_trial_list_of_the_wrong_length_is_refused(setup):
    """Positional pairing: a mismatch fits one condition to another's observations."""
    trials = [(jnp.asarray(v[:, 0]), jnp.asarray(v[:, 1]))
              for v in setup["datasets"].values()]
    with pytest.raises(ValueError, match="trial arrays for"):
        _fit(setup, objective="likelihood", condition_trials=trials[:-1])


def test_every_start_outcome_is_recorded_not_just_its_loss(setup):
    """Losses alone cannot say whether the winner converged or merely stopped."""
    result = _fit(setup, n_starts=3)
    assert len(result["start_outcomes"]) == 3
    for outcome in result["start_outcomes"]:
        for key in ("start", "solution", "loss", "success", "status",
                    "n_iterations", "n_evaluations", "at_bound"):
            assert key in outcome, key
    # Only a converged start may be the reported winner.
    winners = [o for o in result["start_outcomes"]
               if np.isclose(o["loss"], result["best_loss"], rtol=1e-9)]
    assert any(o["success"] for o in winners)


def test_the_search_box_is_recorded_with_the_result(setup):
    """Two runs with the same seed but different artifacts search different boxes,
    and a railed parameter means nothing without the bound it railed against."""
    settings = _fit(setup, n_starts=2)["search_settings"]
    assert "bounds" in settings and "bound_names" in settings
    assert len(settings["bounds"]) == len(settings["bound_names"])
    assert settings["bounds"][0] == [2.5, 200.0]     # sd_feat, from the WNM domain
    assert settings["bounds"][-1] == [5.0, 200.0]    # sd_spat
