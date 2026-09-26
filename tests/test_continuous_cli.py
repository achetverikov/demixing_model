"""The continuous backend through the command production actually runs.

Everything below this file is unit-tested, but a test that pins the helper while
the driver calls it differently proves nothing. What can only be checked here is
that `--search continuous` reaches the gradient search, that the run writes
*every* objective's score rather than only the one it fitted, and that its
fingerprint describes the search that ran -- so a run at one start budget cannot
resume into results produced at another.

These go through `run_fitting` rather than a subprocess so a failure gives a
usable traceback; the CLI parsing above it is thin and separately exercised.
"""
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared import surrogate  # noqa: E402

WNM = surrogate.WNM_DEFAULTS[20]
SURFACE = ROOT / "pretrained" / "model_epoch1425_10ktrain_20samples.pkl"
pytestmark = pytest.mark.skipif(not WNM.exists(), reason="no packaged WNM artifact")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Two conditions with a feature-dependent bias, enough trials to fit."""
    rng = np.random.default_rng(20260906)
    rows = []
    for condition, amplitude in (("first", 6.0), ("second", -4.0)):
        feat = rng.uniform(2.0, 178.0, 60)
        bias = amplitude * np.sin(np.radians(feat)) + rng.normal(0.0, 15.0, 60)
        rows.append(pd.DataFrame({
            "expName": "synthetic", "subject": "S1", "condition": condition,
            "abs_td_dist": feat, "bias_to_distr_corr": ((bias + 180) % 360) - 180,
            "is_outlier": False}))
    path = tmp_path_factory.mktemp("data") / "two_conditions.csv"
    invalid = rows[0].iloc[:20].copy()
    invalid["bias_to_distr_corr"] = np.nan
    pd.concat([*rows, invalid]).to_csv(path, index=False)
    return path


def _run(dataset, output_dir, **overrides):
    import fit_model_to_data as F

    kwargs = dict(
        data_path=str(dataset), checkpoint_path=str(WNM), output_dir=str(output_dir),
        results_dir=str(output_dir.parent), search="continuous", continuous_starts=2,
        continuous_seed=0, methods=["density"], max_subjects=1, circ_space=360)
    kwargs.update(overrides)
    return F.run_fitting(**kwargs)


def _fingerprint(output_dir):
    import json

    payload = json.loads((output_dir / "extended_run_fingerprint.json").read_text())
    return payload.get("payload", payload)


@pytest.fixture(scope="module")
def baseline_run(dataset, tmp_path_factory):
    """One maintained WNM fit reused by all baseline contract assertions."""
    out = tmp_path_factory.mktemp("baseline_run") / "run"
    _run(dataset, out)
    return out


def _copy_baseline(baseline_run, tmp_path):
    out = tmp_path / "run"
    shutil.copytree(baseline_run, out)
    return out


def test_a_continuous_run_completes_and_writes_results(baseline_run):
    out = baseline_run

    results = pickle.loads((out / "extended_fit_results.pkl").read_bytes())
    assert results, "no conditions were written"
    for entry in results.values():
        params = np.asarray(entry["density_fitted_params"])
        assert entry["n_trials"] == len(entry["data_df"]) == 60
        assert params.shape[-1] >= 3
        assert np.all(np.isfinite(params))
        # The shared spatial SD and the search bounds it came from.
        assert 5.0 <= float(params[2]) <= 200.0
        assert 2.5 <= float(params[0]) <= 200.0


def test_every_objective_is_scored_not_just_the_fitted_one(baseline_run):
    """A results row with some columns from one surrogate and some from another,
    with nothing in the file saying so, is the failure this guards."""
    import fit_model_to_data as F

    out = baseline_run

    results = pickle.loads((out / "extended_fit_results.pkl").read_bytes())
    for entry in results.values():
        losses = entry["density_evaluation_losses"]
        assert set(losses) == set(F.LOSS_EVALUATION_METHODS)
        assert all(np.isfinite(v) for v in losses.values())


def test_the_fingerprint_describes_the_search_that_ran(baseline_run):
    out = baseline_run
    payload = _fingerprint(out)

    assert payload["search_backend"] == "continuous"
    assert payload["surrogate_family"] == "wnm"
    assert payload["continuous_spec"]["n_starts"] == 2
    assert payload["continuous_spec"]["sd_feat_bounds"] == [2.5, 200.0]
    for absent in ("grid_spec", "feat_step_schedule", "param_bounds"):
        assert absent not in payload, f"{absent} describes a lattice this run never walked"
    # The distributional objectives are versioned for this family.
    assert payload["objective_versions"]["likelihood"] == "trial_loglik_continuous@1"


def test_the_same_settings_resume_rather_than_refit(dataset, baseline_run, tmp_path):
    out = _copy_baseline(baseline_run, tmp_path)
    before = (out / "extended_fit_results.pkl").read_bytes()
    _run(dataset, out)
    assert (out / "extended_fit_results.pkl").exists()
    assert pickle.loads(before).keys() == pickle.loads(
        (out / "extended_fit_results.pkl").read_bytes()).keys()


def test_a_different_start_budget_refuses_to_resume(dataset, baseline_run, tmp_path):
    """Two budgets are two different fits. Resuming across them would mix
    parameters found under searches of different strength into one result set.
    """
    from model_fit_to_data.run_fingerprint import StaleResultsError

    out = _copy_baseline(baseline_run, tmp_path)
    with pytest.raises(StaleResultsError, match="n_starts"):
        _run(dataset, out, continuous_starts=16)


def test_the_backends_refuse_each_other_s_checkpoints(dataset, tmp_path):
    """The surface backend emits a sampled grid and has no gradients; the
    lattice backends cannot drive a mixture."""
    with pytest.raises(ValueError, match="wrapped-normal-mixture"):
        _run(dataset, tmp_path / "a", checkpoint_path=str(SURFACE))
    with pytest.raises(ValueError, match="cannot drive"):
        _run(dataset, tmp_path / "b", search="hierarchical", checkpoint_path=str(WNM))


# ---------------------------------------------------------------------------
# Regressions from the step 3c/3d audit
# ---------------------------------------------------------------------------

def test_a_motor_enabled_run_actually_searches_the_motor_sd(dataset, tmp_path):
    """The dispatch used to pass sd_motor=0.0 unconditionally, so every
    motor-enabled continuous run fitted at zero while the command and the
    fingerprint both labelled it motor-enabled. Feature and spatial SDs can
    absorb response noise, so the other parameters come out wrong too.
    """
    out = tmp_path / "motor"
    _run(dataset, out, skip_motor_noise=False)

    results = pickle.loads((out / "extended_fit_results.pkl").read_bytes())
    fitted_motor = {float(np.asarray(e["density_fitted_params"])[3]) for e in results.values()}
    assert fitted_motor != {0.0}, "motor noise was enabled but every fit came back at zero"
    # It is searched within the empirical cap, not fixed.
    assert all(0.0 < value <= 50.0 for value in fitted_motor)

    payload = _fingerprint(out)
    assert payload["motor"]["mode"] == "enabled"
    assert payload["continuous_spec"]["motor"] == "searched"


def test_a_no_motor_run_reports_a_fixed_zero(baseline_run):
    out = baseline_run
    results = pickle.loads((out / "extended_fit_results.pkl").read_bytes())
    assert all(float(np.asarray(e["density_fitted_params"])[3]) == 0.0
               for e in results.values())
    assert _fingerprint(out)["continuous_spec"]["motor"] == "fixed_zero"


def test_the_fingerprint_records_every_setting_that_moves_the_parameters(baseline_run):
    """Tolerances and the iteration cap change which starts converge and which
    parameters win, so two runs differing only in those must not share a digest.
    """
    out = baseline_run
    spec = _fingerprint(out)["continuous_spec"]
    for field in ("method", "parameterisation", "n_starts", "seed", "sd_feat_bounds",
                  "sd_spat_bounds", "max_iterations", "tolerance", "gradient_tolerance",
                  "optimizer_version", "batch_size", "dtype", "matmul_precision", "motor"):
        assert field in spec, field


def test_the_recorded_spec_comes_from_the_engine_that_runs(baseline_run):
    """It was built twice -- once for the fingerprint, once on the engine -- and
    the two could drift, which is how a changed tolerance would alter the fitted
    parameters while the digest stayed put.
    """
    import continuous_fit
    import inspect

    out = baseline_run
    spec = _fingerprint(out)["continuous_spec"]

    defaults = inspect.signature(continuous_fit.minimize_continuous).parameters
    assert spec["max_iterations"] == defaults["max_iterations"].default
    assert spec["tolerance"] == defaults["tolerance"].default
    assert spec["gradient_tolerance"] == defaults["gradient_tolerance"].default
    assert spec["batch_size"] == defaults["batch_size"].default
    assert spec["dtype"] == defaults["dtype"].default
    assert spec["matmul_precision"] == defaults["matmul_precision"].default


def test_search_diagnostics_reach_the_saved_results(baseline_run):
    """A run where one start of eight converged must not be stored
    indistinguishably from one where all eight agreed."""
    out = baseline_run
    results = pickle.loads((out / "extended_fit_results.pkl").read_bytes())
    for entry in results.values():
        assert entry["density_search_backend"] == "continuous"
        assert entry["density_n_starts"] == 2
        assert 0 <= entry["density_n_converged"] <= 2
        assert np.isfinite(entry["density_loss_spread"])
        assert isinstance(entry["density_at_bound"], list)
        assert len(entry["density_start_losses"]) == 2
        assert len(entry["density_start_outcomes"]) == 2
        assert entry["density_search_settings"]["optimizer_version"] == "jax-lbfgsb@0350da1"


def test_cli_allows_wnm_on_csv_and_explains_when_bundles_are_preferred(dataset, tmp_path):
    """CSV is the normal end-user path; bundles are for controlled comparisons."""
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(ROOT / "model_fit_to_data" / "fit_model_to_data.py"),
         "--data-path", str(dataset),
         "--checkpoint-path", str(WNM),
         "--search", "continuous",
         "--continuous-starts", "1",
         "--max-subjects", "1",
         "--include-methods", "density",
         "--output-dir", str(tmp_path / "run")],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    combined = completed.stdout + completed.stderr
    assert "ordinary single-model analyses" in combined
    assert (tmp_path / "run" / "extended_fit_results.pkl").exists()


def test_min_trials_counts_only_scored_rows(dataset, tmp_path, monkeypatch):
    """CLI contract: invalid CSV rows cannot make a condition eligible."""
    import fit_model_to_data as F
    def unexpected_fit(*args, **kwargs):
        raise AssertionError("No condition has 70 usable trials")
    monkeypatch.setattr(F, "process_subject", unexpected_fit)
    _run(dataset, tmp_path / "below_threshold", min_trials=70)
