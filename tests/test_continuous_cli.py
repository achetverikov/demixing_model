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
        feat = rng.uniform(2.0, 178.0, 400)
        bias = amplitude * np.sin(np.radians(feat)) + rng.normal(0.0, 15.0, 400)
        rows.append(pd.DataFrame({
            "expName": "synthetic", "subject": "S1", "condition": condition,
            "abs_td_dist": feat, "bias_to_distr_corr": ((bias + 180) % 360) - 180,
            "is_outlier": False}))
    path = tmp_path_factory.mktemp("data") / "two_conditions.csv"
    pd.concat(rows).to_csv(path, index=False)
    return path


def _run(dataset, output_dir, **overrides):
    import fit_model_to_data as F

    kwargs = dict(
        data_path=str(dataset), checkpoint_path=str(WNM), output_dir=str(output_dir),
        results_dir=str(output_dir.parent), search="continuous", continuous_starts=3,
        continuous_seed=0, methods=["density"], max_subjects=1, circ_space=360)
    kwargs.update(overrides)
    return F.run_fitting(**kwargs)


def _fingerprint(output_dir):
    import json

    payload = json.loads((output_dir / "extended_run_fingerprint.json").read_text())
    return payload.get("payload", payload)


def test_a_continuous_run_completes_and_writes_results(dataset, tmp_path):
    out = tmp_path / "run"
    _run(dataset, out)

    results = pickle.loads((out / "extended_fit_results.pkl").read_bytes())
    assert results, "no conditions were written"
    for entry in results.values():
        params = np.asarray(entry["density_fitted_params"])
        assert params.shape[-1] >= 3
        assert np.all(np.isfinite(params))
        # The shared spatial SD and the search bounds it came from.
        assert 5.0 <= float(params[2]) <= 200.0
        assert 2.5 <= float(params[0]) <= 200.0


def test_every_objective_is_scored_not_just_the_fitted_one(dataset, tmp_path):
    """A results row with some columns from one surrogate and some from another,
    with nothing in the file saying so, is the failure this guards."""
    import fit_model_to_data as F

    out = tmp_path / "run"
    _run(dataset, out)

    results = pickle.loads((out / "extended_fit_results.pkl").read_bytes())
    for entry in results.values():
        losses = entry["density_evaluation_losses"]
        assert set(losses) == set(F.LOSS_EVALUATION_METHODS)
        assert all(np.isfinite(v) for v in losses.values())


def test_the_fingerprint_describes_the_search_that_ran(dataset, tmp_path):
    out = tmp_path / "run"
    _run(dataset, out)
    payload = _fingerprint(out)

    assert payload["search_backend"] == "continuous"
    assert payload["surrogate_family"] == "wnm"
    assert payload["continuous_spec"]["n_starts"] == 3
    assert payload["continuous_spec"]["sd_feat_bounds"] == [2.5, 200.0]
    for absent in ("grid_spec", "feat_step_schedule", "param_bounds"):
        assert absent not in payload, f"{absent} describes a lattice this run never walked"
    # The distributional objectives are versioned for this family.
    assert payload["objective_versions"]["likelihood"] == "trial_loglik_continuous@1"


def test_the_same_settings_resume_rather_than_refit(dataset, tmp_path):
    out = tmp_path / "run"
    _run(dataset, out)
    before = (out / "extended_fit_results.pkl").read_bytes()
    _run(dataset, out)
    assert (out / "extended_fit_results.pkl").exists()
    assert pickle.loads(before).keys() == pickle.loads(
        (out / "extended_fit_results.pkl").read_bytes()).keys()


def test_a_different_start_budget_refuses_to_resume(dataset, tmp_path):
    """Two budgets are two different fits. Resuming across them would mix
    parameters found under searches of different strength into one result set.
    """
    from run_fingerprint import StaleResultsError

    out = tmp_path / "run"
    _run(dataset, out)
    with pytest.raises(StaleResultsError, match="n_starts"):
        _run(dataset, out, continuous_starts=16)


def test_a_hierarchical_run_cannot_resume_into_a_continuous_one(dataset, tmp_path):
    from run_fingerprint import StaleResultsError

    out = tmp_path / "run"
    _run(dataset, out)
    with pytest.raises((StaleResultsError, ValueError)):
        _run(dataset, out, search="hierarchical", checkpoint_path=str(SURFACE))


def test_the_backends_refuse_each_other_s_checkpoints(dataset, tmp_path):
    """The surface backend emits a sampled grid and has no gradients; the
    lattice backends cannot drive a mixture."""
    with pytest.raises(ValueError, match="wrapped-normal-mixture"):
        _run(dataset, tmp_path / "a", checkpoint_path=str(SURFACE))
    with pytest.raises(ValueError, match="cannot drive"):
        _run(dataset, tmp_path / "b", search="hierarchical", checkpoint_path=str(WNM))
