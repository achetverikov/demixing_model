"""Rescoring must identify its surrogate by content, not by path text.

`infer_checkpoint_path` used to substring-match the fits CSV path for
``20samples`` / ``100samples``. Those name two different observer models, not two
settings of one, so any results directory whose name carried the other token --
a relocated run, a subset directory named after a comparison -- rescored a fit
under the wrong model. The reproduction gate would usually catch it, but the tool
someone reaches for when that gate "fails" is `repair_stale_reproduction.py`,
which deletes results.

The replacement resolves against the run's own recorded checkpoint digest. These
tests pin that, and pin the refusal when nothing identifies the surrogate at all:
guessing is what was removed, and falling back to it under any condition would
put the defect straight back.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import postprocess_fitted_likelihoods as P  # noqa: E402
from run_fingerprint import file_sha256  # noqa: E402
from shared import surrogate  # noqa: E402

WNM = surrogate.WNM_DEFAULTS[20]
SURFACE_20 = ROOT / "pretrained" / "model_epoch1425_10ktrain_20samples.pkl"
SURFACE_100 = ROOT / "pretrained" / "model_epoch1500_10ktrain_100samples.pkl"

pytestmark = pytest.mark.skipif(
    not (WNM.exists() and SURFACE_20.exists()),
    reason="needs installed artifacts to resolve digests against")


def _run_dir(tmp_path, checkpoint, name="results_dir"):
    """A results directory with a fingerprint recording `checkpoint`."""
    run = tmp_path / name
    run.mkdir(parents=True)
    (run / "extended_run_fingerprint.json").write_text(json.dumps({
        "checkpoint_sha256": file_sha256(checkpoint),
        "surrogate_family": "wnm" if "wnm" in checkpoint.name else "surface_nn"}))
    fits = run / "fitted_parameters.csv"
    fits.write_text("subject\n")
    return fits


def test_the_path_name_does_not_decide_the_checkpoint(tmp_path):
    """The exact defect: a directory named for one observer model holding a run
    fitted with another."""
    fits = _run_dir(tmp_path, WNM, name="rerun_20samples_subset")
    assert P.infer_checkpoint_path(fits, None) == WNM

    misleading = _run_dir(tmp_path, SURFACE_100, name="analysis_20samples_vs_100samples")
    assert P.infer_checkpoint_path(misleading, None) == SURFACE_100


def test_an_explicit_checkpoint_is_verified_not_merely_accepted(tmp_path):
    """Passing --checkpoint-path is not a licence to rescore under any model."""
    fits = _run_dir(tmp_path, WNM)
    assert P.infer_checkpoint_path(fits, str(WNM)) == WNM
    with pytest.raises(ValueError, match="not the checkpoint this run was fitted with"):
        P.infer_checkpoint_path(fits, str(SURFACE_20))


def test_without_a_fingerprint_or_an_explicit_path_it_refuses(tmp_path):
    """Guessing is what this replaced; falling back to it restores the defect."""
    run = tmp_path / "old_20samples_run"
    run.mkdir(parents=True)
    fits = run / "fitted_parameters.csv"
    fits.write_text("subject\n")
    with pytest.raises(ValueError, match="no run fingerprint near"):
        P.infer_checkpoint_path(fits, None)
    # An explicit path still works for results predating the fingerprint.
    assert P.infer_checkpoint_path(fits, str(SURFACE_20)) == SURFACE_20


def test_a_digest_matching_nothing_installed_says_so(tmp_path):
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "extended_run_fingerprint.json").write_text(json.dumps(
        {"checkpoint_sha256": "0" * 64, "surrogate_family": "wnm"}))
    fits = run / "fitted_parameters.csv"
    fits.write_text("subject\n")
    with pytest.raises(ValueError, match="matches none of the installed artifacts"):
        P.infer_checkpoint_path(fits, None)


def test_the_fingerprint_is_found_from_a_nested_fits_file(tmp_path):
    """Exports often sit in a subdirectory of the run they came from."""
    fits = _run_dir(tmp_path, WNM)
    nested = fits.parent / "exports" / "csv"
    nested.mkdir(parents=True)
    moved = nested / "fitted_parameters.csv"
    moved.write_text("subject\n")
    assert P.infer_checkpoint_path(moved, None) == WNM


def test_wnm_fit_rows_export_both_likelihood_conventions(monkeypatch):
    scored = pd.DataFrame({
        "feat_diff_model_deg": [2.0, 4.0],
        "bias_model_deg": [-1.0, 3.0],
    })
    fit = pd.Series({
        "subject": "S10", "experiment": "color_2", "condition": "low - low",
        "optimizer": "likelihood", "sd_feat1": 20.0, "sd_feat2": 30.0,
        "sd_spat": 10.0, "sd_motor": 0.0, "eval_likelihood_loss": 3.0,
    })
    def fake_likelihood(*_args, **_kwargs):
        return {
            "loglik_density_model_deg": np.array([-1.0, -2.0]),
            "nll_density_model_deg": np.array([1.0, 2.0]),
            "loglik_mass": np.array([-1.0, -2.0]) + np.log(2.0),
            "nll_mass": np.array([1.0, 2.0]) - np.log(2.0),
            "loglik_density_deg": np.array([-1.0, -2.0]),
            "nll_density_deg": np.array([1.0, 2.0]),
            "loglik_cell_probability": np.array([-0.8, -1.8]),
            "bin_width_deg": 2.0,
            "loglik_convention": "continuous_at_observation",
        }
    monkeypatch.setattr(P, "evaluate_trial_likelihoods", fake_likelihood)

    out, check = P._score_fit_row_wnm(object(), scored, "prepared.csv", fit, 360)

    assert out["loglik_convention"].unique().tolist() == ["continuous_at_observation"]
    np.testing.assert_array_equal(out["loglik_cell_probability"], [-0.8, -1.8])
    assert check["rescored_nll_density_model_deg"] == pytest.approx(3.0)
    assert check["n_floor_trials"] == 0
