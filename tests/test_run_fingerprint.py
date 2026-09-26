"""Stale-result refusal: the fingerprint sidecar and the rules built on it.

The failure this guards against is silent: a resumed fit sees a condition key it
already has and skips it, so a refit after an objective/grid/data change leaves a
pickle mixing results computed different ways, and nothing downstream can tell.
These tests pin the four refusal states, the fields that must move the digest,
and — the one that is not self-referential — that the feature-step schedule the
fingerprint records is the schedule the fit actually walks.
"""
import json
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_fingerprint as rf


BASE_KWARGS = dict(
    circ_space=360,
    evaluation_methods=[
        "density", "density_legacy", "expectation", "smoothed_exp", "likelihood",
        "crps", "balanced_crps", "bias_weighted_crps",
    ],
    search_backend="continuous",
    surrogate_family="wnm",
    continuous_spec={"n_starts": 64, "seed": 0, "method": "BatchedLbfgsb"},
    skip_motor_noise=True,
    exp_col="expName",
    subject_col="subject",
    condition_col="condition",
    x_col="abs_td_dist",
    y_col="bias_to_distr_corr",
    outlier_col="is_outlier",
    include_outliers=False,
    min_trials=30,
    corr_weight=0.25,
    density_curve_spec={"emp_density_weights_sd": 20.0, "density_smoothing_sigma": None,
                        "density_bandwidth_mode": "pooled", "density_bandwidth_rule": "sj"},
)


@pytest.fixture
def run_files(tmp_path):
    """A dataset and a checkpoint file to hash, plus an output directory."""
    data = tmp_path / "data.csv"
    data.write_text("subject,expName,condition,abs_td_dist,bias_to_distr_corr\n1,e,c,10,1\n")
    checkpoint = tmp_path / "model.pkl"
    checkpoint.write_bytes(b"weights-v1")
    out = tmp_path / "out"
    out.mkdir()
    return data, checkpoint, out


def make_payload(data, checkpoint, **overrides):
    kwargs = dict(BASE_KWARGS, data_path=data, checkpoint_path=checkpoint)
    kwargs.update(overrides)
    return rf.compute_run_fingerprint(**kwargs)


def test_digest_is_stable_and_field_order_independent(run_files):
    data, checkpoint, _ = run_files
    payload = make_payload(data, checkpoint)
    shuffled = dict(reversed(list(payload.items())))
    assert rf.fingerprint_digest(payload) == rf.fingerprint_digest(shuffled)


@pytest.mark.integration
def test_compiled_fingerprint_pins_bundle_products(run_files, tmp_path):
    _, checkpoint, _ = run_files
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle.yaml").write_text("bundle_id: example\n")
    manifest = {
        "bundle_id": "b1", "population": "signed_bias",
        "canonical_trial_sha256": "canonical",
        "analysis_spec_sha256": "analysis",
        "ordered_scored_row_id_sha256": "rows",
        "objective_versions": {"density": "density-v1"},
        "products": {"shared_empirical_targets": {"sha256": "targets"}},
    }
    payload = rf.compute_compiled_run_fingerprint(
        bundle_path=bundle, bundle_manifest=manifest,
        checkpoint_path=checkpoint, continuous_spec={"n_starts": 8},
        skip_motor_noise=True,
        evaluation_methods=["density", "smoothed_exp", "likelihood",
                            "bias_weighted_crps"], corr_weight=0.25,
        density_curve_spec={"emp_density_weights_sd": 20.0,
                            "density_smoothing_sigma": None})
    assert payload["bundle_id"] == "b1"
    assert payload["empirical_targets_sha256"] == "targets"
    assert payload["density_curve_spec"]["emp_density_weights_sd"] == 20.0
    changed = dict(payload, bundle_id="b2")
    assert rf.fingerprint_digest(changed) != rf.fingerprint_digest(payload)


@pytest.mark.parametrize("overrides", [
    {"circ_space": 180},
    {"min_trials": 40},
    {"corr_weight": 0.5},
    {"include_outliers": True},
    {"outlier_col": None},
    {"skip_motor_noise": False},
    {"x_col": "other_col"},
    {"density_curve_spec": dict(BASE_KWARGS["density_curve_spec"],
                                density_bandwidth_rule="silverman")},
    {"degenerate_eps": 1e-10},
])
def test_every_result_changing_setting_moves_the_digest(run_files, overrides):
    data, checkpoint, _ = run_files
    base = rf.fingerprint_digest(make_payload(data, checkpoint))
    assert rf.fingerprint_digest(make_payload(data, checkpoint, **overrides)) != base


def test_data_and_checkpoint_contents_move_the_digest(run_files):
    data, checkpoint, _ = run_files
    base = rf.fingerprint_digest(make_payload(data, checkpoint))
    data.write_text("subject,expName,condition,abs_td_dist,bias_to_distr_corr\n1,e,c,10,2\n")
    changed_data = rf.fingerprint_digest(make_payload(data, checkpoint))
    assert changed_data != base
    checkpoint.write_bytes(b"weights-v2")
    assert rf.fingerprint_digest(make_payload(data, checkpoint)) != changed_data


def test_requested_methods_are_not_in_the_fingerprint(run_files):
    """Adding a method to an existing run must keep resuming, so the method list
    is a coverage question and must not enter the validity digest."""
    data, checkpoint, _ = run_files
    payload = make_payload(data, checkpoint)
    flat = json.dumps(payload)
    assert "methods" not in payload
    # Every objective is pinned regardless of which the run was asked to fit.
    assert set(payload["objective_versions"]) == set(rf.OBJECTIVE_VERSIONS)
    assert "density" in flat


def test_objective_version_change_invalidates_other_methods_results(run_files):
    """Cross-objective evaluation writes a density loss onto a likelihood fit, so
    a change to the density definition must invalidate that run too."""
    data, checkpoint, _ = run_files
    payload = make_payload(data, checkpoint, evaluation_methods=["likelihood", "density"])
    bumped = json.loads(json.dumps(payload))
    bumped["objective_versions"]["density"] = "ccc@2"
    assert rf.fingerprint_digest(bumped) != rf.fingerprint_digest(payload)


def test_unknown_objective_is_refused(run_files):
    data, checkpoint, _ = run_files
    with pytest.raises(ValueError, match="No objective version recorded"):
        make_payload(data, checkpoint, evaluation_methods=["density", "brand_new_loss"])


# --- the four enforcement states -------------------------------------------------

def test_fresh_directory_passes(run_files):
    data, checkpoint, out = run_files
    rf.enforce_fingerprint(out, make_payload(data, checkpoint), results_present=False)


def test_matching_sidecar_passes_even_with_partial_results(run_files):
    """A matching digest on an incomplete run is the normal mid-run state
    (`save_results` runs after every subject) and must resume, not raise."""
    data, checkpoint, out = run_files
    payload = make_payload(data, checkpoint)
    rf.write_fingerprint_sidecar(out, payload)
    rf.enforce_fingerprint(out, payload, results_present=True)


def test_results_without_a_sidecar_are_refused(run_files):
    data, checkpoint, out = run_files
    with pytest.raises(rf.StaleResultsError, match="predate run fingerprinting"):
        rf.enforce_fingerprint(out, make_payload(data, checkpoint), results_present=True)


def test_mismatched_sidecar_is_refused_and_names_the_field(run_files):
    data, checkpoint, out = run_files
    rf.write_fingerprint_sidecar(out, make_payload(data, checkpoint))
    with pytest.raises(rf.StaleResultsError) as excinfo:
        rf.enforce_fingerprint(out, make_payload(data, checkpoint, circ_space=180),
                               results_present=True)
    message = str(excinfo.value)
    assert "circ_space" in message and "360" in message and "180" in message


def test_corrupt_sidecar_is_not_treated_as_absent(run_files):
    """'Absent' means pre-fingerprint results, a state with its own message;
    damage must not be able to forge it."""
    data, checkpoint, out = run_files
    rf.sidecar_path(out).write_text("{not json")
    with pytest.raises(ValueError, match="could not be read"):
        rf.enforce_fingerprint(out, make_payload(data, checkpoint), results_present=True)


def test_sidecar_round_trips_payload_and_digest(run_files):
    data, checkpoint, out = run_files
    payload = make_payload(data, checkpoint)
    rf.write_fingerprint_sidecar(out, payload)
    sidecar = rf.read_fingerprint_sidecar(out)
    assert sidecar["payload"] == json.loads(json.dumps(payload))
    assert sidecar["digest"] == rf.fingerprint_digest(payload)
    assert not list(Path(out).glob(".*tmp")), "temp file left behind"


# --- the schedule the fingerprint records is the schedule the fit walks ----------





def test_payload_records_the_live_mu1_grid_size(run_files):
    """Ties a run to the circularity fix: the half-open 180-point axis and the
    pre-fix 181-point one must not share a digest."""
    data, checkpoint, _ = run_files
    from shared.config import config

    assert make_payload(data, checkpoint)["mu1_grid_size"] == config.mu1_bias_grid_size


# ---------------------------------------------------------------------------
# Continuous runs must not be fingerprinted as though they walked the lattice
# ---------------------------------------------------------------------------

def _common(**overrides):
    base = dict(
        data_path=ROOT / "tests" / "data" / "bw_sj_reference.tsv",
        checkpoint_path=ROOT / "pretrained" / "current_wnm_k12_20samples.pkl",
        circ_space=360, evaluation_methods=["density", "likelihood", "crps"],
        skip_motor_noise=True, exp_col="e", subject_col="s",
        condition_col="c", x_col="x", y_col="y", outlier_col=None,
        include_outliers=False, min_trials=30, corr_weight=0.25,
        density_curve_spec={"emp_density_weights_sd": 20.0})
    base.update(overrides)
    return base


CONTINUOUS = {"n_starts": 64, "seed": 0, "parameterisation": "log",
              "bounds": [[2.5, 200.0], [5.0, 200.0]], "method": "BatchedLbfgsb",
              "optimizer_version": "jax-lbfgsb@0350da1", "batch_size": 32,
              "dtype": "float32", "matmul_precision": "highest"}


def test_a_continuous_run_records_its_settings_and_omits_the_grid_schedule():
    """Filling the lattice fields with defaults a gradient search never walked
    would let two genuinely different runs share a digest and resume into each
    other's results."""
    payload = rf.compute_run_fingerprint(
        search_backend="continuous", surrogate_family="wnm",
        continuous_spec=CONTINUOUS, **_common())

    assert payload["continuous_spec"]["n_starts"] == 64
    assert payload["continuous_spec"]["seed"] == 0
    assert payload["continuous_spec"]["bounds"] == [[2.5, 200.0], [5.0, 200.0]]
    for absent in ("grid_spec", "feat_step_schedule", "param_bounds", "refinement_spec"):
        assert absent not in payload, f"{absent} describes a search that did not run"

def test_the_optimizer_configuration_is_part_of_the_identity():
    """Two runs at different budgets are different fits, not resumable halves."""
    base = rf.compute_run_fingerprint(
        search_backend="continuous", surrogate_family="wnm",
        continuous_spec=CONTINUOUS, **_common())
    for changed in ({**CONTINUOUS, "n_starts": 32}, {**CONTINUOUS, "seed": 1},
                    {**CONTINUOUS, "optimizer_version": "jax-lbfgsb@future"},
                    {**CONTINUOUS, "batch_size": 16},
                    {**CONTINUOUS, "dtype": "float64"},
                    {**CONTINUOUS, "matmul_precision": "default"}):
        other = rf.compute_run_fingerprint(
            search_backend="continuous", surrogate_family="wnm",
            continuous_spec=changed, **_common())
        assert rf.fingerprint_digest(base) != rf.fingerprint_digest(other)


def test_a_missing_or_spurious_continuous_spec_raises():
    with pytest.raises(ValueError, match="requires continuous_spec"):
        rf.compute_run_fingerprint(search_backend="continuous", surrogate_family="wnm",
                                   **_common())
    with pytest.raises(ValueError, match="unsupported search backend"):
        rf.compute_run_fingerprint(search_backend="hierarchical",
                                   surrogate_family="wnm", continuous_spec=CONTINUOUS,
                                   **_common())
