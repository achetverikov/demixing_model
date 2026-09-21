"""Run identity for fits: fingerprint construction, sidecar IO, stale-result refusal.

Nothing in ``extended_fit_results.pkl`` records *how* a fit was produced, so a
resumed run cannot tell a result computed under one objective/grid/dataset from
one computed under another: it just sees a condition key it already has and skips
it.  A refit after an objective or grid change therefore silently leaves a pickle
mixing old and new fits, which no downstream check can detect.

This module builds a canonical description of the run ("fingerprint"), stores it
in a sidecar next to the pickle, and refuses to resume onto results whose
fingerprint differs or is missing.

Why a sidecar and not a key in the pickle: ``fit_model_to_data.load_results``
derives the completed-group set from ``{key.rsplit('#', 1)[0] for key in results}``,
so any reserved key injected into that flat dict becomes a phantom condition.

The sidecar is *not* a coverage record.  Which methods were requested, and which
conditions are done, remain the job of the existing coverage checks in
``run_fitting``; a matching fingerprint on a partially complete run is the normal
mid-run state and must resume.  The fingerprint answers a validity question only:
"were the results on disk produced the same way as the run about to append to
them?"
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Bump on ANY change to the payload field set or to how a field is derived.
#: v2: `degenerate_eps` became live and `density_legacy` joined the objective map
#: when the density objective moved to CCC.
#: A bump invalidates every existing sidecar, which is the point: an unbumped
#: schema change would let differently-computed runs share a digest.
SCHEMA_VERSION = 3

FINGERPRINT_FILENAME = "extended_run_fingerprint.json"

#: How each objective is computed, independent of which objectives a given run
#: was asked to fit.  Cross-objective evaluation (`evaluate_parameter_losses`)
#: writes a loss for *every* entry here at each fitted method's parameters, so a
#: change to any one of them invalidates results nominally fitted under another.
#: Bump the individual string when an objective's definition changes.
OBJECTIVE_VERSIONS: Dict[str, str] = {
    "density": "ccc_matched_pooled_sj_kde_observed_design@1",
    # The pre-2026-08 density objective, 0.75 * MSE/range + 0.25 * (1 - r), kept
    # so published numbers stay reproducible -- `loss_type="combined"`.
    "density_legacy": "combined_range_scaled_mse_plus_corr@1",
    "expectation": "binned_circular_mean_mse@1",
    "smoothed_exp": "observed_design_complex_moment_mse@1",
    "likelihood": "trial_loglik@1",
    "crps": "crps@1",
    "balanced_crps": "balanced_crps@1",
    "bias_weighted_crps": "bias_weighted_crps_circular_weight@2",
}

#: Objectives whose *evaluation convention* differs on the wrapped-normal mixture,
#: with the version that applies there.  The surrogate family is part of what an
#: objective means, not merely of how fast it is computed: the surface backend
#: reads a trial's log density out of the 180-row bias grid at the centre of the
#: cell the observation falls in, while the mixture evaluates its density at the
#: observation itself, and the CRPS variants integrate cell mass exactly rather
#: than renormalising sampled grid densities. Scoring the two under one version
#: string would make a head-to-head information criterion compare numbers
#: computed under different conventions.
WNM_OBJECTIVE_VERSIONS: Dict[str, str] = {
    "likelihood": "trial_loglik_continuous@1",
    "crps": "crps_integrated_cells@1",
    # @2: the curve-level CRPS objectives pool the prediction onto the observed
    # design before scoring, matching the operator that built their target
    # (contextual_biases_database SHARED_PREDICTIVE_CONTRACT, observed-design-
    # pooled-v1). This is a different objective, not a better implementation of
    # the same one -- its argmin moves -- so results fitted under @1 must not be
    # resumed into or compared with results fitted under @2. The per-trial `crps`
    # above is deliberately unpooled and keeps @1.
    "balanced_crps": "balanced_crps_pooled_design@2",
    # @3 additionally weights feature locations by the squared circular empirical
    # mean, so observations around -180/+180 retain their large bias magnitude.
    "bias_weighted_crps": "bias_weighted_crps_pooled_design_circular_weight@3",
}


def objective_versions_for(family: str, methods) -> Dict[str, str]:
    """Objective versions as computed by one surrogate family.

    The curve objectives are defined identically for both families: same target,
    observed-design operator and loss, with only the family prediction changing.
    The distributional objectives have family-specific evaluation conventions.
    """
    unknown = sorted(set(methods) - set(OBJECTIVE_VERSIONS))
    if unknown:
        raise ValueError(
            f"No objective version recorded for {unknown}; add them to "
            "run_fingerprint.OBJECTIVE_VERSIONS (and bump SCHEMA_VERSION) before "
            "results computed with them can be fingerprinted.")
    if family == "wnm":
        return {method: WNM_OBJECTIVE_VERSIONS.get(method, OBJECTIVE_VERSIONS[method])
                for method in sorted(methods)}
    if family == "surface_nn":
        return {method: OBJECTIVE_VERSIONS[method] for method in sorted(methods)}
    raise ValueError(f"unknown surrogate family {family!r}")

#: The fixed part of the hierarchical feature-grid step schedule.  The effective
#: schedule can differ (see `effective_feat_step_schedule`), and it is the
#: effective one that identifies a run.
BASE_FEAT_STEP_SCHEDULE: Tuple[float, ...] = (10.0, 6.0, 4.0, 2.0, 1.0)


def effective_feat_step_schedule(
    feat_grid_size: int, param_low: float, param_high: float
) -> List[float]:
    """Return the feature-grid step schedule actually used by a hierarchical fit.

    The first pass must span the whole supported parameter domain; when the
    fixed schedule's coarsest step is too fine to do that with ``feat_grid_size``
    points, the exact spanning step is prepended.  Production's
    ``feat_grid_size=20`` triggers that, so the base literal alone does not
    identify a run.

    This is the single definition of the schedule: ``fit_hierarchical_grid``
    calls it, and so does the fingerprint, so the two cannot drift.
    """
    schedule = list(BASE_FEAT_STEP_SCHEDULE)
    full_span_step = (param_high - param_low) / (feat_grid_size - 1)
    if full_span_step > schedule[0]:
        schedule = [full_span_step] + schedule
    return schedule


def file_sha256(path: os.PathLike | str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, streamed so large CSVs do not land in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_run_fingerprint(
    *,
    data_path: os.PathLike | str,
    checkpoint_path: os.PathLike | str,
    circ_space: int,
    evaluation_methods: Sequence[str],
    search_backend: str,
    curve_cache_key: Optional[str],
    surrogate_family: str = "surface_nn",
    continuous_spec: Optional[Dict[str, Any]] = None,
    skip_motor_noise: bool,
    exp_col: str,
    subject_col: str,
    condition_col: str,
    x_col: str,
    y_col: str,
    outlier_col: Optional[str],
    include_outliers: bool,
    min_trials: int,
    corr_weight: float,
    density_curve_spec: Dict[str, Any],
    grid_spec: Optional[Dict[str, Any]] = None,
    refinement_spec: Optional[Dict[str, Any]] = None,
    degenerate_eps: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the canonical fingerprint payload for a fitting run.

    Every field is something that changes the numbers a fit produces.  Note what
    is deliberately *absent*: the requested method list.  Including it would
    contradict incremental method addition -- adding a seventh method to a
    six-method run would change the digest and refuse the very resume it is
    supposed to permit.  What is included is ``objective_versions``, which pins
    how each objective is computed rather than which were asked for.

    Args:
        data_path: The prepared dataset file actually read.
        checkpoint_path: The surrogate ``.pkl`` actually loaded.
        evaluation_methods: Every objective the run may evaluate (not just fit).
            Each must have an entry in `OBJECTIVE_VERSIONS`.
        search_backend: ``"hierarchical"``, ``"exhaustive_1deg"`` or ``"continuous"``.
        curve_cache_key: Cache identity when cache-backed, else ``None``.
        surrogate_family: which family computed the objectives, since some of
            them are evaluated under different conventions per family.
        continuous_spec: settings of the continuous search -- parameterisation,
            starts, seed, bounds, tolerances, iteration cap. Required for
            ``search_backend="continuous"`` and rejected otherwise. The
            hierarchical grid fields are omitted for a continuous run rather than
            filled with the defaults it never walked: recording a schedule the
            search did not follow would let two genuinely different runs share a
            digest and resume into each other.
        grid_spec: Hierarchical search settings (sizes, ``min_grid_step``,
            ``zoom_factor``) -- pass the values actually used, not defaults.
        density_curve_spec: Weighting/smoothing/bandwidth settings of the
            empirical density-asymmetry target.
        refinement_spec: Sub-degree refinement settings, ``None`` when not used.
        degenerate_eps: Degenerate-target exclusion threshold, ``None`` when the
            policy is not active.

    Returns:
        A JSON-serializable payload with sorted-key semantics; feed it to
        `fingerprint_digest`.

    Raises:
        ValueError: if an evaluation method has no recorded objective version.
    """
    from shared.config import config as _cfg

    is_continuous = str(search_backend) == "continuous"
    if is_continuous and continuous_spec is None:
        raise ValueError(
            "search_backend='continuous' requires continuous_spec: the starts, seed, bounds "
            "and tolerances are what produced the parameters, so a run recorded without them "
            "cannot be reproduced or told apart from one at a different budget.")
    if not is_continuous and continuous_spec is not None:
        raise ValueError(
            f"continuous_spec was given for search_backend={search_backend!r}, which does not "
            "use one; recording it would describe a search that did not run.")

    versions = objective_versions_for(surrogate_family, evaluation_methods)

    if skip_motor_noise:
        motor: Dict[str, Any] = {"mode": "skip"}
    else:
        motor = {
            "mode": "enabled",
            "sd_motor_low": 0.1,
            "sd_motor_hard_max": 50.0,
            # The per-subject cap and axis size are data-dependent; what
            # identifies the run is the rule that derives them.
            "cap_rule": "min_condition_circ_sd_x1.1_clipped_0.1_50",
            "grid_sizing_rule": "clip(ceil(span/(2*min_grid_step))+1, 4, shared_grid_size)",
        }

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "data_sha256": file_sha256(data_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "circ_space": int(circ_space),
        "objective_versions": versions,
        "search_backend": str(search_backend),
        "curve_cache_key": curve_cache_key,
        "motor": motor,
        "mu1_grid_size": int(_cfg.mu1_bias_grid_size),
        "column_mapping": {
            "exp_col": exp_col,
            "subject_col": subject_col,
            "condition_col": condition_col,
            "x_col": x_col,
            "y_col": y_col,
        },
        "outlier_policy": {
            "outlier_col": outlier_col,
            "include_outliers": bool(include_outliers),
        },
        "min_trials": int(min_trials),
        "corr_weight": float(corr_weight),
        "model_grids": {
            "feat_diff_range": list(_cfg.feat_diff_range),
            "feat_diff_step": int(_cfg.feat_diff_step),
            "mu1_bias_range": list(_cfg.mu1_bias_range),
            "mu1_bias_step": int(_cfg.mu1_bias_step),
        },
        "density_curve_spec": dict(density_curve_spec),
        "degenerate_eps": None if degenerate_eps is None else float(degenerate_eps),
    }

    # The historical payload is left byte-identical for a surface-backed lattice
    # run. New fields appear only for configurations that did not exist under
    # schema 2, so every in-progress run keeps resuming instead of being told its
    # fingerprint no longer matches by a change that did not affect its numbers.
    if str(surrogate_family) != "surface_nn":
        payload["surrogate_family"] = str(surrogate_family)

    if is_continuous:
        # A continuous run walks no lattice, so it records the settings that did
        # produce its parameters and omits the ones that did not. Filling the
        # grid fields with defaults it never used would let a gradient run and a
        # hierarchical one share a digest and resume into each other's results.
        payload["continuous_spec"] = {
            key: continuous_spec[key] for key in sorted(continuous_spec)
        }
    else:
        payload["grid_spec"] = {
            "shared_grid_size": int(grid_spec["shared_grid_size"]),
            "feat_grid_size": int(grid_spec["feat_grid_size"]),
            "min_grid_step": float(grid_spec["min_grid_step"]),
            "zoom_factor": float(grid_spec["zoom_factor"]),
        }
        payload["refinement_spec"] = refinement_spec
        payload["param_bounds"] = {
            "param_grid_low": float(_cfg.param_grid_low),
            "param_range_high": float(_cfg.param_range_high),
        }
        payload["feat_step_schedule"] = [
            float(step) for step in effective_feat_step_schedule(
                int(grid_spec["feat_grid_size"]), _cfg.param_grid_low,
                _cfg.param_range_high)
        ]
    return payload


def compute_compiled_run_fingerprint(
    *, bundle_path, bundle_manifest, checkpoint_path, continuous_spec,
    skip_motor_noise, evaluation_methods, corr_weight, density_curve_spec,
) -> Dict[str, Any]:
    """Identify a WNM run whose complete empirical contract is a compiled bundle."""
    bundle_path = Path(bundle_path)
    if bundle_manifest["population"] != "signed_bias":
        raise ValueError("production DM-WNM fits require a signed_bias bundle")
    motor = ({"mode": "skip"} if skip_motor_noise else {
        "mode": "enabled",
        "sd_motor_low": 0.1,
        "sd_motor_hard_max": 50.0,
        "cap_rule": "min_condition_circ_sd_x1.1_clipped_0.1_50",
    })
    from contextual_biases_database import SHARED_PREDICTIVE_CONTRACT

    return {
        # v2: records the cross-family predictive contract. The pooled curve-level
        # CRPS objectives implement it, and a run computed under a different one is
        # not the same run even at identical parameters and bundle.
        "schema_version": 2,
        "input_contract": "contextual_biases_compiled_bundle",
        "shared_predictive_contract": SHARED_PREDICTIVE_CONTRACT,
        "bundle_id": bundle_manifest["bundle_id"],
        "bundle_manifest_sha256": file_sha256(bundle_path / "bundle.yaml"),
        "canonical_trial_sha256": bundle_manifest["canonical_trial_sha256"],
        "analysis_spec_sha256": bundle_manifest["analysis_spec_sha256"],
        "ordered_scored_row_id_sha256": bundle_manifest["ordered_scored_row_id_sha256"],
        "empirical_targets_sha256": bundle_manifest["products"]
        ["shared_empirical_targets"]["sha256"],
        "population": bundle_manifest["population"],
        "compiled_objective_versions": dict(bundle_manifest["objective_versions"]),
        "dm_objective_versions": objective_versions_for("wnm", evaluation_methods),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "surrogate_family": "wnm",
        "search_backend": "continuous",
        "continuous_spec": {
            key: continuous_spec[key] for key in sorted(continuous_spec)
        },
        "density_curve_spec": dict(density_curve_spec),
        "motor": motor,
        "corr_weight": float(corr_weight),
    }


def fingerprint_digest(payload: Dict[str, Any]) -> str:
    """Canonical SHA-256 digest of a fingerprint payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flatten(payload: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested payload to dotted paths so diffs name the field."""
    if isinstance(payload, dict):
        flat: Dict[str, Any] = {}
        for key, value in payload.items():
            flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: payload}


def fingerprint_diff(expected: Dict[str, Any], found: Dict[str, Any]) -> List[str]:
    """Human-readable per-field diff of two fingerprint payloads."""
    flat_expected = _flatten(expected)
    flat_found = _flatten(found)
    lines = []
    for key in sorted(set(flat_expected) | set(flat_found)):
        left = flat_found.get(key, "<absent>")
        right = flat_expected.get(key, "<absent>")
        if left != right:
            lines.append(f"  {key}: on disk={left!r}  this run={right!r}")
    return lines


def sidecar_path(output_dir: os.PathLike | str) -> Path:
    return Path(output_dir) / FINGERPRINT_FILENAME


def read_fingerprint_sidecar(output_dir: os.PathLike | str) -> Optional[Dict[str, Any]]:
    """Read the sidecar, or ``None`` when absent.

    Raises:
        ValueError: if the sidecar exists but is unreadable or malformed.  A
            corrupt sidecar is not treated as absent: "absent" is a meaningful
            state (pre-fingerprint results) and must not be forged by damage.
    """
    path = sidecar_path(output_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r") as handle:
            sidecar = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Run fingerprint {path} exists but could not be read: {exc}") from exc
    if not isinstance(sidecar, dict) or "digest" not in sidecar or "payload" not in sidecar:
        raise ValueError(
            f"Run fingerprint {path} is malformed (expected 'digest' and 'payload' keys)."
        )
    return sidecar


def write_fingerprint_sidecar(output_dir: os.PathLike | str, payload: Dict[str, Any]) -> None:
    """Atomically write the sidecar.

    Callers MUST install the results pickle before calling this.  Each file is
    individually atomic, but two replaces are not one transaction: a sidecar
    installed ahead of its pickle leaves, on a crash in between, a matching
    digest sitting on stale results -- exactly the failure this module exists to
    prevent.  Pickle-first fails closed instead (new pickle, old-or-absent
    sidecar, which raises).
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    sidecar = {"digest": fingerprint_digest(payload), "payload": payload}
    target = sidecar_path(output_path)
    tmp = output_path / f".{target.name}.tmp"
    with open(tmp, "w") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)


class StaleResultsError(RuntimeError):
    """Raised when results on disk were not produced by the run about to extend them."""


def enforce_fingerprint(
    output_dir: os.PathLike | str,
    expected_payload: Dict[str, Any],
    results_present: bool,
) -> None:
    """Refuse to continue when on-disk results do not match this run.

    Rules:
      - sidecar present, digest matches            -> return (resume as usual);
      - sidecar present, digest differs            -> raise with a per-field diff;
      - sidecar absent, results present            -> raise (pre-fingerprint results);
      - sidecar absent, no results                 -> return (fresh run; the
        sidecar is written by the first `save_results`).

    A matching digest says nothing about *coverage*; partial results are the
    normal mid-run state and the existing condition/method checks handle them.

    Args:
        results_present: whether the results pickle holds any entries.

    Raises:
        StaleResultsError: on any of the refusal cases above.
    """
    sidecar = read_fingerprint_sidecar(output_dir)
    expected_digest = fingerprint_digest(expected_payload)

    if sidecar is None:
        if results_present:
            raise StaleResultsError(
                f"{Path(output_dir)} holds fit results but no {FINGERPRINT_FILENAME}: they predate "
                "run fingerprinting, so there is no way to tell whether they were produced the "
                "same way as this run. Re-run with --force-refit to discard and refit them, or "
                "point --output-dir somewhere else."
            )
        return

    if sidecar["digest"] == expected_digest:
        return

    diff = fingerprint_diff(expected_payload, sidecar["payload"]) or [
        "  <no field differs; digest mismatch implies a schema or serialization change>"
    ]
    raise StaleResultsError(
        f"Results in {Path(output_dir)} were produced by a different run configuration.\n"
        f"  on disk: {sidecar['digest']}\n"
        f"  this run: {expected_digest}\n"
        + "\n".join(diff)
        + "\nResuming would mix results computed different ways. Re-run with --force-refit to "
        "discard them and refit, or point --output-dir somewhere else."
    )
