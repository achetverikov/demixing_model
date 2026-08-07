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
SCHEMA_VERSION = 2

FINGERPRINT_FILENAME = "extended_run_fingerprint.json"

#: How each objective is computed, independent of which objectives a given run
#: was asked to fit.  Cross-objective evaluation (`evaluate_parameter_losses`)
#: writes a loss for *every* entry here at each fitted method's parameters, so a
#: change to any one of them invalidates results nominally fitted under another.
#: Bump the individual string when an objective's definition changes.
OBJECTIVE_VERSIONS: Dict[str, str] = {
    # 1 - CCC, with constant-target conditions excluded -- `loss_type="ccc"`.
    "density": "ccc_excluding_constant_targets@2",
    # The pre-2026-08 density objective, 0.75 * MSE/range + 0.25 * (1 - r), kept
    # so published numbers stay reproducible -- `loss_type="combined"`.
    "density_legacy": "combined_range_scaled_mse_plus_corr@1",
    "expectation": "binned_circular_mean_mse@1",
    "smoothed_exp": "smoothed_circular_mean_mse@1",
    "likelihood": "trial_loglik@1",
    "crps": "crps@1",
    "balanced_crps": "balanced_crps@1",
    "bias_weighted_crps": "bias_weighted_crps@1",
}

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
    grid_spec: Dict[str, Any],
    density_curve_spec: Dict[str, Any],
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
        search_backend: ``"hierarchical"`` or ``"exhaustive_1deg"``.
        curve_cache_key: Cache identity when cache-backed, else ``None``.
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

    unknown = sorted(set(evaluation_methods) - set(OBJECTIVE_VERSIONS))
    if unknown:
        raise ValueError(
            f"No objective version recorded for {unknown}; add them to "
            "run_fingerprint.OBJECTIVE_VERSIONS (and bump SCHEMA_VERSION) before "
            "results computed with them can be fingerprinted."
        )

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

    feat_schedule = effective_feat_step_schedule(
        int(grid_spec["feat_grid_size"]), _cfg.param_grid_low, _cfg.param_range_high
    )

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "data_sha256": file_sha256(data_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "circ_space": int(circ_space),
        "objective_versions": {
            method: OBJECTIVE_VERSIONS[method] for method in sorted(evaluation_methods)
        },
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
        "grid_spec": {
            "shared_grid_size": int(grid_spec["shared_grid_size"]),
            "feat_grid_size": int(grid_spec["feat_grid_size"]),
            "min_grid_step": float(grid_spec["min_grid_step"]),
            "zoom_factor": float(grid_spec["zoom_factor"]),
        },
        "model_grids": {
            "feat_diff_range": list(_cfg.feat_diff_range),
            "feat_diff_step": int(_cfg.feat_diff_step),
            "mu1_bias_range": list(_cfg.mu1_bias_range),
            "mu1_bias_step": int(_cfg.mu1_bias_step),
        },
        "density_curve_spec": dict(density_curve_spec),
        "refinement_spec": refinement_spec,
        "param_bounds": {
            "param_grid_low": float(_cfg.param_grid_low),
            "param_range_high": float(_cfg.param_range_high),
        },
        "feat_step_schedule": [float(step) for step in feat_schedule],
        "degenerate_eps": None if degenerate_eps is None else float(degenerate_eps),
    }
    return payload


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
