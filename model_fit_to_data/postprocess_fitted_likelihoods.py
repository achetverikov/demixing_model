#!/usr/bin/env python3
"""Postprocess fitted demixing-model likelihoods without refitting.

Re-scores each fitted condition on the cleaned prepared-trial stream and exports
per-trial likelihood contributions in the demixing model's native observation
space: rounded (bias_to_distr_corr, abs_td_dist) grid cells.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from grid_based_multi_condition_optimizer_jax_loops import (
        GridBasedMultiConditionOptimizer,
        apply_motor_noise_with_precomputed_kernel,
        create_motor_noise_kernel_fft,
    )
except ModuleNotFoundError:
    from model_fit_to_data.grid_based_multi_condition_optimizer_jax_loops import (
        GridBasedMultiConditionOptimizer,
        apply_motor_noise_with_precomputed_kernel,
        create_motor_noise_kernel_fft,
    )
from shared.config import config
from shared.utils import filter_data_for_fitting


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_20 = REPO_ROOT / "pretrained/model_epoch1500_10ktrain_20samples.pkl"
DEFAULT_100 = REPO_ROOT / "pretrained/model_epoch1500_10ktrain_100samples.pkl"

# Motor-noise density floor (B1 floor-aware reproduction gate).
#
# apply_motor_noise_with_precomputed_kernel clips the convolved probability at 0
# and re-logs it as log(prob + MOTOR_NOISE_FLOOR_EPS) + log_max. This is a HARD
# cliff at log_max + log(eps): any trial whose convolved density underflows to ~0
# pins to that value. In the deep tail of a peaked kernel the pre-log probability
# is near float32 underflow, so which trials pin to the floor is backend-dependent
# (CPU vs GPU vs the fit's own build differ by whole nats there). The stored
# eval_likelihood_loss and this rescore can therefore disagree by tens of nats on
# such conditions even though every parameter is bit-identical - an inherently
# ill-conditioned quantity, not a bug. See TODO.md
# "Motor-noise likelihood reproduction is ill-conditioned at the density floor".
#
# MOTOR_NOISE_FLOOR_EPS MUST match the epsilon in
# grid_based_multi_condition_optimizer_jax_loops.apply_motor_noise_with_precomputed_kernel.
MOTOR_NOISE_FLOOR_EPS = 1e-10
# A trial counts as "floor region" (its likelihood is non-reproducible) when its
# rescored log-density sits within this many nats of the per-surface floor. 6 nats
# comfortably covers the observed backend-to-backend flip band (trials seen moving
# between ~-25 and ~-19).
MOTOR_NOISE_FLOOR_BAND_NATS = 6.0
# Provable per-trial swing bound: a floor-region trial's log-density can range from
# the floor (log_max + log(eps)) up to at most the surface max (log_max), a span of
# -log(eps) nats. Each such trial therefore contributes up to this much reproduction
# slack; a condition with n floor-region trials is allowed n * this much disagreement.
MOTOR_NOISE_FLOOR_TRIAL_NATS = float(-np.log(MOTOR_NOISE_FLOOR_EPS))

FIT_META_COLS = [
    "fit_subject",
    "fit_experiment",
    "fit_condition",
    "optimizer",
    "sd_feat1",
    "sd_feat2",
    "sd_spat",
    "sd_motor",
    "prepared_data_source",
]

LIKELIHOOD_COLS = [
    "feat_diff_model_deg",
    "bias_model_deg",
    "feat_idx",
    "bias_idx",
    "trial_index_within_fit",
    "valid_model_eval",
    "include_common_eval",
    "loglik_density_model_deg",
    "nll_density_model_deg",
    "loglik_mass",
    "nll_mass",
    "loglik_density_deg",
    "nll_density_deg",
    "bin_width_deg",
]


def _write_parquet(df: pd.DataFrame, path: Path, compression: str) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression=compression, use_dictionary=True)


def write_split_trial_loglik(out: pd.DataFrame, output_dir: Path, compression: str = "zstd") -> None:
    """Write demixing per-trial likelihoods as normalized split Parquet tables."""
    output_dir = Path(output_dir)
    fit_cols = [col for col in FIT_META_COLS if col in out.columns]
    likelihood_cols = [col for col in LIKELIHOOD_COLS if col in out.columns]
    trial_cols = [col for col in out.columns if col not in set(fit_cols + likelihood_cols)]
    if not fit_cols or not likelihood_cols:
        raise ValueError("trial likelihood table is missing fit or likelihood columns")

    trial_meta = out[trial_cols].drop_duplicates().reset_index(drop=True)
    trial_meta.insert(0, "trial_id", np.arange(len(trial_meta), dtype=np.int64))
    fit_meta = out[fit_cols].drop_duplicates().reset_index(drop=True)
    fit_meta.insert(0, "fit_id", np.arange(len(fit_meta), dtype=np.int64))

    likelihood = out[trial_cols + fit_cols + likelihood_cols].merge(
        trial_meta, on=trial_cols, how="left", validate="many_to_one"
    ).merge(
        fit_meta, on=fit_cols, how="left", validate="many_to_one"
    )
    if likelihood["trial_id"].isna().any() or likelihood["fit_id"].isna().any():
        raise RuntimeError("failed to assign trial_id/fit_id while writing split likelihoods")
    likelihood = likelihood[["trial_id", "fit_id"] + likelihood_cols]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    backup_dir = None
    try:
        _write_parquet(trial_meta, tmp_dir / "trial_meta.parquet", compression)
        _write_parquet(fit_meta, tmp_dir / "fit_meta.parquet", compression)
        _write_parquet(likelihood, tmp_dir / "likelihood.parquet", compression)
        manifest = {
            "format": "split_demixing_trial_loglik",
            "version": 1,
            "source": "postprocess_fitted_likelihoods.py",
            "rows": int(len(out)),
            "trial_meta_rows": int(len(trial_meta)),
            "fit_meta_rows": int(len(fit_meta)),
            "trial_columns": trial_cols,
            "fit_columns": fit_cols,
            "likelihood_columns": likelihood_cols,
            "compression": compression,
        }
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output_dir.exists():
            backup_dir = Path(tempfile.mkdtemp(
                prefix=f".{output_dir.name}.backup.", dir=output_dir.parent
            ))
            backup_dir.rmdir()
            output_dir.rename(backup_dir)
        try:
            tmp_dir.rename(output_dir)
        except Exception:
            if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
                backup_dir.rename(output_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


REPRODUCTION_VALUE_COLS = [
    "stored_eval_likelihood_loss",
    "rescored_nll_density_model_deg",
    "per_trial_sum_nll_density_model_deg",
    "rescored_nll_mass",
    "abs_diff",
    "per_trial_sum_abs_diff",
]


def prepare_reproduction_checks(checks: pd.DataFrame, max_abs_diff: float) -> pd.DataFrame:
    """Add finite-value and floor-aware tolerance results to reproduction checks."""
    checks = checks.copy()
    if "n_floor_trials" not in checks.columns:
        checks["n_floor_trials"] = 0
    missing = [col for col in REPRODUCTION_VALUE_COLS if col not in checks.columns]
    if missing:
        raise ValueError(f"reproduction checks are missing numeric columns: {missing}")
    numeric_values = checks[REPRODUCTION_VALUE_COLS].apply(pd.to_numeric, errors="coerce")
    checks[REPRODUCTION_VALUE_COLS] = numeric_values
    checks["values_finite"] = np.isfinite(numeric_values.to_numpy()).all(axis=1)
    checks["floor_tolerance"] = (
        max_abs_diff
        + checks["n_floor_trials"].fillna(0) * MOTOR_NOISE_FLOOR_TRIAL_NATS
    )
    checks["within_tolerance"] = (
        checks["values_finite"]
        & np.isfinite(checks["floor_tolerance"])
        & (checks["abs_diff"] <= checks["floor_tolerance"])
    )
    return checks


def validate_reproduction_checks(checks: pd.DataFrame) -> None:
    """Raise when any reproduction value is non-finite or outside tolerance."""
    nonfinite = checks[~checks["values_finite"]]
    if not nonfinite.empty:
        identities = ", ".join(
            f"{r.optimizer} {r.subject}/{r.experiment}/{r.condition}"
            for r in nonfinite.itertuples()
        )
        raise RuntimeError(
            "Stored eval_likelihood_loss reproduction produced non-finite values for "
            f"{len(nonfinite)} condition(s): {identities}"
        )

    failed = checks[~checks["within_tolerance"]]
    if not failed.empty:
        worst = failed.loc[failed["abs_diff"].idxmax()]
        raise RuntimeError(
            "Stored eval_likelihood_loss reproduction failed: "
            f"{len(failed)} condition(s) exceed the floor-aware tolerance; worst "
            f"abs_diff {float(worst['abs_diff']):.6g} > tolerance "
            f"{float(worst['floor_tolerance']):.6g} "
            f"({worst['optimizer']} {worst['subject']}/{worst['experiment']}/"
            f"{worst['condition']}, n_floor_trials={int(worst['n_floor_trials'])})"
        )


def normalize_label(value: object) -> str:
    return "_".join(str(value).strip().split()).lower()


def normalize_condition_label(value: object) -> str:
    return re.sub(r"[\s_-]+", "", str(value).strip().lower())


def infer_checkpoint_path(fits_csv: Path, checkpoint_path: str | None) -> Path:
    if checkpoint_path:
        return Path(checkpoint_path)
    text = str(fits_csv)
    if "20samples" in text:
        return Path(DEFAULT_20)
    if "100samples" in text:
        return Path(DEFAULT_100)
    raise ValueError(
        "Could not infer checkpoint path from fits CSV path; pass --checkpoint-path explicitly."
    )


def load_fit_rows(path: Path, optimizers: Iterable[str] | None) -> pd.DataFrame:
    dt = pd.read_csv(path)
    required = {
        "subject", "experiment", "condition", "optimizer",
        "sd_feat1", "sd_feat2", "sd_spat", "sd_motor", "eval_likelihood_loss",
    }
    missing = sorted(required.difference(dt.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if optimizers:
        dt = dt[dt["optimizer"].isin(list(optimizers))].copy()
    return dt.reset_index(drop=True)


def load_trial_data(path: Path, outlier_col: str | None, include_outliers: bool) -> pd.DataFrame:
    dt = pd.read_csv(path, low_memory=False)
    dt["_source_row_index"] = np.arange(len(dt), dtype=int)
    if "is_combined" in dt.columns:
        dt = dt[dt["is_combined"] != True].copy()
    if not include_outliers and outlier_col and outlier_col in dt.columns:
        dt = dt[dt[outlier_col] != 1].copy()
    return dt.reset_index(drop=True)


def angle_scale_to_model(circ_space: int) -> float:
    return (2 * config.feat_diff_range[1]) / float(circ_space)


def physical_bin_width_deg(circ_space: int) -> float:
    return config.mu1_bias_step / angle_scale_to_model(circ_space)


def model_grid_indices(
    values: np.ndarray,
    grid_min: float,
    grid_step: float,
    grid_size: int,
    circular: bool = False,
) -> np.ndarray:
    """Match the fitter's JAX/float32 round-to-grid indexing path.

    ``circular=True`` wraps out-of-range indices instead of clipping them, which
    is what the mu1_bias axis needs — it is a circle, so an angle just past the
    last grid point belongs in bin 0, not in the last bin.  feat_diff is a
    bounded interval and keeps the clip.
    """
    indices = jnp.round(
        (jnp.asarray(values, dtype=jnp.float32) - np.float32(grid_min))
        / np.float32(grid_step)
    ).astype(jnp.int32)
    indices = (jnp.mod(indices, grid_size) if circular
               else jnp.clip(indices, 0, grid_size - 1))
    return np.asarray(indices, dtype=int)


def prepare_condition_rows(
    data: pd.DataFrame,
    fit_row: pd.Series,
    exp_col: str,
    subject_col: str,
    condition_col: str,
    x_col: str,
    y_col: str,
    circ_space: int,
) -> pd.DataFrame:
    fit_subject = normalize_label(fit_row["subject"])
    fit_experiment = normalize_label(fit_row["experiment"])
    fit_condition = normalize_condition_label(fit_row["condition"])
    composite_experiment = normalize_label(f"{fit_row['experiment']}_{fit_row['condition']}")

    subject_match = data[subject_col].map(normalize_label) == fit_subject
    condition_match = data[condition_col].map(normalize_condition_label) == fit_condition
    experiment_match = data[exp_col].map(normalize_label).isin([fit_experiment, composite_experiment])

    subset = data[subject_match & condition_match & experiment_match].copy()
    if subset.empty:
        raise ValueError(
            "No prepared-data rows matched "
            f"subject={fit_row['subject']}, experiment={fit_row['experiment']}, "
            f"condition={fit_row['condition']}"
        )

    # Match the fitter's model-space dissimilarity clamp (see fit_model_to_data /
    # codex_audit.md report-level #3): raw bounds = feat_diff_range / scale.
    scale = angle_scale_to_model(circ_space)
    cleaned = filter_data_for_fitting(
        subset, feat_diff_col=x_col, bias_col=y_col, verbose=False,
        min_diss=config.feat_diff_range[0] / scale,
        max_diss=config.feat_diff_range[1] / scale)
    if cleaned.empty:
        raise ValueError(
            "All rows were removed by filter_data_for_fitting for "
            f"subject={fit_row['subject']}, experiment={fit_row['experiment']}, "
            f"condition={fit_row['condition']}"
        )
    cleaned = cleaned.reset_index(drop=True)
    cleaned["feat_diff_model_deg"] = cleaned[x_col].to_numpy(float) * scale
    cleaned["bias_model_deg"] = cleaned[y_col].to_numpy(float) * scale

    feat_idx = model_grid_indices(
        cleaned["feat_diff_model_deg"].to_numpy(float),
        grid_min=config.feat_diff_range[0],
        grid_step=config.feat_diff_step,
        grid_size=config.feat_diff_grid_size,
    )

    bias_idx = model_grid_indices(
        cleaned["bias_model_deg"].to_numpy(float),
        grid_min=config.mu1_bias_range[0],
        grid_step=config.mu1_bias_step,
        grid_size=config.mu1_bias_grid_size,
        circular=True,
    )

    cleaned["feat_idx"] = feat_idx
    cleaned["bias_idx"] = bias_idx
    cleaned["trial_index_within_fit"] = np.arange(len(cleaned), dtype=int)
    cleaned["valid_model_eval"] = True
    cleaned["include_common_eval"] = (
        np.isfinite(cleaned[x_col].to_numpy(float))
        & np.isfinite(cleaned[y_col].to_numpy(float))
    )
    return cleaned


def prepare_condition_rows_from_sources(
    data_sources: list[tuple[str, pd.DataFrame]],
    fit_row: pd.Series,
    exp_col: str,
    subject_col: str,
    condition_col: str,
    x_col: str,
    y_col: str,
    circ_space: int,
) -> tuple[pd.DataFrame, str]:
    last_error: ValueError | None = None
    for source_name, data in data_sources:
        try:
            return (
                prepare_condition_rows(
                    data=data,
                    fit_row=fit_row,
                    exp_col=exp_col,
                    subject_col=subject_col,
                    condition_col=condition_col,
                    x_col=x_col,
                    y_col=y_col,
                    circ_space=circ_space,
                ),
                source_name,
            )
        except ValueError as err:
            if not str(err).startswith("No prepared-data rows matched "):
                raise
            last_error = err
    if last_error is None:
        raise RuntimeError("No trial data sources were available for postprocessing.")
    raise last_error


def predict_log_surface(
    optimizer: GridBasedMultiConditionOptimizer,
    fit_row: pd.Series,
) -> tuple[np.ndarray, float | None]:
    params = jnp.asarray([[
        float(fit_row["sd_feat1"]),
        float(fit_row["sd_feat2"]),
        float(fit_row["sd_spat"]),
    ]], dtype=jnp.float32)
    log_surface = optimizer._predict_batch_fixed_size(params, verbosity=0)
    sd_motor = float(fit_row.get("sd_motor", 0.0))
    floor_log_density = None
    if sd_motor > 0:
        floor_log_density = float(np.asarray(log_surface[0]).max() + np.log(MOTOR_NOISE_FLOOR_EPS))
        kernel_fft = create_motor_noise_kernel_fft(sd_motor, optimizer.n_mu1_bias)
        log_surface = apply_motor_noise_with_precomputed_kernel(log_surface, kernel_fft)
    return np.asarray(log_surface[0]), floor_log_density


def score_fit_row(
    optimizer: GridBasedMultiConditionOptimizer,
    data_sources: list[tuple[str, pd.DataFrame]],
    fit_row: pd.Series,
    exp_col: str,
    subject_col: str,
    condition_col: str,
    x_col: str,
    y_col: str,
    circ_space: int,
) -> tuple[pd.DataFrame, dict]:
    scored, data_source = prepare_condition_rows_from_sources(
        data_sources=data_sources,
        fit_row=fit_row,
        exp_col=exp_col,
        subject_col=subject_col,
        condition_col=condition_col,
        x_col=x_col,
        y_col=y_col,
        circ_space=circ_space,
    )
    log_surface, floor_log_density = predict_log_surface(optimizer, fit_row)
    # The NN surface is a continuous density per model-degree (see
    # shared.surface_functions.normalize_to_density), not a discrete cell mass.
    # Approximate the cell probability as density × cell width — a midpoint/rectangle
    # rule at the cell centre, NOT the exact integral of the density over the cell.
    # The log(cell width) term is a constant offset that cancels in AIC/BIC
    # differences; "mass" here denotes this approximation. See codex_audit.md
    # report-level #4.
    loglik_density_model_deg = log_surface[
        scored["bias_idx"].to_numpy(int),
        scored["feat_idx"].to_numpy(int),
    ]
    model_bin_width_deg = float(config.mu1_bias_step)
    loglik_mass = loglik_density_model_deg + np.log(model_bin_width_deg)
    bin_width_deg = physical_bin_width_deg(circ_space)
    scored["loglik_density_model_deg"] = loglik_density_model_deg
    scored["nll_density_model_deg"] = -scored["loglik_density_model_deg"]
    scored["loglik_mass"] = loglik_mass
    scored["nll_mass"] = -scored["loglik_mass"]
    scored["loglik_density_deg"] = scored["loglik_mass"] - np.log(bin_width_deg)
    scored["nll_density_deg"] = scored["nll_mass"] + np.log(bin_width_deg)
    scored["bin_width_deg"] = bin_width_deg
    scored["fit_subject"] = fit_row["subject"]
    scored["fit_experiment"] = fit_row["experiment"]
    scored["fit_condition"] = fit_row["condition"]
    scored["optimizer"] = fit_row["optimizer"]
    scored["sd_feat1"] = float(fit_row["sd_feat1"])
    scored["sd_feat2"] = float(fit_row["sd_feat2"])
    scored["sd_spat"] = float(fit_row["sd_spat"])
    scored["sd_motor"] = float(fit_row["sd_motor"])
    scored["prepared_data_source"] = data_source

    # The fitter's stored eval_likelihood_loss indexes the NN log-density directly
    # and reduces with JAX segment_sum. Validate against that reduction, not a
    # pandas/NumPy sum of exported rows, because float32 reduction order can differ
    # by ~1e-2 on large/high-NLL conditions.
    flat_indices = (
        scored["bias_idx"].to_numpy(int) * optimizer.n_feat_diff
        + scored["feat_idx"].to_numpy(int)
    )
    fit_log_probs = log_surface.reshape(-1)[flat_indices]
    rescored_nll_density_model_deg = -float(jax.ops.segment_sum(
        jnp.asarray(fit_log_probs, dtype=jnp.float32),
        jnp.zeros(len(fit_log_probs), dtype=jnp.int32),
        num_segments=1,
    )[0])
    per_trial_sum_nll_density_model_deg = float(scored["nll_density_model_deg"].sum())
    rescored_nll_mass = float(scored["nll_mass"].sum())
    stored_nll = float(fit_row["eval_likelihood_loss"])

    # Floor-aware reproduction gate (B1). Motor noise introduces a hard density floor
    # at log_max + log(eps); trials pinned near it are non-reproducible across compute
    # backends. Count how many of THIS rescore's trials sit in that floor region so the
    # gate can allow the resulting (bounded) disagreement. Only motor-noise fits have
    # the floor: the raw NN surface has no such cliff, so sd_motor == 0 => zero floor
    # trials and the strict 0.01 tolerance is preserved unchanged.
    sd_motor = float(fit_row["sd_motor"])
    if sd_motor > 0 and floor_log_density is not None:
        n_floor_trials = int(np.sum(
            loglik_density_model_deg <= floor_log_density + MOTOR_NOISE_FLOOR_BAND_NATS
        ))
    else:
        n_floor_trials = 0

    check = {
        "subject": fit_row["subject"],
        "experiment": fit_row["experiment"],
        "condition": fit_row["condition"],
        "optimizer": fit_row["optimizer"],
        "prepared_data_source": data_source,
        "stored_eval_likelihood_loss": stored_nll,
        "rescored_nll_density_model_deg": rescored_nll_density_model_deg,
        "per_trial_sum_nll_density_model_deg": per_trial_sum_nll_density_model_deg,
        "rescored_nll_mass": rescored_nll_mass,
        "abs_diff": abs(rescored_nll_density_model_deg - stored_nll),
        "per_trial_sum_abs_diff": abs(per_trial_sum_nll_density_model_deg - stored_nll),
        "n_obs_scored": int(len(scored)),
        "n_floor_trials": n_floor_trials,
        "model_bin_width_deg": model_bin_width_deg,
        "bin_width_deg": bin_width_deg,
    }
    return scored, check


def postprocess(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    fits_csv = Path(args.fits_csv)
    checkpoint_path = infer_checkpoint_path(fits_csv, args.checkpoint_path)
    fit_rows = load_fit_rows(fits_csv, args.optimizers)
    trial_data = load_trial_data(Path(args.data_path), args.outlier_col, args.include_outliers)
    data_sources = [(str(Path(args.data_path)), trial_data)]

    dummy = jnp.asarray(np.zeros((1, 2), dtype=np.float32))
    optimizer = GridBasedMultiConditionOptimizer(
        str(checkpoint_path),
        {"dummy": dummy},
        skip_motor_noise=args.skip_motor_noise,
    )

    score_rows = []
    checks = []
    for _, fit_row in fit_rows.iterrows():
        scored, check = score_fit_row(
            optimizer=optimizer,
            data_sources=data_sources,
            fit_row=fit_row,
            exp_col=args.exp_col,
            subject_col=args.subject_col,
            condition_col=args.condition_col,
            x_col=args.x_col,
            y_col=args.y_col,
            circ_space=args.circ_space,
        )
        score_rows.append(scored)
        checks.append(check)

    out = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    checks_dt = pd.DataFrame(checks)
    return out, checks_dt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess demixing fitted likelihoods without refitting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", required=True, help="Prepared trial CSV used for fitting.")
    parser.add_argument("--fits-csv", required=True, help="Path to fitted_parameters.csv.")
    parser.add_argument("--checkpoint-path", default=None, help="NN checkpoint path; inferred from fits path when omitted.")
    parser.add_argument("--output", default=None, help="Output split-Parquet directory for per-trial likelihoods.")
    parser.add_argument("--check-output", default=None, help="Output CSV path for stored-vs-rescored checks.")
    parser.add_argument("--exp-col", default="expName")
    parser.add_argument("--subject-col", default="subject")
    parser.add_argument("--condition-col", default="condition")
    parser.add_argument("--x-col", default="abs_td_dist")
    parser.add_argument("--y-col", default="bias_to_distr_corr")
    parser.add_argument("--outlier-col", default="is_outlier")
    parser.add_argument("--include-outliers", action="store_true")
    parser.add_argument("--circ-space", type=int, choices=[180, 360], default=360)
    parser.add_argument("--skip-motor-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizers", nargs="*", default=None)
    parser.add_argument(
        "--max-check-abs-diff",
        type=float,
        default=0.01,
        help="Fail if stored-vs-rescored density-scale NLL differs by more than this.",
    )
    parser.add_argument(
        "--parquet-compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip"],
        help="Compression codec for split-Parquet likelihood output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fits_csv = Path(args.fits_csv)
    csv_dir = fits_csv.parent
    output = Path(args.output) if args.output else csv_dir / "trial_loglik_split"
    check_output = Path(args.check_output) if args.check_output else csv_dir / "trial_loglik_checks.csv"

    out, checks = postprocess(args)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Floor-aware reproduction gate (B1). Each condition is allowed the strict
    # --max-check-abs-diff, plus extra slack for trials pinned to the motor-noise
    # density floor, whose likelihood is not reproducible across compute backends.
    # A condition with no floor trials keeps the strict tolerance unchanged, so
    # genuine reproduction bugs in the well-conditioned regime still fail hard.
    checks = prepare_reproduction_checks(checks, args.max_check_abs_diff)
    checks.to_csv(check_output, index=False)

    finite_abs_diff = checks.loc[checks["values_finite"], "abs_diff"]
    if not finite_abs_diff.empty:
        max_abs_diff = float(finite_abs_diff.max())
        print(f"stored-vs-rescored max abs diff: {max_abs_diff:.6g}")
        floor_tolerated = checks[
            (~checks["within_tolerance"].isna())
            & checks["within_tolerance"]
            & (checks["abs_diff"] > args.max_check_abs_diff)
        ]
        if not floor_tolerated.empty:
            print(
                f"{len(floor_tolerated)} condition(s) exceeded the strict "
                f"{args.max_check_abs_diff:.6g} tolerance but are explained by "
                "motor-noise floor trials (see within_tolerance/n_floor_trials columns):"
            )
            for _, r in floor_tolerated.iterrows():
                print(
                    f"  {r['optimizer']} {r['subject']}/{r['experiment']}/{r['condition']}: "
                    f"abs_diff={r['abs_diff']:.4g} n_floor_trials={int(r['n_floor_trials'])} "
                    f"floor_tolerance={r['floor_tolerance']:.4g}"
                )
    validate_reproduction_checks(checks)
    write_split_trial_loglik(out, output, compression=args.parquet_compression)
    print(f"wrote {len(out)} rows -> {output}")
    print(f"wrote {len(checks)} checks -> {check_output}")


if __name__ == "__main__":
    main()
