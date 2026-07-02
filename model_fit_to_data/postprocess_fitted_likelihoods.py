#!/usr/bin/env python3
"""Postprocess fitted demixing-model likelihoods without refitting.

Re-scores each fitted condition on the cleaned prepared-trial stream and exports
per-trial likelihood contributions in the demixing model's native observation
space: rounded (bias_to_distr_corr, abs_td_dist) grid cells.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

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
) -> np.ndarray:
    """Match the fitter's JAX/float32 round-to-grid indexing path."""
    indices = jnp.round(
        (jnp.asarray(values, dtype=jnp.float32) - np.float32(grid_min))
        / np.float32(grid_step)
    ).astype(jnp.int32)
    indices = jnp.clip(indices, 0, grid_size - 1)
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
) -> np.ndarray:
    params = jnp.asarray([[
        float(fit_row["sd_feat1"]),
        float(fit_row["sd_feat2"]),
        float(fit_row["sd_spat"]),
    ]], dtype=jnp.float32)
    log_surface = optimizer._predict_batch_fixed_size(params, verbosity=0)
    sd_motor = float(fit_row.get("sd_motor", 0.0))
    if sd_motor > 0:
        kernel_fft = create_motor_noise_kernel_fft(sd_motor, optimizer.n_mu1_bias)
        log_surface = apply_motor_noise_with_precomputed_kernel(log_surface, kernel_fft)
    return np.asarray(log_surface[0])


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
    log_surface = predict_log_surface(optimizer, fit_row)
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
    parser.add_argument("--output", default=None, help="Output CSV path for per-trial likelihoods.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fits_csv = Path(args.fits_csv)
    csv_dir = fits_csv.parent
    output = Path(args.output) if args.output else csv_dir / "trial_loglik.csv"
    check_output = Path(args.check_output) if args.check_output else csv_dir / "trial_loglik_checks.csv"

    out, checks = postprocess(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    checks.to_csv(check_output, index=False)

    finite_abs_diff = checks["abs_diff"].dropna()
    if not finite_abs_diff.empty:
        max_abs_diff = float(finite_abs_diff.max())
        print(f"stored-vs-rescored max abs diff: {max_abs_diff:.6g}")
        if max_abs_diff > args.max_check_abs_diff:
            raise RuntimeError(
                "Stored eval_likelihood_loss reproduction failed: "
                f"max abs diff {max_abs_diff:.6g} > {args.max_check_abs_diff:.6g}"
            )
    print(f"wrote {len(out)} rows -> {output}")
    print(f"wrote {len(checks)} checks -> {check_output}")


if __name__ == "__main__":
    main()
