#!/usr/bin/env python3
"""Export and plot fitted WNM curves directly from analytic mixture predictions."""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from model_fit_to_data.postprocess_fitted_likelihoods import write_split_trial_loglik
from model_fit_to_data.run_fingerprint import file_sha256, read_fingerprint_sidecar
from shared import surrogate
from shared.config import DENSITY_CURVE_SPEC, config
from shared.prediction import mixture_plot_curves, predictor_from_surrogate
from model_fit_to_data.wnm_scoring import trial_log_density

SELECTED_METHODS = ("likelihood", "bias_weighted_crps", "density", "smoothed_exp")
MAX_LIKELIHOOD_REPLAY_ABS_DIFF = 0.01


def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def compiled_trial_likelihoods(results, predictor, identity, methods):
    """Score compiled trial coordinates without reconstructing trial variables."""
    rows = []
    checks = []
    for analysis_cell_id, result in results.items():
        if "ordered_row_ids" not in result:
            continue
        data = np.asarray(result["data_df"], dtype=np.float32)
        row_ids = np.asarray(result["ordered_row_ids"])
        if data.shape != (len(row_ids), 2):
            raise ValueError(f"compiled coordinates and row IDs differ for {analysis_cell_id}")
        scale = float(result["angle_scale_to_model"])
        period = float(result["circ_space"])
        values = result.get("analysis_cell_values", {})
        for method in methods:
            key = f"{method}_fitted_params"
            if key not in result:
                continue
            parameters = np.asarray(result[key], dtype=float)
            log_density = np.asarray(trial_log_density(
                predictor, *parameters[:3], jnp.asarray(data[:, 0]),
                jnp.asarray(data[:, 1]), sd_motor=float(parameters[3])))
            if not np.isfinite(log_density).all():
                raise RuntimeError(
                    f"non-finite WNM likelihood for {analysis_cell_id}/{method}")
            log_mass = log_density + np.log(float(config.mu1_bias_step))
            physical_bin_width = float(config.mu1_bias_step) / scale
            metadata = {
                "analysis_cell_id": analysis_cell_id,
                "fit_group_id": result["fit_group_id"],
                "fit_subject": str(values.get("subject_id", "")),
                "fit_experiment": str(values.get("experiment_id", "")),
                "fit_condition": str(values.get("condition_id", "")),
                "optimizer": method,
                "sd_feat1": parameters[0], "sd_feat2": parameters[1],
                "sd_spat": parameters[2], "sd_motor": parameters[3],
                **result["bundle_identity"], **identity,
            }
            rows.append(pd.DataFrame({
                "row_id": row_ids,
                "experiment_id": str(values.get("experiment_id", "")),
                "subject_id": str(values.get("subject_id", "")),
                "condition_id": str(values.get("condition_id", "")),
                "report_order": int(values.get("report_order", 1)),
                "circular_period_deg": period,
                "dissimilarity_deg": data[:, 0] / scale,
                "bias_toward_context_deg": data[:, 1] / scale,
                "include_signed_loss": True,
                "feat_diff_model_deg": data[:, 0],
                "bias_model_deg": data[:, 1],
                "trial_index_within_fit": np.arange(len(data), dtype=np.int64),
                "valid_model_eval": True,
                "include_common_eval": True,
                "loglik_density_model_deg": log_density,
                "nll_density_model_deg": -log_density,
                "loglik_mass": log_mass,
                "nll_mass": -log_mass,
                "loglik_density_deg": log_density + np.log(scale),
                "nll_density_deg": -log_density - np.log(scale),
                "bin_width_deg": physical_bin_width,
                "loglik_convention": "continuous_at_observation",
                **metadata,
            }))
            checks.append({
                "analysis_cell_id": analysis_cell_id,
                "optimizer": method,
                "n_obs_scored": len(data),
                "stored_eval_likelihood_loss": result[f"{method}_eval_likelihood_loss"],
                "per_trial_sum_nll_density_model_deg": float((-log_density).sum()),
                "abs_diff": abs(float((-log_density).sum()) -
                                float(result[f"{method}_eval_likelihood_loss"])),
                **result["bundle_identity"], **identity,
            })
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
            pd.DataFrame(checks))


def export_curves(results_dir: Path, checkpoint: Path, output_dir: Path,
                  methods=SELECTED_METHODS):
    sidecar = read_fingerprint_sidecar(results_dir)
    if sidecar is None:
        raise ValueError(f"no run fingerprint in {results_dir}")
    payload = sidecar["payload"]
    if payload.get("surrogate_family") != surrogate.FAMILY_WNM:
        raise ValueError("direct WNM export requires a WNM fit fingerprint")
    if payload["checkpoint_sha256"] != file_sha256(checkpoint):
        raise ValueError("checkpoint does not match the fit fingerprint")

    with open(results_dir / "extended_fit_results.pkl", "rb") as handle:
        results = pickle.load(handle)
    predictor = predictor_from_surrogate(
        surrogate.load_surrogate(checkpoint_path=checkpoint))
    identity = predictor.identity().as_dict()
    feat_grid = np.asarray(config.create_grid("feat_diff"), dtype=np.float32)
    density_curve_spec = payload.get("density_curve_spec", DENSITY_CURVE_SPEC)
    matmul_precision = payload["continuous_spec"]["matmul_precision"]

    rows = []
    parameter_rows = []
    for condition, result in results.items():
        empirical = result["empirical_curves"]
        values = result.get("analysis_cell_values", {})
        experiment = str(values.get("experiment_id", condition))
        subject = str(values.get("subject_id", result.get("fit_group_id", "")))
        source_condition = str(values.get("condition_id", condition))
        report_order = int(values.get("report_order", 1))
        cell_values = json.dumps(values, sort_keys=True)
        bundle_identity = result.get("bundle_identity", {})
        operator = np.asarray(empirical["feature_operator"])
        bandwidth = float(empirical["density_bandwidth"])
        angle_scale = float(result.get("angle_scale_to_model", 1.0))
        for method in methods:
            key = f"{method}_fitted_params"
            if key not in result:
                continue
            parameters = np.asarray(result[key], dtype=float)
            parameter_rows.append({
                "analysis_cell_id": condition,
                "fit_group_id": result.get("fit_group_id", condition),
                "experiment": experiment,
                "subject": subject, "condition": source_condition,
                "report_order": report_order, "analysis_cell_values": cell_values,
                "optimizer": method, "n_trials": result["n_trials"],
                "sd_feat1": parameters[0], "sd_feat2": parameters[1],
                "sd_spat": parameters[2], "sd_motor": parameters[3],
                "loss": result[f"{method}_loss"],
                f"{method}_loss": result[f"{method}_loss"],
                **{f"eval_{objective}_loss": result[f"{method}_eval_{objective}_loss"]
                   for objective in SELECTED_METHODS},
                **bundle_identity, **identity,
            })
            with jax.default_matmul_precision(matmul_precision):
                curves = mixture_plot_curves(
                    predictor, parameters[None, :3], feat_grid,
                    sd_motor_by_row=[parameters[3]],
                    emp_density_weights_sd=density_curve_spec["emp_density_weights_sd"],
                    density_smoothing_sigma=density_curve_spec["density_smoothing_sigma"],
                    feature_operators=operator[None, :, :],
                    density_bandwidths=[bandwidth],
                    operator_feature_coordinates=empirical["prediction_coordinates"])
            for index, x_model in enumerate(feat_grid):
                rows.append({
                    "analysis_cell_id": condition,
                    "fit_group_id": result.get("fit_group_id", condition),
                    "experiment": experiment,
                    "subject": subject, "condition": source_condition,
                    "report_order": report_order, "analysis_cell_values": cell_values,
                    "optimizer": method,
                    "x_model_deg": float(x_model),
                    "x_deg": float(x_model / angle_scale),
                    "feat_diff": float(x_model / angle_scale),
                    "bias_deg": float(curves["bias"][0, index] / angle_scale),
                    "mu_bias": float(curves["bias"][0, index] / angle_scale),
                    "density_asymmetry": float(curves["asymmetry"][0, index]),
                    "sd_deg": float(curves["sd"][0, index] / angle_scale),
                    "empirical_bias_deg": float(
                        np.asarray(empirical["target_bias_curve"])[index] / angle_scale),
                    "empirical_density_asymmetry": float(
                        np.asarray(empirical["matched_density_target"])[index]),
                    "sd_feat1": float(parameters[0]), "sd_feat2": float(parameters[1]),
                    "sd_spat": float(parameters[2]), "sd_motor": float(parameters[3]),
                    "density_bandwidth": bandwidth,
                    **bundle_identity, **identity,
                })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no selected fitted methods found")
    with jax.default_matmul_precision(matmul_precision):
        likelihoods, checks = compiled_trial_likelihoods(
            results, predictor, identity, methods)
    if not likelihoods.empty:
        if checks["abs_diff"].max() > MAX_LIKELIHOOD_REPLAY_ABS_DIFF:
            worst = checks.loc[checks["abs_diff"].idxmax()]
            raise RuntimeError(
                "compiled likelihood replay differs from the fitted objective by "
                f"{worst['abs_diff']:.6g} in "
                f"{worst['analysis_cell_id']}/{worst['optimizer']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "fitted_curves.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output_dir / "fitted_parameters.csv", index=False)
    if not likelihoods.empty:
        write_split_trial_loglik(likelihoods, output_dir / "trial_loglik_split")
        checks.to_csv(output_dir / "trial_loglik_checks.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps({
        "source_results": str(results_dir), "checkpoint": str(checkpoint),
        "run_fingerprint_digest": sidecar["digest"], "methods": list(methods),
        **identity,
        "prediction": "direct analytic WNM; no reconstructed NN surface",
        "density_curve": "subject-experiment pooled-SJ KDE plus observed-design feature operator",
        "bias_curve": "observed-design pooled complex first moment",
        "trial_likelihood": ("continuous density at compiled trial coordinates; "
                             "mass uses the two-model-degree reporting cell"
                             if not likelihoods.empty else None),
    }, indent=2) + "\n")

    for condition, condition_frame in frame.groupby("analysis_cell_id", sort=False):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        for method, values in condition_frame.groupby("optimizer", sort=False):
            axes[0].plot(values.x_deg, values.bias_deg, label=method)
            axes[1].plot(values.x_deg, values.density_asymmetry, label=method)
            axes[2].plot(values.x_deg, values.sd_deg, label=method)
        first = condition_frame[condition_frame.optimizer == condition_frame.optimizer.iloc[0]]
        axes[0].plot(first.x_deg, first.empirical_bias_deg, "k--", label="empirical")
        axes[1].plot(first.x_deg, first.empirical_density_asymmetry, "k--", label="empirical")
        for axis, title, ylabel in zip(
                axes, ("Mean bias", "Density asymmetry", "Circular SD"),
                ("degrees", "signed mass", "degrees")):
            axis.set(title=title, xlabel="dissimilarity (degrees)", ylabel=ylabel)
        axes[0].legend(fontsize=7)
        fig.suptitle(condition)
        fig.savefig(output_dir / f"{_safe(condition)}.png", dpi=160)
        plt.close(fig)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", choices=SELECTED_METHODS,
                        default=list(SELECTED_METHODS))
    args = parser.parse_args(argv)
    frame = export_curves(args.results_dir, args.checkpoint, args.output_dir, args.methods)
    print(f"Wrote {len(frame)} direct WNM curve rows to {args.output_dir}")


if __name__ == "__main__":
    main()
