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
import numpy as np
import pandas as pd

from model_fit_to_data.run_fingerprint import file_sha256, read_fingerprint_sidecar
from shared import surrogate
from shared.config import config
from shared.prediction import mixture_plot_curves, predictor_from_surrogate

SELECTED_METHODS = ("likelihood", "bias_weighted_crps", "density", "smoothed_exp")


def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


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
    density_curve_spec = payload["density_curve_spec"]
    matmul_precision = payload["continuous_spec"]["matmul_precision"]

    rows = []
    for condition, result in results.items():
        empirical = result["empirical_curves"]
        operator = np.asarray(empirical["feature_operator"])
        bandwidth = float(empirical["density_bandwidth"])
        angle_scale = float(result.get("angle_scale_to_model", 1.0))
        for method in methods:
            key = f"{method}_fitted_params"
            if key not in result:
                continue
            parameters = np.asarray(result[key], dtype=float)
            with jax.default_matmul_precision(matmul_precision):
                curves = mixture_plot_curves(
                    predictor, parameters[None, :3], feat_grid,
                    sd_motor_by_row=[parameters[3]],
                    emp_density_weights_sd=density_curve_spec["emp_density_weights_sd"],
                    density_smoothing_sigma=density_curve_spec["density_smoothing_sigma"],
                    feature_operators=operator[None, :, :],
                    density_bandwidths=[bandwidth])
            for index, x_model in enumerate(feat_grid):
                rows.append({
                    "condition": condition, "optimizer": method,
                    "x_model_deg": float(x_model),
                    "x_deg": float(x_model / angle_scale),
                    "bias_deg": float(curves["bias"][0, index] / angle_scale),
                    "density_asymmetry": float(curves["asymmetry"][0, index]),
                    "sd_deg": float(curves["sd"][0, index] / angle_scale),
                    "empirical_bias_deg": float(
                        np.asarray(empirical["target_bias_curve"])[index] / angle_scale),
                    "empirical_density_asymmetry": float(
                        np.asarray(empirical["matched_density_target"])[index]),
                    "sd_feat1": float(parameters[0]), "sd_feat2": float(parameters[1]),
                    "sd_spat": float(parameters[2]), "sd_motor": float(parameters[3]),
                    "density_bandwidth": bandwidth,
                    **identity,
                })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no selected fitted methods found")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "wnm_fitted_curves.csv", index=False)
    (output_dir / "manifest.json").write_text(json.dumps({
        "source_results": str(results_dir), "checkpoint": str(checkpoint),
        "run_fingerprint_digest": sidecar["digest"], "methods": list(methods),
        **identity,
        "prediction": "direct analytic WNM; no reconstructed NN surface",
        "density_curve": "pooled-SJ KDE plus observed-design feature operator",
        "bias_curve": "observed-design pooled complex first moment",
    }, indent=2) + "\n")

    for condition, condition_frame in frame.groupby("condition", sort=False):
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
