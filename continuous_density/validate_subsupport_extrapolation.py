#!/usr/bin/env python3
"""Validate WNM predictions below 0.5 degrees against the direct DM simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jax
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "surface_computation"))

import jax_fit_main  # noqa: E402
from shared import surrogate  # noqa: E402
from shared.prediction import predictor_from_surrogate  # noqa: E402


FEATURE_DIFFERENCES = (0.05, 0.25, 0.49, 0.5)
PARAMETER_PANEL = (
    (5.0, 5.0, 5.0),
    (10.0, 10.0, 20.0),
    (10.0, 60.0, 20.0),
    (60.0, 10.0, 20.0),
    (60.0, 60.0, 60.0),
    (180.0, 30.0, 150.0),
    (200.0, 200.0, 200.0),
)
BIAS_EDGES = np.arange(-181.0, 181.0, 2.0)
ACCEPTANCE_TOLERANCES = {
    "first_moment_abs_error": 0.01,
    "asymmetry_abs_error": 0.015,
    "cell_total_variation": 0.02,
    "stable_mean_bias_abs_error_deg": 0.35,
    "stable_circular_sd_abs_error_deg": 1.0,
}


def circular_summary(samples):
    moment = np.mean(np.exp(1j * np.deg2rad(samples)))
    mean = np.rad2deg(np.angle(moment))
    sd = np.rad2deg(np.sqrt(-2 * np.log(np.clip(abs(moment), 1e-12, 1))))
    asymmetry = np.mean(samples > 0) - np.mean(samples < 0)
    return moment, float(mean), float(sd), float(asymmetry)


def artifact_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(n_samples, n_simulations, simulation_batch, seed):
    loaded = surrogate.load_surrogate(checkpoint_path=surrogate.WNM_DEFAULTS[n_samples])
    predictor = predictor_from_surrogate(loaded)
    rows = []
    key = jax.random.PRNGKey(seed)
    for sd_feat1, sd_feat2, sd_spat in PARAMETER_PANEL:
        for feature_difference in FEATURE_DIFFERENCES:
            key, simulation_key = jax.random.split(key)
            batches = []
            remaining = n_simulations
            while remaining:
                simulation_key, batch_key = jax.random.split(simulation_key)
                count = min(simulation_batch, remaining)
                simulated, _ = jax_fit_main.simulate_dual_component_bias_distribution(
                    batch_key, sd_feat1, sd_feat2, sd_spat, feature_difference, 42.0,
                    n_simulations=count, n_samples=n_samples,
                    return_full_results=False, fix_weights=False, algorithm="EM",
                    diagonal_covariance=True)
                batches.append(np.asarray(simulated[:, 0]))
                remaining -= count
            simulated = np.concatenate(batches)
            params = np.asarray([[sd_feat1, sd_feat2, sd_spat, feature_difference]],
                                dtype=np.float32)
            predicted_mean, predicted_resultant = predictor.mean_and_resultant(
                params, validate=False)
            predicted_sd = predictor.circular_sd(params, validate=False)
            predicted_asymmetry = predictor.signed_arc_asymmetry(
                params, validate=False)
            predicted_cells = np.asarray(
                predictor.cell_probabilities(
                    params, edges=BIAS_EDGES, validate=False))[0]
            observed_cells = np.histogram(simulated, bins=BIAS_EDGES)[0] / len(simulated)
            direct_moment, direct_mean, direct_sd, direct_asymmetry = circular_summary(simulated)
            predicted_moment = complex(
                float(predicted_resultant[0])
                * np.exp(1j * np.deg2rad(float(predicted_mean[0]))))
            mean_error = np.rad2deg(np.angle(np.exp(
                1j * np.deg2rad(float(predicted_mean[0]) - direct_mean))))
            rows.append({
                "n_samples": n_samples,
                "sd_feat1": sd_feat1,
                "sd_feat2": sd_feat2,
                "sd_spat": sd_spat,
                "feature_difference_deg": feature_difference,
                "direct_mean_bias_deg": direct_mean,
                "predicted_mean_bias_deg": float(predicted_mean[0]),
                "mean_bias_error_deg": float(mean_error),
                "direct_circular_sd_deg": direct_sd,
                "predicted_circular_sd_deg": float(predicted_sd[0]),
                "circular_sd_error_deg": float(predicted_sd[0] - direct_sd),
                "direct_asymmetry": direct_asymmetry,
                "predicted_asymmetry": float(predicted_asymmetry[0]),
                "asymmetry_error": float(predicted_asymmetry[0] - direct_asymmetry),
                "cell_total_variation": float(
                    0.5 * np.abs(predicted_cells - observed_cells).sum()),
                "first_moment_abs_error": float(abs(predicted_moment - direct_moment)),
                "direct_resultant": float(abs(direct_moment)),
                "predicted_resultant": float(predicted_resultant[0]),
            })
    return rows, {"path": str(loaded.path), "sha256": artifact_sha256(loaded.path)}


def summarize(rows):
    summary = {}
    for n_samples in sorted({row["n_samples"] for row in rows}):
        sample_rows = [row for row in rows if row["n_samples"] == n_samples]
        by_region = {}
        for name, selected in {
            "subsupport": [row for row in sample_rows
                           if row["feature_difference_deg"] < 0.5],
            "boundary_control": [row for row in sample_rows
                                 if row["feature_difference_deg"] == 0.5],
        }.items():
            stable_direction = [row for row in selected
                                if min(row["direct_resultant"],
                                       row["predicted_resultant"]) >= 0.1]
            by_region[name] = {
                "cases": len(selected),
                "stable_mean_bias_cases": len(stable_direction),
                "stable_mean_bias_mae_deg": float(np.mean(
                    np.abs([row["mean_bias_error_deg"] for row in stable_direction]))),
                "stable_mean_bias_max_abs_deg": float(np.max(
                    np.abs([row["mean_bias_error_deg"] for row in stable_direction]))),
                "first_moment_mae": float(np.mean(
                    [row["first_moment_abs_error"] for row in selected])),
                "first_moment_max_error": float(np.max(
                    [row["first_moment_abs_error"] for row in selected])),
                "circular_sd_mae_deg": float(np.mean(
                    np.abs([row["circular_sd_error_deg"] for row in selected]))),
                "circular_sd_max_abs_deg": float(np.max(
                    np.abs([row["circular_sd_error_deg"] for row in selected]))),
                "asymmetry_mae": float(np.mean(
                    np.abs([row["asymmetry_error"] for row in selected]))),
                "asymmetry_max_abs": float(np.max(
                    np.abs([row["asymmetry_error"] for row in selected]))),
                "cell_total_variation_mean": float(np.mean(
                    [row["cell_total_variation"] for row in selected])),
                "cell_total_variation_max": float(np.max(
                    [row["cell_total_variation"] for row in selected])),
            }
        summary[str(n_samples)] = by_region
    return summary


def acceptance(rows):
    """Require extrapolated cases to stay close to their matched 0.5° control."""
    keys = ("n_samples", "sd_feat1", "sd_feat2", "sd_spat")
    controls = {
        tuple(row[key] for key in keys): row for row in rows
        if row["feature_difference_deg"] == 0.5
    }
    degradation = {name: [] for name in ACCEPTANCE_TOLERANCES}
    for row in rows:
        if row["feature_difference_deg"] >= 0.5:
            continue
        control = controls[tuple(row[key] for key in keys)]
        degradation["first_moment_abs_error"].append(
            row["first_moment_abs_error"] - control["first_moment_abs_error"])
        degradation["asymmetry_abs_error"].append(
            abs(row["asymmetry_error"]) - abs(control["asymmetry_error"]))
        degradation["cell_total_variation"].append(
            row["cell_total_variation"] - control["cell_total_variation"])
        if min(row["direct_resultant"], row["predicted_resultant"],
               control["direct_resultant"], control["predicted_resultant"]) >= 0.1:
            degradation["stable_mean_bias_abs_error_deg"].append(
                abs(row["mean_bias_error_deg"]) - abs(control["mean_bias_error_deg"]))
            degradation["stable_circular_sd_abs_error_deg"].append(
                abs(row["circular_sd_error_deg"])
                - abs(control["circular_sd_error_deg"]))
    checks = {
        name: {
            "maximum_extrapolation_minus_boundary_error": float(max(values)),
            "tolerance": tolerance,
            "passed": bool(max(values) <= tolerance),
        }
        for name, tolerance in ACCEPTANCE_TOLERANCES.items()
        for values in [degradation[name]]
    }
    return {"passed": all(check["passed"] for check in checks.values()),
            "rule": ("Each sub-support case is paired with the same parameter case at "
                     "0.5 degrees; low-resultant (<0.1) mean direction and circular-SD "
                     "comparisons are omitted as unstable."),
            "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-simulations", type=int, default=10000)
    parser.add_argument("--simulation-batch", type=int, default=1000)
    parser.add_argument("--sample-counts", type=int, nargs="+", default=[20, 100])
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, artifacts = [], {}
    for n_samples in args.sample_counts:
        result, artifact = evaluate(
            n_samples, args.n_simulations, args.simulation_batch, args.seed + n_samples)
        rows.extend(result)
        artifacts[str(n_samples)] = artifact
    document = {
        "validation": "wnm_subsupport_extrapolation_v1",
        "n_simulations_per_case": args.n_simulations,
        "simulation_batch": args.simulation_batch,
        "seed": args.seed,
        "feature_differences_deg": FEATURE_DIFFERENCES,
        "parameter_panel": PARAMETER_PANEL,
        "artifacts": artifacts,
        "source_sha256": {
            "validator": artifact_sha256(Path(__file__)),
            "direct_simulator": artifact_sha256(REPO / "surface_computation/jax_fit_main.py"),
            "em_implementation": artifact_sha256(
                REPO / "surface_computation/jax_fit_functions.py"),
        },
        "summary": summarize(rows),
        "acceptance": acceptance(rows),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"summary": document["summary"],
                      "acceptance": document["acceptance"]}, indent=2))
    if not document["acceptance"]["passed"]:
        raise SystemExit("sub-support extrapolation validation failed")


if __name__ == "__main__":
    main()
