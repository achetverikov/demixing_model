#!/usr/bin/env python3
"""Covariance-aware equivalence intervals for fitted local core metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from development.wnm_representation import equivalence
from development.wnm_representation import local_fit_common as common


def _result_dict(result: equivalence.EquivalenceResult) -> dict:
    return {"pointwise_pass": result.pointwise_pass,
            "coherent_estimate": result.coherent_estimate,
            "coherent_lower": result.coherent_lower,
            "coherent_upper": result.coherent_upper,
            "coherent_pass": result.coherent_pass,
            "simultaneous_radius": result.simultaneous_radius,
            "max_abs_estimate": float(np.max(np.abs(result.estimate))),
            "max_interval_excursion": float(np.max(np.maximum(
                np.abs(result.lower), np.abs(result.upper))))}


def _complex_result(error_draws: np.ndarray, point_margin: float = 0.005,
                    coherent_margin: float = 0.002, alpha: float = 0.05) -> dict:
    """Simultaneous radial bounds for complex-first-moment errors."""
    draws = np.asarray(error_draws)
    estimate = draws.mean(axis=0)
    radius = float(np.quantile(
        np.max(np.abs(draws - estimate), axis=1), 1.0 - alpha))
    coherent_draws = draws.mean(axis=1)
    coherent_estimate = coherent_draws.mean()
    coherent_radius = float(np.quantile(
        np.abs(coherent_draws - coherent_estimate), 1.0 - alpha))
    point_upper = np.abs(estimate) + radius
    coherent_upper = float(abs(coherent_estimate) + coherent_radius)
    return {"pointwise_pass": bool(np.all(point_upper < point_margin)),
            "max_abs_estimate": float(np.max(np.abs(estimate))),
            "max_upper": float(np.max(point_upper)),
            "simultaneous_radius": radius,
            "coherent_abs_estimate": float(abs(coherent_estimate)),
            "coherent_upper": coherent_upper,
            "coherent_pass": coherent_upper < coherent_margin}


def bootstrap_candidate(group: pd.DataFrame, reference_draws: dict[str, np.ndarray]) -> dict:
    output = {}
    specs = (("density_asymmetry", np.ones(len(group), dtype=bool), 0.005, 0.002),
             ("response_sd", np.ones(len(group), dtype=bool), 0.25, 0.10),
             ("mean_bias_r_ge_0.8", group.raw_resultant.to_numpy() >= 0.8, 0.25, 0.10),
             ("mean_bias_r_0.5_0.8",
              group.raw_resultant.between(0.5, 0.8, inclusive="left").to_numpy(),
              0.50, 0.25))
    for label, mask, point_margin, coherent_margin in specs:
        if not mask.any():
            continue
        metric = "mean_bias" if label.startswith("mean_bias") else label
        draws = group[f"pred_{metric}"].to_numpy()[None, :] - reference_draws[metric]
        if metric == "mean_bias":
            draws = equivalence.wrap_deg(draws)
        draws = draws[:, mask]
        result = equivalence.equivalence_from_bootstrap(
            draws, point_margin, coherent_margin)
        output[label] = _result_dict(result)
    low_r = group.raw_resultant.to_numpy() < 0.5
    if low_r.any():
        pred = (group.pred_resultant.to_numpy()
                * np.exp(1j * np.radians(group.pred_mean_bias.to_numpy())))
        reference = (reference_draws["resultant"]
                     * np.exp(1j * np.radians(reference_draws["mean_bias"])))
        output["complex_first_moment_r_lt_0.5"] = _complex_result(
            (pred[None, :] - reference)[:, low_r])
    return output


def split_reference_summary(train: np.ndarray, test: np.ndarray) -> dict:
    """Report fitting-half minus evaluation-half empirical core differences."""
    a, b = common.empirical_core_metrics(train), common.empirical_core_metrics(test)
    output = {}
    for metric in ("density_asymmetry", "mean_bias", "response_sd"):
        output[metric] = equivalence.curve_summary(
            a[metric], b[metric], circular=metric == "mean_bias")
    za = a["resultant"] * np.exp(1j * np.radians(a["mean_bias"]))
    zb = b["resultant"] * np.exp(1j * np.radians(b["mean_bias"]))
    output["complex_first_moment"] = {
        "mean_abs_difference": float(np.mean(np.abs(za - zb))),
        "max_abs_difference": float(np.max(np.abs(za - zb))),
        "abs_trajectory_mean_difference": float(abs(np.mean(za - zb))),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--sd-feat1", type=float, required=True)
    parser.add_argument("--sd-feat2", type=float, required=True)
    parser.add_argument("--sd-ident", type=float, required=True)
    parser.add_argument("--component", type=int, choices=(1, 2), required=True)
    parser.add_argument("--candidate-row", type=int,
                        help="evaluate one saved trajectory row against a flagship point")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--independent-indices", action="store_true",
                        help="Use only when evaluation rows used independent streams")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    blob = np.load(args.reference, allow_pickle=True)
    _, samples = common.select_trajectory(
        blob["design"], blob["bias"], args.sd_feat1, args.sd_feat2,
        args.sd_ident, args.component)
    train, test = common.split_outcomes(samples, seed=args.split_seed)
    bootstrap = (equivalence.bootstrap_circular_statistics_independent
                 if args.independent_indices
                 else equivalence.bootstrap_circular_statistics_shared)
    reference_draws = bootstrap(test, args.bootstrap, args.split_seed)
    frame = pd.concat([pd.read_csv(path) for path in args.metrics], ignore_index=True)
    results = []
    for (family, size), group in frame.groupby(["family", "size"]):
        group = group.sort_values("feat_diff")
        if args.candidate_row is not None:
            group = group.iloc[[args.candidate_row]].copy()
            group["raw_resultant"] = common.empirical_core_metrics(test)["resultant"]
        results.append({"family": family, "size": int(size),
                        "metrics": bootstrap_candidate(
                            group, reference_draws)})
    output = {"n_bootstrap": args.bootstrap,
              "shared_simulation_indices": not args.independent_indices,
              "bootstrap_resampling_unit": ("iid outcome blocks (2000 per row)"
                  if test.shape[1] > 20_000 and args.independent_indices
                  else "simulation index"),
              "candidate_row": args.candidate_row,
              "fit_evaluation_split_difference": split_reference_summary(train, test),
              "candidates": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    if not args.quiet:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
