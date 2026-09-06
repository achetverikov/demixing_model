#!/usr/bin/env python3
"""Measure reference and trajectory uncertainty from raw simulation corpora.

This script reports precision; it never turns Monte Carlo standard errors into
acceptance thresholds.  Use the fixed margins in the Task 2.0 tolerance memo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from continuous_density import equivalence
from continuous_density import evaluate


def split_half_differences(samples: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """Random split-half differences for the raw circular statistics."""
    x = np.asarray(samples)
    if x.ndim != 2 or x.shape[1] < 4:
        raise ValueError("samples must be (points, outcomes) with at least four outcomes")
    rng = np.random.default_rng(seed)
    order = rng.permuted(np.broadcast_to(np.arange(x.shape[1]), x.shape), axis=1)
    shuffled = np.take_along_axis(x, order, axis=1)
    half = x.shape[1] // 2
    a, b = shuffled[:, :half], shuffled[:, half:2 * half]
    ma, mb = evaluate.empirical_moments(a), evaluate.empirical_moments(b)
    return {
        "density_asymmetry": (evaluate.empirical_density_asymmetry(a)
                               - evaluate.empirical_density_asymmetry(b)),
        "mean_bias_deg": equivalence.wrap_deg(ma["mean_bias"] - mb["mean_bias"]),
        "response_sd_deg": ma["circ_sd"] - mb["circ_sd"],
        "resultant": evaluate.empirical_moments(x)["resultant"],
    }


def summarize_split_half(differences: dict[str, np.ndarray]) -> dict:
    """Convert 50k-vs-50k disagreement to per-full-N precision."""
    out = {}
    for name in ("density_asymmetry", "mean_bias_deg", "response_sd_deg"):
        values = np.asarray(differences[name])
        out[name] = {
            "split_half_sd": float(np.nanstd(values)),
            "per_full_n_se": float(np.nanstd(values) / 2.0),
            "max_abs_split_half_difference": float(np.nanmax(np.abs(values))),
        }
    return out


def trajectory_mean_se(error_draws: np.ndarray) -> dict[str, float]:
    """Covariance-aware se and the independence approximation for comparison."""
    draws = np.asarray(error_draws, dtype=np.float64)
    covariance = np.cov(draws, rowvar=False, ddof=1)
    n = draws.shape[1]
    weights = np.full(n, 1.0 / n)
    covariance_se = float(np.sqrt(weights @ covariance @ weights))
    pointwise_se = np.std(draws, axis=0, ddof=1)
    independent_se = float(np.sqrt(np.sum(pointwise_se ** 2)) / n)
    return {"covariance_aware_se": covariance_se,
            "independence_approximation_se": independent_se,
            "inflation_factor": covariance_se / independent_se}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    blob = np.load(args.reference, allow_pickle=True)
    bias = blob["bias"]
    rows = []
    for component in range(bias.shape[2]):
        differences = split_half_differences(bias[:, :, component], args.seed + component)
        summary = summarize_split_half(differences)
        summary["component"] = component + 1
        rows.append(summary)
    output = {"reference": str(args.reference), "n_rows": int(bias.shape[0]),
              "n_outcomes": int(bias.shape[1]), "components": rows,
              "note": "Standard errors describe precision; fixed margins define gates."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

