#!/usr/bin/env python3
"""Plan simulation N from fixed equivalence margins and measured precision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import norm


def required_n(se_at_reference_n: float, reference_n: int, margin: float,
               n_comparisons: int = 1, expected_error_fraction: float = 0.5,
               alpha: float = 0.05) -> int:
    """N making a simultaneous CI fit in the remaining equivalence margin."""
    if not 0 <= expected_error_fraction < 1:
        raise ValueError("expected_error_fraction must be in [0, 1)")
    available = margin * (1.0 - expected_error_fraction)
    z = norm.ppf(1.0 - alpha / (2.0 * n_comparisons))
    unit_sd = se_at_reference_n * math.sqrt(reference_n)
    return math.ceil((z * unit_sd / available) ** 2)


def planning_table(trajectory_points: int = 90,
                   covariance_inflation: float = 1.0) -> list[dict]:
    measured = {
        "density_asymmetry": (0.00316, 0.005, 0.002),
        "mean_bias_r_ge_0.8_deg": (0.035, 0.25, 0.10),
        "mean_bias_r_0.5_0.8_deg": (0.120, 0.50, 0.25),
        "response_sd_deg": (0.080, 0.25, 0.10),
    }
    rows = []
    for metric, (se, point_margin, coherent_margin) in measured.items():
        point_n = required_n(se, 100_000, point_margin, trajectory_points)
        coherent_se = se * covariance_inflation / math.sqrt(trajectory_points)
        coherent_n = required_n(coherent_se, 100_000, coherent_margin)
        rows.append({"metric": metric, "pointwise_margin": point_margin,
                     "coherent_margin": coherent_margin,
                     "pointwise_n_with_50pct_error_headroom": point_n,
                     "coherent_n_with_50pct_error_headroom": coherent_n,
                     "covariance_inflation": covariance_inflation})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-points", type=int, default=90)
    parser.add_argument("--covariance-inflation", type=float, default=1.0)
    parser.add_argument("--simulations-per-second", type=float)
    parser.add_argument("--flagship-points", type=int, default=5)
    parser.add_argument("--trajectory-count", type=int, default=20)
    parser.add_argument("--trajectory-n", type=int, default=100_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = planning_table(args.trajectory_points, args.covariance_inflation)
    output = {"trajectory_points": args.trajectory_points,
              "expected_error_fraction_of_margin": 0.5,
              "familywise_alpha": 0.05, "metrics": rows}
    if args.simulations_per_second:
        for row in rows:
            row["pointwise_seconds_per_point"] = (
                row["pointwise_n_with_50pct_error_headroom"]
                / args.simulations_per_second)
        output["simulations_per_second"] = args.simulations_per_second
        asymmetry_n = next(row["pointwise_n_with_50pct_error_headroom"]
                           for row in rows if row["metric"] == "density_asymmetry")
        total = (args.trajectory_count * args.trajectory_points * args.trajectory_n
                 + args.flagship_points * max(0, asymmetry_n - args.trajectory_n))
        output["runtime_projection"] = {
            "flagship_points": args.flagship_points,
            "trajectory_count": args.trajectory_count,
            "base_n_per_trajectory_point": args.trajectory_n,
            "total_simulator_outcomes": total,
            "wall_seconds": total / args.simulations_per_second,
            "wall_hours": total / args.simulations_per_second / 3600.0,
            "excludes_fitting_and_analysis": True,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
