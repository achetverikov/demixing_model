#!/usr/bin/env python3
"""Write the frozen-margin uncertainty and power table for Task 2.0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "density_asymmetry": (0.002, 0.005, 0.00316),
    "mean_bias_r_ge_0.8_deg": (0.10, 0.25, 0.035),
    "mean_bias_r_0.5_0.8_deg": (0.25, 0.50, 0.120),
    "response_sd_deg": (0.10, 0.25, 0.080),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("power_plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    planned = {item["metric"]: item for item in
               json.loads(args.power_plan.read_text())["metrics"]}
    rows = []
    for tier, n, points in (("selection_evaluation", 50_000, 90),
                            ("flagship_evaluation", 1_000_000, 1)):
        for metric, (coherent_margin, pointwise_margin, se_100k) in METRICS.items():
            per_point_se = se_100k * np.sqrt(100_000 / n)
            plan = planned[metric]
            rows.append({"tier": tier, "metric": metric, "n": n,
                "coherent_margin": coherent_margin,
                "pointwise_margin": pointwise_margin,
                "planned_pointwise_n_50pct_headroom":
                    plan["pointwise_n_with_50pct_error_headroom"],
                "planned_coherent_n_50pct_headroom":
                    plan["coherent_n_with_50pct_error_headroom"],
                "projected_per_point_se": per_point_se,
                "projected_independent_trajectory_se": per_point_se / np.sqrt(points),
                "interval_method": ("whole-trajectory/simulation-index bootstrap"
                                    if points > 1 else "independent outcome-block bootstrap"),
                "interpretation": ("selection precision screen; not pointwise powered"
                    if tier.startswith("selection") else
                    ("planned 50% pointwise-headroom N reached" if
                     n >= plan["pointwise_n_with_50pct_error_headroom"] else
                     "below planned 50% headroom N; use achieved interval"))})
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
