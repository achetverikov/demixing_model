#!/usr/bin/env python3
"""Compare local density families on all Task 2.0 metrics as trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_density import equivalence


METRICS = {
    "mean_bias": ("pred_mean_bias", "raw_mean_bias", True),
    "response_sd": ("pred_response_sd", "raw_response_sd", False),
    "density_asymmetry": ("pred_density_asymmetry", "raw_density_asymmetry", False),
}


def comparison_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One curve-level row per family/size/metric, never pooled over dissimilarity."""
    data = frame.copy()
    data["best_point_nll"] = data.groupby("feat_diff").heldout_nll.transform("min")
    data["excess_nll"] = data.heldout_nll - data.best_point_nll
    rows = []
    for (family, size), group in data.groupby(["family", "size"], sort=False):
        group = group.sort_values("feat_diff")
        for metric, (pred, raw, circular) in METRICS.items():
            summary = equivalence.curve_summary(group[pred], group[raw], circular)
            rows.append({"family": family, "size": size, "metric": metric, **summary})
        nll_summary = equivalence.curve_summary(group.excess_nll, np.zeros(len(group)))
        rows.append({"family": family, "size": size, "metric": "excess_nll",
                     **nll_summary})
        w1_summary = equivalence.curve_summary(
            group.circular_wasserstein_deg, np.zeros(len(group)))
        rows.append({"family": family, "size": size,
                     "metric": "circular_wasserstein_deg", **w1_summary})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.metrics], ignore_index=True)
    table = comparison_table(frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print("Point-estimate screen only: equivalence requires bootstrap intervals.")


if __name__ == "__main__":
    main()

