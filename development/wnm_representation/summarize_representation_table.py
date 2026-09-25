#!/usr/bin/env python3
"""Build the wide Phase A representation table required by PLAN section 12."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from continuous_density.compare_local_representations import comparison_table


def parameter_count(family: str, size: int) -> int:
    return {"wrapped_mixture": 3 * size - 1,
            "maxent_fourier": 2 * size,
            "periodic_spline": size - 1}[family]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_root", type=Path)
    parser.add_argument("gate_summary", type=Path)
    parser.add_argument("speed", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case in sorted(args.fit_root.glob("*/component_*")):
        paths = sorted(case.glob("*/local_*_metrics.csv"))
        if not paths:
            continue
        metrics = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        metrics = metrics.drop_duplicates(["family", "size", "feat_diff"])
        comparison = comparison_table(metrics)
        for (family, size), group in metrics.groupby(["family", "size"]):
            row = {"trajectory": case.parent.name,
                   "component": int(case.name.rsplit("_", 1)[1]),
                   "family": family, "size": int(size),
                   "density_parameters": parameter_count(family, int(size)),
                   "mean_heldout_nll": float(group.heldout_nll.mean())}
            summary = comparison[(comparison.family == family)
                                 & (comparison["size"] == size)]
            for result in summary.itertuples():
                prefix = result.metric
                row[f"{prefix}_max_abs_error"] = result.max_abs_error
                row[f"{prefix}_coherent_error"] = result.mean_signed_error
                row[f"{prefix}_amplitude_slope"] = result.amplitude_slope
                row[f"{prefix}_longest_sign_run"] = result.longest_sign_run
            rows.append(row)
    output = pd.DataFrame(rows)
    gates = pd.read_csv(args.gate_summary)
    output = output.merge(gates, on=["trajectory", "component", "family", "size"],
                          how="left", suffixes=("", "_gate"))
    speed = pd.read_csv(args.speed).drop(columns=["label"])
    output = output.merge(speed, on=["family", "size", "density_parameters"], how="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"Wrote {len(output)} representation/case rows")


if __name__ == "__main__":
    main()
