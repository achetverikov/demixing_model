#!/usr/bin/env python3
"""Classify Phase A trajectories from simultaneous all-metric gates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def classify(group: pd.DataFrame, pass_column: str = "all_coherent_pass") -> dict:
    passing = group[group[pass_column]]
    if ((passing.family == "wrapped_mixture") & (passing["size"] == 12)).any():
        label = "K=12 sufficient"
    elif ((passing.family == "wrapped_mixture") & (passing["size"] < 12)).any():
        label = "optimizer-unresolved (smaller K passes, K12 fails)"
    elif ((passing.family == "wrapped_mixture") & (passing["size"] > 12)).any():
        label = "larger mixture required"
    elif (passing.family == "maxent_fourier").any():
        label = "Fourier preferred"
    elif (passing.family == "periodic_spline").any():
        label = "periodic spline preferred"
    else:
        label = "unresolved"
    criterion = "pointwise" if pass_column == "all_pointwise_pass" else "coherent"
    binding_column = f"binding_{criterion}_metrics"
    if binding_column not in group:
        binding_column = "binding_coherent_metrics"
    ranked = passing if len(passing) else group
    parameter_count = ranked.apply(lambda row: {
        "wrapped_mixture": 3 * row["size"] - 1,
        "maxent_fourier": 2 * row["size"],
        "periodic_spline": row["size"] - 1}[row.family], axis=1)
    best = ranked.assign(parameter_count=parameter_count,
        n_binding=ranked[binding_column].map(
        lambda value: 0 if not value else len(value.split(";")))).sort_values(
            ["n_binding", "parameter_count", "wasserstein_upper_deg",
             "excess_nll_coherent_upper"]).iloc[0]
    return {"classification": label, "best_screen_family": best.family,
            "best_screen_size": int(best["size"]),
            "binding_metrics": best[binding_column]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--criterion", choices=("coherent", "pointwise"),
                        default="coherent")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.summary).fillna({
        "binding_coherent_metrics": "", "binding_pointwise_metrics": ""})
    rows = []
    pass_column = f"all_{args.criterion}_pass"
    for (trajectory, component), group in frame.groupby(["trajectory", "component"]):
        rows.append({"trajectory": trajectory, "component": component,
                     "criterion": args.criterion,
                     **classify(group, pass_column)})
    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
