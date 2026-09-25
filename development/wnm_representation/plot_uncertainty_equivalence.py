#!/usr/bin/env python3
"""Plot Phase A pointwise intervals and the separately measured uncertainty sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


MARGINS = {
    "density_asymmetry": 0.005,
    "response_sd": 0.25,
    "mean_bias_r_ge_0.8": 0.25,
    "mean_bias_r_0.5_0.8": 0.50,
    "complex_first_moment_r_lt_0.5": 0.005,
    "excess_nll": 0.002,
    "circular_wasserstein_deg": 0.50,
}
FIT_METRICS = {
    "density_asymmetry": "pred_density_asymmetry",
    "response_sd": "pred_response_sd",
    "mean_bias_r_ge_0.8": "pred_mean_bias",
    "mean_bias_r_0.5_0.8": "pred_mean_bias",
    "complex_first_moment_r_lt_0.5": "pred_resultant",
    "excess_nll": "heldout_nll",
    "circular_wasserstein_deg": "circular_wasserstein_deg",
}


def _candidate_rows(case: Path) -> list[dict]:
    core = json.loads((case / "core_equivalence.json").read_text())
    distribution = json.loads((case / "distribution_gate.json").read_text())
    dist = {(item["family"], item["size"]): item
            for item in distribution["candidates"]}
    rows = []
    for item in core["candidates"]:
        other = dist[(item["family"], item["size"])]
        failures, score = 0, 0.0
        for name, metric in item["metrics"].items():
            failures += not metric["pointwise_pass"]
            bound = metric.get("max_interval_excursion", metric.get("max_upper"))
            score += bound / MARGINS[name]
        for name in ("excess_nll", "circular_wasserstein_deg"):
            passed = other["nll_pointwise_pass"] if name == "excess_nll" else other["wasserstein_pointwise_pass"]
            failures += not passed
            bound = (max(other[name]["simultaneous_upper"]) if name == "excess_nll"
                     else max(other[name]["simultaneous_upper"]))
            score += bound / MARGINS[name]
        rows.append({"core": item, "dist": other, "rank": (failures, score, item["size"])})
    return rows


def _fit_variation(frame: pd.DataFrame, trajectory: str, component: int,
                   family: str, size: int, metric: str) -> float:
    if frame.empty:
        return np.nan
    selected = frame[(frame.trajectory == trajectory) & (frame.component == component)
                     & (frame.family == family) & (frame["size"] == size)
                     & (frame.metric == FIT_METRICS[metric])]
    return (float(selected.fitting_half_max_abs_difference.iloc[0])
            if len(selected) else np.nan)


def collect(root: Path, variation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case in sorted(root.glob("*/component_*")):
        if not (case / "core_equivalence.json").exists():
            continue
        choices = _candidate_rows(case)
        repeated_rows = variation[
            (variation.trajectory == case.parent.name)
            & (variation.component == int(case.name.rsplit("_", 1)[1]))]
        repeated = set(zip(repeated_rows.family, repeated_rows["size"].astype(int)))
        if repeated:
            choices = [row for row in choices
                       if (row["core"]["family"], int(row["core"]["size"])) in repeated]
        selected = min(choices, key=lambda row: row["rank"])
        core, dist = selected["core"], selected["dist"]
        trajectory = case.parent.name
        component = int(case.name.rsplit("_", 1)[1])
        family, size = core["family"], int(core["size"])
        for metric, result in core["metrics"].items():
            estimate = result.get("coherent_estimate", result.get("coherent_abs_estimate"))
            radius = result.get("simultaneous_radius", result.get("max_upper", 0) - estimate)
            rows.append({"trajectory": trajectory, "component": component,
                         "family": family, "size": size, "metric": metric,
                         "estimate": estimate, "lower": estimate - radius,
                         "upper": estimate + radius, "margin": MARGINS[metric],
                         "evaluation_radius": radius,
                         "fitting_half_max_difference": _fit_variation(
                             variation, trajectory, component, family, size, metric)})
        for metric, key in (("excess_nll", "excess_nll"),
                            ("circular_wasserstein_deg", "circular_wasserstein_deg")):
            result = dist[key]
            estimate = float(np.asarray(result["estimate"])[0])
            upper = float(np.asarray(result["simultaneous_upper"])[0])
            rows.append({"trajectory": trajectory, "component": component,
                         "family": family, "size": size, "metric": metric,
                         "estimate": estimate, "lower": estimate,
                         "upper": upper, "margin": MARGINS[metric],
                         "evaluation_radius": upper - estimate,
                         "fitting_half_max_difference": _fit_variation(
                             variation, trajectory, component, family, size, metric)})
    return pd.DataFrame(rows)


def plot(frame: pd.DataFrame, out: Path) -> None:
    labels = [f"{r.trajectory.replace('phase_a_', '')} c{r.component}: {r.metric}"
              for r in frame.itertuples()]
    y = np.arange(len(frame))
    estimate = frame.estimate.to_numpy() / frame.margin.to_numpy()
    lower = frame.lower.to_numpy() / frame.margin.to_numpy()
    upper = frame.upper.to_numpy() / frame.margin.to_numpy()
    fitting = frame.fitting_half_max_difference.to_numpy() / frame.margin.to_numpy()
    fig, axis = plt.subplots(figsize=(11, max(5, 0.28 * len(frame))))
    finite = np.isfinite(fitting)
    axis.errorbar(estimate[finite], y[finite], xerr=fitting[finite], fmt="none",
                  ecolor="tab:orange", lw=4, alpha=0.35, label="fitting-half change")
    axis.errorbar(estimate, y, xerr=np.vstack((estimate - lower, upper - estimate)),
                  fmt="o", color="tab:blue", ecolor="tab:blue", capsize=2,
                  label="estimate + evaluation interval")
    axis.axvspan(-1, 1, color="tab:green", alpha=0.10, label="pointwise margin")
    axis.axvline(-1, color="tab:green", lw=1)
    axis.axvline(1, color="tab:green", lw=1)
    axis.axvline(0, color="0.35", lw=0.8)
    axis.set(yticks=y, yticklabels=labels, xlabel="Error / frozen pointwise margin")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--fit-variation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-table", type=Path, required=True)
    args = parser.parse_args()
    variation = (pd.read_csv(args.fit_variation) if args.fit_variation else pd.DataFrame())
    frame = collect(args.root, variation)
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_table, index=False)
    plot(frame, args.out)
    print(f"Wrote {len(frame)} uncertainty rows")


if __name__ == "__main__":
    main()
