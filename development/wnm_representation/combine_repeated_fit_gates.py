#!/usr/bin/env python3
"""Combine independent-fit Phase A intervals without pooling uncertainty sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def _core_interval(result: dict) -> tuple[float, float, float]:
    if "coherent_abs_estimate" in result:
        estimate = float(result["coherent_abs_estimate"])
        return estimate, 0.0, float(result["max_upper"])
    estimate = float(result["coherent_estimate"])
    radius = float(result["simultaneous_radius"])
    return estimate, estimate - radius, estimate + radius


def _read_case(case: Path) -> dict[tuple[str, int], dict]:
    core = json.loads((case / "core_equivalence.json").read_text())
    distribution = json.loads((case / "distribution_gate.json").read_text())
    output = {(item["family"], int(item["size"])): {"core": item["metrics"]}
              for item in core["candidates"]}
    for item in distribution["candidates"]:
        key = item["family"], int(item["size"])
        if key in output:
            output[key]["distribution"] = item
    return output


def combine(primary: Path, repeat: Path) -> pd.DataFrame:
    rows = []
    for first_case in sorted(primary.glob("*/component_*")):
        relative = first_case.relative_to(primary)
        second_case = repeat / relative
        if not (second_case / "core_equivalence.json").exists():
            continue
        first, second = _read_case(first_case), _read_case(second_case)
        trajectory, component = relative.parts[0], int(relative.parts[1].rsplit("_", 1)[1])
        for family, size in sorted(first.keys() & second.keys()):
            candidate_pass = True
            for metric in first[family, size]["core"].keys() & second[family, size]["core"].keys():
                estimate_a, lower_a, upper_a = _core_interval(first[family, size]["core"][metric])
                estimate_b, lower_b, upper_b = _core_interval(second[family, size]["core"][metric])
                margin = MARGINS[metric]
                lower, upper = min(lower_a, lower_b), max(upper_a, upper_b)
                passed = lower > -margin and upper < margin
                candidate_pass &= passed
                rows.append({"trajectory": trajectory, "component": component,
                    "family": family, "size": size, "metric": metric,
                    "fit0_estimate": estimate_a, "fit1_estimate": estimate_b,
                    "combined_lower": lower, "combined_upper": upper,
                    "margin": margin, "combined_pass": passed})
            for metric, field in (("excess_nll", "excess_nll"),
                                  ("circular_wasserstein_deg", "circular_wasserstein_deg")):
                a = first[family, size]["distribution"][field]
                b = second[family, size]["distribution"][field]
                estimate_a, estimate_b = float(a["estimate"][0]), float(b["estimate"][0])
                upper = max(float(a["simultaneous_upper"][0]),
                            float(b["simultaneous_upper"][0]))
                passed = upper < MARGINS[metric]
                candidate_pass &= passed
                rows.append({"trajectory": trajectory, "component": component,
                    "family": family, "size": size, "metric": metric,
                    "fit0_estimate": estimate_a, "fit1_estimate": estimate_b,
                    "combined_lower": min(estimate_a, estimate_b),
                    "combined_upper": upper, "margin": MARGINS[metric],
                    "combined_pass": passed})
            for row in rows[-(len(first[family, size]["core"]) + 2):]:
                row["candidate_all_metrics_pass"] = candidate_pass
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("repeat", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = combine(args.primary, args.repeat)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = frame.groupby(["trajectory", "component", "family", "size"])[
        "candidate_all_metrics_pass"].first()
    print(summary.to_string())


if __name__ == "__main__":
    main()
