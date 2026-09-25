#!/usr/bin/env python3
"""Flatten local benchmark gates into candidate and trajectory tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_case(case_dir: Path) -> list[dict]:
    core_path, dist_path = case_dir / "core_equivalence.json", case_dir / "distribution_gate.json"
    if not core_path.exists() or not dist_path.exists():
        return []
    core = json.loads(core_path.read_text())
    distribution = json.loads(dist_path.read_text())
    dist_by_label = {item["label"]: item for item in distribution["candidates"]}
    rows = []
    for candidate in core["candidates"]:
        family, size = candidate["family"], int(candidate["size"])
        prefix = {"wrapped_mixture": "w", "maxent_fourier": "f",
                  "periodic_spline": "s"}[family]
        label = prefix + str(size)
        dist = dist_by_label[label]
        metrics = candidate["metrics"]
        mean_metrics = [value for key, value in metrics.items() if key.startswith("mean_bias")]
        complex_metric = metrics.get("complex_first_moment_r_lt_0.5")
        core_coherent = [metrics["density_asymmetry"]["coherent_pass"],
                         metrics["response_sd"]["coherent_pass"]]
        core_pointwise = [metrics["density_asymmetry"]["pointwise_pass"],
                          metrics["response_sd"]["pointwise_pass"]]
        core_coherent.extend(value["coherent_pass"] for value in mean_metrics)
        core_pointwise.extend(value["pointwise_pass"] for value in mean_metrics)
        if complex_metric:
            core_coherent.append(complex_metric["coherent_pass"])
            core_pointwise.append(complex_metric["pointwise_pass"])
        binding = []
        binding_pointwise = []
        if not metrics["density_asymmetry"]["coherent_pass"]:
            binding.append("density_asymmetry")
        if not metrics["density_asymmetry"]["pointwise_pass"]:
            binding_pointwise.append("density_asymmetry")
        if not metrics["response_sd"]["coherent_pass"]:
            binding.append("response_sd")
        if not metrics["response_sd"]["pointwise_pass"]:
            binding_pointwise.append("response_sd")
        if not all(value["coherent_pass"] for value in mean_metrics):
            binding.append("mean_bias")
        if not all(value["pointwise_pass"] for value in mean_metrics):
            binding_pointwise.append("mean_bias")
        if complex_metric and not complex_metric["coherent_pass"]:
            binding.append("complex_first_moment")
        if complex_metric and not complex_metric["pointwise_pass"]:
            binding_pointwise.append("complex_first_moment")
        if not dist["nll_coherent_pass"]:
            binding.append("excess_nll")
        if not dist["nll_pointwise_pass"]:
            binding_pointwise.append("excess_nll")
        if not dist["wasserstein_coherent_pass"]:
            binding.append("circular_wasserstein")
        if not dist["wasserstein_pointwise_pass"]:
            binding_pointwise.append("circular_wasserstein")
        rows.append({
            "trajectory": case_dir.parent.name,
            "component": int(case_dir.name.rsplit("_", 1)[1]),
            "family": family, "size": size,
            "asymmetry_coherent_estimate": metrics["density_asymmetry"]["coherent_estimate"],
            "asymmetry_coherent_pass": metrics["density_asymmetry"]["coherent_pass"],
            "asymmetry_pointwise_pass": metrics["density_asymmetry"]["pointwise_pass"],
            "response_sd_coherent_pass": metrics["response_sd"]["coherent_pass"],
            "response_sd_pointwise_pass": metrics["response_sd"]["pointwise_pass"],
            "mean_coherent_pass": all(value["coherent_pass"] for value in mean_metrics),
            "mean_pointwise_pass": all(value["pointwise_pass"] for value in mean_metrics),
            "complex_moment_coherent_pass": (complex_metric or {}).get("coherent_pass", np.nan),
            "complex_moment_pointwise_pass": (complex_metric or {}).get("pointwise_pass", np.nan),
            "excess_nll_coherent_upper": dist["excess_nll"]["coherent_upper"],
            "nll_coherent_pass": dist["nll_coherent_pass"],
            "wasserstein_estimate_deg": dist["circular_wasserstein_deg"]["coherent_estimate"],
            "wasserstein_upper_deg": dist["circular_wasserstein_deg"]["coherent_upper"],
            "wasserstein_coherent_pass": dist["wasserstein_coherent_pass"],
            "all_coherent_pass": (all(core_coherent) and dist["nll_coherent_pass"]
                                  and dist["wasserstein_coherent_pass"]),
            "all_pointwise_pass": (all(core_pointwise) and dist["nll_pointwise_pass"]
                                   and dist["wasserstein_pointwise_pass"]),
            "w1_split_half_floor_deg": distribution["wasserstein_split_half_floor"]["mean_deg"],
            "nll_reference": distribution["selected_nll_reference"],
            "binding_coherent_metrics": ";".join(binding),
            "binding_pointwise_metrics": ";".join(binding_pointwise),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case in sorted(args.root.glob("*/component_*")):
        rows.extend(summarize_case(case))
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame)} candidates across {frame[['trajectory', 'component']].drop_duplicates().shape[0]} cases")


if __name__ == "__main__":
    main()
