#!/usr/bin/env python3
"""Task 2.0 local-family trajectory and size-tradeoff figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from development.wnm_representation import equivalence


METRICS = (("mean_bias", "Circular mean bias, deg", True),
           ("response_sd", "Response SD, deg", False),
           ("density_asymmetry", "Density asymmetry", False))


def _reference_radius(equivalence_path: Path | None, metric: str) -> float | None:
    if equivalence_path is None:
        return None
    candidates = json.loads(equivalence_path.read_text())["candidates"]
    labels = ([metric] if metric != "mean_bias" else
              ["mean_bias_r_ge_0.8", "mean_bias_r_0.5_0.8"])
    radii = [candidate["metrics"][label]["simultaneous_radius"]
             for candidate in candidates for label in labels
             if label in candidate["metrics"]]
    return max(radii) if radii else None


def plot_best_curves(frame: pd.DataFrame, out: Path,
                     equivalence_path: Path | None = None) -> None:
    """Raw, largest candidate per family, and residual curves for core moments."""
    selected = frame.loc[frame.groupby("family")["size"].transform("max") == frame["size"]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col",
                             gridspec_kw={"height_ratios": [2, 1]})
    first = selected.sort_values("feat_diff").iloc[:selected.feat_diff.nunique()]
    for column, (metric, label, circular) in enumerate(METRICS):
        axes[0, column].plot(first.feat_diff, first[f"raw_{metric}"], color="black",
                             label="raw held-out", lw=2)
        radius = _reference_radius(equivalence_path, metric)
        if radius is not None:
            raw = first[f"raw_{metric}"].to_numpy()
            axes[0, column].fill_between(first.feat_diff, raw - radius, raw + radius,
                                         color="black", alpha=0.12,
                                         label="simultaneous MC band")
        for (family, size), group in selected.groupby(["family", "size"]):
            group = group.sort_values("feat_diff")
            name = f"{family} {size:g}"
            axes[0, column].plot(group.feat_diff, group[f"pred_{metric}"], label=name)
            error = group[f"pred_{metric}"].to_numpy() - group[f"raw_{metric}"].to_numpy()
            if circular:
                error = equivalence.wrap_deg(error)
            axes[1, column].plot(group.feat_diff, error, label=name)
        axes[0, column].set_ylabel(label)
        axes[1, column].set(xlabel="Feature difference, deg", ylabel="pred - raw")
        axes[1, column].axhline(0, color="0.4", lw=0.8)
        margin = {"mean_bias": 0.25, "response_sd": 0.25,
                  "density_asymmetry": 0.005}[metric]
        axes[1, column].axhspan(-margin, margin, color="tab:green", alpha=0.08)
        for axis in axes[:, column]:
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    resultant_axis = axes[0, 0].twinx()
    resultant_axis.plot(first.feat_diff, first.raw_resultant, color="0.5",
                        ls=":", lw=1, label="raw resultant")
    resultant_axis.set_ylabel("Resultant length", color="0.4")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_size_tradeoff(summary: pd.DataFrame, out: Path) -> None:
    metrics = ("density_asymmetry", "mean_bias", "response_sd", "excess_nll",
               "circular_wasserstein_deg")
    fig, axes = plt.subplots(1, len(metrics), figsize=(19, 3.8))
    for axis, metric in zip(axes, metrics):
        subset = summary[summary.metric == metric]
        for family, group in subset.groupby("family"):
            group = group.sort_values("size")
            axis.plot(group["size"], group.max_abs_error, marker="o", label=family)
        axis.set(xlabel="representation size", ylabel="max error", title=metric)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--equivalence", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.metrics], ignore_index=True)
    summary = pd.read_csv(args.summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_best_curves(frame, args.out_dir / "local_core_trajectories.png",
                     args.equivalence)
    plot_size_tradeoff(summary, args.out_dir / "local_error_vs_size.png")


if __name__ == "__main__":
    main()
