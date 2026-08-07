#!/usr/bin/env python3
"""Generate the two data-derived figures embedded in the main README."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["svg.hashsalt"] = "demixing-readme"


BLUE = "#2166ac"
INK = "#18212b"
GRID = "#d8dee6"


def prepare_fitting_data(trials_path: Path, curves_path: Path, objective: str):
    """Return binned behavioral data and mean fitted curves for plotting."""
    trials = pd.read_csv(trials_path)
    if "is_combined" in trials:
        trials = trials[~trials["is_combined"].astype(bool)]
    if "is_outlier" in trials:
        trials = trials[trials["is_outlier"] != 1]

    edges = np.arange(0, 95, 5)
    trials = trials.assign(bin=pd.cut(trials["abs_td_dist"], edges, include_lowest=True))
    subject_bins = trials.groupby(["subject", "bin"], observed=True).agg(
        x=("abs_td_dist", "mean"),
        bias=("bias_to_distr_corr", "mean"),
    ).reset_index()
    empirical = subject_bins.groupby("bin", observed=True).agg(
        x=("x", "mean"),
        bias=("bias", "mean"),
        sem=("bias", "sem"),
    )

    curves = pd.read_csv(curves_path)
    curves = curves[curves["optimizer"] == objective]
    if curves.empty:
        raise ValueError(f"No {objective!r} curves found in {curves_path}")
    fitted = curves.groupby("feat_diff", as_index=False).agg(
        bias=("mu_bias", "mean"),
        sem=("mu_bias", "sem"),
    )
    return empirical, fitted


def plot_fitting_example(trials_path: Path, curves_path: Path, objective: str, output: Path):
    empirical, fitted = prepare_fitting_data(trials_path, curves_path, objective)

    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    ax.axhline(0, color="#7b8794", linewidth=1, zorder=0)
    ax.fill_between(fitted["feat_diff"], fitted["bias"] - fitted["sem"],
                    fitted["bias"] + fitted["sem"], color=BLUE, alpha=0.17, linewidth=0)
    ax.plot(fitted["feat_diff"], fitted["bias"], color=BLUE, linewidth=2.8,
            label="Model prediction (mean-bias fit)")
    ax.errorbar(empirical["x"], empirical["bias"], yerr=empirical["sem"],
                fmt="o", markersize=5.2, color=INK, ecolor="#657180", capsize=2.5,
                label="Behavioral data (5° bins)")
    ax.set(xlim=(0, 90), xlabel="Target–distractor dissimilarity (°)",
           ylabel="Bias toward distractor (°)",
           title="Empirical data and representative model fit")
    ax.grid(color=GRID, linewidth=0.8, alpha=0.75)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _first_list(series):
    for value in series:
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return np.asarray(value, dtype=float)
    raise ValueError("No grid metadata found in prediction output")


def plot_prediction_example(predictions_path: Path, output: Path):
    predictions = pd.read_parquet(predictions_path)
    feat_diff = _first_list(predictions["feat_diff_grid"])

    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    colors = ["#2c7bb6", "#f28e2b", "#b2182b"]
    for (_, row), color in zip(predictions.iterrows(), colors):
        curve = np.asarray(row["mu1_expectation_curve"], dtype=float)
        ax.plot(feat_diff, curve, linewidth=2.7, color=color,
                label=f"Identifiability noise = {row['sd_spat']:g}°")

    ax.axhline(0, color="#7b8794", linewidth=1, zorder=0)
    ax.set(xlim=(feat_diff.min(), feat_diff.max()),
           xlabel="Item dissimilarity (model degrees)",
           ylabel="Predicted bias toward the other item (°)",
           title="How does identifiability noise change the predicted bias?")
    ax.text(0.02, 0.96, "Target noise = 10°, non-target noise = 30°, motor noise = 0°",
            transform=ax.transAxes, va="top", fontsize=10, color="#4b5563",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2})
    ax.grid(color=GRID, linewidth=0.8, alpha=0.75)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path,
                        default=Path("example_data/fischer_whitney_prepared.csv"))
    parser.add_argument("--fit-curves", type=Path,
                        default=Path("results/fischer_whitney_20samples_circular/csv_exports/fitted_curves.csv"))
    parser.add_argument("--fit-objective", default="expectation")
    parser.add_argument("--predictions", type=Path,
                        default=Path("results/prediction_example.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/images"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_fitting_example(args.trials, args.fit_curves, args.fit_objective,
                         args.output_dir / "data_fitting_example.png")
    plot_prediction_example(args.predictions,
                            args.output_dir / "prediction_generation_example.png")
    print(f"README figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
