#!/usr/bin/env python3
"""Generate the three purpose-built figures embedded in the main README."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

plt.rcParams["svg.hashsalt"] = "demixing-readme"


BLUE = "#2166ac"
LIGHT_BLUE = "#d9ebf7"
ORANGE = "#c75b12"
LIGHT_ORANGE = "#fbe3cf"
GREEN = "#14866d"
INK = "#18212b"
GRID = "#d8dee6"


def _box(ax, xy, width, height, text, facecolor, edgecolor, fontsize=11):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.7, facecolor=facecolor, edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, color=INK)


def _arrow(ax, start, end, color=INK):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.8, color=color, shrinkA=4, shrinkB=4,
    ))


def plot_workflow_overview(output: Path):
    fig, ax = plt.subplots(figsize=(13.2, 5.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "Two ways to use the Demixing Model",
            ha="center", va="center", fontsize=21, fontweight="bold", color=INK)
    ax.text(0.245, 0.82, "ESTIMATE PARAMETERS FROM DATA", ha="center", fontsize=12,
            fontweight="bold", color=BLUE)
    ax.text(0.755, 0.82, "GENERATE THEORETICAL PREDICTIONS", ha="center", fontsize=12,
            fontweight="bold", color=ORANGE)

    _box(ax, (0.035, 0.49), 0.18, 0.19,
         "Experimental data\nStimulus difference +\nsigned response bias", LIGHT_BLUE, BLUE)
    _box(ax, (0.285, 0.49), 0.18, 0.19,
         "Estimate parameters\nFind noise values that\nbest describe the data", "#eef5fb", BLUE)
    _box(ax, (0.535, 0.49), 0.18, 0.19,
         "Calculate predictions\nBias + response variability\nacross dissimilarity", "#fff5ec", ORANGE)
    _box(ax, (0.785, 0.49), 0.18, 0.19,
         "Hypothesized noise levels\nTarget + non-target item\nIdentifiability (+ motor)", LIGHT_ORANGE, ORANGE)

    _box(ax, (0.405, 0.18), 0.19, 0.17,
         "Trained Demixing Model\nIncluded with the repository", "#dff3ed", GREEN, fontsize=12)
    _box(ax, (0.055, 0.14), 0.25, 0.15,
         "Estimated noise parameters\nModel–data comparison", "white", BLUE)
    _box(ax, (0.695, 0.14), 0.25, 0.15,
         "Predicted bias and variability\nfor the chosen assumptions", "white", ORANGE)

    _arrow(ax, (0.215, 0.585), (0.285, 0.585), BLUE)
    _arrow(ax, (0.375, 0.49), (0.445, 0.35), BLUE)
    _arrow(ax, (0.405, 0.265), (0.305, 0.215), BLUE)
    _arrow(ax, (0.785, 0.585), (0.715, 0.585), ORANGE)
    _arrow(ax, (0.625, 0.49), (0.555, 0.35), ORANGE)
    _arrow(ax, (0.595, 0.265), (0.695, 0.215), ORANGE)

    ax.text(0.5, 0.055,
            "Both workflows are ready to use — no simulation files need to be downloaded",
            ha="center", va="center", fontsize=10.5, color="#4b5563")
    fig.savefig(output, bbox_inches="tight", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def plot_fitting_example(trials_path: Path, curves_path: Path, objective: str, output: Path):
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
    plot_workflow_overview(args.output_dir / "workflow_overview.svg")
    plot_fitting_example(args.trials, args.fit_curves, args.fit_objective,
                         args.output_dir / "data_fitting_example.png")
    plot_prediction_example(args.predictions,
                            args.output_dir / "prediction_generation_example.png")
    print(f"README figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
