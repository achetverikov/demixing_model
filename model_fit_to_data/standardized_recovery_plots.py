"""Plots for artifacts written by :mod:`standardized_recovery`.

Plotting consumes the standardized long tables; it never reevaluates a model.
Consequently a future optimizer participates by writing the same contract, with
no objective-specific plotting branch.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def plot_artifact(artifact: Path) -> list[Path]:
    """Render parameter and curve comparisons, returning the written paths."""
    artifact = Path(artifact)
    parameters = pd.read_csv(artifact / "recovery_parameters.csv")
    curves = pd.read_csv(artifact / "recovery_curves.csv")
    output = artifact / "plots"
    output.mkdir(exist_ok=True)
    written = []

    grid = sns.relplot(
        data=parameters, x="true", y="recovered", hue="objective",
        col="parameter", col_wrap=3, kind="scatter", facet_kws={"sharex": False,
                                                                  "sharey": False})
    for axis in grid.axes.flat:
        low = min(axis.get_xlim()[0], axis.get_ylim()[0])
        high = max(axis.get_xlim()[1], axis.get_ylim()[1])
        axis.plot([low, high], [low, high], color="black", linewidth=0.8,
                  linestyle="--")
        axis.set(xscale="log", yscale="log")
    path = output / "parameter_recovery.png"
    grid.figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(grid.figure)
    written.append(path)

    means = (curves.groupby(
        ["case", "objective", "trial_count", "condition", "metric", "source",
         "dissimilarity"], as_index=False, dropna=False)["value"].mean())
    for (case, objective), frame in means.groupby(["case", "objective"], sort=False):
        grid = sns.relplot(
            data=frame, x="dissimilarity", y="value", hue="source",
            style="trial_count", row="metric", col="condition", kind="line",
            facet_kws={"sharex": False, "sharey": False}, height=2.4,
            aspect=1.35)
        grid.figure.suptitle(f"{case} · fitted by {objective}", y=1.01)
        path = output / f"curves__{_safe_filename(case)}__{_safe_filename(objective)}.png"
        grid.figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(grid.figure)
        written.append(path)
    return written
