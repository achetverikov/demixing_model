#!/usr/bin/env python3
"""Predicted-versus-raw amplitude slopes for all core trajectory metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.metrics], ignore_index=True)
    selected = frame[frame["size"] == frame.groupby("family")["size"].transform("max")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric in zip(axes, ("mean_bias", "response_sd", "density_asymmetry")):
        for (family, size), group in selected.groupby(["family", "size"]):
            raw, pred = group[f"raw_{metric}"], group[f"pred_{metric}"]
            slope = ((raw - raw.mean()) * (pred - pred.mean())).sum() / (
                (raw - raw.mean()) ** 2).sum()
            axis.scatter(raw, pred, s=9, alpha=0.5, label=f"{family} {size:g}: {slope:.3f}")
        low = min(axis.get_xlim()[0], axis.get_ylim()[0])
        high = max(axis.get_xlim()[1], axis.get_ylim()[1])
        axis.plot([low, high], [low, high], color="black", lw=1)
        axis.set(title=metric, xlabel="raw", ylabel="predicted")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
