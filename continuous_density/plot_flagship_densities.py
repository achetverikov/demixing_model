#!/usr/bin/env python3
"""High-N raw histograms with the best saved candidate from each local family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from continuous_density import local_distribution_gate as gate
from continuous_density import local_fit_common as common


def parameter_path(root: Path, family: str, size: int) -> Path:
    for metrics in root.glob("*/local_*_metrics.csv"):
        frame = pd.read_csv(metrics, usecols=["family", "size"])
        if ((frame.family == family) & (frame["size"] == size)).any():
            return metrics.with_name(metrics.name.replace("metrics.csv", "parameters.npz"))
    raise ValueError(f"no parameters for {family} {size} in {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("design_manifest", type=Path)
    parser.add_argument("gate_root", type=Path)
    parser.add_argument("fit_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    blob = np.load(args.reference, allow_pickle=True)
    points = json.loads(args.design_manifest.read_text())["points"]
    fig, axes = plt.subplots(len(points), 1, figsize=(10, 2.7 * len(points)), sharex=True)
    for axis, point in zip(np.atleast_1d(axes), points):
        component = point["component"]
        _, samples = common.select_trajectory(blob["design"], blob["bias"],
            *point["design"][:3], component)
        axis.hist(samples[0], bins=np.linspace(-180, 180, 361), density=True,
                  color="0.75", alpha=0.7, label="raw 2M")
        gate_path = (args.gate_root / point["trajectory"] /
                     f"component_{component}/distribution_gate.json")
        result = json.loads(gate_path.read_text())
        for family in sorted({candidate["family"] for candidate in result["candidates"]}):
            choices = [candidate for candidate in result["candidates"]
                       if candidate["family"] == family]
            best = min(choices, key=lambda candidate: candidate["confirmation_nll"])
            source = args.fit_root / point["trajectory"] / f"component_{component}"
            candidate = gate.Candidate(family, family, best["size"],
                                       parameter_path(source, family, best["size"]))
            # Flagship-local fits contain one row, whereas fits inherited from the
            # 90-point selection trajectory use the original candidate-row index.
            row = 0 if samples.shape[0] == 1 else point["candidate_row"]
            density = gate.candidate_mass(candidate, row)[0] / common.CELL_WIDTH
            axis.plot(common.GRID, density, lw=1.4,
                      label=f"{family} {best['size']}")
        axis.set(title=f"{point['trajectory']}, component {component}, "
                       f"feature difference {point['feat_diff']:g}°",
                 ylabel="density / degree", xlim=(-180, 180))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=4)
    axes[-1].set_xlabel("Bias, degrees")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
