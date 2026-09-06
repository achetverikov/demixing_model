#!/usr/bin/env python3
"""Exploratory circular mode diagnostics for raw reference trajectories.

The smoothed histogram is used only to flag visibly multimodal cases for Phase A
and to report approximate mode positions/masses.  It is never a fitted likelihood
or a reference for the core metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def circular_modes(samples: np.ndarray, bin_width_deg: float = 1.0,
                   smoothing_sd_deg: float = 3.0,
                   min_prominence_fraction: float = 0.08,
                   min_mass: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return approximate positions and basin masses of substantial modes."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]
    edges = np.arange(-180.0, 180.0 + bin_width_deg, bin_width_deg)
    mass = np.histogram(x, edges)[0].astype(np.float64)
    mass /= mass.sum()
    centers = edges[:-1] + bin_width_deg / 2.0
    return circular_modes_from_mass(mass, centers, smoothing_sd_deg,
                                    min_prominence_fraction, min_mass)


def circular_modes_from_mass(mass: np.ndarray, centers: np.ndarray,
                             smoothing_sd_deg: float = 3.0,
                             min_prominence_fraction: float = 0.08,
                             min_mass: float = 0.05
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Approximate circular mode positions and basin masses from grid mass."""
    mass = np.asarray(mass, dtype=np.float64)
    mass = mass / mass.sum()
    centers = np.asarray(centers, dtype=np.float64)
    bin_width_deg = 360.0 / len(mass)
    smooth = gaussian_filter1d(mass, smoothing_sd_deg / bin_width_deg, mode="wrap")
    extended = np.tile(smooth, 3)
    peaks, properties = find_peaks(
        extended, prominence=min_prominence_fraction * smooth.max())
    n = len(smooth)
    central = (peaks >= n) & (peaks < 2 * n)
    peaks = peaks[central] - n
    prominences = properties["prominences"][central]
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(smooth))])
        prominences = np.array([smooth.max()])
    peaks = peaks[np.argsort(peaks)]
    boundaries = []
    for left, right in zip(peaks, np.roll(peaks, -1)):
        indices = np.arange(left, right + (n if right <= left else 0) + 1) % n
        boundaries.append(int(indices[np.argmin(smooth[indices])]))
    basin_mass = np.empty(len(peaks))
    for index in range(len(peaks)):
        left = boundaries[index - 1]
        right = boundaries[index]
        indices = np.arange(left + 1, right + (n if right <= left else 0) + 1) % n
        basin_mass[index] = mass[indices].sum()
    keep = basin_mass >= min_mass
    if not keep.any():
        keep[np.argmax(prominences)] = True
    order = np.argsort(basin_mass[keep])[::-1]
    return centers[peaks[keep]][order], basin_mass[keep][order]


def diagnose_corpus(reference: Path) -> pd.DataFrame:
    blob = np.load(reference, mmap_mode="r", allow_pickle=True)
    design, bias = blob["design"], blob["bias"]
    rows = []
    for component in range(bias.shape[2]):
        for row in range(len(design)):
            positions, masses = circular_modes(bias[row, :, component])
            rows.append({"sd_feat1": design[row, 0], "sd_feat2": design[row, 1],
                         "sd_ident": design[row, 2], "feat_diff": design[row, 3],
                         "component": component + 1, "mode_count": len(positions),
                         "mode_positions_deg": json.dumps(positions.tolist()),
                         "mode_masses": json.dumps(masses.tolist()),
                         "secondary_mode_mass": float(masses[1] if len(masses) > 1 else 0)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.concat([diagnose_corpus(path) for path in args.reference], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame)} rows; {int((frame.mode_count > 1).sum())} are multimodal")


if __name__ == "__main__":
    main()
