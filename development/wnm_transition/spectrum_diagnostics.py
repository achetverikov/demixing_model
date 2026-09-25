#!/usr/bin/env python3
"""Sweep full Fourier/asymmetry bandwidth over raw UEV trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from development.wnm_representation import fourier_moments as fm


def _histogram_mass(samples: np.ndarray, edges: np.ndarray) -> np.ndarray:
    finite = np.asarray(samples)[np.isfinite(samples)]
    return np.histogram(finite, bins=edges)[0].astype(np.float64)


def diagnose_rows(samples: np.ndarray, kmax: int, asymmetry_margin: float,
                  wasserstein_margin: float, grid_size: int = 720) -> pd.DataFrame:
    """Full-spectrum diagnostics for ``(rows, outcomes)`` raw samples."""
    coefficients = fm.empirical_coefficients(samples, kmax)
    direct = fm.direct_asymmetry(samples)
    grid = np.linspace(-180.0, 180.0, grid_size, endpoint=False)
    edges = np.linspace(-180.0, 180.0, grid_size + 1)
    reference_mass = np.stack([_histogram_mass(row, edges) for row in samples])
    required_asym = fm.required_asymmetry_bandwidth(
        coefficients, direct, asymmetry_margin)
    w1_errors = []
    for k in range(1, kmax + 1):
        density = fm.fejer_density(coefficients, grid, k)
        mass = density * (360.0 / grid_size)
        error = fm.circular_wasserstein_grid(reference_mass, mass, 360.0 / grid_size)
        w1_errors.append(error)
    w1_errors = np.stack(w1_errors, axis=1)
    stable_w1 = np.maximum.accumulate(w1_errors[:, ::-1], axis=1)[:, ::-1]
    within_w1 = stable_w1 <= wasserstein_margin
    required_w1 = np.where(within_w1.any(axis=1),
                           np.argmax(within_w1, axis=1) + 1, -1)
    magnitude = np.abs(coefficients[:, 1:])
    peak = np.argmax(magnitude, axis=1) + 1
    rows = np.arange(len(samples))
    return pd.DataFrame({
        "direct_asymmetry": direct,
        "required_k_asymmetry": required_asym,
        "required_k_fejer_wasserstein": required_w1,
        "peak_k_magnitude": peak,
        "peak_magnitude": magnitude[rows, peak - 1],
        "peak_real_abs": np.max(np.abs(coefficients[:, 1:].real), axis=1),
        "peak_imag_abs": np.max(np.abs(coefficients[:, 1:].imag), axis=1),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kmax", type=int, default=151)
    parser.add_argument("--asymmetry-margin", type=float, default=0.005)
    parser.add_argument("--wasserstein-margin", type=float, required=True,
                        help="Frozen full-distribution margin in degrees")
    parser.add_argument("--grid-size", type=int, default=720)
    parser.add_argument("--row-block", type=int, default=4,
                        help="Rows processed together; keep small for 100k corpora")
    parser.add_argument("--max-outcomes", type=int,
                        help="Deterministic spectral pilot subsample; omit for all outcomes")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    blob = np.load(args.reference, allow_pickle=True)
    design, bias = blob["design"], blob["bias"]
    frames = []
    for component in range(bias.shape[2]):
        for first in range(0, len(design), args.row_block):
            stop = min(first + args.row_block, len(design))
            samples = bias[first:stop, :, component]
            if args.max_outcomes is not None:
                samples = samples[:, :args.max_outcomes]
            frame = diagnose_rows(samples, args.kmax,
                                  args.asymmetry_margin, args.wasserstein_margin,
                                  args.grid_size)
            frame.insert(0, "component", component + 1)
            for column, name in enumerate(
                    ("sd_feat1", "sd_feat2", "sd_ident", "feat_diff")):
                frame[name] = design[first:stop, column]
            frames.append(frame)
            if stop == len(design) or stop % args.progress_every < args.row_block:
                print(f"component {component + 1}: {stop}/{len(design)}", flush=True)
    output = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"Wrote {len(output)} rows to {args.out}")


if __name__ == "__main__":
    main()
