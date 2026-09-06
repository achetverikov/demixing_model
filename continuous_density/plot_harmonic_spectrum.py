#!/usr/bin/env python3
"""Raw and fitted complex spectra, asymmetry sums, and reconstruction error."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from continuous_density import fourier_moments as fm
from continuous_density import local_distribution_gate as gate
from continuous_density import local_fit_common as common


def mass_coefficients(mass: np.ndarray, kmax: int) -> np.ndarray:
    angle = np.radians(common.GRID)
    k = np.arange(kmax + 1)
    return mass @ np.exp(1j * angle[:, None] * k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--sd-feat1", type=float, required=True)
    parser.add_argument("--sd-feat2", type=float, required=True)
    parser.add_argument("--sd-ident", type=float, required=True)
    parser.add_argument("--feat-diff", type=float, required=True)
    parser.add_argument("--component", type=int, choices=(1, 2), required=True)
    parser.add_argument("--candidate-row", type=int, required=True)
    parser.add_argument("--candidate", action="append", type=gate.parse_candidate,
                        required=True)
    parser.add_argument("--kmax", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    blob = np.load(args.reference, allow_pickle=True)
    design, samples = common.select_trajectory(blob["design"], blob["bias"],
        args.sd_feat1, args.sd_feat2, args.sd_ident, args.component)
    row = int(np.argmin(np.abs(design[:, 3] - args.feat_diff)))
    raw = fm.empirical_coefficients(samples[row:row + 1], args.kmax)[0]
    spectra = {"raw": raw}
    for candidate in args.candidate:
        spectra[candidate.label] = mass_coefficients(
            gate.candidate_mass(candidate, args.candidate_row)[0], args.kmax)
    k = np.arange(1, args.kmax + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(k, raw[1:].real, label="Re raw", color="black")
    axes[0, 0].plot(k, raw[1:].imag, label="Im raw", color="black", ls="--")
    axes[0, 1].plot(k, np.abs(raw[1:]), label="raw", color="black", lw=2)
    for label, coefficients in list(spectra.items())[1:]:
        axes[0, 1].plot(k, np.abs(coefficients[1:]), label=label, alpha=0.8)
    odd = np.arange(1, args.kmax + 1, 2)
    for label, coefficients in spectra.items():
        cumulative = np.cumsum((4 / np.pi) * coefficients[odd].imag / odd)
        axes[1, 0].plot(odd, cumulative, label=label)
    reference_mass = common.histogram_mass(samples[row:row + 1])
    w1 = []
    for cutoff in k:
        density = fm.fejer_density(raw[None, :], common.GRID, cutoff)
        w1.append(fm.circular_wasserstein_grid(
            reference_mass, density * common.CELL_WIDTH, common.CELL_WIDTH)[0])
    axes[1, 1].plot(k, w1, color="black")
    axes[1, 1].axhline(0.5, color="tab:red", ls="--", label="0.5° margin")
    titles = ("Raw complex coefficients", "Spectrum magnitude",
              "Cumulative asymmetry partial sum", "Raw Fejér reconstruction W1")
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("Harmonic k")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[1, 1].set_ylabel("Circular W1, degrees")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)


if __name__ == "__main__":
    main()
