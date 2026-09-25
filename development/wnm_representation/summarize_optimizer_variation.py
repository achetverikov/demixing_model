#!/usr/bin/env python3
"""Propagate wrapped-mixture restart variation to Phase A output metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from development.wnm_representation import fourier_moments as fm
from development.wnm_representation import local_fit_common as common
from shared import wnm as wm


def _stats(mass: np.ndarray) -> dict[str, float]:
    z = mass @ np.exp(1j * np.radians(common.GRID))
    resultant = float(abs(z))
    positive = (common.GRID > 0) & (np.abs(common.GRID) < 180)
    negative = (common.GRID < 0) & (np.abs(common.GRID) < 180)
    return {"pred_mean_bias": float(np.degrees(np.angle(z))),
            "pred_resultant": resultant,
            "pred_response_sd": float(np.degrees(np.sqrt(-2 * np.log(
                np.clip(resultant, 1e-12, 1))))),
            "pred_density_asymmetry": float(mass[positive].sum() - mass[negative].sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("parameters", type=Path)
    parser.add_argument("--sd-feat1", type=float, required=True)
    parser.add_argument("--sd-feat2", type=float, required=True)
    parser.add_argument("--sd-ident", type=float, required=True)
    parser.add_argument("--component", type=int, choices=(1, 2), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = np.load(args.reference, allow_pickle=True)
    _, samples = common.select_trajectory(raw["design"], raw["bias"],
        args.sd_feat1, args.sd_feat2, args.sd_ident, args.component)
    _, test = common.split_outcomes(samples, seed=args.split_seed)
    test = test[args.row:args.row + 1]
    reference_mass = common.histogram_mass(test)
    blob = np.load(args.parameters)
    prefix = f"k{args.size}_all_"
    log_pi = blob[prefix + "log_pi"][:, args.row:args.row + 1]
    mu = blob[prefix + "mu"][:, args.row:args.row + 1]
    sigma = blob[prefix + "sigma"][:, args.row:args.row + 1]
    values = []
    for start in range(len(log_pi)):
        dist = {"log_pi": log_pi[start], "mu": mu[start], "sigma": sigma[start]}
        density = np.exp(np.asarray(wm.mixture_logpdf_grid(
            jnp.asarray(common.GRID), dist)))[0]
        mass = density * common.CELL_WIDTH
        mass /= mass.sum()
        row = {"start": start, **_stats(mass)}
        row["circular_wasserstein_deg"] = float(fm.circular_wasserstein_grid(
            reference_mass, mass[None, :], common.CELL_WIDTH)[0])
        total, count = 0.0, 0
        for first in range(0, test.shape[1], 5000):
            block = test[:, first:first + 5000]
            finite = np.isfinite(block)
            logpdf = np.asarray(wm.mixture_logpdf_samples(
                jnp.asarray(np.where(finite, block, 0)), dist))
            total += float(np.where(finite, logpdf, 0).sum())
            count += int(finite.sum())
        row["heldout_nll"] = -total / count
        values.append(row)
    frame = pd.DataFrame(values)
    rows = []
    for metric in frame.columns.drop("start"):
        metric_values = frame[metric].to_numpy()
        if metric == "pred_mean_bias":
            center = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(metric_values)))))
            metric_values = (metric_values - center + 180) % 360 - 180
        rows.append({"metric": metric, "optimizer_start_sd": float(metric_values.std()),
                     "optimizer_start_range": float(metric_values.max() - metric_values.min())})
    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
