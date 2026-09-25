#!/usr/bin/env python3
"""Fit independent periodic cubic-spline log densities along one trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from development.wnm_representation import local_fit_common as common
from development.wnm_representation import periodic_spline as spline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--sd-feat1", type=float, required=True)
    parser.add_argument("--sd-feat2", type=float, required=True)
    parser.add_argument("--sd-ident", type=float, required=True)
    parser.add_argument("--component", type=int, choices=(1, 2), required=True)
    parser.add_argument("--knots", type=int, nargs="+", default=[24, 48, 96])
    parser.add_argument("--quadrature-size", type=int, default=2880)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    blob = np.load(args.reference, allow_pickle=True)
    design, samples = common.select_trajectory(blob["design"], blob["bias"],
        args.sd_feat1, args.sd_feat2, args.sd_ident, args.component)
    train, test = common.split_outcomes(samples, seed=args.split_seed)
    raw = common.empirical_core_metrics(test)
    train_raw = common.empirical_core_metrics(train)
    frames, parameters = [], {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for knots in args.knots:
        fits = [spline.fit_periodic_spline(row, knots, args.quadrature_size, args.maxiter)
                for row in train]
        theta = np.stack([fit.density.theta for fit in fits])
        parameters[f"k{knots}_theta"] = theta
        predicted = {key: [] for key in ("mean_bias", "resultant", "response_sd",
                                          "density_asymmetry", "nll", "wasserstein")}
        for row, fit in enumerate(fits):
            stats = fit.density.circular_stats()
            for key in ("mean_bias", "resultant", "response_sd"):
                predicted[key].append(stats[key])
            predicted["density_asymmetry"].append(fit.density.asymmetry())
            predicted["nll"].append(float(-fit.density.logpdf(test[row]).mean()))
            density = fit.density.density_grid(common.GRID)[None, :]
            predicted["wasserstein"].append(float(
                common.circular_wasserstein_to_density(test[row:row + 1], density)[0]))
        frame = pd.DataFrame({"feat_diff": design[:, 3], "family": "periodic_spline",
            "size": knots, "fit_success": [fit.success for fit in fits],
            "fit_iterations": [fit.n_iterations for fit in fits],
            "fit_gradient_norm": [fit.gradient_norm for fit in fits]})
        for metric in ("mean_bias", "resultant", "response_sd", "density_asymmetry"):
            frame[f"raw_{metric}"] = raw[metric]
            frame[f"train_raw_{metric}"] = train_raw[metric]
            frame[f"pred_{metric}"] = predicted[metric]
        frame["heldout_nll"] = predicted["nll"]
        frame["circular_wasserstein_deg"] = predicted["wasserstein"]
        frame["train_nll"] = [-fit.density.logpdf(row).mean()
                              for fit, row in zip(fits, train)]
        frame["train_circular_wasserstein_deg"] = [float(
            common.circular_wasserstein_to_density(
                train[row:row + 1], fits[row].density.density_grid(common.GRID)[None, :])[0])
            for row in range(len(train))]
        frames.append(frame)
        print(f"knots={knots}: NLL={frame.heldout_nll.mean():.6f}", flush=True)
    pd.concat(frames, ignore_index=True).to_csv(args.out_dir / "local_spline_metrics.csv", index=False)
    np.savez(args.out_dir / "local_spline_parameters.npz", **parameters)


if __name__ == "__main__":
    main()
