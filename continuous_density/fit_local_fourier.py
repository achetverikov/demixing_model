#!/usr/bin/env python3
"""Fit independent maxent-Fourier densities along one raw trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_density import equivalence
from continuous_density import local_fit_common as common
from continuous_density import maxent_fourier as mf


def fit_trajectory(train: np.ndarray, test: np.ndarray, feat_diff: np.ndarray,
                   harmonics: list[int], quadrature_size: int, maxiter: int
                   ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frames, parameters = [], {}
    previous = None
    for k in sorted(harmonics):
        frame, theta = fit_size(train, test, feat_diff, k, quadrature_size,
                                maxiter, previous)
        parameters[f"k{k}_theta"] = theta
        previous = theta
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), parameters


def _expand_initial(previous: np.ndarray | None, k: int, row: int) -> np.ndarray:
    initial = np.zeros(2 * k)
    if previous is not None:
        previous_k = previous.shape[1] // 2
        initial[:previous_k] = previous[row, :previous_k]
        initial[k:k + previous_k] = previous[row, previous_k:]
    return initial


def fit_size(train: np.ndarray, test: np.ndarray, feat_diff: np.ndarray, k: int,
             quadrature_size: int, maxiter: int,
             previous: np.ndarray | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit and evaluate one K, warm-starting from the previous smaller K."""
    raw = common.empirical_core_metrics(test)
    train_raw = common.empirical_core_metrics(train)
    fits = []
    theta = np.empty((len(train), 2 * k))
    for row in range(len(train)):
        fit = mf.fit_maxent_fourier(
            train[row], k, quadrature_size, maxiter=maxiter,
            initial=_expand_initial(previous, k, row))
        fits.append(fit)
        theta[row] = fit.density.theta
        if (row + 1) % 10 == 0 or row + 1 == len(train):
            print(f"K={k}: fitted {row + 1}/{len(train)}", flush=True)
    pred = {name: [] for name in ("mean_bias", "resultant", "response_sd",
                                  "density_asymmetry", "nll", "wasserstein")}
    for row, fit in enumerate(fits):
        stats = fit.density.circular_stats()
        pred["mean_bias"].append(stats["mean_bias"])
        pred["resultant"].append(stats["resultant"])
        pred["response_sd"].append(stats["response_sd"])
        pred["density_asymmetry"].append(fit.density.asymmetry())
        finite = test[row][np.isfinite(test[row])]
        pred["nll"].append(float(-fit.density.logpdf(finite).mean()))
        density = fit.density.density_grid(common.GRID)[None, :]
        pred["wasserstein"].append(float(
            common.circular_wasserstein_to_density(test[row:row + 1], density)[0]))
    frame = pd.DataFrame({"feat_diff": feat_diff, "family": "maxent_fourier",
                          "size": k, "fit_success": [f.success for f in fits],
                          "fit_iterations": [f.n_iterations for f in fits],
                          "fit_gradient_norm": [f.gradient_norm for f in fits]})
    for metric in ("mean_bias", "resultant", "response_sd", "density_asymmetry"):
        frame[f"raw_{metric}"] = raw[metric]
        frame[f"train_raw_{metric}"] = train_raw[metric]
        frame[f"pred_{metric}"] = pred[metric]
    frame["heldout_nll"] = pred["nll"]
    frame["circular_wasserstein_deg"] = pred["wasserstein"]
    frame["train_nll"] = [-fit.density.logpdf(row).mean()
                          for fit, row in zip(fits, train)]
    frame["train_circular_wasserstein_deg"] = [float(
        common.circular_wasserstein_to_density(
            train[row:row + 1], fits[row].density.density_grid(common.GRID)[None, :])[0])
        for row in range(len(train))]
    return frame, theta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--sd-feat1", type=float, required=True)
    parser.add_argument("--sd-feat2", type=float, required=True)
    parser.add_argument("--sd-ident", type=float, required=True)
    parser.add_argument("--component", type=int, choices=(1, 2), required=True)
    parser.add_argument("--harmonics", type=int, nargs="+",
                        default=[4, 8, 16, 24, 32, 48, 64, 96, 128])
    parser.add_argument("--quadrature-size", type=int, default=4096)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    blob = np.load(args.reference, allow_pickle=True)
    design, samples = common.select_trajectory(
        blob["design"], blob["bias"], args.sd_feat1, args.sd_feat2,
        args.sd_ident, args.component)
    train, test = common.split_outcomes(samples, seed=args.split_seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"reference": str(args.reference), "sd_feat1": args.sd_feat1,
                  "sd_feat2": args.sd_feat2, "sd_ident": args.sd_ident,
                  "component": args.component, "harmonics": args.harmonics,
                  "quadrature_size": args.quadrature_size, "maxiter": args.maxiter,
                  "split_seed": args.split_seed}
    (args.out_dir / "local_fourier_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    frames, parameters, previous = [], {}, None
    for k in sorted(args.harmonics):
        metric_path = args.out_dir / f"k{k}_metrics.csv"
        theta_path = args.out_dir / f"k{k}_theta.npy"
        if args.resume and metric_path.exists() and theta_path.exists():
            group, theta = pd.read_csv(metric_path), np.load(theta_path)
            print(f"K={k}: resumed saved result", flush=True)
        else:
            group, theta = fit_size(train, test, design[:, 3], k,
                                    args.quadrature_size, args.maxiter, previous)
            temporary_csv = metric_path.with_suffix(".tmp.csv")
            temporary_npy = theta_path.with_suffix(".tmp.npy")
            group.to_csv(temporary_csv, index=False)
            np.save(temporary_npy, theta)
            temporary_csv.replace(metric_path)
            temporary_npy.replace(theta_path)
        frames.append(group)
        parameters[f"k{k}_theta"] = theta
        previous = theta
        summary = equivalence.curve_summary(
            group.pred_density_asymmetry, group.raw_density_asymmetry)
        print(f"K={k}: NLL={group.heldout_nll.mean():.6f}; "
              f"asym max={summary['max_abs_error']:.5f}; "
              f"asym coherent={summary['mean_signed_error']:.5f}")
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(args.out_dir / "local_fourier_metrics.csv", index=False)
    np.savez(args.out_dir / "local_fourier_parameters.npz", **parameters)


if __name__ == "__main__":
    main()
