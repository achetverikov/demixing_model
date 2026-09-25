#!/usr/bin/env python3
"""Fit multi-start independent wrapped-normal mixtures along one trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import softmax

from development.wnm_representation import diagnose_local_capacity as local
from development.wnm_representation import equivalence
from development.wnm_representation import local_fit_common as common
from shared import wnm as wm


def initialize_from_samples(samples: np.ndarray, n_components: int,
                            seed: int = 0) -> dict[str, np.ndarray]:
    """Quantile initialization with randomized circular jitter for one restart."""
    rng = np.random.default_rng(seed)
    quantiles = (np.arange(n_components) + 0.5) / n_components
    mu = np.stack([np.quantile(row[np.isfinite(row)], quantiles) for row in samples])
    if seed:
        mu = np.asarray(wm.wrap_deg(mu + rng.normal(
            scale=min(15.0, 90.0 / n_components), size=mu.shape)))
    resultant = np.abs(np.nanmean(np.exp(1j * np.radians(samples)), axis=1))
    circ_sd = np.degrees(np.sqrt(-2 * np.log(np.clip(resultant, 1e-6, 1))))
    scale = np.maximum(circ_sd / np.sqrt(n_components), 1.0)
    sigma = np.repeat(scale[:, None], n_components, axis=1)
    log_pi = np.full_like(mu, -np.log(n_components))
    return {"log_pi": log_pi, "mu": mu, "sigma": sigma}


def _take_rows(stacked: np.ndarray, choices: np.ndarray) -> np.ndarray:
    return stacked[choices, np.arange(stacked.shape[1])]


def fit_multistart(train: np.ndarray, n_components: int, starts: int,
                   steps: int, batch_size: int, lr: float,
                   min_scale: float = 0.25, n_wraps: int = 4
                   ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit every start and select each row by training NLL; retain all starts."""
    distributions, train_nll = [], []
    for start in range(starts):
        initial = initialize_from_samples(train, n_components, start)
        fitted = local.fit_free_mixtures(
            initial, train, min_scale=min_scale, steps=steps,
            batch_size=batch_size, lr=lr, seed=start, n_wraps=n_wraps)
        distributions.append(fitted)
        train_nll.append(local._dist_nll(fitted, train, n_wraps))
    all_starts = {name: np.stack([dist[name] for dist in distributions])
                  for name in ("log_pi", "mu", "sigma")}
    all_starts["train_nll"] = np.stack(train_nll)
    choice = np.argmin(all_starts["train_nll"], axis=0)
    selected = {name: _take_rows(all_starts[name], choice)
                for name in ("log_pi", "mu", "sigma")}
    selected["selected_start"] = choice
    return selected, all_starts


def fitted_metrics(dist: dict[str, np.ndarray], test: np.ndarray,
                   n_wraps: int = 4, train: np.ndarray | None = None
                   ) -> dict[str, np.ndarray]:
    raw = common.empirical_core_metrics(test)
    mean, resultant = (np.asarray(x) for x in wm.mean_and_resultant(dist))
    sd = np.asarray(wm.circular_sd(dist))
    log_density = np.asarray(wm.mixture_logpdf_grid(
        jnp.asarray(common.GRID), dist, n_wraps))
    density = np.exp(log_density)
    asymmetry = np.asarray(wm.density_asymmetry(dist))
    output = {**{f"raw_{key}": value for key, value in raw.items()},
            "pred_mean_bias": mean, "pred_resultant": resultant,
            "pred_response_sd": sd, "pred_density_asymmetry": asymmetry,
            "heldout_nll": local._dist_nll(dist, test, n_wraps),
            "circular_wasserstein_deg": common.circular_wasserstein_to_density(test, density)}
    if train is not None:
        train_raw = common.empirical_core_metrics(train)
        output.update({f"train_raw_{key}": value for key, value in train_raw.items()})
        output["train_nll"] = local._dist_nll(dist, train, n_wraps)
        output["train_circular_wasserstein_deg"] = (
            common.circular_wasserstein_to_density(train, density))
    return output


def component_diagnostics(dist: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    weights = softmax(dist["log_pi"], axis=1)
    delta = np.abs(wm.wrap_deg(dist["mu"][:, :, None] - dist["mu"][:, None, :]))
    scale_ratio = np.abs(np.log(dist["sigma"][:, :, None] /
                                dist["sigma"][:, None, :]))
    upper = np.triu(np.ones(delta.shape[1:], dtype=bool), 1)
    duplicate = ((delta < 1.0) & (scale_ratio < 0.1) & upper).sum(axis=(1, 2))
    return {"min_component_scale": dist["sigma"].min(axis=1),
            "mass_scale_le_3deg": np.sum(weights * (dist["sigma"] <= 3.0), axis=1),
            "effective_components": np.sum(weights >= 1e-3, axis=1),
            "vanishing_components": np.sum(weights < 1e-4, axis=1),
            "near_floor_components": np.sum(dist["sigma"] < 0.30, axis=1),
            "duplicate_component_pairs": duplicate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--sd-feat1", type=float, required=True)
    parser.add_argument("--sd-feat2", type=float, required=True)
    parser.add_argument("--sd-ident", type=float, required=True)
    parser.add_argument("--component", type=int, choices=(1, 2), required=True)
    parser.add_argument("--components", type=int, nargs="+", default=[4, 8, 12, 24, 48])
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.02)
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
    frames, arrays = [], {}
    for k in args.components:
        metric_path = args.out_dir / f"k{k}_metrics.csv"
        parameter_path = args.out_dir / f"k{k}_parameters.npz"
        if args.resume and metric_path.exists() and parameter_path.exists():
            frame = pd.read_csv(metric_path)
            saved = np.load(parameter_path)
            selected = {name: saved[f"selected_{name}"] for name in
                        ("log_pi", "mu", "sigma", "selected_start")}
            starts = {name: saved[f"all_{name}"] for name in
                      ("log_pi", "mu", "sigma", "train_nll")}
            metrics = {name: frame[name].to_numpy() for name in
                       frame.columns if name.startswith(("raw_", "pred_"))
                       or name in {"heldout_nll", "circular_wasserstein_deg"}}
            print(f"K={k}: resumed saved result", flush=True)
        else:
            selected, starts = fit_multistart(
                train, k, args.starts, args.steps, args.batch_size, args.lr)
            metrics = fitted_metrics(selected, test, train=train)
            frame = pd.DataFrame({
                "feat_diff": design[:, 3], "family": "wrapped_mixture", "size": k,
                "selected_start": selected["selected_start"],
                "train_nll_start_sd": np.std(starts["train_nll"], axis=0)})
            for name, values in component_diagnostics(selected).items():
                frame[name] = values
            for name, values in metrics.items():
                frame[name] = values
            temporary_csv = metric_path.with_suffix(".tmp.csv")
            temporary_npz = parameter_path.with_suffix(".tmp.npz")
            frame.to_csv(temporary_csv, index=False)
            np.savez(temporary_npz,
                     **{f"all_{name}": value for name, value in starts.items()},
                     **{f"selected_{name}": value for name, value in selected.items()})
            temporary_csv.replace(metric_path)
            temporary_npz.replace(parameter_path)
        frames.append(frame)
        for name, values in starts.items():
            arrays[f"k{k}_all_{name}"] = values
        for name, values in selected.items():
            arrays[f"k{k}_selected_{name}"] = values
        summary = equivalence.curve_summary(
            metrics["pred_density_asymmetry"], metrics["raw_density_asymmetry"])
        print(f"K={k}: NLL={metrics['heldout_nll'].mean():.6f}; "
              f"asym max={summary['max_abs_error']:.5f}; "
              f"asym coherent={summary['mean_signed_error']:.5f}")
    pd.concat(frames, ignore_index=True).to_csv(
        args.out_dir / "local_wrapped_metrics.csv", index=False)
    np.savez(args.out_dir / "local_wrapped_parameters.npz", **arrays)
    provenance = vars(args).copy()
    provenance = {key: str(value) if isinstance(value, Path) else value
                  for key, value in provenance.items()}
    (args.out_dir / "local_wrapped_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
