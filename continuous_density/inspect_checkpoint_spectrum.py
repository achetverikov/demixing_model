#!/usr/bin/env python3
"""A0 scale-placement and full-spectrum audit of a conditional checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from continuous_density import fit_local_wrapped_mixture as wrapped
from continuous_density import fourier_moments as fm
from continuous_density import local_fit_common as common
from continuous_density import wrapped_mixture_model as wm


def inspect(model, variables, design: np.ndarray, samples: np.ndarray,
            component: int, kmax: int, n_wraps: int = 4
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = design if component == 1 else np.asarray(wm.mirror_params(design))
    dist = {name: np.asarray(value) for name, value in
            model.apply(variables, jnp.asarray(params)).items()}
    raw_c = fm.empirical_coefficients(samples, kmax)
    model_c = np.column_stack([np.ones(len(design), dtype=complex)] + [
        np.asarray(wm.circular_moment(dist, k)) for k in range(1, kmax + 1)])
    metrics = wrapped.fitted_metrics(dist, samples, n_wraps)
    weight = np.exp(dist["log_pi"])
    summary = pd.DataFrame({"feat_diff": design[:, 3],
                            "min_sigma_deg": dist["sigma"].min(axis=1),
                            "weight_sigma_le_3deg": np.sum(
                                weight * (dist["sigma"] <= 3.0), axis=1)})
    for name, values in metrics.items():
        summary[name] = values
    spectra = []
    for row in range(len(design)):
        for k in range(1, kmax + 1):
            spectra.append({"feat_diff": design[row, 3], "k": k,
                            "raw_real": raw_c[row, k].real,
                            "raw_imag": raw_c[row, k].imag,
                            "raw_magnitude": abs(raw_c[row, k]),
                            "model_real": model_c[row, k].real,
                            "model_imag": model_c[row, k].imag,
                            "model_magnitude": abs(model_c[row, k])})
    return summary, pd.DataFrame(spectra)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sd-feat1", type=float, required=True)
    parser.add_argument("--sd-feat2", type=float, required=True)
    parser.add_argument("--sd-ident", type=float, required=True)
    parser.add_argument("--component", type=int, choices=(1, 2), required=True)
    parser.add_argument("--kmax", type=int, default=151)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    model, variables, meta = wm.load_model(args.model)
    blob = np.load(args.reference, allow_pickle=True)
    design, samples = common.select_trajectory(
        blob["design"], blob["bias"], args.sd_feat1, args.sd_feat2,
        args.sd_ident, args.component)
    summary, spectra = inspect(model, variables, design, samples, args.component,
                               args.kmax, meta.get("n_wraps", 4))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "checkpoint_scale_and_metrics.csv", index=False)
    spectra.to_csv(args.out_dir / "checkpoint_full_spectrum.csv", index=False)
    print(summary[["min_sigma_deg", "weight_sigma_le_3deg"]].describe())
    print(f"Wrote A0 checkpoint audit to {args.out_dir}")


if __name__ == "__main__":
    main()

