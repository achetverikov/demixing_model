#!/usr/bin/env python3
"""Fit shared generated datasets with JAX EM variants."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "results" / "osf_density_asymmetry" / "shared_data"

FITTER_MAP = {
    "train_em":    "JAX Lk_Bk",   # Gaussian_pk_Lk_Bk equivalent
    "lk_b":        "JAX Lk_B",    # Gaussian_pk_Lk_B equivalent
    "diag_common": "JAX L_B",     # Gaussian_pk_L_B equivalent
}


def main() -> None:
    prefix = os.environ.get("SHARED_OUTPUT_PREFIX", "shared_osf_2d_1000rep")
    fitter = os.environ.get("JAX_SHARED_FITTER", "train_em").lower()
    if fitter not in FITTER_MAP:
        raise ValueError(f"JAX_SHARED_FITTER must be one of: {list(FITTER_MAP)}")

    npz  = np.load(SHARED_DIR / f"{prefix}.npz")
    meta = pd.read_csv(SHARED_DIR / f"{prefix}_meta.csv")
    samples = npz["samples"]   # (n_cond, n_reps, n_samples, 2)  col0=feat, col1=spat

    sys.path.insert(0, str(REPO_ROOT / "surface_computation"))
    import jax_fit_functions as jf

    dataset = FITTER_MAP[fitter]

    n_init = int(os.environ.get("JAX_SHARED_N_INIT", "1024"))

    if fitter == "train_em":
        @jax.jit
        def fit_one(key, observed, feat_diff):
            out = jf.train_em_jax(key, observed, num_comp=2, n_init=n_init)
            true_mu2 = jnp.array([-0.5, 0.5]) * 30.0   # spat means
            res = jf.rearrange_components(out[:5], true_mu2)
            mean_1 = res[:, 3]   # feat means, col index: ids(2) + weight(1) + mu1(1)
            true_mu1 = jnp.array([-0.5, 0.5]) * feat_diff
            return -jnp.sign(true_mu1) * (mean_1 - true_mu1)

    elif fitter == "lk_b":
        @jax.jit
        def fit_one(key, observed, feat_diff):
            out = jf.train_em_jax_Lk_B(key, observed, num_comp=2, n_init=n_init)
            true_mu2 = jnp.array([-0.5, 0.5]) * 30.0
            res = jf.rearrange_components(out[:5], true_mu2)
            mean_1 = res[:, 3]
            true_mu1 = jnp.array([-0.5, 0.5]) * feat_diff
            return -jnp.sign(true_mu1) * (mean_1 - true_mu1)

    elif fitter == "diag_common":
        @jax.jit
        def fit_one(key, observed, feat_diff):
            out = jf.train_em_jax_diag_common(key, observed, num_comp=2, n_init=n_init)
            true_mu2 = jnp.array([-0.5, 0.5]) * 30.0
            res = jf.rearrange_components(out[:5], true_mu2)
            mean_1 = res[:, 3]
            true_mu1 = jnp.array([-0.5, 0.5]) * feat_diff
            return -jnp.sign(true_mu1) * (mean_1 - true_mu1)

    rep_batch = int(os.environ.get("JAX_SHARED_REP_BATCH", "1000"))
    batch_fit = jax.jit(jax.vmap(lambda k, obs, fd: fit_one(k, obs, fd), in_axes=(0, 0, None)))
    key = jax.random.PRNGKey(int(os.environ.get("JAX_SHARED_SEED", "20260521")))
    rows = []

    for cond_i, row in meta.iterrows():
        feat_diff = float(row["feat_diff"])
        key, subkey = jax.random.split(key)
        n_reps = samples.shape[1]
        keys = jax.random.split(subkey, n_reps)
        samps = jnp.asarray(samples[cond_i])
        bias_chunks = []
        for start in range(0, n_reps, rep_batch):
            sl = slice(start, start + rep_batch)
            bias_chunks.append(np.asarray(batch_fit(keys[sl], samps[sl], feat_diff)))
        bias = np.concatenate(bias_chunks).reshape(-1)
        rows.append({
            "dataset":      dataset,
            "sd_feat":      float(row["sd_feat"]),
            "sd_spat":      float(row["sd_spat"]),
            "spat_dprime":  float(row["spat_dprime"]),
            "feat_diff":    feat_diff,
            "N":            int(bias.size),
            "mean_bias":    float(np.mean(bias)),
            "bias_sd":      float(np.std(bias, ddof=1)),
            "sign_asymmetry": float(np.mean(bias > 0) - np.mean(bias < 0)),
        })
        print(f"fit {dataset} cond {cond_i + 1}/{len(meta)} "
              f"sd_feat={row['sd_feat']:g} feat_diff={feat_diff:g}", flush=True)

    suffix = {"train_em": "Lk_Bk", "lk_b": "Lk_B", "diag_common": "L_B"}[fitter]
    out_path = SHARED_DIR / f"{prefix}_jax_{suffix}_summary.csv"
    pd.DataFrame(rows).sort_values(["sd_feat", "feat_diff"]).to_csv(out_path, index=False)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
