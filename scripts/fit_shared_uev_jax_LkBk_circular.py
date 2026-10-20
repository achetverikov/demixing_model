#!/usr/bin/env python3
"""UEV simulation using Lk_Bk with the current circular-feature defaults."""

from __future__ import annotations

import os
import sys
import time
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "results" / "osf_density_asymmetry" / "shared_data"

sys.path.insert(0, str(REPO_ROOT / "surface_computation"))
import jax_fit_functions as jf


def main() -> None:
    n_reps    = int(os.environ.get("SHARED_N_REPS",        "4000"))
    n_samples = int(os.environ.get("SHARED_N_SAMPLES",     "100"))
    n_init    = int(os.environ.get("JAX_SHARED_N_INIT",    "400"))
    rep_batch = int(os.environ.get("JAX_SHARED_REP_BATCH", "1000"))
    gen_seed  = int(os.environ.get("SHARED_SEED",          "20260521"))
    jax_seed  = int(os.environ.get("JAX_SHARED_SEED",      "20260521"))
    sd_spat   = float(os.environ.get("SHARED_SD_SPAT",     "30.0"))

    sd_feat_vals   = [5.0, 10.0, 20.0, 30.0, 60.0]
    dist_feat_vals = list(np.arange(0, 181, 5, dtype=float))
    dprime_vals    = [0.5, 1.0, 2.0]

    conditions = [
        (sf1, sf2, dp, dist)
        for sf1, sf2, dp, dist in product(sd_feat_vals, sd_feat_vals, dprime_vals, dist_feat_vals)
        if sf1 <= sf2
    ]
    n_cond = len(conditions)

    @jax.jit
    def fit_one(key, observed, feat_diff, spat_diff):
        out      = jf.train_em_jax(key, observed, num_comp=2, n_init=n_init)
        true_mu2 = jnp.array([-0.5, 0.5]) * spat_diff
        res      = jf.rearrange_components(out[:5], true_mu2)
        mean_1   = res[:, 3]
        true_mu1 = jnp.array([-0.5, 0.5]) * feat_diff
        return -jnp.sign(true_mu1) * (mean_1 - true_mu1)

    batch_fit = jax.jit(jax.vmap(
        jax.jit(fit_one), in_axes=(0, 0, None, None)
    ))

    rng     = np.random.default_rng(gen_seed)
    jax_key = jax.random.PRNGKey(jax_seed)
    rows: list[dict] = []
    t0 = time.perf_counter()

    for cond_i, (sd_feat1, sd_feat2, spat_dprime, dist_feat) in enumerate(conditions):
        spat_diff = sd_spat * spat_dprime

        samps_np = np.empty((n_reps, n_samples, 2), dtype=np.float32)
        for sim_i in range(n_reps):
            n1    = rng.binomial(n_samples, 0.5)
            n2    = n_samples - n1
            comp1 = rng.normal(loc=(-dist_feat / 2, -spat_diff / 2),
                               scale=(sd_feat1, sd_spat), size=(n1, 2))
            comp2 = rng.normal(loc=(dist_feat / 2,  spat_diff / 2),
                               scale=(sd_feat2, sd_spat), size=(n2, 2))
            samps_np[sim_i] = np.vstack([comp1, comp2]).astype(np.float32)

        samps   = jnp.asarray(samps_np)
        jax_key, subkey = jax.random.split(jax_key)
        keys    = jax.random.split(subkey, n_reps)

        bias_chunks = []
        for start in range(0, n_reps, rep_batch):
            sl = slice(start, start + rep_batch)
            bias_chunks.append(np.asarray(
                batch_fit(keys[sl], samps[sl], dist_feat, spat_diff)
            ))
        bias = np.concatenate(bias_chunks)  # (n_reps, 2)

        for which_comp in (1, 2):
            b = bias[:, which_comp - 1]
            rows.append({
                "cur_sd_feat1":    sd_feat1,
                "cur_sd_feat2":    sd_feat2,
                "cur_sd_spat":     sd_spat,
                "cur_spat_dprime": spat_dprime,
                "cur_dist_feat":   dist_feat,
                "model":           "Gaussian_pk_Lk_Bk_circular",
                "which_comp1":     which_comp,
                "N":               int(b.size),
                "mu2_bias":        float(np.mean(b)),
                "mu2_bias_sd":     float(np.std(b, ddof=1)),
                "mu":              float(np.mean(b)),
            })

        if (cond_i + 1) % 100 == 0 or cond_i == n_cond - 1:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (cond_i + 1) * (n_cond - cond_i - 1)
            print(f"cond {cond_i+1}/{n_cond}  "
                  f"sf1={sd_feat1:g} sf2={sd_feat2:g} "
                  f"d'={spat_dprime} dist={dist_feat:g}  "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s left]", flush=True)

    out_path = SHARED_DIR / "shared_osf_2d_uev_4000rep_jax_LkBk_circular_uev_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}  (total {time.perf_counter()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
