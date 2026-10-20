#!/usr/bin/env python3
"""Fit shared UEV datasets with JAX train_em_jax (Lk_Bk model).

Outputs per-component bias rows (which_comp ∈ {1, 2}):
  which_comp=1: smaller-spat component (sd_feat1, true feat mean = -dist_feat/2)
  which_comp=2: larger-spat component  (sd_feat2, true feat mean = +dist_feat/2)

Positive bias = attractive (estimated mean closer to 0 than true mean).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


REPO_ROOT  = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "results" / "osf_density_asymmetry" / "shared_data"


def main() -> None:
    prefix   = os.environ.get("SHARED_OUTPUT_PREFIX", "shared_osf_2d_uev_1000rep")
    n_init   = int(os.environ.get("JAX_SHARED_N_INIT",    "1024"))
    rep_batch = int(os.environ.get("JAX_SHARED_REP_BATCH", "1000"))

    npz  = np.load(SHARED_DIR / f"{prefix}.npz")
    meta = pd.read_csv(SHARED_DIR / f"{prefix}_meta.csv")
    samples = npz["samples"]   # (n_cond, n_reps, n_samples, 2)

    sys.path.insert(0, str(REPO_ROOT / "surface_computation"))
    import jax_fit_functions as jf

    @jax.jit
    def fit_one(key, observed, feat_diff, spat_diff):
        out = jf.train_em_jax(key, observed, num_comp=2, n_init=n_init)
        true_mu2 = jnp.array([-0.5, 0.5]) * spat_diff
        res = jf.rearrange_components(out[:5], true_mu2)
        mean_1 = res[:, 3]                              # feat means, (2,)
        true_mu1 = jnp.array([-0.5, 0.5]) * feat_diff
        return -jnp.sign(true_mu1) * (mean_1 - true_mu1)  # (2,): [bias_comp1, bias_comp2]

    batch_fit = jax.jit(jax.vmap(
        lambda k, obs, fd, sd: fit_one(k, obs, fd, sd),
        in_axes=(0, 0, None, None),
    ))

    key = jax.random.PRNGKey(int(os.environ.get("JAX_SHARED_SEED", "20260521")))
    rows = []
    t0 = time.perf_counter()

    for cond_i, row in meta.iterrows():
        dist_feat  = float(row["dist_feat"])
        spat_diff  = float(row["sd_spat"]) * float(row["spat_dprime"])
        n_reps     = samples.shape[1]

        key, subkey = jax.random.split(key)
        keys  = jax.random.split(subkey, n_reps)
        samps = jnp.asarray(samples[cond_i])

        bias_chunks = []
        for start in range(0, n_reps, rep_batch):
            sl = slice(start, start + rep_batch)
            bias_chunks.append(np.asarray(
                batch_fit(keys[sl], samps[sl], dist_feat, spat_diff)
            ))
        bias = np.concatenate(bias_chunks)   # (n_reps, 2)

        for which_comp in (1, 2):
            b = bias[:, which_comp - 1]
            sd_feat = float(row[f"sd_feat{which_comp}"])
            rows.append({
                "sd_feat1":    float(row["sd_feat1"]),
                "sd_feat2":    float(row["sd_feat2"]),
                "sd_spat":     float(row["sd_spat"]),
                "spat_dprime": float(row["spat_dprime"]),
                "dist_feat":   dist_feat,
                "which_comp":  which_comp,
                "sd_feat":     sd_feat,
                "N":           int(b.size),
                "mean_bias":   float(np.mean(b)),
                "bias_sd":     float(np.std(b, ddof=1)),
                "sign_asymmetry": float(np.mean(b > 0) - np.mean(b < 0)),
            })

        if (cond_i + 1) % 100 == 0 or cond_i == len(meta) - 1:
            elapsed = time.perf_counter() - t0
            print(f"cond {cond_i+1}/{len(meta)}  "
                  f"sf1={row['sd_feat1']:g} sf2={row['sd_feat2']:g} "
                  f"d'={row['spat_dprime']} dist={dist_feat:g}  "
                  f"[{elapsed:.0f}s]", flush=True)

    out_path = SHARED_DIR / f"{prefix}_jax_Lk_Bk_uev_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    total = time.perf_counter() - t0
    print(f"Wrote {out_path}  (total {total:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
