#!/usr/bin/env python3
"""UEV simulation using Lk_Bk model with main-branch (old) initialization.

Runs the same UEV conditions as fit_shared_uev_jax_LkB_LB.py but with
train_em_jax patched to use the grid-angle / fixed-sigma init from main,
so the effect of initialization can be compared.
"""

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
from functools import partial

REPO_ROOT  = Path(__file__).resolve().parents[1]
SHARED_DIR = REPO_ROOT / "results" / "osf_density_asymmetry" / "shared_data"

sys.path.insert(0, str(REPO_ROOT / "surface_computation"))
import jax_fit_functions as jf


def train_em_jax_old_init(key, observed, num_comp=2, n_init=400,
                           rtol=1e-6, max_iter=5000):
    """Lk_Bk EM with main-branch grid-angle / fixed-sigma initialisation."""

    def cond_fn(state):
        i, thetas, loss, loss_diff = state
        return jnp.any((i < max_iter) & (loss_diff > rtol))

    @partial(jax.jit, static_argnames=['min_val'])
    def project_to_simplex_with_min(x, min_val=0.01):
        x = jnp.maximum(x, min_val)
        excess = jnp.sum(x) - 1.0
        x -= excess * (x - min_val) / jnp.sum(x - min_val + 1e-8)
        return x

    def generate_inits(key):
        raw_pi = jax.random.uniform(key, shape=(n_init, num_comp))
        pi_init = raw_pi / raw_pi.sum(-1, keepdims=True)
        key, subkey = jax.random.split(key)

        # --- OLD INIT: grid of angles ----------------------------------------
        angles   = jnp.linspace(0, 180, n_init)
        mu_init  = jnp.stack([-angles, angles], axis=1)        # (n_init, 2)
        n_trials = observed.shape[-1]
        mu_init  = jnp.broadcast_to(
            mu_init[:, :, None], (n_init, 2, n_trials)
        )
        mu1_init = mu_init[..., 0]
        mu2_init = mu_init[..., 1]

        key, subkey = jax.random.split(key)
        # --- OLD INIT: fixed sigma range 40-120 ------------------------------
        sigma_init  = jax.random.uniform(
            subkey, minval=40, maxval=120,
            shape=(n_init, num_comp, observed.shape[-1])
        )
        sigma1_init = sigma_init[..., 0]
        sigma2_init = sigma_init[..., 1]

        loss_init = jf.initial_em_vlb(observed,
            (pi_init, mu1_init, mu2_init, sigma1_init, sigma2_init),
            circular=False)
        loss_upd_init =  jnp.ones([n_init]) * jnp.inf
        init_val = (jnp.zeros([n_init], jnp.int32),
                    (pi_init, mu1_init, mu2_init, sigma1_init, sigma2_init),
                    loss_init, loss_upd_init)
        return key, init_val

    def one_step_norm(state):
        i, (pi, mu1, mu2, sigma1, sigma2), loss_i, loss_diff_i = state
        membership_weight = jf.e_step(observed, pi, mu1, mu2, sigma1, sigma2)
        pi_upd, mu1_upd, mu2_upd, sigma1_upd, sigma2_upd = jf.m_step(
            observed, membership_weight
        )
        membership_weight_upd = jf.e_step(
            observed, pi_upd, mu1_upd, mu2_upd, sigma1_upd, sigma2_upd
        )
        loss_upd = jf.compute_vlb(
            observed, pi_upd, mu1_upd, mu2_upd,
            sigma1_upd, sigma2_upd, membership_weight_upd
        )
        loss_diff_upd = jnp.abs(loss_upd - loss_i) / (1e-12 + jnp.abs(loss_i))
        return (i + 1,
                (pi_upd, mu1_upd, mu2_upd, sigma1_upd, sigma2_upd),
                loss_upd, loss_diff_upd)

    def run_one_init(init_state):
        return jax.lax.while_loop(cond_fn, one_step_norm, init_state)

    key, init_val = generate_inits(key)
    final_states  = jax.vmap(run_one_init)(init_val)

    _, (pi_all, mu1_all, mu2_all, sigma1_all, sigma2_all), loss_all, _ = final_states
    best_idx = jnp.argmax(loss_all)
    return (pi_all[best_idx], mu1_all[best_idx], mu2_all[best_idx],
            sigma1_all[best_idx], sigma2_all[best_idx], loss_all[best_idx])


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

    def fit_one(key, observed, feat_diff, spat_diff):
        out      = train_em_jax_old_init(key, observed, num_comp=2, n_init=n_init)
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
                "model":           "Gaussian_pk_Lk_Bk_old_init",
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

    out_path = SHARED_DIR / "shared_osf_2d_uev_4000rep_jax_LkBk_old_init_uev_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}  (total {time.perf_counter()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
