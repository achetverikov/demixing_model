#!/usr/bin/env python3
"""Run actual circular-feature JAX mixture simulations for the OSF d'=1 setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "osf_density_asymmetry"


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_float_list(name: str, default: list[float]) -> list[float]:
    value = os.environ.get(name)
    if not value:
        return default
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    sd_feats = env_float_list(
        "JAX_CIRC_SD_FEATS",
        [env_float("JAX_CIRC_SD_FEAT", 30.0)],
    )
    sd_spat = env_float("JAX_CIRC_SD_SPAT", 30.0)
    spat_dprime = env_float("JAX_CIRC_SPAT_DPRIME", 1.0)
    n_samples = env_int("JAX_CIRC_N_SAMPLES", 100)
    n_simulations = env_int("JAX_CIRC_N_REPS", 10000)
    seed = env_int("JAX_CIRC_SEED", 20260520)
    feat_step = env_int("JAX_CIRC_FEAT_STEP", 15)
    feat_diffs = env_float_list(
        "JAX_CIRC_FEAT_DIFFS",
        sorted(set(np.arange(0, 180 + feat_step, feat_step, dtype=float).tolist() + [50.0])),
    )

    sys.path.insert(0, str(REPO_ROOT / "surface_computation"))
    import jax_fit_functions as jf

    from jax_fit_main import simulate_dual_component_bias_distribution

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "jax_circular_feat_full_raw.csv.gz"
    summary_path = OUT_DIR / "jax_circular_feat_full_summary.csv"

    rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []
    key = jax.random.PRNGKey(seed)
    spat_diff = sd_spat * spat_dprime

    print(
        "Running actual circular-feature JAX simulations: "
        f"sd_feats={','.join(f'{x:g}' for x in sd_feats)} sd_spat={sd_spat:g} dprime={spat_dprime:g} "
        f"n_samples={n_samples} n_reps={n_simulations}",
        flush=True,
    )

    total = len(sd_feats) * len(feat_diffs)
    done = 0
    for sd_feat in sd_feats:
        for feat_diff in feat_diffs:
            key, subkey = jax.random.split(key)
            mu1_bias, _, full_results = simulate_dual_component_bias_distribution(
                subkey,
                sd_feat,
                sd_feat,
                sd_spat,
                float(feat_diff),
                spat_diff,
                n_simulations=n_simulations,
                n_samples=n_samples,
                return_full_results=True,
            )
            bias = np.asarray(mu1_bias).reshape(-1)
            full_np = np.asarray(full_results).reshape(-1, full_results.shape[-1])
            comp_df = pd.DataFrame(
                {
                    "dataset": "Actual JAX circular-feature wrapped EM",
                    "sd_feat": sd_feat,
                    "sd_spat": sd_spat,
                    "spat_dprime": spat_dprime,
                    "n_samples": n_samples,
                    "feat_diff": feat_diff,
                    "bias": bias,
                    "comp_id": full_np[:, jf.ResCol.comp_id],
                    "orig_comp_id": full_np[:, jf.ResCol.orig_comp_id],
                    "weight": full_np[:, jf.ResCol.weight],
                    "mu1_est": full_np[:, jf.ResCol.mu1_est],
                    "mu2_est": full_np[:, jf.ResCol.mu2_est],
                    "sigma1_est": full_np[:, jf.ResCol.sigma1_est],
                    "sigma2_est": full_np[:, jf.ResCol.sigma2_est],
                }
            )
            rows.append(comp_df)
            summary_rows.append(
                {
                    "dataset": "Actual JAX circular-feature wrapped EM",
                    "sd_feat": sd_feat,
                    "sd_spat": sd_spat,
                    "spat_dprime": spat_dprime,
                    "n_samples": n_samples,
                    "feat_diff": feat_diff,
                    "N": int(bias.size),
                    "mean_bias": float(np.mean(bias)),
                    "bias_sd": float(np.std(bias, ddof=1)),
                    "sign_asymmetry": float(np.mean(bias > 0) - np.mean(bias < 0)),
                    "mean_pos": float(np.mean(bias[bias > 0])) if np.any(bias > 0) else np.nan,
                    "mean_neg": float(np.mean(bias[bias < 0])) if np.any(bias < 0) else np.nan,
                }
            )
            done += 1
            print(f"Finished {done}/{total} sd_feat={sd_feat:g} feat_diff={feat_diff:g}", flush=True)

    raw = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("feat_diff")
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {raw_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
