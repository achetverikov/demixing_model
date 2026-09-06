#!/usr/bin/env python3
"""Record the plotting estimators' outputs, as a frozen reference.

Run this **before** the plotting path is routed through the shared prediction
layer, and never again after. The plots are not decoration: row 3 compares a
model circular SD against an empirical one, and the report-order panel pools
distributions before a nonlinear CRPS score. Both are estimator definitions the
transition plan names explicitly -- pool complex moments rather than averaging
SDs or angles, preserve report-order pooling before the nonlinear score -- so a
routing change that moved either would be a silent redefinition, not a refactor.

Same discipline as ``record_fitting_targets_golden.py``: capture what the
deployed code produces while it is still the deployed code. Verifying a rerouted
estimator against code that has already been rerouted proves only that the new
code agrees with itself.

The fixtures exercise the cases where these estimators differ from each other:

* **continuous feature sampling** -- trials spread across a bin, so the pooled
  model SD genuinely mixes several columns.
* **discrete feature levels** -- a Moors-shaped design where every trial in a bin
  sits at one feature difference, so the pooling must collapse to the unpooled
  value rather than inflating the model with columns no trial visited.
* **sparse bins** -- few trials per bin, where the small-sample resultant
  correction matters most (roughly -27% at n=3).
* **an empty bin** -- which must stay NaN on both curves rather than becoming a
  zero that plots as a real measurement.

Usage::

    JAX_PLATFORMS=cpu PYTHONPATH=. python tests/record_plot_estimators_golden.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

GOLDEN = Path(__file__).resolve().parent / "data" / "plot_estimators_golden.npz"
CHECKPOINT = ROOT / "pretrained" / "model_epoch1425_10ktrain_20samples.pkl"

#: Every recorded array and what it pins.
RECORDED = {
    "empirical_sd": "row 3 data curve: binned circular SD with the small-sample correction",
    "bin_weights": "per-bin mixture weights over feature columns, read off the data",
    "pooled_model_sd": "row 3 model curve: SD of the weight-mixed per-column densities",
    "unpooled_model_sd": "the same model SD without pooling, to show pooling changes it",
}


def fixtures():
    """Fixed synthetic conditions. The seed is part of the record."""
    rng = np.random.default_rng(20260906)
    cases = {}

    def condition(feat, spread):
        bias = 8.0 * np.sin(np.radians(feat)) + rng.normal(0.0, spread, len(feat))
        return feat.astype(np.float64), (((bias + 180) % 360) - 180).astype(np.float64)

    cases["continuous"] = condition(rng.uniform(2.0, 178.0, 600), 18.0)
    # Discrete levels: a bin's trials all sit at one feature difference, so the
    # pooled model SD must reduce to the unpooled one.
    cases["discrete_levels"] = condition(
        rng.choice([10.0, 30.0, 60.0, 90.0, 140.0], 600), 18.0)
    cases["sparse"] = condition(rng.uniform(2.0, 178.0, 45), 25.0)
    # Trials confined to the low half, leaving the upper bins empty.
    cases["empty_bins"] = condition(rng.uniform(2.0, 60.0, 300), 15.0)
    return cases


def build(feat, bias):
    import jax.numpy as jnp
    from grid_based_multi_condition_optimizer_jax_loops import GridBasedMultiConditionOptimizer
    from shared.config import config

    import create_unified_subject_plots as plots

    feat_vals = np.asarray(config.create_grid('feat_diff'), dtype=float)

    empirical = np.asarray(plots.compute_empirical_sd_curve(feat, bias), dtype=float)
    weights = np.asarray(plots.compute_feat_bin_weights(feat, feat_vals), dtype=float)

    optimizer = GridBasedMultiConditionOptimizer(
        str(CHECKPOINT), {"dummy": jnp.zeros((4, 2))}, skip_motor_noise=True)
    params = jnp.asarray([[25.0, 40.0, 30.0]], dtype=jnp.float32)
    log_surfaces = optimizer._predict_batch_fixed_size(params, verbosity=0)

    pooled = np.asarray(plots.compute_predicted_sd_curves_batch_pooled(
        log_surfaces, jnp.asarray(weights)[None, :, :]), dtype=float)

    # The unpooled comparison: the model at each bin's centre column alone. Kept
    # so the record shows the pooling is doing something, rather than pinning a
    # number that would look the same either way.
    from surface_simulator_for_predictions.surface_simulator import (
        compute_predicted_sd_curves_batch)
    centres = np.asarray(
        [feat_vals[np.argmax(row)] if row.sum() > 0 else np.nan for row in weights])
    finite = np.isfinite(centres)
    unpooled = np.full(len(centres), np.nan)
    if finite.any():
        values = np.asarray(compute_predicted_sd_curves_batch(
            log_surfaces, jnp.asarray(centres[finite], jnp.float32)), dtype=float)
        unpooled[finite] = values[0]

    return {"empirical_sd": empirical, "bin_weights": weights,
            "pooled_model_sd": pooled[0], "unpooled_model_sd": unpooled}


def main():
    if not CHECKPOINT.exists():
        raise SystemExit(f"needs the production surface checkpoint at {CHECKPOINT}")
    if GOLDEN.exists():
        raise SystemExit(
            f"{GOLDEN} already exists. This is the pre-routing reference and is written "
            "once; re-recording it against rerouted code would make the check vacuous. "
            "Delete it deliberately if an estimator definition is genuinely changing, and "
            "say so in the commit.")

    payload = {}
    for case, (feat, bias) in fixtures().items():
        print(f"building {case} ({len(feat)} trials)", flush=True)
        for name, array in build(feat, bias).items():
            payload[f"{case}/{name}"] = array
        payload[f"{case}/input/feat"] = feat
        payload[f"{case}/input/bias"] = bias

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN, **payload)
    print(f"\nwrote {GOLDEN} with {len(payload)} arrays from {len(fixtures())} fixtures")


if __name__ == "__main__":
    main()
