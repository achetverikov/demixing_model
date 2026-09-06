#!/usr/bin/env python3
"""Record the report-order BWCRPS pooling, as a frozen reference.

Run this **before** the pooling is routed anywhere, and never again after.

The estimator is the one the transition plan singles out with "preserve the
report-order distribution pooling before nonlinear CRPS scoring". The order
matters and is not interchangeable: the report orders are pooled into one
predicted distribution per feature location *first*, and the energy score is
applied to that. Scoring each report order separately and averaging the scores
would be a different quantity, because the score is nonlinear in the
distribution -- and it would look entirely plausible.

Two further details this pins, both easy to lose in a rewrite:

* the bias axis is **wrapped, never clipped**, when trials are binned;
* feature locations are weighted by squared smoothed mean bias times a support
  mask, and the routine *raises* rather than returning a number when every weight
  is zero, because a pooled score with no identified feature locations is not a
  small score, it is no score.

Usage::

    JAX_PLATFORMS=cpu PYTHONPATH=. python tests/record_pooled_bwcrps_golden.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

GOLDEN = Path(__file__).resolve().parent / "data" / "pooled_bwcrps_golden.npz"
CHECKPOINT = ROOT / "pretrained" / "model_epoch1425_10ktrain_20samples.pkl"


def fixtures():
    """Report-order pairs, seeded so the record is reproducible."""
    rng = np.random.default_rng(20260906)

    def order(n, amplitude, spread, feat_low=2.0, feat_high=178.0):
        feat = rng.uniform(feat_low, feat_high, n)
        bias = amplitude * np.sin(np.radians(feat)) + rng.normal(0.0, spread, n)
        return np.stack([feat, ((bias + 180) % 360) - 180], axis=-1)

    return {
        # The ordinary case: two report orders with different bias amplitudes,
        # so pooling them is not the same as scoring either alone.
        "two_orders": [order(500, 9.0, 16.0), order(500, -5.0, 22.0)],
        # Unequal support: one order contributes far fewer trials, so the
        # support-weighted pooling has to favour the other.
        "unequal_support": [order(600, 8.0, 15.0), order(80, -6.0, 20.0)],
        # Disjoint feature coverage: each order occupies half the axis, which is
        # where a pooling that ignored support would misweight badly.
        "disjoint_coverage": [order(400, 7.0, 14.0, 2.0, 90.0),
                              order(400, -7.0, 14.0, 90.0, 178.0)],
        # Bias near zero everywhere: the squared-bias weights get small, probing
        # the support mask and the zero-weight refusal without tripping it.
        "flat_bias": [order(400, 0.4, 25.0), order(400, -0.3, 25.0)],
    }


def build(datasets):
    import jax.numpy as jnp
    from grid_based_multi_condition_optimizer_jax_loops import GridBasedMultiConditionOptimizer
    from shared.config import config

    import create_unified_subject_plots as plots

    feat_grid = np.asarray(config.create_grid('feat_diff'), dtype=float)
    bias_grid = np.asarray(config.create_grid('mu1_bias'), dtype=float)
    difference = np.abs(bias_grid[:, None] - bias_grid[None, :])
    distance = np.minimum(difference, 360.0 - difference)

    optimizer = GridBasedMultiConditionOptimizer(
        str(CHECKPOINT), {"dummy": jnp.zeros((4, 2))}, skip_motor_noise=True)
    # One surface per report order, at deliberately different parameters so a
    # pooling that dropped one of them would change the answer.
    params = jnp.asarray([[25.0, 40.0, 30.0], [45.0, 20.0, 30.0]], dtype=jnp.float32)
    log_surfaces = optimizer._predict_batch_fixed_size(params, verbosity=0)[:len(datasets)]

    score = plots._pooled_bias_weighted_crps(
        log_surfaces, datasets, feat_grid, distance, weights_sd=20.0)

    # Scoring each order separately, for the record: the pooled value must not
    # equal the mean of these, or the pooling is doing nothing and the plan's
    # instruction to pool before scoring would be untestable.
    separate = np.array([
        plots._pooled_bias_weighted_crps(
            log_surfaces[index:index + 1], [datasets[index]], feat_grid, distance,
            weights_sd=20.0)
        for index in range(len(datasets))], dtype=float)

    return {"pooled_score": np.asarray([score], dtype=float),
            "separate_scores": separate,
            "log_surfaces_digest": np.asarray(
                [float(np.asarray(log_surfaces).sum())], dtype=float)}


def main():
    if not CHECKPOINT.exists():
        raise SystemExit(f"needs the production surface checkpoint at {CHECKPOINT}")
    if GOLDEN.exists():
        raise SystemExit(
            f"{GOLDEN} already exists. This is the pre-routing reference and is written "
            "once; re-recording it against routed code would make the check vacuous.")

    payload = {}
    for case, datasets in fixtures().items():
        print(f"building {case} ({[len(d) for d in datasets]} trials per order)", flush=True)
        for name, array in build(datasets).items():
            payload[f"{case}/{name}"] = array
        for index, dataset in enumerate(datasets):
            payload[f"{case}/input/order{index}"] = dataset

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN, **payload)
    print(f"\nwrote {GOLDEN} with {len(payload)} arrays from {len(fixtures())} fixtures")


if __name__ == "__main__":
    main()
