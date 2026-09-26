#!/usr/bin/env python3
"""Record the fitting targets the current optimizer builds, as a frozen reference.

Run this **before** target preparation is moved anywhere, and never again after.
The point is to capture what the deployed entry point produces while it is still
the deployed entry point: verifying an extraction against code that has already
been extracted proves only that the new code agrees with itself.

The fixtures are chosen to exercise the parts of target construction that a
careless extraction would quietly change:

* **unequal lengths** -- conditions are padded to a common width for the vmapped
  bias core, and the padding is the repeated last observation. If the density or
  CRPS targets ever start seeing padded rows, they gain a mass clump at the last
  trial's coordinates and their bandwidths shift.
* **sparse** -- few trials spread thinly, where a per-condition bandwidth and a
  pooled one differ most.
* **narrow errors** -- bias values tightly concentrated, which drives Silverman's
  rule toward zero and is where the bandwidth floor is reachable.
* **sign reversal** -- bias that changes sign with feature difference, so a
  transposed or reordered feature axis shows up as a sign error rather than as a
  permutation with the same summary statistics.
* **both periods** -- 180 and 360 degree data, since the angle scaling into model
  space is applied before any target is built.

Usage::

    JAX_PLATFORMS=cpu python tests/record_fitting_targets_golden.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

GOLDEN = Path(__file__).resolve().parent / "data" / "fitting_targets_golden.npz"
CHECKPOINT = ROOT / "pretrained" / "surface_legacy_epoch1425_10ktrain_20samples.pkl"

#: Every recorded array, and the objective each one serves. A target that is not
#: listed here is not covered by the extraction check.
RECORDED = {
    "unified_feat_indices": "expectation: which feature-grid points the target occupies",
    "unified_target_bias": "expectation: circular mean per bin",
    "unified_bias_weights": "expectation: trial-count weights per bin",
    "unified_target_density": "density / density_legacy: empirical signed-mass curve",
    "unified_target_bias_curve": "smoothed_exp: rolling circular mean curve",
    "unified_target_d": "crps variants: expected circular distance per bias bin",
    "unified_fd_weights": "balanced_crps: support mask over feature locations",
    "unified_bias_fd_weights": "bias_weighted_crps: squared smoothed-bias weights",
    "density_target_var": "density refusal: variance of the target curve",
}


def fixtures():
    """Fixed synthetic conditions. Deterministic: the seed is part of the record."""
    rng = np.random.default_rng(20260906)
    cases = {}

    for circ_space in (180, 360):
        half = circ_space / 2.0
        scale = 360.0 / circ_space  # data degrees -> model degrees

        def condition(n, spread, reversal=False, feat_max=None):
            feat = rng.uniform(2.0, feat_max or half, size=n)
            centre = np.where(feat > (feat_max or half) / 2, -1.0, 1.0) if reversal else 1.0
            bias = centre * 6.0 * np.sin(np.radians(feat * scale)) + rng.normal(0, spread, n)
            bias = (bias + half) % circ_space - half
            return np.stack([feat * scale, bias * scale], axis=-1).astype(np.float32)

        cases[f"unequal_{circ_space}"] = {
            "long": condition(400, 12.0),
            "short": condition(60, 12.0),
            "medium": condition(150, 20.0),
        }
        cases[f"sparse_{circ_space}"] = {
            "a": condition(35, 25.0),
            "b": condition(31, 25.0),
        }
        cases[f"narrow_{circ_space}"] = {
            "tight": condition(200, 0.4),
            "tighter": condition(200, 0.15),
        }
        cases[f"reversal_{circ_space}"] = {
            "flip": condition(300, 10.0, reversal=True),
            "plain": condition(300, 10.0),
        }
    return cases


def build(datasets):
    """Construct the optimizer exactly as production does and take its targets."""
    import jax.numpy as jnp
    from grid_based_multi_condition_optimizer_jax_loops import GridBasedMultiConditionOptimizer
    import fit_model_to_data as F

    optimizer = GridBasedMultiConditionOptimizer(
        checkpoint_path=str(CHECKPOINT),
        condition_datasets={name: jnp.asarray(values) for name, values in datasets.items()},
        emp_density_weights_sd=F.DENSITY_CURVE_SPEC["emp_density_weights_sd"],
        density_smoothing_sigma=F.DENSITY_CURVE_SPEC["density_smoothing_sigma"],
        density_bandwidth_rule=F.DENSITY_CURVE_SPEC["density_bandwidth_rule"],
        density_bandwidth_mode=F.DENSITY_CURVE_SPEC["density_bandwidth_mode"],
    )
    return {name: np.asarray(getattr(optimizer, name)) for name in RECORDED}


def main():
    if not CHECKPOINT.exists():
        raise SystemExit(f"needs the production checkpoint at {CHECKPOINT}")
    if GOLDEN.exists():
        raise SystemExit(
            f"{GOLDEN} already exists. This record is the pre-extraction reference and is "
            "written once; re-recording it against extracted code would make the check "
            "vacuous. Delete it deliberately if the target definition itself is changing, "
            "and say so in the commit.")

    payload = {}
    for case, datasets in fixtures().items():
        print(f"building {case} ({len(datasets)} conditions, "
              f"{[len(v) for v in datasets.values()]} trials)", flush=True)
        for name, array in build(datasets).items():
            payload[f"{case}/{name}"] = array
        for cond, values in datasets.items():
            payload[f"{case}/input/{cond}"] = values

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN, **payload)
    print(f"\nwrote {GOLDEN} with {len(payload)} arrays from {len(fixtures())} fixtures")


if __name__ == "__main__":
    main()
