#!/usr/bin/env python3
"""Build the density-asymmetry curve cache the exhaustive backend scans.

One curve per ``(sd_spat, sd_feat1, sd_feat2)`` lattice point: the surrogate's
log surface for that triple, collapsed to a signed density-asymmetry curve over
the feat_diff grid. Collapsing at build time is what makes the exhaustive search
affordable -- the scan never touches a surface, only a curve.

Usage (from repo root)::

    python cloud/build_curve_cache.py \\
        --checkpoint-path pretrained/model_epoch1500_10ktrain_20samples.pkl \\
        --out-root results/curve_caches --step 1.0 --verify

Size: at ``--step 1.0`` over [5, 200] the lattice is 196 sd_spat x 196^2 pairs =
7,529,536 curves of 90 float32 = 2.71 GB, plus ~60 MB of per-curve means and
variances. The build is dominated by surrogate evaluations, not by IO.

The cache directory is named by a key digesting everything that changes a curve's
value (checkpoint, both lattices, the feat_diff and mu1_bias grids, the
density-target settings, the encoding), so two builds that differ in any of it
cannot overwrite each other -- see ``model_fit_to_data/curve_cache.py``.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax.numpy as jnp  # noqa: E402

import curve_cache as cc  # noqa: E402
from grid_based_multi_condition_optimizer_jax_loops import (  # noqa: E402
    GridBasedMultiConditionOptimizer,
    generate_nn_density_asymmetry_batch,
    predict_nn,
)
from run_fingerprint import file_sha256  # noqa: E402
from shared.config import config  # noqa: E402
from shared.utils import resolve_input_path  # noqa: E402


def build_curves(optimizer, sd_spat_values, feat_pairs, *, chunk_size=4096, verbosity=1):
    """Generate the curve lattice, one ``sd_spat`` slab at a time.

    Slab-at-a-time so peak memory is one slab of surfaces rather than the whole
    lattice: a slab of 196^2 surfaces at 180x90 float32 is already ~2.4 GB, and
    the full lattice of surfaces would be 470 GB. The curves that survive are
    three orders of magnitude smaller, which is the entire point of the cache.

    NN parameter order is ``[sd_feat1, sd_feat2, sd_spat]`` -- the order
    ``fit_hierarchical_grid`` feeds the surrogate. Getting it wrong here would
    misattribute every curve while leaving every checksum valid.
    """
    n_points = len(config.create_grid('feat_diff'))
    curves = np.empty((len(sd_spat_values), len(feat_pairs), n_points), dtype=np.float32)

    for spat_index, sd_spat in enumerate(sd_spat_values):
        started = time.time()
        for start in range(0, len(feat_pairs), chunk_size):
            chunk = feat_pairs[start:start + chunk_size]
            nn_params = jnp.column_stack([
                jnp.asarray(chunk[:, 0]),                       # sd_feat1
                jnp.asarray(chunk[:, 1]),                       # sd_feat2
                jnp.full(len(chunk), float(sd_spat)),           # sd_spat
            ])
            surfaces = predict_nn(optimizer, nn_params)
            asymmetry = generate_nn_density_asymmetry_batch(
                surfaces,
                weights_sd=optimizer.emp_density_weights_sd,
                smoothing_sigma=optimizer.density_smoothing_sigma,
            )
            curves[spat_index, start:start + len(chunk)] = np.asarray(asymmetry, dtype=np.float32)
        if verbosity > 0:
            elapsed = time.time() - started
            remaining = elapsed * (len(sd_spat_values) - spat_index - 1)
            print(f"  sd_spat {sd_spat:6.1f}  ({spat_index + 1}/{len(sd_spat_values)})  "
                  f"{elapsed:5.1f}s  ETA {remaining / 60:.1f}m", flush=True)
    return curves


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint-path', required=True,
                        help='Surrogate checkpoint. Enters the cache key by content hash.')
    parser.add_argument('--out-root', required=True,
                        help='Directory holding cache directories, one per key.')
    parser.add_argument('--step', type=float, default=1.0,
                        help='Lattice step, in degrees, for both sd_spat and the features.')
    parser.add_argument('--low', type=float, default=None,
                        help='Lattice lower bound (default: the surfaces\' own param_grid_low).')
    parser.add_argument('--high', type=float, default=None,
                        help='Lattice upper bound (default: param_range_high).')
    parser.add_argument('--chunk-size', type=int, default=4096,
                        help='Surrogate batch size within a slab.')
    parser.add_argument('--verify', action='store_true',
                        help='Re-hash every array after writing and compare to the manifest. '
                             'Costs a full read; worth it for a cache that will be reused.')
    parser.add_argument('--results-dir', default='results',
                        help='Base directory for resolving a relative checkpoint path.')
    args = parser.parse_args()

    low = config.param_grid_low if args.low is None else args.low
    high = config.param_range_high if args.high is None else args.high
    checkpoint = resolve_input_path(args.checkpoint_path, args.results_dir)

    sd_spat_values = cc.lattice(low, high, args.step)
    feat_values = cc.lattice(low, high, args.step)
    feat_pairs = cc.feat_pair_lattice(feat_values)

    cache_key = cc.compute_cache_key(
        checkpoint_sha256=file_sha256(checkpoint),
        sd_spat_grid=(low, high, args.step),
        feat_grid=(low, high, args.step),
        feat_diff_range=config.feat_diff_range,
        feat_diff_step=config.feat_diff_step,
        mu1_bias_range=config.mu1_bias_range,
        mu1_bias_step=config.mu1_bias_step,
        emp_density_weights_sd=20.0,
        density_smoothing_sigma=None,
        encoding={"kind": "exact", "dtype": "float32"},
    )
    cache_dir = cc.cache_dir_for(args.out_root, cache_key)

    n_points = len(config.create_grid('feat_diff'))
    n_curves = len(sd_spat_values) * len(feat_pairs)
    print(f"Checkpoint:  {checkpoint}")
    print(f"Lattice:     {len(sd_spat_values)} sd_spat x {len(feat_pairs)} feature pairs "
          f"= {n_curves:,} curves of {n_points} points")
    print(f"Size:        {n_curves * n_points * 4 / 2**30:.2f} GB of curves "
          f"+ {n_curves * 8 / 2**30:.2f} GB of statistics")
    print(f"Cache key:   {cache_key}")
    print(f"Destination: {cache_dir}")

    if (cache_dir / cc.COMPLETION_MARKER).exists():
        print("Already built; nothing to do. (Delete the directory to force a rebuild.)")
        return

    optimizer = GridBasedMultiConditionOptimizer(str(checkpoint), None, skip_motor_noise=True)

    started = time.time()
    curves = build_curves(optimizer, sd_spat_values, feat_pairs, chunk_size=args.chunk_size)
    print(f"Generated in {(time.time() - started) / 60:.1f}m; writing...")

    cc.write_cache(
        cache_dir, cache_key=cache_key, sd_spat_values=sd_spat_values,
        feat_pairs=feat_pairs, curves=curves,
        manifest_extra={
            "checkpoint": str(checkpoint),
            "sd_spat_grid": [low, high, args.step],
            "feat_grid": [low, high, args.step],
            "emp_density_weights_sd": 20.0,
            "density_smoothing_sigma": None,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    if args.verify:
        cc.read_manifest(cache_dir, verify=True)
        print("Verified: every array matches its manifest checksum.")
    print(f"Done in {(time.time() - started) / 60:.1f}m -> {cache_dir}")


if __name__ == '__main__':
    main()
