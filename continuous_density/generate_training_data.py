#!/usr/bin/env python3
"""Generate raw EM bias samples on a continuous design.

Saves parameters and raw per-simulation biases only — never a KDE, histogram or
surface.

Usage (from the repo root)::

    PYTHONPATH=. python continuous_density/generate_training_data.py \
        --n-points 4000 --n-simulations 200 --out $DEMIXING_ARTIFACT_ROOT/continuous_density/train.npz

    PYTHONPATH=. python continuous_density/generate_training_data.py \
        --validation --n-simulations 100000 --shard-rows 1 --resume \
        --out .../validation.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import jax
import numpy as np

from continuous_density import design as design_mod
from continuous_density import sim_interface


def _save_npz(path: Path, compress: bool, **arrays):
    """Atomically write an NPZ so an interrupted shard is never considered done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    writer = np.savez_compressed if compress else np.savez
    with open(tmp, 'wb') as handle:
        writer(handle, **arrays)
    os.replace(tmp, path)


def _design_digest(design: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(design).view(np.uint8)).hexdigest()


def _shard_ranges(n_rows: int, shard_rows: int):
    return [(start, min(start + shard_rows, n_rows))
            for start in range(0, n_rows, shard_rows)]


def _valid_shard(path: Path, expected_design: np.ndarray, n_simulations: int) -> bool:
    try:
        with np.load(path) as blob:
            return (np.array_equal(blob['design'], expected_design)
                    and blob['bias'].shape == (len(expected_design), n_simulations, 2))
    except (OSError, ValueError, KeyError):
        return False


def _assemble_shards(paths):
    """Load completed shards in row order into the final bias array."""
    biases = []
    for path in paths:
        with np.load(path) as blob:
            biases.append(np.asarray(blob['bias'], dtype=np.float32))
    return np.concatenate(biases, axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--n-points', type=int, default=4000,
                   help='design points (training mode)')
    p.add_argument('--n-simulations', type=int, default=200,
                   help='EM runs per design point')
    p.add_argument('--n-samples', type=int, default=100,
                   help="observer's internal evidence samples per trial (20 or 100)")
    p.add_argument('--validation', action='store_true',
                   help='use an off-grid validation design instead of training Sobol')
    p.add_argument('--validation-design', choices=['points', 'trajectories'],
                   default='points', help='scattered points or difficult fixed-SD curves')
    p.add_argument('--per-stratum', type=int, default=12)
    p.add_argument('--trajectory-curves', type=int, default=15)
    p.add_argument('--trajectory-points', type=int, default=24)
    p.add_argument('--sd-scale', choices=['log', 'linear'], default='log')
    p.add_argument('--fix-weights', action='store_true')
    p.add_argument('--crn', action='store_true',
                   help='common random numbers across design rows')
    p.add_argument('--block-rows', type=int, default=64)
    p.add_argument('--shard-rows', type=int, default=0,
                   help='rows per resumable output shard; 0 writes one final file')
    p.add_argument('--resume', action='store_true',
                   help='reuse verified completed shards from an interrupted run')
    p.add_argument('--compress', action='store_true',
                   help='compress NPZ output (slower; raw storage is the default)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--design-seed', type=int, default=None,
                   help='design seed; set identically across independent simulation repeats')
    args = p.parse_args()
    design_seed = args.seed if args.design_seed is None else args.design_seed

    if args.validation:
        if args.validation_design == 'points':
            design, strata = design_mod.validation_design(args.per_stratum, seed=design_seed)
        else:
            design, strata = design_mod.stress_trajectory_design(
                args.trajectory_curves, args.trajectory_points, seed=design_seed)
    else:
        design, strata = design_mod.sobol_design(args.n_points, seed=design_seed,
                                                 sd_scale=args.sd_scale), None

    if args.resume and args.shard_rows <= 0:
        p.error('--resume requires --shard-rows')

    print(f"Design: {design.shape[0]} points x {args.n_simulations} simulations "
          f"(n_samples={args.n_samples}, crn={args.crn})")
    t0 = time.time()
    base_key = jax.random.PRNGKey(args.seed + 1)
    shard_paths = []
    if args.shard_rows > 0:
        shard_dir = args.out.with_suffix(args.out.suffix + '.shards')
        shard_dir.mkdir(parents=True, exist_ok=True)
        signature = {
            'design_sha256': _design_digest(design),
            'n_rows': len(design), 'n_simulations': args.n_simulations,
            'n_samples': args.n_samples, 'fix_weights': args.fix_weights,
            'crn': args.crn, 'seed': args.seed, 'design_seed': design_seed,
            'shard_rows': args.shard_rows,
        }
        manifest = shard_dir / 'manifest.json'
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            if previous != signature:
                raise ValueError(f'existing shard manifest does not match this run: {manifest}')
        else:
            manifest.write_text(json.dumps(signature, indent=2) + '\n')

        for start, stop in _shard_ranges(len(design), args.shard_rows):
            path = shard_dir / f'rows_{start:05d}_{stop:05d}.npz'
            expected = design[start:stop]
            if args.resume and path.exists() and _valid_shard(
                    path, expected, args.n_simulations):
                print(f'  reusing {path.name}', flush=True)
            else:
                key = base_key if args.crn else jax.random.fold_in(base_key, start)
                shard_bias = np.asarray(sim_interface.simulate(
                    key, expected, n_simulations=args.n_simulations,
                    n_samples=args.n_samples, fix_weights=args.fix_weights,
                    common_random_numbers=args.crn,
                    block_rows=min(args.block_rows, len(expected)), progress=True),
                    dtype=np.float32)
                _save_npz(path, args.compress, design=expected, bias=shard_bias)
                print(f'  saved {path.name}', flush=True)
            shard_paths.append(path)
        bias = None
    else:
        bias = sim_interface.simulate(
            base_key, design, n_simulations=args.n_simulations,
            n_samples=args.n_samples, fix_weights=args.fix_weights,
            common_random_numbers=args.crn, block_rows=args.block_rows, progress=True)
        bias = np.asarray(bias, dtype=np.float32)
    simulation_elapsed = time.time() - t0
    if shard_paths:
        bias = _assemble_shards(shard_paths)
    n_bad = int(np.sum(~np.isfinite(bias)))
    elapsed = time.time() - t0
    meta = dict(vars(args) | {'out': str(args.out), 'elapsed_s': elapsed,
                              'simulation_elapsed_s': simulation_elapsed,
                              'n_nonfinite': n_bad,
                              'effective_design_seed': design_seed,
                              'spat_diff': sim_interface.SPAT_DIFF,
                              'param_names': list(design_mod.PARAM_NAMES)})
    _save_npz(args.out, args.compress, design=design, bias=bias,
              strata=np.asarray(strata if strata else [], dtype=object),
              meta=json.dumps(meta))
    print(f"Done in {elapsed:.1f}s ({elapsed / design.shape[0]:.2f}s/point); "
          f"{n_bad} non-finite biases")
    print(f"Saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
