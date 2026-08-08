#!/usr/bin/env python3
"""Generate raw EM bias samples on a continuous design.

Saves parameters and raw per-simulation biases only — never a KDE, histogram or
surface.

Usage (from the repo root)::

    PYTHONPATH=. python continuous_density/generate_training_data.py \
        --n-points 4000 --n-simulations 200 --out $DEMIXING_ARTIFACT_ROOT/continuous_density/train.npz

    PYTHONPATH=. python continuous_density/generate_training_data.py \
        --validation --n-simulations 100000 --simulation-chunk 250 \
        --shard-rows 1 --resume \
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


def _chunk_ranges(size: int, chunk_size: int):
    return [(start, min(start + chunk_size, size))
            for start in range(0, size, chunk_size)]


def _shard_ranges(n_rows: int, shard_rows: int):
    """Backward-compatible name for row chunk ranges."""
    return _chunk_ranges(n_rows, shard_rows)


def _valid_shard(path: Path, expected_design: np.ndarray, n_simulations: int,
                 expected_coordinates=None) -> bool:
    try:
        with np.load(path) as blob:
            valid = (np.array_equal(blob['design'], expected_design)
                     and blob['bias'].shape == (len(expected_design), n_simulations, 2))
            if expected_coordinates is not None:
                valid = (valid and 'coordinates' in blob
                         and np.array_equal(blob['coordinates'], expected_coordinates))
            return valid
    except (OSError, ValueError, KeyError):
        return False


def _assemble_chunks(records, n_rows: int, n_simulations: int):
    """Assemble verified two-axis chunks with only the final array in memory."""
    bias = np.empty((n_rows, n_simulations, 2), dtype=np.float32)
    for path, row_start, row_stop, sim_start, sim_stop in records:
        with np.load(path) as blob:
            bias[row_start:row_stop, sim_start:sim_stop] = blob['bias']
    return bias


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
    p.add_argument('--block-rows', type=int, default=1,
                   help='design rows simultaneously processed on device')
    p.add_argument('--simulation-chunk', type=int, default=250,
                   help='maximum EM runs per device call; bounds GPU memory')
    p.add_argument('--shard-rows', type=int, default=16,
                   help='design rows per resumable disk shard')
    p.add_argument('--resume', action='store_true',
                   help='reuse verified completed shards from an interrupted run')
    p.add_argument('--compress', action='store_true',
                   help='compress NPZ output (slower; raw storage is the default)')
    p.add_argument('--progress-every', type=int, default=100,
                   help='report every N completed chunks (0 disables progress)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--design-seed', type=int, default=None,
                   help='design seed; set identically across independent simulation repeats')
    args = p.parse_args()
    design_seed = args.seed if args.design_seed is None else args.design_seed

    if args.n_simulations < 1 or args.simulation_chunk < 1:
        p.error('--n-simulations and --simulation-chunk must be positive')
    if args.shard_rows < 1 or args.block_rows < 1:
        p.error('--shard-rows and --block-rows must be positive')
    if args.progress_every < 0:
        p.error('--progress-every cannot be negative')
    if args.block_rows > args.shard_rows:
        p.error('--block-rows cannot exceed --shard-rows')

    if args.validation:
        if args.validation_design == 'points':
            design, strata = design_mod.validation_design(args.per_stratum, seed=design_seed)
        else:
            design, strata = design_mod.stress_trajectory_design(
                args.trajectory_curves, args.trajectory_points, seed=design_seed)
    else:
        design, strata = design_mod.sobol_design(args.n_points, seed=design_seed,
                                                 sd_scale=args.sd_scale), None

    print(f"Design: {design.shape[0]} points x {args.n_simulations} simulations "
          f"(n_samples={args.n_samples}, crn={args.crn}, "
          f"device chunk <= {args.block_rows} rows x {args.simulation_chunk} simulations)")
    t0 = time.time()
    base_key = jax.random.PRNGKey(args.seed + 1)
    shard_dir = args.out.with_suffix(args.out.suffix + '.shards')
    shard_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        'format_version': 2,
        'design_sha256': _design_digest(design),
        'n_rows': len(design), 'n_simulations': args.n_simulations,
        'n_samples': args.n_samples, 'fix_weights': args.fix_weights,
        'crn': args.crn, 'seed': args.seed, 'design_seed': design_seed,
        'shard_rows': args.shard_rows,
        'simulation_chunk': args.simulation_chunk,
    }
    manifest = shard_dir / 'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous != signature:
            raise ValueError(f'existing shard manifest does not match this run: {manifest}')
    else:
        manifest.write_text(json.dumps(signature, indent=2) + '\n')

    records = []
    total_chunks = (len(_chunk_ranges(len(design), args.shard_rows))
                    * len(_chunk_ranges(args.n_simulations, args.simulation_chunk)))
    completed_chunks = 0
    for row_start, row_stop in _chunk_ranges(len(design), args.shard_rows):
        expected = design[row_start:row_stop]
        for sim_start, sim_stop in _chunk_ranges(
                args.n_simulations, args.simulation_chunk):
            path = shard_dir / (f'rows_{row_start:05d}_{row_stop:05d}_'
                                f'sims_{sim_start:06d}_{sim_stop:06d}.npz')
            n_chunk = sim_stop - sim_start
            coordinates = np.asarray([row_start, row_stop, sim_start, sim_stop])
            if args.resume and path.exists() and _valid_shard(
                    path, expected, n_chunk, coordinates):
                action = 'reused'
            else:
                chunk_bias = np.asarray(sim_interface.simulate(
                    base_key, expected, n_simulations=n_chunk,
                    n_samples=args.n_samples, fix_weights=args.fix_weights,
                    common_random_numbers=args.crn,
                    block_rows=min(args.block_rows, len(expected)), progress=False,
                    row_offset=row_start, simulation_offset=sim_start), dtype=np.float32)
                _save_npz(path, args.compress, design=expected, bias=chunk_bias,
                          coordinates=coordinates)
                action = 'saved'
            records.append((path, row_start, row_stop, sim_start, sim_stop))
            completed_chunks += 1
            if (args.progress_every and
                    (completed_chunks % args.progress_every == 0
                     or completed_chunks == total_chunks)):
                print(f'  chunks {completed_chunks}/{total_chunks}; '
                      f'{action} {path.name}', flush=True)
    simulation_elapsed = time.time() - t0
    bias = _assemble_chunks(records, len(design), args.n_simulations)
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
