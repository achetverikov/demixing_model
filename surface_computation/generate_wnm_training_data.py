#!/usr/bin/env python3
"""Generate raw EM bias samples on a continuous design.

Saves parameters and raw per-simulation biases only — never a KDE, histogram or
surface.

Usage (from the repo root)::

    PYTHONPATH=. python -m surface_computation.generate_wnm_training_data \
        --n-points 4000 --n-simulations 200 --out $DEMIXING_ARTIFACT_ROOT/wnm/train.npz

    PYTHONPATH=. python -m surface_computation.generate_wnm_training_data \
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

from surface_computation import wnm_design as design_mod
from surface_computation import wnm_simulation
from surface_computation import likelihood_em, em_batch, cuda_em


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


def _code_digest(paths=None) -> str:
    """Digest every in-repository source that can change generated shards."""
    root = Path(__file__).resolve().parents[1]
    paths = paths or (
        Path(__file__),
        root / 'surface_computation' / 'wnm_design.py',
        root / 'surface_computation' / 'wnm_simulation.py',
        root / 'surface_computation' / 'jax_fit_main.py',
        root / 'surface_computation' / 'jax_fit_functions.py',
        root / 'surface_computation' / 'likelihood_em.py',
        root / 'surface_computation' / 'em_batch.py',
        root / 'surface_computation' / 'em_identity.py',
        root / 'surface_computation' / 'cuda_em.py',
        *(root / 'surface_computation' / name for name in cuda_em.SOURCE_NAMES),
        root / 'shared' / 'wnm.py',
    )
    digest = hashlib.sha256()
    for path in sorted(map(Path, paths), key=lambda item: str(item.resolve())):
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, 'big'))
        digest.update(content)
    return digest.hexdigest()


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
                     and blob['bias'].shape == (len(expected_design), n_simulations, 2)
                     and np.isfinite(blob['bias']).all()
                     and blob['em_converged'].shape == (len(expected_design), n_simulations)
                     and blob['em_converged'].all())
            valid = valid and all(blob['em_'+name].shape == (len(expected_design), n_simulations)
                and np.isfinite(blob['em_'+name]).all() for name in likelihood_em.DIAGNOSTIC_FIELDS)
            valid = valid and not (blob['em_nonfinite_starts'].any() or blob['em_material_decreases'].any())
            valid = valid and blob['em_parameters'].shape == (len(expected_design),n_simulations,5,2)
            valid = valid and np.isfinite(blob['em_parameters']).all()
            if expected_coordinates is not None:
                valid = (valid and 'coordinates' in blob
                         and np.array_equal(blob['coordinates'], expected_coordinates))
            return valid
    except (OSError, ValueError, KeyError):
        return False


def _assemble_chunks(records, n_rows: int, n_simulations: int, return_diagnostics=False):
    """Assemble verified two-axis chunks with only the final array in memory."""
    bias = np.empty((n_rows, n_simulations, 2), dtype=np.float32)
    diagnostics = {}
    for path, row_start, row_stop, sim_start, sim_stop in records:
        with np.load(path) as blob:
            bias[row_start:row_stop, sim_start:sim_stop] = blob['bias']
            if return_diagnostics:
                for name in ('em_'+k for k in (*likelihood_em.DIAGNOSTIC_FIELDS,'parameters')):
                    if name not in diagnostics:
                        diagnostics[name] = np.empty((n_rows, n_simulations)+blob[name].shape[2:], dtype=blob[name].dtype)
                    diagnostics[name][row_start:row_stop, sim_start:sim_stop] = blob[name]
    return (bias, diagnostics) if return_diagnostics else bias


def _load_design_file(path: Path) -> tuple[np.ndarray, list[str] | None]:
    """Load an exact N x 4 design, optionally with one stratum label per row."""
    if path.suffix == '.npy':
        design = np.load(path)
        strata = None
    elif path.suffix == '.npz':
        with np.load(path, allow_pickle=True) as blob:
            design = np.asarray(blob['design'])
            strata = (np.asarray(blob['strata']).astype(str).tolist()
                      if 'strata' in blob and len(blob['strata']) else None)
    else:
        raise ValueError('--design-file must be an .npy or .npz file')
    design = np.asarray(design, dtype=np.float32)
    if design.ndim != 2 or design.shape[1] != len(design_mod.PARAM_NAMES):
        raise ValueError('explicit design must have shape (N, 4)')
    if not np.all(np.isfinite(design)):
        raise ValueError('explicit design contains non-finite values')
    if strata is not None and len(strata) != len(design):
        raise ValueError('explicit strata must have one label per design row')
    return design, strata


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--design-file', type=Path,
                   help='exact .npy/.npz design; overrides generated designs')
    p.add_argument('--n-points', type=int, default=4000,
                   help='design points (training mode)')
    p.add_argument('--training-design', choices=['sobol', 'low-dprime', 'trajectories'],
                   default='sobol', help='training design when --validation is absent')
    p.add_argument('--n-simulations', type=int, default=200,
                   help='EM runs per design point')
    p.add_argument('--n-samples', type=int, default=100,
                   help="observer's total internal evidence samples per trial")
    p.add_argument('--validation', action='store_true',
                   help='use an off-grid validation design instead of training Sobol')
    p.add_argument('--validation-design',
                   choices=['points', 'trajectories', 'low-dprime-trajectories',
                            'phase-a-trajectories', 'uev', 'uev-extension'],
                   default='points',
                   help='scattered points, difficult trajectories, or the UEV figure grid')
    p.add_argument('--per-stratum', type=int, default=12)
    p.add_argument('--trajectory-curves', type=int, default=15)
    p.add_argument('--trajectory-points', type=int, default=24)
    p.add_argument('--uev-feature-step', type=float, default=2.0)
    p.add_argument('--uev-added-sd-feat', type=float, nargs='+', default=[90., 120.])
    p.add_argument('--uev-idf-dprime', type=float, nargs='+', default=[2.])
    p.add_argument('--sd-scale', choices=['log', 'linear'], default='log')
    p.add_argument('--fix-weights', action='store_true')
    p.add_argument('--crn', action='store_true',
                   help='common random numbers across design rows')
    p.add_argument('--crn-within-trajectories', action='store_true',
                   help='reuse random streams only within each labeled trajectory')
    p.add_argument('--block-rows', type=int, default=1,
                   help='design rows simultaneously processed on device')
    p.add_argument('--simulation-chunk', type=int, default=250,
                   help='maximum EM runs per device call; bounds GPU memory')
    p.add_argument('--em-backend', choices=['jax', 'cuda'], default='jax',
                   help='persistent CUDA EM requires nvcc and N=4,10,20,100,500; default: jax')
    p.add_argument('--em-stages', type=int, nargs='+',
                   help='pool unfinished starts across parameter rows at these cumulative caps; end at 5000')
    p.add_argument('--continuation-batch', type=int,
                   help='maximum pooled starts per continuation call; default pools all active starts')
    p.add_argument('--em-max-iter', type=int, default=likelihood_em.MAX_ITER,
                   help='total iteration cap for continuation of a selected capped winner')
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
    if args.crn and args.crn_within_trajectories:
        p.error('--crn and --crn-within-trajectories are mutually exclusive')
    if args.em_max_iter < 1:
        p.error('--em-max-iter must be positive')
    if args.em_backend == 'cuda':
        if args.fix_weights or args.n_samples not in cuda_em.SAMPLE_COUNTS:
            p.error('CUDA EM requires free weights and N=4,10,20,100,500')
        if args.em_stages is not None or args.continuation_batch is not None:
            p.error('CUDA EM uses a persistent work queue; omit --em-stages and --continuation-batch')
        if args.em_max_iter > likelihood_em.MAX_ITER:
            p.error('CUDA EM supports --em-max-iter up to 25000')
    if args.em_stages and (args.em_stages[-1] != min(likelihood_em.FIRST_CAP, args.em_max_iter)
            or args.em_stages[0] < 1 or any(a >= b for a,b in zip(args.em_stages,args.em_stages[1:]))):
        p.error('--em-stages must strictly increase and end at min(5000, --em-max-iter)')
    if args.continuation_batch is not None and (args.continuation_batch < 1 or not args.em_stages):
        p.error('--continuation-batch must be positive and requires --em-stages')

    strata = None
    if args.design_file is not None:
        design, strata = _load_design_file(args.design_file)
    elif args.validation:
        if args.validation_design == 'points':
            design, strata = design_mod.validation_design(args.per_stratum, seed=design_seed)
        elif args.validation_design == 'trajectories':
            design, strata = design_mod.stress_trajectory_design(
                args.trajectory_curves, args.trajectory_points, seed=design_seed)
        elif args.validation_design == 'low-dprime-trajectories':
            design, strata = design_mod.low_dprime_trajectory_design(
                args.trajectory_curves, args.trajectory_points, seed=design_seed)
        elif args.validation_design == 'phase-a-trajectories':
            design, strata = design_mod.phase_a_trajectory_design(
                args.trajectory_curves, args.trajectory_points, seed=design_seed)
        elif args.validation_design == 'uev':
            design, strata = design_mod.uev_design(args.uev_feature_step)
        else:
            design, strata = design_mod.uev_extension_design(
                args.uev_added_sd_feat, args.uev_idf_dprime,
                args.uev_feature_step)
    else:
        if args.training_design == 'sobol':
            design = design_mod.sobol_design(args.n_points, seed=design_seed,
                                             sd_scale=args.sd_scale)
        elif args.training_design == 'low-dprime':
            design = design_mod.low_dprime_augmentation_design(
                args.n_points, seed=design_seed)
            strata = None
        else:
            design, strata = design_mod.trajectory_training_design(
                args.n_points, args.trajectory_points, seed=design_seed,
                sd_scale=args.sd_scale)

    crn_groups = None
    if args.crn_within_trajectories:
        if not strata:
            p.error('--crn-within-trajectories requires a trajectory design')
        _, crn_groups = np.unique(np.asarray(strata), return_inverse=True)

    print(f"Design: {design.shape[0]} points x {args.n_simulations} simulations "
          f"(n_samples={args.n_samples}, crn={args.crn}, "
          f"trajectory_crn={args.crn_within_trajectories}, "
          f"device chunk <= {args.block_rows} rows x {args.simulation_chunk} simulations)")
    t0 = time.time()
    base_key = jax.random.PRNGKey(args.seed + 1)
    shard_dir = args.out.with_suffix(args.out.suffix + '.shards')
    shard_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        'format_version': 3,
        'estimator': likelihood_em.ESTIMATOR_ID,
        'simulation_code_sha256': _code_digest(),
        'design_sha256': _design_digest(design),
        'n_rows': len(design), 'n_simulations': args.n_simulations,
        'n_samples': args.n_samples, 'fix_weights': args.fix_weights,
        'crn': args.crn, 'seed': args.seed, 'design_seed': design_seed,
        'crn_within_trajectories': args.crn_within_trajectories,
        'shard_rows': args.shard_rows,
        'simulation_chunk': args.simulation_chunk,
        'block_rows': args.block_rows,
        'em_stages': args.em_stages,
        'continuation_batch': args.continuation_batch,
        'em_max_iter': args.em_max_iter,
        'em_backend': args.em_backend,
        'em_backend_build': cuda_em.build_identity() if args.em_backend == 'cuda' else None,
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
                chunk_bias, diagnostic = wnm_simulation.simulate(
                    base_key, expected, n_simulations=n_chunk,
                    n_samples=args.n_samples, fix_weights=args.fix_weights,
                    common_random_numbers=args.crn,
                    block_rows=min(args.block_rows, len(expected)), progress=False,
                    row_offset=row_start, simulation_offset=sim_start,
                    common_random_groups=(None if crn_groups is None else
                                          crn_groups[row_start:row_stop]),
                    em_stages=args.em_stages, continuation_batch=args.continuation_batch,
                    em_max_iter=args.em_max_iter, return_diagnostics=True,
                    em_backend=args.em_backend)
                _save_npz(path, args.compress, design=expected, bias=np.asarray(chunk_bias),
                          coordinates=coordinates, **{'em_'+k:np.asarray(v) for k,v in diagnostic.items()})
                action = 'saved'
            records.append((path, row_start, row_stop, sim_start, sim_stop))
            completed_chunks += 1
            if (args.progress_every and
                    (completed_chunks % args.progress_every == 0
                     or completed_chunks == total_chunks)):
                print(f'  chunks {completed_chunks}/{total_chunks}; '
                      f'{action} {path.name}', flush=True)
    simulation_elapsed = time.time() - t0
    bias, diagnostics = _assemble_chunks(records, len(design), args.n_simulations, return_diagnostics=True)
    n_bad = int(np.sum(~np.isfinite(bias)))
    elapsed = time.time() - t0
    meta = dict(vars(args) | {'elapsed_s': elapsed,
                              'simulation_elapsed_s': simulation_elapsed,
                              'n_nonfinite': n_bad,
                              'estimator': likelihood_em.ESTIMATOR_ID,
                              'em_summary': em_batch.summarize({k.removeprefix('em_'):v for k,v in diagnostics.items()}),
                              'effective_design_seed': design_seed,
                              'idf_diff': wnm_simulation.IDF_DIFF,
                              'dprime_definition': 'idf_diff / sd_idf',
                              'param_names': list(design_mod.PARAM_NAMES)})
    meta = {key: str(value) if isinstance(value, Path) else value
            for key, value in meta.items()}
    _save_npz(args.out, args.compress, design=design, bias=bias,
              strata=np.asarray(strata if strata else [], dtype=object),
              meta=json.dumps(meta), **diagnostics)
    print(f"Done in {elapsed:.1f}s ({elapsed / design.shape[0]:.2f}s/point); "
          f"{n_bad} non-finite biases")
    print(f"Saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
