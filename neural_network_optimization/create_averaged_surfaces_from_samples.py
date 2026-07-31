#!/usr/bin/env python3
"""
Create Averaged Surfaces from Samples
=====================================

This script creates averaged surfaces directly from samples using 2D normal weighted KDE.
It combines mirrored samples to create higher-quality surfaces with smooth likelihood transitions.

For each unique pair (sf1, sf2, sp):
- Loads sample files for both (sf1, sf2, sp) and (sf2, sf1, sp) if available
- Combines samples with appropriate mirroring for mu1_comp1 and mu1_comp2
- Uses 2D normal weighted KDE to create smooth density surfaces
- Saves the resulting averaged surfaces

Only processes L1 parameter combinations (coarse grid).
"""

import pickle
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import re
import gzip
from shared.utils import AveragedSurface, resolve_input_path, resolve_results_path, get_git_commit
from shared.config import Config
from shared.averaging import prefilter_isolated_samples, make_averaging_functions
from concurrent.futures import ThreadPoolExecutor
import threading

config = Config()
config.samples_folder = './sim_samples_10k_20samples'
output_folder = "averaged_surfaces_10k_20samples" 

def find_sample_file(folder: Path, sf1: float, sf2: float, sp: float) -> Optional[Path]:
    """Find sample file matching the given parameters.

    For off-diagonal (sf1 != sf2) files this returns the first match.
    For diagonal files use find_diagonal_run_files instead.
    Excludes _r0/_r1 run-indexed files so diagonal runs don't accidentally
    satisfy an off-diagonal lookup.
    """
    pattern = re.compile(rf"samples_sf1_{sf1:.1f}_sf2_{sf2:.1f}_sp_{sp:.1f}_[^r].*\.pkl\.gz")
    for file in folder.glob("samples_sf1_*_sf2_*_sp_*.pkl.gz"):
        if pattern.match(file.name):
            return file
    return None


def find_diagonal_run_files(folder: Path, sf1: float, sp: float) -> List[Path]:
    """Find the two independent-run sample files for a diagonal (sf1==sf2) combination.

    Returns a list of matching files ordered by run index (r0 before r1).
    Returns an empty list if neither run file exists.
    """
    pattern = re.compile(rf"samples_sf1_{sf1:.1f}_sf2_{sf1:.1f}_sp_{sp:.1f}_r\d_.*\.pkl\.gz")
    files = sorted(
        (f for f in folder.glob("samples_sf1_*_sf2_*_sp_*.pkl.gz") if pattern.match(f.name)),
        key=lambda f: f.name
    )
    return files

def load_sample_data(file_path: Path) -> Dict:
    """Load sample data from compressed pickle file."""
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)


def stub_sample_file(file_path: Path) -> None:
    """Replace a sample file with a tiny stub so grid tracking still works.

    The stub keeps the filename (and therefore the hash that simulated_samples_grid.py
    uses for progress tracking) but replaces the large sample arrays with empty ones,
    reducing file size from ~6 MB to ~1 KB.  simulated_samples_grid._samples_exist()
    checks for 'parameters' and 'mu1_samples'/'mu2_samples' keys, so the stub still
    passes that check and the combination will not be re-computed.
    """
    try:
        with gzip.open(file_path, 'rb') as f:
            original = pickle.load(f)
        stub = {
            'parameters': original.get('parameters', {}),
            'mu1_samples': np.empty((0,), dtype=np.float16),
            'mu2_samples': np.empty((0,), dtype=np.float16),
            'stub': True,
        }
        # Write atomically: temp file beside the target, then rename, so
        # simulated_samples_grid never sees a half-written stub.
        tmp = file_path.with_suffix('.pkl.gz.tmp')
        with gzip.open(tmp, 'wb') as f:
            pickle.dump(stub, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(file_path)
    except Exception as e:
        print(f"Warning: could not stub {file_path.name}: {e}")

def is_l1_surface(sf1: float, sf2: float, sp: float) -> bool:
    """Check if surface parameters correspond to L1 grid (coarse grid)."""
    step = config.param_step
    low = config.param_range_low

    def is_on_grid(val):
        return abs(val - round((val - low) / step) * step - low) < 1e-6

    return is_on_grid(sf1) and is_on_grid(sf2) and is_on_grid(sp)


def get_l1_parameter_combinations(folder: Path) -> List[Tuple[float, float, float]]:
    """Get all L1 parameter combinations from sample files."""
    pattern = re.compile(r"samples_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)_.*\.pkl\.gz")
    combinations = set()
    
    for file in folder.glob("samples_sf1_*_sf2_*_sp_*.pkl.gz"):
        match = pattern.match(file.name)
        if match:
            sf1, sf2, sp = map(float, match.groups())
            if is_l1_surface(sf1, sf2, sp):
                combinations.add((sf1, sf2, sp))
    
    return sorted(list(combinations))


def get_all_parameter_combinations(folder: Path) -> List[Tuple[float, float, float]]:
    """Get all parameter combinations from sample files (no grid filtering)."""
    pattern = re.compile(r"samples_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)_.*\.pkl\.gz")
    combinations = set()

    for file in folder.glob("samples_sf1_*_sf2_*_sp_*.pkl.gz"):
        match = pattern.match(file.name)
        if match:
            combinations.add(tuple(map(float, match.groups())))

    return sorted(list(combinations))
def get_surface_filename(sf1: float, sf2: float, sp: float) -> str:
    """Get canonical filename for averaged surface."""
    canonical_sf1 = min(sf1, sf2)
    canonical_sf2 = max(sf1, sf2)
    return f"averaged_sf1_{canonical_sf1:.1f}_sf2_{canonical_sf2:.1f}_sp_{sp:.1f}.pkl"

def get_existing_surfaces(output_folder: Path) -> set:
    """Get set of existing surface parameter combinations."""
    pattern = re.compile(r"averaged_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)\.pkl")
    existing = set()
    
    if not output_folder.exists():
        return existing
    
    for file in output_folder.glob("averaged_sf1_*_sf2_*_sp_*.pkl"):
        match = pattern.match(file.name)
        if match:
            sf1, sf2, sp = map(float, match.groups())
            # Store in canonical form (sf1 <= sf2) and both permutations
            canonical_sf1, canonical_sf2 = min(sf1, sf2), max(sf1, sf2)
            existing.add((canonical_sf1, canonical_sf2, sp))
            existing.add((canonical_sf2, canonical_sf1, sp))  # Both permutations
    
    return existing

def save_averaged_surface(averaged_surface: AveragedSurface, sf1: float, sf2: float, sp: float,
                         output_folder: Path,
                         source_files_meta: List[Dict] = None) -> None:
    """Save averaged surface to output folder.

    Stores parameters, the surface object, creation timestamp, the git commit
    of the averaging script, and provenance metadata for each source sample file.
    """
    filename = get_surface_filename(sf1, sf2, sp)
    output_file = output_folder / filename
    print(f"Saving averaged surface to {output_file}")

    canonical_sf1 = min(sf1, sf2)
    canonical_sf2 = max(sf1, sf2)

    averaging_commit = get_git_commit()
    averaged_data = {
        'parameters': {
            'sd_feat1': canonical_sf1,
            'sd_feat2': canonical_sf2,
            'sd_spat': sp
        },
        'surface': averaged_surface,
        'creation_timestamp': np.datetime64('now').astype(str),
        'git_commit': averaging_commit,
        'source_files': source_files_meta or [],
    }

    with open(output_file, 'wb') as f:
        pickle.dump(averaged_data, f)

    print(f"Saved: {filename} (from {averaged_surface.n_sample_files_used} sample files)")

def process_single_surface(params_tuple, input_path, output_path, compiled_vmap_circular, compiled_vmap_spatial,
                          feat_diff_grid, mu1_bias_grid, mu2_bias_grid, feat_diff_steps,
                          bias_bandwidth, feat_bandwidth, ref_sum, stub_processed_samples=False,
                          dry_run=False):
    """Load, combine, KDE-smooth, and save one averaged surface for (sf1, sf2, sp).

    Skips if the mirror file (sf2, sf1, sp) is absent for sf1 != sf2 cases so
    both files can be processed together on a later run.  With stub_processed_samples
    the raw sample files are replaced with tiny stubs after saving.  With dry_run
    nothing is written to disk.  Returns True on success, False if skipped/errored.
    """
    sf1, sf2, sp = params_tuple
    
    try:
        thread_id = threading.current_thread().ident
        print(f"[Thread {thread_id}] Processing ({sf1}, {sf2}, {sp})...")
        
        n_expected = len(feat_diff_steps)
        mirror_samples = None
        mirror_file = None

        def _trim(data):
            if data['mu1_samples'].shape[0] > n_expected:
                return {**data,
                        'mu1_samples': data['mu1_samples'][1:],
                        'mu2_samples': data['mu2_samples'][1:]}
            return data

        if sf1 != sf2:
            # Off-diagonal: find canonical file + mirror file.
            original_file = find_sample_file(input_path, sf1, sf2, sp)
            if not original_file:
                print(f"[Thread {thread_id}] Original samples not found for ({sf1}, {sf2}, {sp})")
                return False
            print(f"[Thread {thread_id}] Loading original samples from {original_file}")
            original_samples = _trim(load_sample_data(original_file))

            mirror_file = find_sample_file(input_path, sf2, sf1, sp)
            if not mirror_file:
                print(f"[Thread {thread_id}] Mirror file not yet available for ({sf1}, {sf2}, {sp}) — skipping")
                return False
            print(f"[Thread {thread_id}] Loading mirror samples from {mirror_file}")
            mirror_samples = _trim(load_sample_data(mirror_file))
        else:
            # Diagonal: prefer the run-indexed pair (_r0/_r1); fall back to a
            # single plain file for older data that has no run index.
            run_files = find_diagonal_run_files(input_path, sf1, sp)
            if len(run_files) >= 2:
                original_file = run_files[0]
                print(f"[Thread {thread_id}] Loading original samples from {original_file}")
                original_samples = _trim(load_sample_data(original_file))
                mirror_file = run_files[1]
                print(f"[Thread {thread_id}] Loading diagonal run-1 samples from {mirror_file}")
                mirror_samples = _trim(load_sample_data(mirror_file))
            else:
                # Single plain file (old format).
                original_file = find_sample_file(input_path, sf1, sf2, sp)
                if not original_file:
                    print(f"[Thread {thread_id}] Original samples not found for ({sf1}, {sf2}, {sp})")
                    return False
                print(f"[Thread {thread_id}] Loading original samples from {original_file}")
                original_samples = _trim(load_sample_data(original_file))
        
        # Process samples and create surface directly
        print(f"[Thread {thread_id}] Creating averaged surface for ({sf1}, {sf2}, {sp})")
        
        # Combine all mu1 samples along axis 2 (following surfaces_from_samples.py approach)
        if mirror_samples is not None:
            mu1_combined = jnp.concatenate([
                original_samples['mu1_samples'], 
                jnp.flip(mirror_samples['mu1_samples'], axis=2)
            ], axis=1)
        else:
            mu1_combined = original_samples['mu1_samples']
        
        # Combine all mu2 samples (same approach as mu1)
        if mirror_samples is not None:
            mu2_combined = jnp.concatenate([
                original_samples['mu2_samples'], 
                mirror_samples['mu2_samples']
            ], axis=1)
        else:
            mu2_combined = original_samples['mu2_samples']
        
        # Apply prefiltering to remove isolated samples
        print(f"[Thread {thread_id}] Applying prefiltering to remove isolated samples...")
        
        # For sf1 == sf2 case, combine both mu1 components before filtering
        if sf1 == sf2:
            print(f"[Thread {thread_id}] sf1 == sf2, combining mu1 components...")
            # Combine both components into one dataset
            mu1_all_samples = jnp.concatenate([mu1_combined[:, :, 0], mu1_combined[:, :, 1]], axis=1)
            mu1_filtered, removed_mu1 = prefilter_isolated_samples(mu1_all_samples)
            
            print(f"[Thread {thread_id}] Prefiltering removed: mu1_combined={removed_mu1}")
        else:
            # Filter components separately for sf1 != sf2
            mu1_comp1_filtered, removed_mu1_comp1 = prefilter_isolated_samples(mu1_combined[:, :, 0])
            mu1_comp2_filtered, removed_mu1_comp2 = prefilter_isolated_samples(mu1_combined[:, :, 1])
            
            # Reconstruct mu1_combined with filtered samples
            mu1_combined = jnp.stack([mu1_comp1_filtered, mu1_comp2_filtered], axis=2)
            
            print(f"[Thread {thread_id}] Prefiltering removed: mu1_comp1={removed_mu1_comp1}, mu1_comp2={removed_mu1_comp2}")
        
        # Filter mu2 samples (both components together)
        mu2_all_samples_orig = mu2_combined.reshape(mu2_combined.shape[0], -1)
        mu2_filtered, removed_mu2 = prefilter_isolated_samples(mu2_all_samples_orig)
        mu2_combined = mu2_filtered.reshape(mu2_combined.shape)
        
        print(f"[Thread {thread_id}] Prefiltering removed: mu2={removed_mu2}")
        
        # Create density surfaces using vmap functions
        if sf1 == sf2:
            # Create single surface for both components
            print(f"[Thread {thread_id}] Computing combined mu1 surface (sf1 == sf2)...")
            mu1_surface = compiled_vmap_circular(
                jnp.arange(len(feat_diff_steps)), mu1_filtered
            ).T
            mu1_comp1_surface = mu1_surface
            mu1_comp2_surface = mu1_surface
        else:
            # Create separate surfaces for each component
            print(f"[Thread {thread_id}] Computing mu1_comp1 surface...")
            mu1_comp1_surface = compiled_vmap_circular(
                jnp.arange(len(feat_diff_steps)), mu1_combined[:, :, 0]
            ).T
            
            print(f"[Thread {thread_id}] Computing mu1_comp2 surface...")
            mu1_comp2_surface = compiled_vmap_circular(
                jnp.arange(len(feat_diff_steps)), mu1_combined[:, :, 1]
            ).T
        
        print(f"[Thread {thread_id}] Computing mu2 surface...")
        # For mu2, use all samples combined (both components have same spatial parameters)
        mu2_all_samples = mu2_combined.reshape(mu2_combined.shape[0], -1)  # Flatten all samples
        mu2_surface = compiled_vmap_spatial(
            jnp.arange(len(feat_diff_steps)), mu2_all_samples
        ).T
        
        # KDE already returns log probabilities, no conversion needed
        
        n_files_used = 1 + (1 if mirror_samples is not None else 0)
        
        averaged_surface = AveragedSurface(
            feat_diff_grid=feat_diff_grid,
            mu1_bias_grid=mu1_bias_grid,
            mu2_bias_grid=mu2_bias_grid,
            mu1_comp1_surface=mu1_comp1_surface,
            mu1_comp2_surface=mu1_comp2_surface,
            mu2_comp1_surface=mu2_surface,
            mu2_comp2_surface=mu2_surface,
            n_sample_files_used=n_files_used,
            kde_parameters={
                'bias_bandwidth': bias_bandwidth,
                'feat_bandwidth': feat_bandwidth,
                # No weight threshold needed with fixed window approach
            }
        )
        
        print(f"[Thread {thread_id}] Averaged surface created for ({sf1}, {sf2}, {sp})")
        if dry_run:
            mirror_note = f" + {mirror_file.name}" if mirror_file else ""
            print(f"[Thread {thread_id}] DRY RUN: would save averaged surface for ({sf1}, {sf2}, {sp})"
                  f" from {original_file.name}{mirror_note}"
                  + (" [would stub both]" if stub_processed_samples else ""))
            return True

        # Collect source file provenance
        source_files_meta = [{'filename': original_file.name,
                               'parameters': original_samples.get('parameters', {}),
                               'computation_time': original_samples.get('computation_time'),
                               'timestamp': original_samples.get('timestamp')}]
        if mirror_samples is not None:
            source_files_meta.append({'filename': mirror_file.name,
                                      'parameters': mirror_samples.get('parameters', {}),
                                      'computation_time': mirror_samples.get('computation_time'),
                                      'timestamp': mirror_samples.get('timestamp')})

        # Save averaged surface
        save_averaged_surface(averaged_surface, sf1, sf2, sp, output_path,
                              source_files_meta=source_files_meta)
        print(f"[Thread {thread_id}] Completed ({sf1}, {sf2}, {sp})")

        # Optionally replace raw sample files with tiny stubs to free disk space.
        # We only reach here when both files exist (or sf1==sf2), so it is safe to stub both.
        if stub_processed_samples:
            stub_sample_file(original_file)
            print(f"[Thread {thread_id}] Stubbed {original_file.name}")
            if mirror_samples is not None:
                stub_sample_file(mirror_file)
                print(f"[Thread {thread_id}] Stubbed {mirror_file.name}")

        return True

    except Exception as e:
        thread_id = threading.current_thread().ident
        print(f"[Thread {thread_id}] Error processing ({sf1}, {sf2}, {sp}): {e}")
        return False


def create_all_averaged_surfaces(input_folder: str = None, output_folder: str = None,
                                bias_bandwidth: float = 0.075, feat_bandwidth: float = 5.0,
                                n_workers: int = 2, results_dir: str = "results",
                                include_all_params: bool = False,
                                stub_processed_samples: bool = False,
                                dry_run: bool = False) -> None:
    """Main function to create all averaged surfaces from samples.

    Args:
        feat_bandwidth: Gaussian SD in feat_diff grid steps. The CLI default of
            3.0 corresponds to 6 degrees on the standard 2-degree grid — but
            THIS function's default is 5.0 (10 degrees), so programmatic
            callers omitting the argument get wider smoothing than the
            production surfaces (see MODEL_PIPELINE_FOR_AGENTS.md D.6).
        stub_processed_samples: If True, replace raw sample files with tiny stubs after
            averaging.  The stubs preserve the filename/hash so simulated_samples_grid.py
            still counts those combinations as done, but occupy ~1 KB instead of ~6 MB.
            Default False — sample files are left untouched.
        dry_run: If True, print what would be done without writing any files.
    """
    
    if input_folder is None:
        input_folder = config.samples_folder
    if output_folder is None:
        output_folder = "averaged_surfaces_from_samples"
    
    input_path = resolve_input_path(input_folder, results_dir)
    output_path = resolve_results_path(output_folder, results_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating averaged surfaces from samples in: {input_path}")
    print(f"Output folder: {output_path}")
    print(f"KDE parameters: bias_bw={bias_bandwidth}, feat_bw={feat_bandwidth}")
    print("Compiling shared bounded-KDE functions...")
    avg_fns = make_averaging_functions(config, bias_bandwidth, feat_bandwidth)
    compiled_vmap_circular = avg_fns['vmap_circular']
    compiled_vmap_spatial = avg_fns['vmap_spatial']
    feat_diff_steps = avg_fns['feat_diff_steps']
    feat_diff_grid = avg_fns['feat_diff_grid']
    mu1_bias_grid = avg_fns['mu1_bias_grid']
    mu2_bias_grid = avg_fns['mu2_bias_grid']
    ref_sum = avg_fns['ref_sum']
    
    # Get parameter combinations (L1-only by default)
    combinations = (
        get_all_parameter_combinations(input_path)
        if include_all_params
        else get_l1_parameter_combinations(input_path)
    )
    print(f"Found {len(combinations)} L1 parameter combinations")
    
    # Get existing surfaces to avoid reprocessing
    existing_surfaces = get_existing_surfaces(output_path)
    print(f"Found {len(existing_surfaces)} existing averaged surfaces")
    
    # Filter to only sf1 <= sf2 combinations and exclude existing ones
    unique_combinations = []
    skipped_existing = 0
    for sf1, sf2, sp in combinations:
        if sf1 <= sf2:  # Only process canonical pairs
            if (sf1, sf2, sp) not in existing_surfaces:
                unique_combinations.append((sf1, sf2, sp))
            else:
                skipped_existing += 1
    
    print(f"Filtered to {len(unique_combinations)} unique combinations (sf1 <= sf2)")
    print(f"Skipped {skipped_existing} existing surfaces")
    print(f"Using {n_workers} worker threads for parallel processing")
    
    # Process in parallel using ThreadPoolExecutor
    created_count = 0
    error_count = 0
    
    # Create a partial function with all the shared parameters
    from functools import partial
    if stub_processed_samples:
        print("stub-samples enabled: raw sample files will be replaced with stubs after averaging.")

    worker_func = partial(
        process_single_surface,
        input_path=input_path,
        output_path=output_path,
        compiled_vmap_circular=compiled_vmap_circular,
        compiled_vmap_spatial=compiled_vmap_spatial,
        feat_diff_grid=feat_diff_grid,
        mu1_bias_grid=mu1_bias_grid,
        mu2_bias_grid=mu2_bias_grid,
        feat_diff_steps=feat_diff_steps,
        bias_bandwidth=bias_bandwidth,
        feat_bandwidth=feat_bandwidth,
        ref_sum=ref_sum,
        stub_processed_samples=stub_processed_samples,
        dry_run=dry_run,
    )

    # Process with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_params = {executor.submit(worker_func, params): params 
                           for params in unique_combinations}
        
        # Collect results as they complete
        from concurrent.futures import as_completed
        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                success = future.result()
                if success:
                    created_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"Unexpected error with {params}: {e}")
                error_count += 1
    
    print(f"\nSurface creation complete!")
    print(f"Total averaged surfaces created: {created_count}")
    print(f"Overall combinations: {len(combinations)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create averaged surfaces from samples using 2D normal weighted KDE')
    parser.add_argument('--input-folder', type=str, default=config.samples_folder,
                        help='Input folder containing sample files')
    parser.add_argument('--output-folder', type=str, default=output_folder,
                        help='Output folder for averaged surfaces')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Base directory for outputs (relative paths are placed here)')
    parser.add_argument('--bias-bandwidth', type=float, default=0.075,
                        help='Bandwidth for bias dimension KDE')
    parser.add_argument('--feat-bandwidth', type=float, default=3.0,
                        help='Gaussian SD across feat_diff grid steps; 3.0 steps = '
                             '6 degrees on the default grid')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads for parallel processing')
    parser.add_argument('--include-all-params', action='store_true',
                        help='Include all parameter combinations (skip L1 grid filtering)')
    parser.add_argument('--stub-samples', action='store_true', default=False,
                        help='After averaging, replace raw sample files with tiny stubs to free '
                             'disk space while preserving grid progress tracking (default: off)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Print what would be done without writing any files (default: off)')

    args = parser.parse_args()

    create_all_averaged_surfaces(
        args.input_folder, args.output_folder,
        args.bias_bandwidth, args.feat_bandwidth, args.workers,
        results_dir=args.results_dir,
        include_all_params=args.include_all_params,
        stub_processed_samples=args.stub_samples,
        dry_run=args.dry_run,
    )
