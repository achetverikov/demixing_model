#!/usr/bin/env python3
"""
Enhanced Parallel Likelihood Surface Grid Computer with Dynamic Chunking
=======================================================================

Parallel likelihood surface grid computer with dynamic chunking for load balancing
across any number of machines:
1. Dynamic chunk sizes: 100 parameter combinations when many chunks available, 10 when few remain
2. Chunk-based locking system for better coordination
3. Robust progress tracking and load balancing
4. Machine coordination and status monitoring

Works with the new Surface class and compute_empirical_likelihood_surface format.

Usage (run from repo root):
    PYTHONPATH=. python surface_computation/simulated_samples_grid.py --machine-id my_pc
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'

import time
import pickle, gzip
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import hashlib
from typing import Tuple, Dict,  List, Optional
import platform
import json
from datetime import datetime
from jax import Array

# Import required functions and config
from shared.utils import Surface, resolve_input_path, resolve_results_path, get_git_commit
from shared.config import Config
import jax_fit_functions as jf  # Import module to access diagonal_covariance flag
import jax_fit_main as jfm
config = Config()
config.n_samples = 100

def get_grid_level_info(level: int = 1) -> Dict:
    """
    Get grid configuration for different refinement levels.

    Parameters:
    -----------
    level : int
        Grid refinement level:
        1 = Coarse grid (config.param_step spacing, e.g., 10 degrees)
        2 = Fine grid (adds intermediate points at config.param_step/2, e.g., 5 degrees)

    Returns:
    --------
    Dict with step_size, description, and expected_total
    """
    if level == 1:
        step_size = config.param_step
        description = f"Coarse grid (step={step_size})"
        # Level 1: all surfaces at step_size intervals
        param_vals = np.arange(config.param_range_low, config.param_range_high + step_size, step_size)
        expected_total = len(param_vals) ** 3
    elif level == 2:
        step_size = config.param_step // 2
        description = f"Fine grid (step={step_size})"
        # Level 2: all surfaces at step_size/2 intervals MINUS those already in Level 1
        fine_param_vals = np.arange(config.param_range_low, config.param_range_high + step_size, step_size)
        coarse_param_vals = np.arange(config.param_range_low, config.param_range_high + config.param_step, config.param_step)
        
        total_fine_surfaces = len(fine_param_vals) ** 3
        total_coarse_surfaces = len(coarse_param_vals) ** 3
        expected_total = total_fine_surfaces - total_coarse_surfaces
    else:
        raise ValueError(f"Unsupported grid level: {level}")

    return {
        'level': level,
        'step_size': step_size,
        'description': description,
        'expected_total': expected_total,
        'param_range': (config.param_range_low, config.param_range_high)
    }


def create_param_identifier(sd_feat1: float, sd_feat2: float, sd_spat: float) -> Tuple[str, str]:
    """Create human-readable parameter identifier and hash."""
    param_name = f"sf1_{sd_feat1:.1f}_sf2_{sd_feat2:.1f}_sp_{sd_spat:.1f}"
    param_str = f"{sd_feat1:.6f}_{sd_feat2:.6f}_{sd_spat:.6f}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    return param_name, param_hash


def save_samples_checkpoint(output_dir: Path,
                            mu1_samples: Array,
                            mu2_samples: Array,
                            param_name: str,
                            param_hash: str, sd_feat1: float, sd_feat2: float,
                            sd_spat: float, computation_time: float, machine_id: str,
                            n_simulations: int = 0, n_samples: int = 0,
                            random_seed: int = 0,
                            full_results: Optional[Array] = None, save_csv: bool = False) -> None:
    """Save EM sample arrays and provenance metadata to a compressed pickle.

    Provenance fields stored under 'parameters': machine_id, platform,
    n_simulations, n_samples, random_seed, git_commit.
    """
    checkpoint_data = {
        'parameters': {
            'sd_feat1': sd_feat1,
            'sd_feat2': sd_feat2,
            'sd_spat': sd_spat,
            'param_name': param_name,
            'param_hash': param_hash,
            'machine_id': machine_id,
            'platform': platform.node(),
            'n_simulations': n_simulations,
            'n_samples': n_samples,
            'random_seed': random_seed,
            'git_commit': get_git_commit(),
        },
        'mu1_samples': mu1_samples.astype(jnp.float16),
        'mu2_samples': mu2_samples.astype(jnp.float16),
        'computation_time': computation_time,
        'timestamp': time.time()
    }

    # Add full results if provided
    if full_results is not None:
        checkpoint_data['full_results'] = full_results.astype(jnp.float32)

    # Save pickle file
    file = output_dir / f"samples_{param_name}_{param_hash}.pkl.gz"
    with gzip.open(file, 'wb') as f:
        pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Optionally save CSV
    if save_csv and full_results is not None:
        csv_file = output_dir / f"samples_{param_name}_{param_hash}.csv"
        # Always 23 columns total: 21 from jax_generate_and_fit + 2 bias columns
        full_results_2d = full_results.reshape(-1, full_results.shape[-1])

        # Get column names from jax_fit_functions module (single source of truth)
        # jax_generate_and_fit returns 21 columns (see jf.RESULT_COLUMNS)
        # simulate_dual_component_bias_distribution appends 2 bias columns
        columns = jf.RESULT_COLUMNS + ['mu1_bias', 'mu2_bias']

        header = ','.join(columns)
        np.savetxt(csv_file, np.asarray(full_results_2d), delimiter=',',
                   header=header, comments='', fmt='%.6f')


def load_progress_state(output_dir: Path) -> Dict:
    """Load current progress state from saved files (optimized to avoid pickle.load)."""
    completed_hashes = set()
    total_computation_time = 0.0
    machine_surfaces = {}

    # Use os.scandir for efficient directory scanning
    import os, re
    with os.scandir(output_dir) as entries:
        for entry in entries:
            if re.match(r"(surface|samples)_sf1_.*\.pkl.*$", entry.name):
                try:
                    # Extract param_hash from filename instead of loading pickle
                    # Filename format: samples_sf1_X_sf2_Y_sp_Z_HASH.pkl
                    hash_value = re.search(r'_([a-f0-9]+)(?=\.pkl)', entry.name).group(1)
                    completed_hashes.add(hash_value)

                    # Count surfaces per machine (we'll estimate this from progress summaries)
                    # For now, just count total surfaces
                    total_computation_time += 1  # Placeholder - will get real data from summaries
                        
                except (IndexError, ValueError):
                    continue

    # Get actual computation time and machine info from progress summaries
    for summary_file in output_dir.glob("progress_summary_*.json"):
        try:
            import json
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            machine_id = summary.get('machine_id', 'unknown')
            session_stats = summary.get('session_stats', {})
            samples_computed = session_stats.get('samples_computed', 0)
            comp_time = session_stats.get('total_computation_time', 0)
            
            machine_surfaces[machine_id] = samples_computed
            if samples_computed > 0:
                total_computation_time += comp_time
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            continue

    return {
        'completed_hashes': completed_hashes,
        'total_computation_time': total_computation_time,
        'machine_surfaces': machine_surfaces
    }


class ChunkedGridComputer:
    """Enhanced grid computer with dynamic chunking system and multi-level refinement."""

    def __init__(self, machine_id: str = "PC1", grid_level: int = 1,
                 custom_param_list: Optional[List[Tuple[float, float, float]]] = None,
                 save_full_results: bool = False, save_csv: bool = False,
                 diagonal_covariance: bool = True, fix_weights: bool = False,
                 algorithm: str = "EM"):
        """Initialize a grid computer with chunking settings.

        Args:
            machine_id: Identifier for coordinating multi-machine runs.
            grid_level: Grid refinement level (1 or 2).
            custom_param_list: Optional list of parameter tuples to process.
            save_full_results: Whether to store full simulation outputs.
            save_csv: Whether to export CSV alongside pickle results.
            diagonal_covariance: Whether to use diagonal covariance in fitting.
            fix_weights: Whether to fix mixture weights during fitting.
            algorithm: Fitting algorithm name.
        """
        self.machine_id = machine_id
        self.platform = platform.node()
        self.grid_level = grid_level
        self.grid_info = get_grid_level_info(grid_level)
        self.output_dir = Path(config.samples_folder)
        self.custom_param_list = custom_param_list
        self.save_full_results = save_full_results
        self.save_csv = save_csv
        self.fix_weights = fix_weights
        self.diagonal_covariance = diagonal_covariance
        self.algorithm = algorithm.upper()

        # Dynamic chunking parameters
        self.large_chunk_size = 50
        self.small_chunk_size = 5
        self.small_chunk_threshold = 10

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Initialize parameter grid for this level
        self.param_combinations = self._create_param_combinations()
        self.total_combinations = len(self.param_combinations)

        print(f"=== Enhanced Chunked Grid Computer ===")
        print(f"Machine ID: {self.machine_id}")
        print(f"Platform: {self.platform}")
        print(f"Algorithm: {self.algorithm}")
        if custom_param_list:
            print(f"Mode: Custom parameter list ({len(custom_param_list)} combinations)")
        else:
            print(f"Grid Level: {self.grid_info['level']} - {self.grid_info['description']}")
            print(
                f"Parameter range: {config.param_range_low}-{config.param_range_high}, step: {self.grid_info['step_size']}")
        print(f"Total parameter combinations: {self.total_combinations:,}")
        print(f"Large/small chunk sizes: {self.large_chunk_size}/{self.small_chunk_size}")
        print(f"Save full results: {self.save_full_results}, Save CSV: {self.save_csv}")


    def _create_param_combinations(self) -> List[Tuple]:
        """Create parameter combinations for the current grid level or from custom list."""
        if self.custom_param_list:
            # Use custom parameter list
            combinations = []
            for idx, (sd_feat1, sd_feat2, sd_spat) in enumerate(self.custom_param_list):
                # Use idx as placeholder for i, j, k
                combinations.append((idx, idx, idx, sd_feat1, sd_feat2, sd_spat))
            return combinations
        else:
            # Use standard grid
            step_size = self.grid_info['step_size']
            param_vals = np.arange(config.param_range_low, config.param_range_high + step_size, step_size)
            combinations = []

            for i, sd_feat1 in enumerate(param_vals):
                for j, sd_feat2 in enumerate(param_vals):
                    for k, sd_spat in enumerate(param_vals):
                        combinations.append((i, j, k, sd_feat1, sd_feat2, sd_spat))
            return combinations

    def _samples_exist(self, param_name: str, param_hash: str) -> bool:
        """Check if surface file exists and is complete."""
        samples_file = self.output_dir / f"samples_{param_name}_{param_hash}.pkl.gz"

        if not samples_file.exists():
            return False

        try:
            with gzip.open(samples_file, 'rb') as f:
                data = pickle.load(f)
                return ('parameters' in data and ('mu1_samples' in data or 'mu2_samples' in data))
        except (pickle.PickleError, EOFError, FileNotFoundError, gzip.BadGzipFile):
            try:
                samples_file.unlink()
            except:
                pass
            return False

    def _chunk_lock_operations(self, chunk_id: str, operation: str) -> bool:
        """Handle chunk lock creation/removal with improved stale lock detection."""
        lock_file = self.output_dir / f"computing_{chunk_id}.lock"

        if operation == 'create':
            try:
                with open(lock_file, 'x') as f:
                    json.dump({
                        'machine_id': self.machine_id,
                        'platform': self.platform,
                        'start_time': time.time(),
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
                return True
            except FileExistsError:
                # Enhanced stale lock detection
                return self._handle_existing_lock(lock_file, chunk_id)

        elif operation == 'remove':
            try:
                lock_file.unlink()
            except FileNotFoundError:
                pass
            return True

    def _handle_existing_lock(self, lock_file: Path, chunk_id: str) -> bool:
        """Handle existing lock with comprehensive stale detection."""
        try:
            with open(lock_file, 'r') as f:
                lock_data = json.load(f)

            lock_start_time = lock_data.get('start_time', 0)
            lock_machine_id = lock_data.get('machine_id', 'unknown')
            current_time = time.time()
            lock_age_hours = (current_time - lock_start_time) / 3600

            # Multiple criteria for stale lock detection
            is_stale = False
            stale_reason = ""

            # Criterion 1: Age-based (configurable threshold)
            stale_threshold_hours = 3.0  # Increased from 2 to 3 hours
            if lock_age_hours > stale_threshold_hours:
                is_stale = True
                stale_reason = f"age {lock_age_hours:.1f}h > {stale_threshold_hours}h threshold"

            # Criterion 2: Same machine reclaiming its own work
            elif lock_machine_id == self.machine_id:
                is_stale = True
                stale_reason = f"same machine ({self.machine_id}) reclaiming own lock"

            # Criterion 3: Check if lock file is older than any recent surface files
            # (indicates the machine stopped working)
            elif self._is_machine_inactive(lock_machine_id, lock_start_time):
                is_stale = True
                stale_reason = f"machine {lock_machine_id} appears inactive"

            if is_stale:
                print(f"[{self.machine_id}] Removing stale lock {chunk_id}: {stale_reason}")
                try:
                    lock_file.unlink()
                    # Try to create our own lock
                    return self._chunk_lock_operations(chunk_id, 'create')
                except FileNotFoundError:
                    return self._chunk_lock_operations(chunk_id, 'create')
            else:
                # Lock is still valid
                return False

        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            # Corrupted lock file - treat as stale
            print(f"[{self.machine_id}] Removing corrupted lock {chunk_id}: {e}")
            try:
                lock_file.unlink()
                return self._chunk_lock_operations(chunk_id, 'create')
            except FileNotFoundError:
                return self._chunk_lock_operations(chunk_id, 'create')

    def _is_machine_inactive(self, machine_id: str, lock_start_time: float) -> bool:
        """Check if a machine appears to be inactive based on recent surface creation."""
        # Look for surfaces created by this machine after the lock was created
        recent_activity_threshold = lock_start_time + 1800  # 30 minutes after lock

        for samples_file in self.output_dir.glob("samples_sf1_*_sf2_*_sp_*_*.pkl"):
            try:
                # Check file modification time
                if samples_file.stat().st_mtime > recent_activity_threshold:
                    # Check if this surface was created by the locked machine
                    with open(samples_file, 'rb') as f:
                        data = pickle.load(f)
                        if data.get('parameters', {}).get('machine_id') == machine_id:
                            return False  # Machine is still active
            except (OSError, pickle.PickleError, KeyError):
                continue

        # No recent activity found from this machine
        return True

    def _find_available_chunk(self) -> Tuple[Optional[int], Optional[int]]:
        """Find the next available chunk to work on."""
        progress = load_progress_state(self.output_dir)
        completed_hashes = progress['completed_hashes']

        # Find incomplete parameter combinations
        incomplete_indices = []
        for idx, (i, j, k, sd_feat1, sd_feat2, sd_spat) in enumerate(self.param_combinations):
            param_name, param_hash = create_param_identifier(sd_feat1, sd_feat2, sd_spat)
            if param_hash not in completed_hashes and not self._samples_exist(param_name, param_hash):
                incomplete_indices.append(idx)

        if not incomplete_indices:
            return None, None

        # Determine chunk size based on remaining work
        remaining_work = len(incomplete_indices)
        potential_large_chunks = remaining_work // self.large_chunk_size

        chunk_size = (self.small_chunk_size if potential_large_chunks <= self.small_chunk_threshold
                      else self.large_chunk_size)

        if potential_large_chunks <= self.small_chunk_threshold:
            print(f"[{self.machine_id}] Switching to small chunks (size={chunk_size})")

        import random
        # Step 1: Collect candidate chunks
        candidate_chunks = []
        for chunk_start in range(0, self.total_combinations, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.total_combinations)
            chunk_id = f"L{self.grid_level}_chunk_{chunk_start:05d}_{chunk_end:05d}"

            # Check if this chunk has any work to do
            chunk_has_work = any(idx in incomplete_indices for idx in range(chunk_start, chunk_end))
            if chunk_has_work:
                candidate_chunks.append((chunk_start, chunk_end, chunk_id))

        # Step 2: Shuffle candidates and try locking one
        random.shuffle(candidate_chunks)
        for chunk_start, chunk_end, chunk_id in candidate_chunks:
            if self._chunk_lock_operations(chunk_id, 'create'):
                return chunk_start, chunk_size

        # No available chunk found
        return None, None

    def _process_chunk(self, chunk_start_idx: int, chunk_size: int, samples_params: Dict) -> Tuple[int, int, float]:
        """Process a single chunk of parameter combinations."""
        chunk_end_idx = min(chunk_start_idx + chunk_size, self.total_combinations)
        chunk_id = f"L{self.grid_level}_chunk_{chunk_start_idx:05d}_{chunk_end_idx:05d}"

        print(f"\n[{self.machine_id}] Processing {chunk_id}")
        print(f"Parameter indices: {chunk_start_idx} to {chunk_end_idx - 1}")

        chunk_start_time = time.time()
        samples_computed = samples_skipped = 0
        feat_diff_vals = config.create_grid('feat_diff')
        feat_diff_vals = feat_diff_vals[feat_diff_vals > 0]  # feat_diff=0 is trivially 0 bias
        key = config.seed.get_jax_key(purpose='generating samples')

        num_scan_loops = 10
        num_sims_per_loop = samples_params['n_simulations'] // num_scan_loops

        def universal_scan_fn(carry, inputs):
            """Run one scan step for either full results or bias-only mode."""
            subkey, feat_diff, sd_feat1, sd_feat2, sd_spat = inputs
            if self.save_full_results:
                mu_1_bias, mu_2_bias, full_res = jfm.simulate_dual_component_bias_distribution(
                    subkey, sd_feat1, sd_feat2, sd_spat, feat_diff, 42.0,
                    num_sims_per_loop,
                    samples_params['n_samples'],
                    return_full_results=True,
                    fix_weights=self.fix_weights,
                    algorithm=self.algorithm,
                    diagonal_covariance=self.diagonal_covariance
                )
                return carry, (mu_1_bias, mu_2_bias, full_res)
            else:
                mu_1_bias, mu_2_bias = jfm.simulate_dual_component_bias_distribution(
                    subkey, sd_feat1, sd_feat2, sd_spat, feat_diff, 42.0,
                    num_sims_per_loop,
                    samples_params['n_samples'],
                    return_full_results=False,
                    fix_weights=self.fix_weights,
                    algorithm=self.algorithm,
                    diagonal_covariance=self.diagonal_covariance
                )
                return carry, (mu_1_bias, mu_2_bias)

        for idx in range(chunk_start_idx, chunk_end_idx):
            if idx >= len(self.param_combinations):
                break

            i, j, k, sd_feat1, sd_feat2, sd_spat = self.param_combinations[idx]
            param_name, param_hash = create_param_identifier(sd_feat1, sd_feat2, sd_spat)
            sd_feat1_arr = jnp.full(len(feat_diff_vals), sd_feat1)
            sd_feat2_arr = jnp.full(len(feat_diff_vals), sd_feat2)
            sd_spat_arr = jnp.full(len(feat_diff_vals), sd_spat)

            if self._samples_exist(param_name, param_hash):
                samples_skipped += 1
                continue

            samples_start = time.time()
            print(f"  [{self.machine_id}] Computing {samples_computed + 1}: "
                  f"sd_feat1={sd_feat1:.1f}, sd_feat2={sd_feat2:.1f}, sd_spat={sd_spat:.1f}, n_simulations={samples_params['n_simulations']}, n_samples={samples_params['n_samples']}")

            # Run simulations 10 times and combine results
            all_mu1_results = []
            all_mu2_results = []

            all_full_results = [] if self.save_full_results else None

            for rep in range(num_scan_loops):
                key, *subkeys = jax.random.split(key, len(feat_diff_vals)+1)

                tic = time.time()
                if self.save_full_results:
                    _, (mu1_bias_array, mu2_bias_array, full_res_array) = jax.lax.scan(
                        universal_scan_fn, None, (jnp.array(subkeys), feat_diff_vals, sd_feat1_arr, sd_feat2_arr, sd_spat_arr)
                    )
                    all_full_results.append(full_res_array)
                else:
                    _, (mu1_bias_array, mu2_bias_array) = jax.lax.scan(
                        universal_scan_fn, None, (jnp.array(subkeys), feat_diff_vals, sd_feat1_arr, sd_feat2_arr, sd_spat_arr)
                    )
                toc = time.time()
                print(f"    Scan {rep+1}/10: {toc-tic:.1f}s")

                all_mu1_results.append(mu1_bias_array)
                all_mu2_results.append(mu2_bias_array)

            # Combine all results
            mu1_bias_array = jnp.concatenate(all_mu1_results, axis=1)
            mu2_bias_array = jnp.concatenate(all_mu2_results, axis=1)

            # Combine full results if collected
            full_results_combined = None
            if self.save_full_results and all_full_results:
                # all_full_results is list of length num_scan_loops, each element shape (n_feat_diff, num_sims_per_loop, 2, C)
                # Concatenate along the simulation dimension (axis=1)
                full_results_combined = jnp.concatenate(all_full_results, axis=1)  # shape: (n_feat_diff, n_simulations, 2, C)

            samples_time = time.time() - samples_start
            save_samples_checkpoint(
                self.output_dir, mu1_bias_array, mu2_bias_array, param_name, param_hash,
                sd_feat1, sd_feat2, sd_spat, samples_time, self.machine_id,
                n_simulations=samples_params['n_simulations'],
                n_samples=samples_params['n_samples'],
                random_seed=samples_params['random_seed'],
                full_results=full_results_combined, save_csv=self.save_csv
            )

            samples_computed += 1
            print(f"    Completed in {samples_time:.1f}s")

        chunk_time = time.time() - chunk_start_time

        print(f"[{self.machine_id}] {chunk_id} completed:")
        print(f"  Surfaces computed: {samples_computed}, skipped: {samples_skipped}")
        print(f"  Chunk time: {chunk_time:.1f}s")
        if samples_computed > 0:
            print(f"  Average time per surface: {chunk_time / samples_computed:.1f}s")

        self._chunk_lock_operations(chunk_id, 'remove')
        return samples_computed, samples_skipped, chunk_time

    def _save_progress_summary(self, session_stats: Dict) -> None:
        """Save progress summary for this machine."""
        summary_file = self.output_dir / f"progress_summary_{self.machine_id}.json"
        progress = load_progress_state(self.output_dir)

        summary = {
            'machine_id': self.machine_id,
            'platform': self.platform,
            'timestamp': datetime.now().isoformat(),
            'session_stats': session_stats,
            'total_combinations': self.total_combinations,
            'completed_count': len(progress['completed_hashes']),
            'completion_rate': len(progress['completed_hashes']) / self.total_combinations * 100,
            'grid_params': {
                'param_range_low': config.param_range_low,
                'param_range_high': config.param_range_high,
                'param_step': config.param_step
            }
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def compute_grid(self, feat_diff_step: int = 2, mu1_bias_step: int = 2, mu2_bias_step: int = 6,
                     feat_diff_range: Tuple[int, int] = (4, 180),
                     mu1_bias_range: Tuple[int, int] = (-180, 180),
                     mu2_bias_range: Tuple[int, int] = (-498, 498),
                     n_simulations: int = 1000, n_samples: int = 100, random_seed: int = 42) -> Dict:
        """Main computation loop with dynamic chunking."""

        samples_params = {
            'feat_diff_step': feat_diff_step, 'mu1_bias_step': mu1_bias_step,
            'mu2_bias_step': mu2_bias_step, 'feat_diff_range': feat_diff_range,
            'mu1_bias_range': mu1_bias_range, 'mu2_bias_range': mu2_bias_range,
            'n_simulations': n_simulations, 'n_samples': n_samples, 'random_seed': random_seed
        }

        print(f"\n[{self.machine_id}] Starting dynamic chunked computation")
        print(f"Surface parameters: steps=({feat_diff_step},{mu1_bias_step},{mu2_bias_step}), "
              f"sims={n_simulations}×{n_samples}")

        session_start = time.time()
        session_stats = {
            'chunks_processed': 0, 'samples_computed': 0, 'samples_skipped': 0,
            'total_computation_time': 0.0, 'start_time': session_start
        }

        # Main processing loop
        while True:
            chunk_start, chunk_size = self._find_available_chunk()
            if chunk_start is None:
                print(f"\n[{self.machine_id}] No more work available")
                break

            try:
                samples_computed, samples_skipped, chunk_time = self._process_chunk(
                    chunk_start, chunk_size, samples_params
                )

                # Update session stats
                session_stats['chunks_processed'] += 1
                session_stats['samples_computed'] += samples_computed
                session_stats['samples_skipped'] += samples_skipped
                session_stats['total_computation_time'] += chunk_time

                self._save_progress_summary(session_stats)
                time.sleep(1)  # Brief pause between chunks

            except KeyboardInterrupt:
                print(f"\n[{self.machine_id}] Interrupted by user")
                chunk_id = f"L{self.grid_level}_chunk_{chunk_start:05d}_{min(chunk_start + chunk_size, self.total_combinations):05d}"
                self._chunk_lock_operations(chunk_id, 'remove')
                break

        # Final session summary
        session_time = time.time() - session_start
        session_stats['session_time'] = session_time

        print(f"\n=== {self.machine_id} Session Complete ===")
        print(f"Chunks: {session_stats['chunks_processed']}, "
              f"Surfaces: {session_stats['samples_computed']}, "
              f"Skipped: {session_stats['samples_skipped']}")
        print(
            f"Times: {session_time / 3600:.1f}h session, {session_stats['total_computation_time'] / 3600:.1f}h compute")

        if session_stats['samples_computed'] > 0:
            avg_time = session_stats['total_computation_time'] / session_stats['samples_computed']
            print(f"Average time per surface: {avg_time:.1f}s")

        self._save_progress_summary(session_stats)
        return session_stats


def get_grid_status(grid_level: int = None) -> Dict:
    """Get comprehensive status of grid computation across all machines."""
    output_dir = Path(config.samples_folder)

    if not output_dir.exists():
        return {'status': 'error', 'message': 'Output directory does not exist'}

    # If grid_level not specified, determine from existing files or default to 1
    if grid_level is None:
        # Check what grid levels have been worked on
        lock_files = list(output_dir.glob("computing_L*_chunk_*.lock"))
        if lock_files:
            # Extract highest level from existing lock files
            levels = []
            for lock_file in lock_files:
                try:
                    level = int(lock_file.name.split('_')[1][1:])  # Extract number after 'L'
                    levels.append(level)
                except (ValueError, IndexError):
                    pass
            grid_level = max(levels) if levels else 1
        else:
            grid_level = 1

    grid_info = get_grid_level_info(grid_level)

    # Load progress and active chunks for this level
    progress = load_progress_state(output_dir)

    # Filter active chunks for this grid level
    active_chunks = []
    for lock_file in output_dir.glob(f"computing_L{grid_level}_chunk_*.lock"):
        try:
            with open(lock_file, 'r') as f:
                lock_data = json.load(f)
                active_chunks.append({
                    'chunk_id': lock_file.stem.replace('computing_', ''),
                    'machine_id': lock_data.get('machine_id', 'unknown'),
                    'running_time': time.time() - lock_data.get('start_time', time.time())
                })
        except (json.JSONDecodeError, FileNotFoundError):
            continue

    # Load machine summaries
    machine_summaries = {}
    for summary_file in output_dir.glob("progress_summary_*.json"):
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                machine_summaries[summary.get('machine_id', 'unknown')] = summary
        except (json.JSONDecodeError, FileNotFoundError):
            continue

    # Count surfaces that match this grid level
    level_surfaces = 0
    step_size = grid_info['step_size']
    import re

    pattern = re.compile(r"samples_sf1_(?P<sf1>[\d.]+)_sf2_(?P<sf2>[\d.]+)_sp_(?P<sp>[\d.]+)_.*")

    if grid_level == 1:
        # Level 1: count surfaces at coarse step intervals
        param_vals = np.arange(config.param_range_low, config.param_range_high + config.param_step, config.param_step)
        for samples_file in output_dir.glob("samples_sf1_*_sf2_*_sp_*"):
            try:
                match = pattern.match(samples_file.name)
                if not match:
                    continue

                sf1 = float(match.group("sf1"))
                sf2 = float(match.group("sf2"))
                sp = float(match.group("sp"))

                # Check if this surface belongs to Level 1 (coarse grid)
                if (sf1 in param_vals and sf2 in param_vals and sp in param_vals):
                    level_surfaces += 1

            except (ValueError, IndexError):
                continue
                
    elif grid_level == 2:
        # Level 2: count surfaces at fine step intervals BUT exclude those from Level 1
        fine_param_vals = np.arange(config.param_range_low, config.param_range_high + step_size, step_size)
        coarse_param_vals = np.arange(config.param_range_low, config.param_range_high + config.param_step, config.param_step)
        
        for samples_file in output_dir.glob("samples_sf1_*_sf2_*_sp_*"):
            try:
                match = pattern.match(samples_file.name)
                if not match:
                    continue

                sf1 = float(match.group("sf1"))
                sf2 = float(match.group("sf2"))
                sp = float(match.group("sp"))

                # Check if this surface belongs to Level 2 (fine grid excluding coarse grid)
                in_fine_grid = (sf1 in fine_param_vals and sf2 in fine_param_vals and sp in fine_param_vals)
                in_coarse_grid = (sf1 in coarse_param_vals and sf2 in coarse_param_vals and sp in coarse_param_vals)
                
                if in_fine_grid and not in_coarse_grid:
                    level_surfaces += 1

            except (ValueError, IndexError):
                continue

    expected_total = grid_info['expected_total']
    completion_rate = level_surfaces / expected_total * 100

    return {
        'grid_level': grid_level,
        'grid_info': grid_info,
        'expected_total': expected_total,
        'completed_count': level_surfaces,
        'missing_count': expected_total - level_surfaces,
        'completion_rate': completion_rate,
        'total_computation_time_hours': progress['total_computation_time'] / 3600,
        'machine_surfaces': progress['machine_surfaces'],
        'active_chunks': active_chunks,
        'machine_summaries': machine_summaries,
        'status': 'complete' if completion_rate >= 100 else 'in_progress'
    }


def print_grid_status(grid_level: int = None) -> None:
    """Print comprehensive grid status."""
    status = get_grid_status(grid_level)

    print("=== Grid Computation Status ===")
    print(f"Grid Level: {status['grid_level']} - {status['grid_info']['description']}")
    print(f"Expected: {status['expected_total']}, Completed: {status['completed_count']}, "
          f"Missing: {status['missing_count']}")
    print(f"Completion: {status['completion_rate']:.1f}%")
    print(f"Total computation time: {status['total_computation_time_hours']:.1f} hours")

    print(f"\nSurfaces by machine:")
    for machine_id, count in status['machine_surfaces'].items():
        print(f"  {machine_id}: {count}")

    if status['active_chunks']:
        print(f"\nActive chunks:")
        for chunk in status['active_chunks']:
            runtime_hours = chunk['running_time'] / 3600
            print(f"  {chunk['chunk_id']} on {chunk['machine_id']} ({runtime_hours:.1f}h)")
    else:
        print("\nNo active chunks")


def check_grid_level_readiness(target_level: int) -> Dict:
    """
    Check if the grid is ready for the target refinement level.

    Parameters:
    -----------
    target_level : int
        Target grid level to check readiness for

    Returns:
    --------
    Dict with readiness status and recommendations
    """
    if target_level == 1:
        return {
            'ready': True,
            'message': 'Level 1 (coarse grid) can always be started',
            'prerequisite_level': None,
            'prerequisite_completion': 100.0
        }

    elif target_level == 2:
        # Check if level 1 is complete
        level1_status = get_grid_status(grid_level=1)
        level1_complete = level1_status['completion_rate'] >= 99.0  # Allow 1% tolerance

        return {
            'ready': level1_complete,
            'message': f"Level 2 ready: {level1_complete} (Level 1 completion: {level1_status['completion_rate']:.1f}%)",
            'prerequisite_level': 1,
            'prerequisite_completion': level1_status['completion_rate']
        }

    else:
        return {
            'ready': False,
            'message': f"Unsupported grid level: {target_level}",
            'prerequisite_level': None,
            'prerequisite_completion': 0.0
        }


def run_chunked_computation(machine_id: str = "PC1", grid_level: int = 1,
                            auto_advance: bool = True,
                            custom_param_list: Optional[List[Tuple[float, float, float]]] = None,
                            save_full_results: bool = False, save_csv: bool = False,
                            diagonal_covariance: bool = True, fix_weights: bool = False,
                            algorithm: str = "EM", **samples_params) -> Dict:
    """
    Run the enhanced chunked computation system with multi-level refinement.

    Parameters:
    -----------
    machine_id : str
        Unique identifier for this machine (PC1, PC2, PC3, PC4, PC5)
    grid_level : int
        Grid refinement level to work on (1=coarse, 2=fine)
    auto_advance : bool
        If True, automatically advance to next level when current level is complete
    custom_param_list : Optional[List[Tuple[float, float, float]]]
        If provided, run only for these parameter combinations instead of full grid
    save_full_results : bool
        If True, save full simulation results in addition to biases
    save_csv : bool
        If True, also save full results as CSV files
    diagonal_covariance : bool
        If True, use diagonal covariance matrix in simulations (default: True)
    fix_weights : bool
        If True, constrain mixture weights to be equal during fitting (default: False)
    algorithm : str
        Fitting algorithm to use ('EM', 'VBEM' for MAP estimates, or 'VBEM_MIX' for mixture-of-modes means)
    **samples_params : dict
        Surface computation parameters passed to compute_grid()
    """

    # If using custom param list, skip grid level management
    if custom_param_list:
        print(f"✅ Using custom parameter list ({len(custom_param_list)} combinations)")
        computer = ChunkedGridComputer(machine_id, grid_level, custom_param_list=custom_param_list,
                                       save_full_results=save_full_results, save_csv=save_csv,
                                       diagonal_covariance=diagonal_covariance, fix_weights=fix_weights,
                                       algorithm=algorithm)
        result = computer.compute_grid(**samples_params)
        return {'custom_params': result}

    # Standard grid level computation
    current_level = grid_level
    all_results = {}

    while True:
        # Check if current level is ready
        readiness = check_grid_level_readiness(current_level)
        if not readiness['ready']:
            print(f"❌ {readiness['message']}")
            if readiness['prerequisite_level']:
                print(f"💡 Suggestion: Complete level {readiness['prerequisite_level']} first "
                      f"({readiness['prerequisite_completion']:.1f}% done)")
            break

        print(f"✅ {readiness['message']}")

        # Run computation for current level
        computer = ChunkedGridComputer(machine_id, current_level,
                                       save_full_results=save_full_results, save_csv=save_csv,
                                       diagonal_covariance=diagonal_covariance, fix_weights=fix_weights,
                                       algorithm=algorithm)
        level_result = computer.compute_grid(**samples_params)
        all_results[f'level_{current_level}'] = level_result

        # Check if auto-advance is enabled and this level is now complete
        if auto_advance:
            level_status = get_grid_status(current_level)
            if level_status['completion_rate'] >= 99.0:  # Level complete
                next_level = current_level + 1
                next_readiness = check_grid_level_readiness(next_level)

                if next_readiness['ready']:
                    print(f"\n🎯 Level {current_level} complete! Auto-advancing to level {next_level}")
                    current_level = next_level
                    continue
                else:
                    print(f"\n🏁 Level {current_level} complete! No more levels available.")
                    break
            else:
                print(f"\n⏸️  Level {current_level} incomplete ({level_status['completion_rate']:.1f}%), stopping here")
                break
        else:
            break

    return all_results


def cleanup_stale_locks(max_age_hours: float = 3.0, dry_run: bool = True) -> Dict:
    """
    Clean up stale lock files to recover from crashed machines.

    Parameters:
    -----------
    max_age_hours : float
        Maximum age for lock files before considering them stale
    dry_run : bool
        If True, only report what would be cleaned without actually removing files

    Returns:
    --------
    Dict with cleanup results
    """
    output_dir = Path(config.samples_folder)
    current_time = time.time()
    stale_threshold = max_age_hours * 3600  # Convert to seconds

    cleanup_results = {
        'scanned_locks': 0,
        'stale_locks_found': [],
        'cleaned_locks': [],
        'errors': []
    }

    print(f"🔍 Scanning for stale locks (max age: {max_age_hours:.1f}h, dry_run: {dry_run})")

    for lock_file in output_dir.glob("computing_*.lock"):
        cleanup_results['scanned_locks'] += 1

        try:
            with open(lock_file, 'r') as f:
                lock_data = json.load(f)

            lock_start_time = lock_data.get('start_time', 0)
            lock_machine_id = lock_data.get('machine_id', 'unknown')
            lock_age_hours = (current_time - lock_start_time) / 3600
            chunk_id = lock_file.stem.replace('computing_', '')

            if (current_time - lock_start_time) > stale_threshold:
                stale_info = {
                    'chunk_id': chunk_id,
                    'machine_id': lock_machine_id,
                    'age_hours': lock_age_hours,
                    'lock_file': str(lock_file)
                }
                cleanup_results['stale_locks_found'].append(stale_info)

                print(f"🚨 Stale lock: {chunk_id} from {lock_machine_id} ({lock_age_hours:.1f}h old)")

                if not dry_run:
                    try:
                        lock_file.unlink()
                        cleanup_results['cleaned_locks'].append(stale_info)
                        print(f"   ✅ Removed: {lock_file.name}")
                    except FileNotFoundError:
                        print(f"   ⚠️  Already removed: {lock_file.name}")
                    except Exception as e:
                        error_info = {'chunk_id': chunk_id, 'error': str(e)}
                        cleanup_results['errors'].append(error_info)
                        print(f"   ❌ Error removing {lock_file.name}: {e}")
                else:
                    print(f"   🔍 Would remove: {lock_file.name}")

        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            # Corrupted lock file
            stale_info = {
                'chunk_id': lock_file.stem.replace('computing_', ''),
                'machine_id': 'corrupted',
                'age_hours': -1,
                'lock_file': str(lock_file),
                'error': str(e)
            }
            cleanup_results['stale_locks_found'].append(stale_info)

            print(f"🚨 Corrupted lock: {lock_file.name} ({e})")

            if not dry_run:
                try:
                    lock_file.unlink()
                    cleanup_results['cleaned_locks'].append(stale_info)
                    print(f"   ✅ Removed corrupted lock: {lock_file.name}")
                except Exception as e2:
                    error_info = {'chunk_id': stale_info['chunk_id'], 'error': str(e2)}
                    cleanup_results['errors'].append(error_info)
                    print(f"   ❌ Error removing corrupted lock: {e2}")
            else:
                print(f"   🔍 Would remove corrupted: {lock_file.name}")

    # Summary
    print(f"\n📊 Cleanup Summary:")
    print(f"   Scanned: {cleanup_results['scanned_locks']} lock files")
    print(f"   Stale found: {len(cleanup_results['stale_locks_found'])}")
    print(f"   Cleaned: {len(cleanup_results['cleaned_locks'])}")
    print(f"   Errors: {len(cleanup_results['errors'])}")

    if dry_run and cleanup_results['stale_locks_found']:
        print(f"\n💡 To actually clean stale locks, run with dry_run=False")

    return cleanup_results


def recover_crashed_machine(machine_id: str, max_age_hours: float = 2.0,
                            restart_computation: bool = True) -> Dict:
    """
    Recover from a crashed machine by cleaning its stale locks and optionally restarting.

    Parameters:
    -----------
    machine_id : str
        ID of the crashed machine to recover
    max_age_hours : float
        Maximum age for considering locks stale
    restart_computation : bool
        Whether to restart computation after cleanup

    Returns:
    --------
    Dict with recovery results
    """
    output_dir = Path(config.samples_folder)
    current_time = time.time()

    recovery_results = {
        'machine_id': machine_id,
        'crashed_chunks_found': [],
        'cleaned_chunks': [],
        'restart_attempted': False,
        'restart_successful': False
    }

    print(f"🚑 Recovering crashed machine: {machine_id}")

    # Find chunks locked by this machine
    for lock_file in output_dir.glob("computing_*.lock"):
        try:
            with open(lock_file, 'r') as f:
                lock_data = json.load(f)

            if lock_data.get('machine_id') == machine_id:
                lock_age_hours = (current_time - lock_data.get('start_time', 0)) / 3600
                chunk_id = lock_file.stem.replace('computing_', '')

                chunk_info = {
                    'chunk_id': chunk_id,
                    'age_hours': lock_age_hours,
                    'lock_file': str(lock_file)
                }
                recovery_results['crashed_chunks_found'].append(chunk_info)

                print(f"🔍 Found chunk from {machine_id}: {chunk_id} ({lock_age_hours:.1f}h old)")

                # Remove the lock if it's old enough or if we're forcing recovery
                if lock_age_hours > max_age_hours or True:  # Always clean crashed machine locks
                    try:
                        lock_file.unlink()
                        recovery_results['cleaned_chunks'].append(chunk_info)
                        print(f"   ✅ Cleaned lock: {chunk_id}")
                    except FileNotFoundError:
                        print(f"   ⚠️  Lock already gone: {chunk_id}")
                    except Exception as e:
                        print(f"   ❌ Error cleaning lock {chunk_id}: {e}")

        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            continue

    print(f"\n📊 Recovery Summary for {machine_id}:")
    print(f"   Crashed chunks found: {len(recovery_results['crashed_chunks_found'])}")
    print(f"   Cleaned chunks: {len(recovery_results['cleaned_chunks'])}")

    if restart_computation:
        print(f"\n🚀 Attempting to restart computation on {machine_id}...")
        try:
            # This would need to be run on the actual machine
            recovery_results['restart_attempted'] = True
            print(f"💡 To restart {machine_id}, run:")
            print(f"   python simulated_samples_grid.py --machine-id {machine_id} --grid-level 1")

        except Exception as e:
            print(f"❌ Could not restart computation: {e}")

    return recovery_results

def extract_params_from_csv_dir(csv_dir: str = './csv_samples') -> List[Tuple[float, float, float]]:
    """
    Extract parameter combinations from CSV filenames in the specified directory.

    Parameters:
    -----------
    csv_dir : str
        Directory containing CSV files with naming: samples_sf1_X_sf2_Y_sp_Z.csv

    Returns:
    --------
    List of (sd_feat1, sd_feat2, sd_spat) tuples
    """
    import re
    from pathlib import Path

    csv_path = Path(csv_dir)
    if not csv_path.exists():
        print(f"Warning: CSV directory {csv_dir} does not exist")
        return []

    pattern = re.compile(r'samples_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)\.csv')
    param_combos = []

    for csv_file in csv_path.glob('samples_sf1_*.csv'):
        match = pattern.match(csv_file.name)
        if match:
            sd_feat1 = float(match.group(1))
            sd_feat2 = float(match.group(2))
            sd_spat = float(match.group(3))
            param_combos.append((sd_feat1, sd_feat2, sd_spat))

    # Sort for consistent ordering
    param_combos = sorted(set(param_combos))

    print(f"Found {len(param_combos)} parameter combinations in {csv_dir}")
    for params in param_combos:
        print(f"  sf1={params[0]:.1f}, sf2={params[1]:.1f}, sp={params[2]:.1f}")

    return param_combos


def format_count_suffix(value: int) -> str:
    """Format counts like 1000 -> 1k for compact folder names."""
    if value >= 1000 and value % 1000 == 0:
        return f"{value // 1000}k"
    return str(value)


def build_samples_folder_name(n_simulations: int, n_samples: int, algorithm: str,
                              diagonal_covariance: bool, fix_weights: bool) -> str:
    """Construct descriptive folder name based on runtime parameters."""
    geometry = 'circular' if jf.wrap_1st else 'linear'
    sim_part = format_count_suffix(n_simulations)
    algorithm_part = algorithm.lower()
    covariance_part = 'diagcov' if diagonal_covariance else 'fullcov'
    weights_part = 'fix_weights' if fix_weights else 'free_weights'
    return f"sim_samples_{sim_part}_{n_samples}samples_{geometry}_{algorithm_part}_{covariance_part}_{weights_part}"


def configure_samples_folder(n_simulations: int, n_samples: int, algorithm: str,
                             diagonal_covariance: bool, fix_weights: bool,
                             results_dir: str = "results") -> Path:
    """Set config.samples_folder dynamically and ensure it exists."""
    folder_name = build_samples_folder_name(n_simulations, n_samples, algorithm,
                                            diagonal_covariance, fix_weights)
    folder_path = resolve_results_path(folder_name, results_dir)
    folder_path.mkdir(parents=True, exist_ok=True)
    config.samples_folder = folder_path
    return folder_path


def determine_run_dimensions(args) -> Tuple[int, int]:
    """Determine the number of simulations and samples per simulation for this run."""
    if args.test_mode:
        return 100, 50
    return args.n_simulations, config.n_samples


def print_runtime_configuration(args, n_simulations: int, n_samples: int, output_dir: Path) -> None:
    """Display key command-line arguments for transparency."""
    boolean_flag = lambda flag: "Yes" if flag else "No"
    print("\n=== Runtime Configuration ===")
    print(f"Machine ID           : {args.machine_id}")
    print(f"Grid Level           : {args.grid_level}")
    print(f"Auto Advance         : {boolean_flag(args.auto_advance)}")
    print(f"Test Mode            : {boolean_flag(args.test_mode)}")
    print(f"Status Only          : {boolean_flag(args.status_only)}")
    print(f"Save Full Results    : {boolean_flag(args.save_full_results)}")
    print(f"Save CSV             : {boolean_flag(args.save_csv)}")
    print(f"Diagonal Covariance  : {boolean_flag(args.diagonal_covariance)}")
    print(f"Fix Weights          : {boolean_flag(args.fix_weights)}")
    print(f"Algorithm            : {args.algorithm}")
    print(f"Match CSV Params     : {args.match_csv_params or 'None'}")
    print(f"Cleanup Stale Locks  : {boolean_flag(args.cleanup_stale_locks)} (max_age={args.max_lock_age}h)")
    print(f"Recover Machine      : {args.recover_machine or 'None'}")
    print(f"N Simulations        : {n_simulations}")
    print(f"N Samples            : {n_samples}")
    print(f"Output Folder        : {output_dir}")
    print("================================\n")


def parse_arguments():
    """Parse command-line arguments for sample grid computation."""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Chunked Likelihood Surface Grid Computation')
    parser.add_argument('--machine-id', required=False, default='TEST_PC',
                        help='Machine identifier (PC1, PC2, PC3, PC4, PC5)')
    parser.add_argument('--cleanup-stale-locks', action='store_true',
                        help='Clean up stale lock files from crashed machines')
    parser.add_argument('--max-lock-age', type=float, default=3.0,
                        help='Maximum age in hours for lock files before considering them stale')
    parser.add_argument('--recover-machine', type=str,
                        help='Recover specific crashed machine by cleaning its locks')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Only show what would be cleaned without actually removing files')
    parser.add_argument('--force-cleanup', action='store_true',
                        help='Actually perform cleanup operations (overrides --dry-run)')
    parser.add_argument('--grid-level', type=int, default=1, choices=[1, 2],
                        help='Grid refinement level (1=coarse 10°, 2=fine 5°)')
    parser.add_argument('--auto-advance', action='store_true', default=True,
                        help='Automatically advance to next grid level when current is complete')
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode with reduced parameters')
    parser.add_argument('--diagonal-covariance', action=argparse.BooleanOptionalAction, default=False,
                        help='Use diagonal covariance matrix in simulations (default: False)')
    parser.add_argument('--status-only', action='store_true',
                        help='Only show status without running computation')
    parser.add_argument('--match-csv-params', type=str,
                        help='Run only for parameter combinations found in CSV files in this directory (e.g., ./csv_samples)')
    parser.add_argument('--save-full-results', action='store_true',
                        help='Save full simulation results in addition to biases')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save full results as CSV files in addition to pickle')
    parser.add_argument('--fix-weights', action='store_true', default=False,
                        help='Fix mixture weights to equal proportions during fitting (default: False)')
    parser.add_argument('--algorithm', type=str.upper, choices=['EM', 'VBEM', 'VBEM_MIX'], default='EM',
                        help='Fitting algorithm to use for simulations (EM, VBEM MAP, or VBEM_MIX for mixture-of-modes means)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Base directory for outputs (relative paths are placed here)')
    parser.add_argument('--n-simulations', type=int, default=10000,
                        help='Number of simulations per surface (default: 10000)')

    args = parser.parse_args()
    active_n_simulations, active_n_samples = determine_run_dimensions(args)
    output_dir = configure_samples_folder(
        active_n_simulations, active_n_samples,
        args.algorithm, args.diagonal_covariance, args.fix_weights,
        results_dir=args.results_dir
    )
    print_runtime_configuration(args, active_n_simulations, active_n_samples, output_dir)

    # Handle cleanup operations first
    if args.cleanup_stale_locks:
        dry_run = args.dry_run and not args.force_cleanup
        cleanup_stale_locks(max_age_hours=args.max_lock_age, dry_run=dry_run)

    if args.recover_machine:
        recover_crashed_machine(args.recover_machine, max_age_hours=args.max_lock_age)


    if args.status_only:
        if args.grid_level:
            print_grid_status(args.grid_level)
        else:
            # Show status for all levels
            for level in [1, 2]:
                try:
                    print(f"\n" + "=" * 50)
                    print_grid_status(level)
                except:
                    print(f"Level {level}: No data available")
    else:
        print(f"Starting enhanced chunked computation on {args.machine_id}")

        # Check if we should match CSV parameters
        custom_params = None
        if args.match_csv_params:
            csv_param_dir = resolve_input_path(args.match_csv_params, args.results_dir)
            custom_params = extract_params_from_csv_dir(str(csv_param_dir))
            if not custom_params:
                print(f"Error: No parameter combinations found in {csv_param_dir}")
                return

        if args.test_mode:
            print("Running in test mode")
            session_stats = run_chunked_computation(
                machine_id=args.machine_id,
                grid_level=args.grid_level,
                auto_advance=args.auto_advance,
                custom_param_list=custom_params,
                save_full_results=args.save_full_results,
                save_csv=args.save_csv,
                diagonal_covariance=args.diagonal_covariance,
                fix_weights=args.fix_weights,
                algorithm=args.algorithm,
                feat_diff_step=4, mu1_bias_step=4, mu2_bias_step=12,
                n_simulations=active_n_simulations, n_samples=active_n_samples
            )
        else:
            print("Running full computation")
            session_stats = run_chunked_computation(
                machine_id=args.machine_id,
                grid_level=args.grid_level,
                auto_advance=args.auto_advance,
                custom_param_list=custom_params,
                save_full_results=args.save_full_results,
                save_csv=args.save_csv,
                diagonal_covariance=args.diagonal_covariance,
                fix_weights=args.fix_weights,
                algorithm=args.algorithm,
                n_simulations=active_n_simulations,
               n_samples=active_n_samples
            )

        print(f"\nSession completed. Final status:")
        if not custom_params:
            print_grid_status()
if __name__ == "__main__":
    parse_arguments()
