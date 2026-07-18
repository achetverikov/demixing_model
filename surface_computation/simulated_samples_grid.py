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
from typing import Tuple, Dict, List, Optional
import platform
import json
from datetime import datetime
from jax import Array

# Load .env before any other local imports so REDIS_* vars are available
from lock_backend import load_env, make_lock_backend, FileLockBackend, LockBackend
load_env()
try:
    from object_store import ObjectStoreConfig, SurfaceObjectStore
except ImportError:
    from surface_computation.object_store import ObjectStoreConfig, SurfaceObjectStore

# Import required functions and config
from shared.utils import AveragedSurface, Surface, resolve_input_path, resolve_results_path, get_git_commit
from shared.config import Config
from shared.averaging import make_averaging_functions, average_sample_pair
import jax_fit_functions as jf  # Import module to access diagonal_covariance flag
import jax_fit_main as jfm
config = Config()
config.n_samples = 100


class ChunkConflict(Exception):
    """Raised when a write conflict is detected mid-chunk; signals the main loop to release
    the current chunk lock and call _find_available_chunk again."""


def _load_json_with_retry(path: Path, max_attempts: int = 3, delay: float = 2.0):
    """Read and parse a JSON file, retrying on transient I/O errors (e.g. OneDrive sync).

    Raises the last OSError if all attempts fail, so callers can decide whether to
    treat the file as missing/corrupt or to propagate the error.
    """
    last_exc: Exception = FileNotFoundError(path)
    for _ in range(max_attempts):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except OSError as e:
            last_exc = e
            time.sleep(delay)
        except json.JSONDecodeError:
            raise  # Not transient — propagate immediately
    raise last_exc


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
        n = len(param_vals)
        expected_total = n ** 3 + n ** 2  # cube + one extra run per diagonal (sf1==sf2)
    elif level == 2:
        step_size = config.param_step // 2
        description = f"Fine grid (step={step_size})"
        # L2 extends one step below L1 so values like 5 are included.
        fine_start = config.param_range_low - step_size
        fine_param_vals = np.arange(fine_start, config.param_range_high + step_size, step_size)
        coarse_param_vals = np.arange(config.param_range_low, config.param_range_high + config.param_step, config.param_step)

        n_fine = len(fine_param_vals)
        n_coarse = len(coarse_param_vals)
        total_fine_surfaces = n_fine ** 3 + n_fine ** 2
        total_coarse_surfaces = n_coarse ** 3 + n_coarse ** 2
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


def create_param_identifier(sd_feat1: float, sd_feat2: float, sd_spat: float,
                            run_index: Optional[int] = None) -> Tuple[str, str]:
    """Create human-readable parameter identifier and hash.

    For diagonal combinations (sf1==sf2) that are computed twice, pass run_index=0 or 1
    to produce distinct filenames and hashes for each independent run.
    """
    if run_index is not None:
        param_name = f"sf1_{sd_feat1:.1f}_sf2_{sd_feat2:.1f}_sp_{sd_spat:.1f}_r{run_index}"
        param_str = f"{sd_feat1:.6f}_{sd_feat2:.6f}_{sd_spat:.6f}_r{run_index}"
    else:
        param_name = f"sf1_{sd_feat1:.1f}_sf2_{sd_feat2:.1f}_sp_{sd_spat:.1f}"
        param_str = f"{sd_feat1:.6f}_{sd_feat2:.6f}_{sd_spat:.6f}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    return param_name, param_hash


def create_surface_identifier(sd_feat1: float, sd_feat2: float, sd_spat: float) -> str:
    """Create a compact canonical ID for one averaged surface."""
    canonical_sf1 = min(sd_feat1, sd_feat2)
    canonical_sf2 = max(sd_feat1, sd_feat2)
    return f"{canonical_sf1:.1f}|{canonical_sf2:.1f}|{sd_spat:.1f}"


def create_pipeline_surface_ids_for_level(level: int) -> set:
    """Return canonical averaged-surface IDs that belong to a grid level.

    Pipeline mode tracks one completion per canonical averaged surface. This is
    different from standard sample-file accounting, where diagonal runs and
    mirrors are separate sample combinations.
    """
    grid_info = get_grid_level_info(level)
    step_size = grid_info['step_size']
    fine_start = (config.param_range_low - step_size
                  if level == 2 else config.param_range_low)
    param_vals = np.arange(fine_start, config.param_range_high + step_size, step_size)
    ids = {
        create_surface_identifier(float(sf1), float(sf2), float(sp))
        for sf1 in param_vals
        for sf2 in param_vals
        for sp in param_vals
        if sf1 <= sf2
    }

    if level == 2:
        ids -= create_pipeline_surface_ids_for_level(1)

    return ids


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

    # Save pickle file atomically: write to .tmp then rename to avoid partial reads
    file = output_dir / f"samples_{param_name}_{param_hash}.pkl.gz"
    tmp = file.parent / (file.name + '.tmp')
    with gzip.open(tmp, 'wb') as f:
        pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(file)

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
            summary = _load_json_with_retry(summary_file)
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
                 algorithm: str = "EM",
                 lock_backend: Optional[LockBackend] = None,
                 lock_ttl: int = 1800,
                 pipeline: bool = False,
                 averaged_surfaces_dir: Optional[Path] = None,
                 completion_registry: Optional[str] = None,
                 bundle_output: bool = False,
                 bundle_dir: Optional[Path] = None,
                 save_samples: bool = False,
                 bias_bandwidth: float = 0.075,
                 feat_bandwidth: float = 3.0,
                 chunk_size: Optional[int] = None):
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
            lock_backend: LockBackend instance; defaults to FileLockBackend.
            lock_ttl: Lock TTL in seconds (heartbeat keeps it alive).
            pipeline: If True, average samples in-memory and save only averaged
                      surfaces, skipping intermediate sample files.
            averaged_surfaces_dir: Output directory for averaged surfaces when
                                   pipeline=True. Derived from samples_folder if None.
            completion_registry: Redis completed-set registry name. If omitted,
                                 pipeline mode uses the averaged-surfaces folder name.
            bundle_output: If True, upload one compressed chunk bundle instead of
                           uploading each surface file separately.
            bundle_dir: Local directory for chunk bundles when bundle_output=True.
            save_samples: If True (and pipeline=True), also write sample files.
            bias_bandwidth: KDE bandwidth for bias dimension (pipeline mode).
            feat_bandwidth: KDE bandwidth for feat_diff dimension (pipeline mode).
            chunk_size: Optional override for both large and small dynamic chunk
                        sizes, useful for smoke tests.
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
        self.lock_ttl = lock_ttl
        self.pipeline = pipeline
        self.save_samples = save_samples
        self.bundle_output = bundle_output

        # Lock backend — default to file locks in the output directory
        self.lock_backend = lock_backend  # set after mkdir below

        # Dynamic chunking parameters
        self.large_chunk_size = 50
        self.small_chunk_size = 5
        self.small_chunk_threshold = 10
        if chunk_size is not None:
            self.large_chunk_size = chunk_size
            self.small_chunk_size = chunk_size

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Resolve lock backend now that output_dir exists
        if self.lock_backend is None:
            self.lock_backend = FileLockBackend(self.output_dir)

        # Averaged surfaces directory (pipeline mode)
        if pipeline:
            if averaged_surfaces_dir is not None:
                self.averaged_surfaces_dir = Path(averaged_surfaces_dir)
            else:
                folder_name = Path(config.samples_folder).name
                parent = Path(config.samples_folder).parent
                self.averaged_surfaces_dir = parent / folder_name.replace(
                    'sim_samples_', 'averaged_surfaces_', 1
                )
            self.averaged_surfaces_dir.mkdir(parents=True, exist_ok=True)
            if bundle_output:
                if bundle_dir is not None:
                    self.bundle_dir = Path(bundle_dir)
                else:
                    self.bundle_dir = self.averaged_surfaces_dir.parent / (
                        self.averaged_surfaces_dir.name + "_bundles"
                    )
                self.bundle_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.bundle_dir = None
            object_store_config = ObjectStoreConfig.from_env()
            self.object_store = (
                SurfaceObjectStore(object_store_config) if object_store_config else None
            )
            self.completion_registry = (
                completion_registry
                or (self.averaged_surfaces_dir.name if self.object_store else None)
            )

            print("Compiling KDE averaging functions (pipeline mode)...")
            self.avg_fns = make_averaging_functions(
                config, bias_bandwidth=bias_bandwidth,
                feat_bandwidth=feat_bandwidth,
            )
            print("KDE functions ready.")
            self._reconcile_object_store_completion()
        else:
            self.averaged_surfaces_dir = None
            self.completion_registry = completion_registry
            self.object_store = None
            self.bundle_dir = None
            self.avg_fns = None

        # Initialize parameter grid for this level.
        # param_groups is a list of groups; each group is a list of 1 or 2 combinations.
        # Off-diagonal pairs (sf1 != sf2) form groups of 2 so they always land in the
        # same chunk.  Diagonal combinations (sf1 == sf2) form singleton groups.
        self.param_groups = self._create_param_groups()
        self.total_groups = len(self.param_groups)
        self.total_combinations = sum(len(g) for g in self.param_groups)

        lock_type = type(self.lock_backend).__name__
        print(f"=== Enhanced Chunked Grid Computer ===")
        print(f"Machine ID: {self.machine_id}")
        print(f"Platform: {self.platform}")
        print(f"Algorithm: {self.algorithm}")
        print(f"Lock backend: {lock_type}  TTL: {self.lock_ttl}s")
        print(f"Mode: {'pipeline (sample→average in-memory)' if pipeline else 'standard (save sample files)'}")
        if custom_param_list:
            print(f"Input: Custom parameter list ({len(custom_param_list)} combinations)")
        else:
            print(f"Grid Level: {self.grid_info['level']} - {self.grid_info['description']}")
            print(
                f"Parameter range: {config.param_range_low}-{config.param_range_high}, step: {self.grid_info['step_size']}")
        print(f"Total parameter combinations: {self.total_combinations:,} ({self.total_groups:,} groups)")
        print(f"Large/small chunk sizes: {self.large_chunk_size}/{self.small_chunk_size}")
        if pipeline:
            print(f"Averaged surfaces dir: {self.averaged_surfaces_dir}")
            print(f"Completion registry: {self.completion_registry}")
            print(f"Bundle output: {'enabled' if self.bundle_output else 'disabled'}")
            if self.bundle_output:
                print(f"Bundle dir: {self.bundle_dir}")
            if self.object_store:
                print(
                    f"Object store: s3://{self.object_store.config.bucket}/"
                    f"{self.object_store.config.prefix}"
                )
            else:
                print("Object store: disabled")
        else:
            print(f"Save full results: {self.save_full_results}, Save CSV: {self.save_csv}")


    def _create_param_groups(self) -> List[List[Tuple]]:
        """Create parameter combination groups for the current grid level or custom list.

        Each group is a list of 1 or 2 tuples (i, j, k, sf1, sf2, sp).
        Off-diagonal pairs (sf1 != sf2) yield a group of two: canonical (sf1 < sf2)
        followed by its mirror (sf2, sf1).  Diagonal combinations yield a singleton.
        Grouping guarantees that mirrors always fall in the same chunk.
        """
        if self.custom_param_list:
            if not self.pipeline:
                return [[(idx, idx, idx, float(sf1), float(sf2), float(sp), None)]
                        for idx, (sf1, sf2, sp) in enumerate(self.custom_param_list)]

            # Pipeline mode: group mirrors/diagonals into 2-entry groups.
            # Off-diagonal (sf1 != sf2): canonical (min,max) + mirror (max,min).
            # Diagonal (sf1 == sf2): two independent runs r0, r1.
            # Deduplicate by canonical key so mirror entries in the list don't double-count.
            seen: set = set()
            groups: List[List[Tuple]] = []
            for idx, (sf1, sf2, sp) in enumerate(self.custom_param_list):
                sf1, sf2, sp = float(sf1), float(sf2), float(sp)
                if sf1 == sf2:
                    key = (sf1, sf2, sp)
                    if key not in seen:
                        seen.add(key)
                        groups.append([
                            (idx, idx, idx, sf1, sf2, sp, 0),
                            (idx, idx, idx, sf1, sf2, sp, 1),
                        ])
                else:
                    canon_sf1, canon_sf2 = min(sf1, sf2), max(sf1, sf2)
                    key = (canon_sf1, canon_sf2, sp)
                    if key not in seen:
                        seen.add(key)
                        groups.append([
                            (idx, idx, idx, canon_sf1, canon_sf2, sp, None),
                            (idx, idx, idx, canon_sf2, canon_sf1, sp, None),
                        ])
            return groups

        step_size = self.grid_info['step_size']
        fine_start = (config.param_range_low - step_size
                      if self.grid_level == 2 else config.param_range_low)
        param_vals = np.arange(fine_start, config.param_range_high + step_size, step_size)
        groups = []
        for i, sf1 in enumerate(param_vals):
            for j, sf2 in enumerate(param_vals):
                if sf1 > sf2:
                    continue  # covered as the mirror in group (sf2, sf1)
                for k, sp in enumerate(param_vals):
                    if sf1 < sf2:
                        # Off-diagonal: canonical + mirror, no run_index needed
                        groups.append([
                            (i, j, k, float(sf1), float(sf2), float(sp), None),
                            (j, i, k, float(sf2), float(sf1), float(sp), None),
                        ])
                    else:
                        # Diagonal (sf1==sf2): two independent runs with different seeds
                        # so the averaged surface has the same sample count as off-diagonal.
                        groups.append([
                            (i, j, k, float(sf1), float(sf2), float(sp), 0),
                            (i, j, k, float(sf1), float(sf2), float(sp), 1),
                        ])
        return groups

    def _samples_exist(self, param_name: str, param_hash: str) -> bool:
        """Check if surface file exists and is complete."""
        samples_file = self.output_dir / f"samples_{param_name}_{param_hash}.pkl.gz"

        if not samples_file.exists():
            return False

        try:
            with gzip.open(samples_file, 'rb') as f:
                data = pickle.load(f)
                return ('parameters' in data and ('mu1_samples' in data or 'mu2_samples' in data))
        except (pickle.PickleError, EOFError, gzip.BadGzipFile):
            # Corrupt file — delete and recompute
            try:
                samples_file.unlink()
            except Exception:
                pass
            return False
        except (OSError, FileNotFoundError):
            # Transient error (e.g. file being written by another process) — skip
            return False

    def _acquire_chunk_lock(self, chunk_id: str) -> bool:
        return self.lock_backend.acquire(chunk_id, self.machine_id, self.lock_ttl)

    def _log_error(self, param_name: str, param_hash: str,
                   sd_feat1: float, sd_feat2: float, sd_spat: float,
                   error: Exception) -> None:
        """Append a write-failure entry to the machine-specific error log (JSONL)."""
        log_file = self.output_dir / f"errors_{self.machine_id}.jsonl"
        entry = {
            'timestamp': datetime.now().isoformat(),
            'param_name': param_name,
            'param_hash': param_hash,
            'sd_feat1': sd_feat1,
            'sd_feat2': sd_feat2,
            'sd_spat': sd_spat,
            'error': str(error),
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def _check_error_logs(self) -> None:
        """After all work is done, verify that previously errored files are now correct."""
        log_file = self.output_dir / f"errors_{self.machine_id}.jsonl"
        if not log_file.exists():
            return

        errors = []
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        errors.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not errors:
            return

        still_broken = [
            e for e in errors
            if not self._samples_exist(e['param_name'], e['param_hash'])
        ]

        if still_broken:
            print(f"\n[{self.machine_id}] WARNING: {len(still_broken)}/{len(errors)} "
                  f"previously errored files are still missing or corrupt:")
            for e in still_broken:
                print(f"  {e['param_name']} (error at {e['timestamp']}): {e['error']}")
        else:
            print(f"\n[{self.machine_id}] All {len(errors)} previously errored files "
                  f"are now present and correct.")

    def _save_with_retry(self, output_dir: Path, mu1_bias_array, mu2_bias_array,
                         param_name: str, param_hash: str,
                         sd_feat1: float, sd_feat2: float, sd_spat: float,
                         samples_time: float, samples_params: Dict,
                         run_index: Optional[int] = None,
                         full_results_combined=None) -> str:
        """Save samples with retry logic for OneDrive/network filesystem conflicts.

        Returns 'saved' or 'error'.
        Raises ChunkConflict if another machine's file appears during a retry — the caller
        should release the chunk lock and call _find_available_chunk again.
        On persistent failure the error is logged to errors_{machine_id}.jsonl.
        """
        max_retries = 3
        wait_seconds = 60
        last_exc = None

        for attempt in range(max_retries):
            try:
                save_samples_checkpoint(
                    output_dir, mu1_bias_array, mu2_bias_array,
                    param_name, param_hash, sd_feat1, sd_feat2, sd_spat,
                    samples_time, self.machine_id,
                    n_simulations=samples_params['n_simulations'],
                    n_samples=samples_params['n_samples'],
                    random_seed=samples_params['random_seed'],
                    full_results=full_results_combined,
                    save_csv=self.save_csv,
                )
                return 'saved'
            except OSError as e:
                last_exc = e
                print(f"  [{self.machine_id}] Write failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  [{self.machine_id}] Waiting {wait_seconds}s then checking for conflict...")
                time.sleep(wait_seconds)
                if self._samples_exist(param_name, param_hash):
                    print(f"  [{self.machine_id}] File appeared — another machine wrote it.")
                    raise ChunkConflict(
                        f"Write conflict on {param_name}: another machine owns this work"
                    )
                print(f"  [{self.machine_id}] File still absent, retrying write...")

        print(f"  [{self.machine_id}] All {max_retries} retries exhausted for {param_name}, logging error.")
        self._log_error(param_name, param_hash, sd_feat1, sd_feat2, sd_spat, last_exc)
        return 'error'

    def _get_completed_keys(self) -> set:
        """Return the set of completed keys used for progress tracking.

        In standard mode: set of sample file hashes.
        In pipeline mode: set of averaged surface canonical names (sf1_sf2_sp).
        """
        if self.pipeline:
            completed = set()
            if self.bundle_output:
                for manifest_file in self.bundle_dir.glob("surface_bundle_*.manifest.json"):
                    try:
                        with open(manifest_file) as f:
                            manifest = json.load(f)
                        completed.update(manifest.get('surface_ids', []))
                    except (OSError, json.JSONDecodeError):
                        continue
            else:
                import re
                pat = re.compile(
                    r"averaged_sf1_(?P<sf1>[\d.]+)_sf2_(?P<sf2>[\d.]+)_sp_(?P<sp>[\d.]+)\.pkl$"
                )
                for f in self.averaged_surfaces_dir.glob("averaged_sf1_*.pkl"):
                    m = pat.match(f.name)
                    if m:
                        completed.add(create_surface_identifier(
                            float(m['sf1']), float(m['sf2']), float(m['sp'])
                        ))
            if self.completion_registry:
                try:
                    completed.update(self.lock_backend.completed_members(self.completion_registry))
                except Exception as e:
                    print(f"[{self.machine_id}] Warning: could not read completion registry: {e}")
            return completed
        return load_progress_state(self.output_dir)['completed_hashes']

    def _reconcile_object_store_completion(self) -> None:
        """Seed the Redis completed set from durable object storage on startup."""
        if not self.pipeline or not self.object_store or not self.completion_registry:
            return
        try:
            s3_completed = set(self.object_store.list_completed_surface_ids())
        except Exception as e:
            print(f"[{self.machine_id}] Warning: could not list object storage: {e}")
            return
        if not s3_completed:
            return
        # One SMEMBERS call to find what Redis already knows about
        redis_completed = self.lock_backend.completed_members(self.completion_registry)
        missing = s3_completed - redis_completed
        if not missing:
            print(
                f"[{self.machine_id}] Reconciliation: Redis already has all "
                f"{len(redis_completed)} S3 completions — skipping"
            )
            return
        # Single SADD round-trip for all missing entries
        added = self.lock_backend.mark_completed_batch(self.completion_registry, missing)
        print(
            f"[{self.machine_id}] Reconciled {added} new completions into "
            f"{self.completion_registry} (S3={len(s3_completed)}, "
            f"Redis had {len(redis_completed)})"
        )

    def _group_is_complete(self, group: List[Tuple], completed_keys: set) -> bool:
        """Return True if all outputs for this group already exist."""
        if self.pipeline:
            # Pipeline output is one averaged surface per group, keyed by canonical (sf1<=sf2, sp)
            _, _, _, sf1, sf2, sp, _ = group[0]
            return create_surface_identifier(sf1, sf2, sp) in completed_keys
        # Standard mode: check all sample hashes
        for (_, _, _, sf1, sf2, sp, run_index) in group:
            _, param_hash = create_param_identifier(sf1, sf2, sp, run_index)
            if param_hash not in completed_keys:
                return False
        return True

    def _find_available_chunk(self) -> Tuple[Optional[int], Optional[int]]:
        """Find the next available chunk to work on and acquire its lock."""
        completed_keys = self._get_completed_keys()

        incomplete_group_indices = {
            idx for idx, group in enumerate(self.param_groups)
            if not self._group_is_complete(group, completed_keys)
        }

        if not incomplete_group_indices:
            return None, None

        remaining_groups = len(incomplete_group_indices)
        potential_large_chunks = remaining_groups // self.large_chunk_size
        chunk_size = (self.small_chunk_size if potential_large_chunks <= self.small_chunk_threshold
                      else self.large_chunk_size)

        if potential_large_chunks <= self.small_chunk_threshold:
            print(f"[{self.machine_id}] Switching to small chunks (size={chunk_size})")

        import random
        candidate_chunks = []
        for chunk_start in range(0, self.total_groups, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.total_groups)
            chunk_id = f"L{self.grid_level}_chunk_{chunk_start:05d}_{chunk_end:05d}"
            if any(idx in incomplete_group_indices for idx in range(chunk_start, chunk_end)):
                candidate_chunks.append((chunk_start, chunk_end, chunk_id))

        random.shuffle(candidate_chunks)
        for chunk_start, chunk_end, chunk_id in candidate_chunks:
            if self._acquire_chunk_lock(chunk_id):
                return chunk_start, chunk_size

        return None, None

    def _compute_samples_for_entry(self, sd_feat1: float, sd_feat2: float, sd_spat: float,
                                    run_index: Optional[int], samples_params: Dict,
                                    feat_diff_vals) -> Tuple:
        """Generate mu1/mu2 sample arrays for one parameter combination.

        Returns (mu1_bias_array, mu2_bias_array, full_results_combined).
        full_results_combined is None unless save_full_results is True.
        """
        num_scan_loops = 10
        num_sims_per_loop = samples_params['n_simulations'] // num_scan_loops

        def scan_fn(carry, inputs):
            subkey, feat_diff, sf1, sf2, sp = inputs
            if self.save_full_results:
                b1, b2, full = jfm.simulate_dual_component_bias_distribution(
                    subkey, sf1, sf2, sp, feat_diff, 42.0, num_sims_per_loop,
                    samples_params['n_samples'], return_full_results=True,
                    fix_weights=self.fix_weights, algorithm=self.algorithm,
                    diagonal_covariance=self.diagonal_covariance,
                )
                return carry, (b1, b2, full)
            b1, b2 = jfm.simulate_dual_component_bias_distribution(
                subkey, sf1, sf2, sp, feat_diff, 42.0, num_sims_per_loop,
                samples_params['n_samples'], return_full_results=False,
                fix_weights=self.fix_weights, algorithm=self.algorithm,
                diagonal_covariance=self.diagonal_covariance,
            )
            return carry, (b1, b2)

        _, param_hash = create_param_identifier(sd_feat1, sd_feat2, sd_spat, run_index)
        base_key = jax.random.PRNGKey(samples_params['random_seed'])
        param_key = jax.random.fold_in(base_key, int(param_hash, 16))
        key = jax.random.fold_in(param_key, run_index if run_index is not None else 0)

        sd1_arr = jnp.full(len(feat_diff_vals), sd_feat1)
        sd2_arr = jnp.full(len(feat_diff_vals), sd_feat2)
        sp_arr  = jnp.full(len(feat_diff_vals), sd_spat)

        all_mu1, all_mu2, all_full = [], [], []
        for rep in range(num_scan_loops):
            key, *subkeys = jax.random.split(key, len(feat_diff_vals) + 1)
            tic = time.time()
            if self.save_full_results:
                _, (mu1, mu2, full) = jax.lax.scan(
                    scan_fn, None,
                    (jnp.array(subkeys), feat_diff_vals, sd1_arr, sd2_arr, sp_arr)
                )
                all_full.append(full)
            else:
                _, (mu1, mu2) = jax.lax.scan(
                    scan_fn, None,
                    (jnp.array(subkeys), feat_diff_vals, sd1_arr, sd2_arr, sp_arr)
                )
            print(f"    Scan {rep + 1}/{num_scan_loops}: {time.time() - tic:.1f}s")
            all_mu1.append(mu1)
            all_mu2.append(mu2)

        mu1_out = jnp.concatenate(all_mu1, axis=1)
        mu2_out = jnp.concatenate(all_mu2, axis=1)
        full_out = jnp.concatenate(all_full, axis=1) if self.save_full_results else None
        return mu1_out, mu2_out, full_out

    def _process_group_pipeline(self, group: List[Tuple], samples_params: Dict,
                                 feat_diff_vals) -> str:
        """Pipeline mode: compute both entries in a group, average in-memory, save surface.

        Returns 'saved', 'skipped', or 'error'.
        """
        _, _, _, sf1_a, sf2_a, sp, run_index_a = group[0]
        _, _, _, sf1_b, sf2_b, _,  run_index_b = group[1]

        canon_sf1, canon_sf2 = min(sf1_a, sf2_a), max(sf1_a, sf2_a)
        surface_id = create_surface_identifier(sf1_a, sf2_a, sp)
        surface_file = self.averaged_surfaces_dir / (
            f"averaged_sf1_{canon_sf1:.1f}_sf2_{canon_sf2:.1f}_sp_{sp:.1f}.pkl"
        )

        completed_remotely = False
        if self.completion_registry:
            completed_remotely = surface_id in self.lock_backend.completed_members(
                self.completion_registry
            )
        if not completed_remotely and self.object_store and not self.bundle_output:
            completed_remotely = self.object_store.object_exists(surface_file.name)
            if completed_remotely and self.completion_registry:
                self.lock_backend.mark_completed(self.completion_registry, surface_id)

        if surface_file.exists() or completed_remotely:
            return 'skipped'

        print(f"  [{self.machine_id}] Pipeline: sf1={sf1_a:.1f} sf2={sf2_a:.1f} sp={sp:.1f}")

        try:
            mu1_a, mu2_a, full_a = self._compute_samples_for_entry(
                sf1_a, sf2_a, sp, run_index_a, samples_params, feat_diff_vals
            )
            mu1_b, mu2_b, full_b = self._compute_samples_for_entry(
                sf1_b, sf2_b, sp, run_index_b, samples_params, feat_diff_vals
            )
        except Exception as e:
            print(f"  [{self.machine_id}] Simulation error: {e}")
            return 'error'

        # Optionally persist sample files
        if self.save_samples:
            for (sf1, sf2, sp_, ri, mu1, mu2, full) in [
                (sf1_a, sf2_a, sp, run_index_a, mu1_a, mu2_a, full_a),
                (sf1_b, sf2_b, sp, run_index_b, mu1_b, mu2_b, full_b),
            ]:
                pname, phash = create_param_identifier(sf1, sf2, sp_, ri)
                self._save_with_retry(
                    self.output_dir, mu1, mu2, pname, phash,
                    sf1, sf2, sp_, 0.0, samples_params, ri, full,
                )

        try:
            averaged = average_sample_pair(
                mu1_a, mu2_a, mu1_b, mu2_b, sf1_a, sf2_a, self.avg_fns
            )
        except Exception as e:
            print(f"  [{self.machine_id}] Averaging error: {e}")
            return 'error'

        # Atomic write
        tmp = surface_file.with_suffix('.pkl.tmp')
        import pickle as _pickle
        with open(tmp, 'wb') as f:
            _pickle.dump({
                'parameters': {
                    'sd_feat1': canon_sf1, 'sd_feat2': canon_sf2, 'sd_spat': sp,
                    'machine_id': self.machine_id,
                    'n_simulations': samples_params['n_simulations'],
                    'n_samples': samples_params['n_samples'],
                    'random_seed': samples_params['random_seed'],
                    'git_commit': get_git_commit(),
                },
                'surface': averaged,
                'creation_timestamp': str(jnp.datetime64('now') if False else __import__('datetime').datetime.now().isoformat()),
            }, f)
        tmp.replace(surface_file)
        if self.object_store and not self.bundle_output:
            try:
                key = self.object_store.upload_surface(surface_file)
                print(f"  [{self.machine_id}] Uploaded {surface_file.name} to s3://{self.object_store.config.bucket}/{key}")
            except Exception as e:
                print(f"  [{self.machine_id}] Object-store upload failed: {e}")
                surface_file.unlink(missing_ok=True)
                return 'error'
        if self.completion_registry and not self.bundle_output:
            try:
                self.lock_backend.mark_completed(self.completion_registry, surface_id)
            except Exception as e:
                print(f"  [{self.machine_id}] Warning: could not mark completed in Redis: {e}")
        print(f"  [{self.machine_id}] Saved {surface_file.name}")
        return 'saved'

    def _surface_info_for_group(self, group: List[Tuple]) -> Tuple[str, Path]:
        """Return the canonical surface ID and expected local path for a group."""
        _, _, _, sf1, sf2, sp, _ = group[0]
        canon_sf1, canon_sf2 = min(sf1, sf2), max(sf1, sf2)
        surface_id = create_surface_identifier(sf1, sf2, sp)
        surface_file = self.averaged_surfaces_dir / (
            f"averaged_sf1_{canon_sf1:.1f}_sf2_{canon_sf2:.1f}_sp_{sp:.1f}.pkl"
        )
        return surface_id, surface_file

    def _bundle_chunk_outputs(self, chunk_id: str, chunk_start_idx: int,
                              chunk_end_idx: int, samples_params: Dict) -> bool:
        """Create/upload a compressed bundle for local surfaces produced in a chunk."""
        if not (self.pipeline and self.bundle_output):
            return True

        surfaces = {}
        surface_ids = []
        missing = []
        completed = set()
        if self.completion_registry:
            try:
                completed = self.lock_backend.completed_members(self.completion_registry)
            except Exception as e:
                print(f"  [{self.machine_id}] Warning: could not read completion registry before bundling: {e}")

        for group_idx in range(chunk_start_idx, chunk_end_idx):
            if group_idx >= self.total_groups:
                break
            surface_id, surface_file = self._surface_info_for_group(self.param_groups[group_idx])
            if surface_id in completed:
                continue
            if not surface_file.exists():
                missing.append(surface_file.name)
                continue
            surfaces[surface_file.name] = surface_file.read_bytes()
            surface_ids.append(surface_id)

        if missing:
            print(f"  [{self.machine_id}] Bundle skipped; {len(missing)} local surfaces missing")
            return False
        if not surfaces:
            return True

        bundle_name = f"surface_bundle_{chunk_id}.pkl.gz"
        manifest_name = f"surface_bundle_{chunk_id}.manifest.json"
        bundle_path = self.bundle_dir / bundle_name
        manifest_path = self.bundle_dir / manifest_name
        manifest = {
            'chunk_id': chunk_id,
            'machine_id': self.machine_id,
            'surface_count': len(surface_ids),
            'surface_ids': surface_ids,
            'surface_files': sorted(surfaces),
            'parameters': {
                'n_simulations': samples_params['n_simulations'],
                'n_samples': samples_params['n_samples'],
                'random_seed': samples_params['random_seed'],
                'git_commit': get_git_commit(),
            },
            'created_at': datetime.now().isoformat(),
        }

        tmp_bundle = bundle_path.with_suffix(bundle_path.suffix + '.tmp')
        with gzip.open(tmp_bundle, 'wb') as f:
            pickle.dump({'manifest': manifest, 'surfaces': surfaces}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_bundle.replace(bundle_path)

        tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + '.pending')
        with open(tmp_manifest, 'w') as f:
            json.dump(manifest, f, indent=2)

        if self.object_store:
            try:
                bundle_key = self.object_store.upload_file(bundle_path)
                manifest_key = self.object_store.upload_file(
                    tmp_manifest, object_name=manifest_name
                )
                print(
                    f"  [{self.machine_id}] Uploaded bundle s3://{self.object_store.config.bucket}/{bundle_key} "
                    f"and manifest {manifest_key}"
                )
            except Exception as e:
                print(f"  [{self.machine_id}] Bundle upload failed: {e}")
                return False

        # The manifest is the local completion marker. Publish it only after the
        # durable bundle and manifest uploads have both succeeded.
        tmp_manifest.replace(manifest_path)

        if self.completion_registry:
            try:
                for surface_id in surface_ids:
                    self.lock_backend.mark_completed(self.completion_registry, surface_id)
            except Exception as e:
                print(f"  [{self.machine_id}] Warning: could not mark bundled surfaces complete: {e}")
                return False

        print(f"  [{self.machine_id}] Bundled {len(surface_ids)} surfaces into {bundle_name}")
        return True

    def _process_chunk(self, chunk_start_idx: int, chunk_size: int, samples_params: Dict) -> Tuple[int, int, float]:
        """Process a single chunk of parameter combinations."""
        chunk_end_idx = min(chunk_start_idx + chunk_size, self.total_groups)
        chunk_id = f"L{self.grid_level}_chunk_{chunk_start_idx:05d}_{chunk_end_idx:05d}"

        print(f"\n[{self.machine_id}] Processing {chunk_id}")
        print(f"Parameter indices: {chunk_start_idx} to {chunk_end_idx - 1}")

        chunk_start_time = time.time()
        surfaces_computed = surfaces_skipped = 0
        feat_diff_vals = config.create_grid('feat_diff')
        feat_diff_vals = feat_diff_vals[feat_diff_vals > 0]

        for group_idx in range(chunk_start_idx, chunk_end_idx):
            if group_idx >= self.total_groups:
                break

            group = self.param_groups[group_idx]

            if self.pipeline:
                result = self._process_group_pipeline(group, samples_params, feat_diff_vals)
                if result == 'saved':
                    surfaces_computed += 1
                elif result == 'skipped':
                    surfaces_skipped += 1
            else:
                # Standard mode: process each combination independently
                for (i, j, k, sd_feat1, sd_feat2, sd_spat, run_index) in group:
                    param_name, param_hash = create_param_identifier(
                        sd_feat1, sd_feat2, sd_spat, run_index
                    )
                    if self._samples_exist(param_name, param_hash):
                        surfaces_skipped += 1
                        continue

                    samples_start = time.time()
                    print(f"  [{self.machine_id}] Computing {surfaces_computed + 1}: "
                          f"sd_feat1={sd_feat1:.1f}, sd_feat2={sd_feat2:.1f}, "
                          f"sd_spat={sd_spat:.1f}")

                    mu1, mu2, full = self._compute_samples_for_entry(
                        sd_feat1, sd_feat2, sd_spat, run_index, samples_params, feat_diff_vals
                    )
                    samples_time = time.time() - samples_start
                    result = self._save_with_retry(
                        self.output_dir, mu1, mu2, param_name, param_hash,
                        sd_feat1, sd_feat2, sd_spat, samples_time, samples_params,
                        run_index, full,
                    )
                    if result == 'saved':
                        surfaces_computed += 1
                        print(f"    Completed in {samples_time:.1f}s")

        if self.pipeline and self.bundle_output:
            if not self._bundle_chunk_outputs(chunk_id, chunk_start_idx, chunk_end_idx, samples_params):
                raise RuntimeError(f"{chunk_id} bundle was not completed")

        chunk_time = time.time() - chunk_start_time
        label = 'surfaces' if self.pipeline else 'samples'
        print(f"[{self.machine_id}] {chunk_id} completed:")
        print(f"  {label.capitalize()} computed: {surfaces_computed}, skipped: {surfaces_skipped}")
        print(f"  Chunk time: {chunk_time:.1f}s")
        if surfaces_computed > 0:
            print(f"  Average time per {label[:-1]}: {chunk_time / surfaces_computed:.1f}s")

        return surfaces_computed, surfaces_skipped, chunk_time

    def _save_progress_summary(self, session_stats: Dict) -> None:
        """Save progress summary for this machine."""
        summary_file = self.output_dir / f"progress_summary_{self.machine_id}.json"
        progress = load_progress_state(self.output_dir)
        if self.pipeline:
            completed_count = len(self._get_completed_keys())
            total_count = self.total_groups
        else:
            completed_count = len(progress['completed_hashes'])
            total_count = self.total_combinations

        summary = {
            'machine_id': self.machine_id,
            'platform': self.platform,
            'timestamp': datetime.now().isoformat(),
            'session_stats': session_stats,
            'total_combinations': total_count,
            'completed_count': completed_count,
            'completion_rate': completed_count / total_count * 100 if total_count else 0,
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
                     n_simulations: int = 1000, n_samples: int = 100, random_seed: int = 42,
                     max_chunks: Optional[int] = None) -> Dict:
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
            if max_chunks is not None and session_stats['chunks_processed'] >= max_chunks:
                print(f"\n[{self.machine_id}] Reached max_chunks={max_chunks}, stopping")
                break

            chunk_start, chunk_size = self._find_available_chunk()
            if chunk_start is None:
                print(f"\n[{self.machine_id}] No more work available")
                self._check_error_logs()
                break

            chunk_id = f"L{self.grid_level}_chunk_{chunk_start:05d}_{min(chunk_start + chunk_size, self.total_groups):05d}"
            interrupted = False
            with self.lock_backend.maintain(chunk_id, self.machine_id, self.lock_ttl):
                try:
                    surfaces_computed, surfaces_skipped, chunk_time = self._process_chunk(
                        chunk_start, chunk_size, samples_params
                    )

                    # Update session stats
                    session_stats['chunks_processed'] += 1
                    session_stats['samples_computed'] += surfaces_computed
                    session_stats['samples_skipped'] += surfaces_skipped
                    session_stats['total_computation_time'] += chunk_time

                    self._save_progress_summary(session_stats)
                    time.sleep(1)

                except ChunkConflict as e:
                    print(f"\n[{self.machine_id}] Chunk conflict detected: {e}")
                    print(f"[{self.machine_id}] Releasing lock on {chunk_id} and reassigning chunk.")

                except KeyboardInterrupt:
                    print(f"\n[{self.machine_id}] Interrupted by user")
                    interrupted = True
            if interrupted:
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
            lock_data = _load_json_with_retry(lock_file)
            active_chunks.append({
                'chunk_id': lock_file.stem.replace('computing_', ''),
                'machine_id': lock_data.get('machine_id', 'unknown'),
                'running_time': time.time() - lock_data.get('start_time', time.time())
            })
        except (json.JSONDecodeError, OSError):
            continue

    # Load machine summaries
    machine_summaries = {}
    for summary_file in output_dir.glob("progress_summary_*.json"):
        try:
            summary = _load_json_with_retry(summary_file)
            machine_summaries[summary.get('machine_id', 'unknown')] = summary
        except (json.JSONDecodeError, OSError):
            continue

    # Count surfaces that match this grid level
    level_surfaces = 0
    step_size = grid_info['step_size']
    import re

    # Strict pattern: only canonical filenames with 8-char hex hash and .pkl.gz extension.
    # Excludes OneDrive conflict copies, .tmp files, and other spurious matches.
    pattern = re.compile(r"samples_sf1_(?P<sf1>[\d.]+)_sf2_(?P<sf2>[\d.]+)_sp_(?P<sp>[\d.]+)(?:_r\d+)?_[a-f0-9]{8}\.pkl\.gz$")

    if grid_level == 1:
        # Level 1: count surfaces at coarse step intervals
        param_vals = np.arange(config.param_range_low, config.param_range_high + config.param_step, config.param_step)
        for samples_file in output_dir.glob("samples_sf1_*_sf2_*_sp_*.pkl.gz"):
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
        fine_start = config.param_range_low - step_size
        fine_param_vals = np.arange(fine_start, config.param_range_high + step_size, step_size)
        coarse_param_vals = np.arange(config.param_range_low, config.param_range_high + config.param_step, config.param_step)
        
        for samples_file in output_dir.glob("samples_sf1_*_sf2_*_sp_*.pkl.gz"):
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


def check_grid_level_readiness(target_level: int,
                               prerequisite_completion_rate: Optional[float] = None) -> Dict:
    """
    Check if the grid is ready for the target refinement level.

    Parameters:
    -----------
    target_level : int
        Target grid level to check readiness for
    prerequisite_completion_rate : Optional[float]
        Known completion percentage for the prerequisite level. Pipeline mode
        passes Redis/object-store based completion here because local sample
        files are intentionally absent.

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
        # Check if level 1 is complete. Standard mode can scan local sample
        # files. Pipeline mode should pass a Redis/object-store based rate
        # because sample files are intentionally not written there.
        if prerequisite_completion_rate is None:
            level1_status = get_grid_status(grid_level=1)
            completion_rate = level1_status['completion_rate']
        else:
            completion_rate = prerequisite_completion_rate
        level1_complete = completion_rate >= 99.0

        return {
            'ready': level1_complete,
            'message': f"Level 2 {'ready' if level1_complete else 'blocked'}: Level 1 at {completion_rate:.1f}%",
            'prerequisite_level': 1,
            'prerequisite_completion': completion_rate
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
                            algorithm: str = "EM",
                            lock_backend: str = 'auto', lock_ttl: int = 1800,
                            pipeline: bool = False, save_samples: bool = False,
                            averaged_surfaces_dir: Optional[str] = None,
                            completion_registry: Optional[str] = None,
                            bundle_output: bool = False,
                            bundle_dir: Optional[str] = None,
                            bias_bandwidth: float = 0.075, feat_bandwidth: float = 3.0,
                            chunk_size: Optional[int] = None,
                            max_chunks: Optional[int] = None,
                            **samples_params) -> Dict:
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

    coordination_backend = make_lock_backend(lock_backend,
                                             lock_dir=Path(config.samples_folder))

    def _pipeline_registry_name() -> Optional[str]:
        if completion_registry:
            return completion_registry
        if not pipeline:
            return None
        if averaged_surfaces_dir:
            return Path(averaged_surfaces_dir).name
        folder_name = Path(config.samples_folder).name
        return folder_name.replace('sim_samples_', 'averaged_surfaces_', 1)

    def _pipeline_prerequisite_completion_rate(target_level: int) -> Optional[float]:
        if not pipeline or target_level != 2:
            return None

        completed_keys = set()
        registry = _pipeline_registry_name()
        if registry:
            try:
                completed_keys.update(coordination_backend.completed_members(registry))
            except Exception as e:
                print(f"[{machine_id}] Warning: could not read completion registry for readiness: {e}")

        avg_dir = (Path(averaged_surfaces_dir) if averaged_surfaces_dir else None)
        if avg_dir is None:
            folder_name = Path(config.samples_folder).name
            parent = Path(config.samples_folder).parent
            avg_dir = parent / folder_name.replace('sim_samples_', 'averaged_surfaces_', 1)
        local_bundle_dir = (Path(bundle_dir) if bundle_dir else avg_dir.parent / (avg_dir.name + '_bundles'))

        if local_bundle_dir.exists():
            for manifest_file in local_bundle_dir.glob('surface_bundle_*.manifest.json'):
                try:
                    with open(manifest_file) as f:
                        manifest = json.load(f)
                    completed_keys.update(manifest.get('surface_ids', []))
                except (OSError, json.JSONDecodeError):
                    continue

        if avg_dir.exists():
            import re
            pat = re.compile(
                r"averaged_sf1_(?P<sf1>[\d.]+)_sf2_(?P<sf2>[\d.]+)_sp_(?P<sp>[\d.]+)\.pkl$"
            )
            for surface_file in avg_dir.glob('averaged_sf1_*.pkl'):
                match = pat.match(surface_file.name)
                if match:
                    completed_keys.add(create_surface_identifier(
                        float(match['sf1']), float(match['sf2']), float(match['sp'])
                    ))

        level1_ids = create_pipeline_surface_ids_for_level(1)
        if not level1_ids:
            return 0.0
        return 100.0 * len(completed_keys & level1_ids) / len(level1_ids)

    def _make_computer(level):
        return ChunkedGridComputer(
            machine_id, level,
            save_full_results=save_full_results, save_csv=save_csv,
            diagonal_covariance=diagonal_covariance, fix_weights=fix_weights,
            algorithm=algorithm,
            lock_backend=coordination_backend, lock_ttl=lock_ttl,
            pipeline=pipeline, save_samples=save_samples,
            averaged_surfaces_dir=Path(averaged_surfaces_dir) if averaged_surfaces_dir else None,
            completion_registry=completion_registry,
            bundle_output=bundle_output,
            bundle_dir=Path(bundle_dir) if bundle_dir else None,
            bias_bandwidth=bias_bandwidth, feat_bandwidth=feat_bandwidth,
            chunk_size=chunk_size,
        )

    # If using custom param list, skip grid level management
    if custom_param_list:
        print(f"✅ Using custom parameter list ({len(custom_param_list)} combinations)")
        computer = ChunkedGridComputer(
            machine_id, grid_level, custom_param_list=custom_param_list,
            save_full_results=save_full_results, save_csv=save_csv,
            diagonal_covariance=diagonal_covariance, fix_weights=fix_weights,
            algorithm=algorithm,
            lock_backend=coordination_backend, lock_ttl=lock_ttl,
            pipeline=pipeline, save_samples=save_samples,
            averaged_surfaces_dir=Path(averaged_surfaces_dir) if averaged_surfaces_dir else None,
            completion_registry=completion_registry,
            bundle_output=bundle_output,
            bundle_dir=Path(bundle_dir) if bundle_dir else None,
            bias_bandwidth=bias_bandwidth, feat_bandwidth=feat_bandwidth,
            chunk_size=chunk_size,
        )
        result = computer.compute_grid(max_chunks=max_chunks, **samples_params)
        return {'custom_params': result}

    # Standard grid level computation
    current_level = grid_level
    all_results = {}

    while True:
        # Check if current level is ready. In pipeline mode, use the
        # Redis/object-store completion view instead of local sample files.
        prerequisite_rate = _pipeline_prerequisite_completion_rate(current_level)
        readiness = check_grid_level_readiness(
            current_level, prerequisite_completion_rate=prerequisite_rate
        )
        if not readiness['ready']:
            print(f"❌ {readiness['message']}")
            if readiness['prerequisite_level']:
                print(f"💡 Suggestion: Complete level {readiness['prerequisite_level']} first "
                      f"({readiness['prerequisite_completion']:.1f}% done)")
            break

        print(f"✅ {readiness['message']}")

        # Run computation for current level
        computer = _make_computer(current_level)
        level_result = computer.compute_grid(max_chunks=max_chunks, **samples_params)
        all_results[f'level_{current_level}'] = level_result

        if max_chunks is not None and level_result.get('chunks_processed', 0) >= max_chunks:
            print(f"\nReached max_chunks={max_chunks}; not advancing to another grid level")
            break

        # Check if auto-advance is enabled and this level is now complete.
        # Use the computer's completion view (Redis + local manifests) rather than
        # get_grid_status() which scans local sample files — those don't exist in
        # pipeline mode, causing it to always report 0% and block level advance.
        if auto_advance:
            completed_keys = computer._get_completed_keys()
            completion_rate = (100.0 * len(completed_keys) / computer.total_groups
                               if computer.total_groups else 0.0)
            if completion_rate >= 99.0:
                next_level = current_level + 1
                # Level 1 being complete is the prerequisite for Level 2 — we just
                # confirmed it, so skip the redundant file-scan readiness check.
                if next_level <= 2:
                    print(f"\n🎯 Level {current_level} complete ({completion_rate:.1f}%)!"
                          f" Auto-advancing to level {next_level}")
                    current_level = next_level
                    continue
                else:
                    print(f"\n🏁 Level {current_level} complete ({completion_rate:.1f}%)!"
                          f" No more levels available.")
                    break
            else:
                print(f"\n⏸️  Level {current_level} incomplete ({completion_rate:.1f}%), stopping here")
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
            lock_data = _load_json_with_retry(lock_file)

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
            lock_data = _load_json_with_retry(lock_file)

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
    n_samples = args.n_samples if args.n_samples is not None else config.n_samples
    return args.n_simulations, n_samples


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
    print(f"Random Seed          : {args.random_seed}")
    print(f"Lock Backend         : {args.lock_backend}")
    print(f"Pipeline Mode        : {boolean_flag(args.pipeline)}")
    if args.pipeline:
        print(f"Save Samples         : {boolean_flag(args.save_samples)}")
        print(f"Completion Registry : {args.completion_registry or 'auto with object storage; disabled otherwise'}")
        print(f"Bundle Output       : {boolean_flag(args.bundle_output)}")
        if args.bundle_output:
            print(f"Bundle Dir          : {args.bundle_dir or 'derived from averaged-surfaces dir'}")
        print(f"Bias Bandwidth       : {args.bias_bandwidth}")
        print(f"Feat Bandwidth       : {args.feat_bandwidth}")
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
    parser.add_argument('--auto-advance', action=argparse.BooleanOptionalAction, default=True,
                        help='Automatically advance to the next grid level when complete '
                             '(default: enabled; use --no-auto-advance to stop after this level)')
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode with reduced parameters')
    parser.add_argument('--diagonal-covariance', action=argparse.BooleanOptionalAction, default=True,
                        help='Use diagonal covariance in the fit (default: True). This is the only '
                             'implemented mode; --no-diagonal-covariance raises NotImplementedError. '
                             'Matches build_samples_folder_name so folders are truthfully labeled diagcov.')
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
    parser.add_argument('--n-samples', type=int, default=None,
                        help='Samples per simulation (default: config value = 100)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Base random seed for reproducible simulations (default: 42)')
    parser.add_argument('--lock-backend', choices=['redis', 'file', 'auto'], default='auto',
                        help='Lock backend: redis, file, or auto (redis if REDIS_HOST set, else file)')
    parser.add_argument('--pipeline', action='store_true', default=False,
                        help='Average samples in-memory and save averaged surfaces directly, '
                             'skipping intermediate sample files')
    parser.add_argument('--save-samples', action='store_true', default=False,
                        help='When --pipeline is set, also write sample files to disk')
    parser.add_argument('--averaged-surfaces-dir', type=str, default=None,
                        help='Output directory for averaged surfaces in pipeline mode '
                             '(default: derived from samples folder name)')
    parser.add_argument('--completion-registry', type=str, default=None,
                        help='Redis completed-set registry name for pipeline mode '
                             '(default: averaged-surfaces folder name)')
    parser.add_argument('--bundle-output', action='store_true', default=False,
                        help='In pipeline mode, upload one compressed bundle per chunk '
                             'instead of individual surface files')
    parser.add_argument('--bundle-dir', type=str, default=None,
                        help='Local directory for chunk bundles '
                             '(default: <averaged-surfaces-dir>_bundles)')
    parser.add_argument('--bias-bandwidth', type=float, default=0.075,
                        help='KDE bandwidth for bias dimension in pipeline mode (default: 0.075)')
    parser.add_argument('--feat-bandwidth', type=float, default=3.0,
                        help='KDE bandwidth for feat_diff dimension in pipeline mode (default: 3.0)')
    parser.add_argument('--lock-ttl', type=int, default=1800,
                        help='Lock TTL in seconds; heartbeat keeps it alive (default: 1800)')
    parser.add_argument('--chunk-size', type=int, default=None,
                        help='Override dynamic chunk size; useful for smoke tests')
    parser.add_argument('--max-chunks', type=int, default=None,
                        help='Stop after processing this many chunks; useful for smoke tests')

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

        shared_kwargs = dict(
            machine_id=args.machine_id,
            grid_level=args.grid_level,
            auto_advance=args.auto_advance,
            custom_param_list=custom_params,
            save_full_results=args.save_full_results,
            save_csv=args.save_csv,
            diagonal_covariance=args.diagonal_covariance,
            fix_weights=args.fix_weights,
            algorithm=args.algorithm,
            lock_backend=args.lock_backend,
            lock_ttl=args.lock_ttl,
            pipeline=args.pipeline,
            save_samples=args.save_samples,
            averaged_surfaces_dir=args.averaged_surfaces_dir,
            completion_registry=args.completion_registry,
            bundle_output=args.bundle_output,
            bundle_dir=args.bundle_dir,
            bias_bandwidth=args.bias_bandwidth,
            feat_bandwidth=args.feat_bandwidth,
            chunk_size=args.chunk_size,
            max_chunks=args.max_chunks,
            random_seed=args.random_seed,
        )

        if args.test_mode:
            print("Running in test mode")
            session_stats = run_chunked_computation(
                **shared_kwargs,
                n_simulations=active_n_simulations,
                n_samples=active_n_samples,
                feat_diff_step=4, mu1_bias_step=4, mu2_bias_step=12,
            )
        else:
            print("Running full computation")
            session_stats = run_chunked_computation(
                **shared_kwargs,
                n_simulations=active_n_simulations,
                n_samples=active_n_samples,
            )

        print(f"\nSession completed. Final status:")
        if args.pipeline:
            print(
                "Pipeline mode writes averaged surfaces/bundles; use "
                "surface_computation/grid_status_monitor.py or cloud/cloud_status.py "
                "for pipeline completion status."
            )
        elif not custom_params:
            print_grid_status()
if __name__ == "__main__":
    parse_arguments()
