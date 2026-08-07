"""Shared utility helpers for checkpoints, surfaces, and data handling."""

import json
import random
import re
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import itertools
import numpy as np
from flax.training import train_state
import inspect, pickle
import gzip
from shared.surface_folder_parsing import load_surface
from shared.config import config
from shared.mu1_axis import (assert_mu1_axis, legacy_mu1_axis, mu1_cell_width,
                            mu1_size, periodic_integral, sign_masks,
                            trim_legacy_rows, GRID_CONVENTION,
                            LEGACY_MU1_GRID_SIZE)
from shared.plotting import plot_surface_comparison
from jax.scipy.special import logsumexp

from shared.surface_functions import compute_expectation
import jax.numpy as jnp
import jax


# Periodic images summed on each side when a KDE over a circular error axis is
# built. One is already far past double precision: these bandwidths are a few
# degrees against a 360-degree period, so the nearest omitted image sits >100
# sigma away. Raise it only if a bandwidth ever approaches the period.
KDE_WRAPS = 1


def get_git_commit() -> Optional[str]:
    """Return the current HEAD commit hash, or None if not in a git repo."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def resolve_results_path(path: str, results_dir: str = "results") -> Path:
    """Place relative outputs under results_dir while preserving folder name."""
    candidate = Path(path)
    if candidate.is_absolute() or not results_dir:
        return candidate
    if candidate.parts and candidate.parts[0] == results_dir:
        return candidate
    return Path(results_dir) / candidate


def resolve_input_path(path: str, results_dir: str = "results") -> Path:
    """Resolve inputs, checking results_dir fallback for relative paths."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    resolved = resolve_results_path(path, results_dir)
    return resolved if resolved.exists() else candidate


def save_checkpoint(state, epoch, loss, save_dir="checkpoints", prefix="model"):
    """
    Save model checkpoint.

    Parameters:
    -----------
    state : TrainState
        The training state to save
    epoch : int
        Current epoch number
    loss : float
        Current loss value
    save_dir : str
        Directory to save checkpoints
    prefix : str
        Prefix for checkpoint filenames
    """
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Create checkpoint filename
    checkpoint_path = Path(save_dir) / f"{prefix}_epoch_{epoch:04d}.pkl"

    # Prepare checkpoint data
    checkpoint_data = {
        'params': state.params,
        'opt_state': state.opt_state,
        'step': state.step,
        'epoch': epoch,
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'apply_fn': state.apply_fn,  # Keep reference to model
        # Declare the mu1_bias axis convention this model emits.  Legacy
        # checkpoints predate this key; its ABSENCE is the only available signal
        # that a file is the old dual-endpoint (181-row) model, so load_checkpoint
        # treats a missing key as legacy — see _wrap_legacy_mu1_apply_fn.
        'grid_convention': GRID_CONVENTION,
        'mu1_bias_grid_size': mu1_size(),
    }

    # Save checkpoint
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def _renormalize_mu1_log_density(log_density):
    """Renormalize a log-density over the mu1_bias axis (the second-to-last one)."""
    discrete = log_density - logsumexp(log_density, axis=-2, keepdims=True)
    return discrete - jnp.log(mu1_cell_width())


def _wrap_legacy_mu1_apply_fn(apply_fn):
    """Run a legacy (dual-endpoint) checkpoint and trim its output to the circle.

    The model's output shape is read from config at call time, so with the
    periodic axis in force a legacy checkpoint would silently *interpolate*
    181 -> 180 rows — exactly the resampling this migration exists to avoid.  So
    the forward pass is run under the legacy inclusive axis, at the row count the
    model was trained on, and the duplicated +180 row is then **trimmed** and the
    result renormalized.  Trim and resize are different operations; that is the
    whole point.
    """
    def legacy_apply(*args, **kwargs):
        with legacy_mu1_axis():
            out = apply_fn(*args, **kwargs)
        if out.ndim < 2 or out.shape[-2] != LEGACY_MU1_GRID_SIZE:
            raise ValueError(
                f"Checkpoint carries no grid_convention metadata, so it is "
                f"treated as legacy ({LEGACY_MU1_GRID_SIZE} mu1 rows), but its "
                f"output has shape {out.shape}. Refusing to guess.")
        trimmed = out[..., :-1, :]
        return _renormalize_mu1_log_density(trimmed)

    return legacy_apply


def _wrap_periodic_mu1_apply_fn(apply_fn):
    """Assert a modern checkpoint really emits the periodic row count."""
    def checked_apply(*args, **kwargs):
        out = apply_fn(*args, **kwargs)
        if out.ndim >= 2 and out.shape[-2] != mu1_size():
            raise ValueError(
                f"Checkpoint metadata declares grid_convention="
                f"{GRID_CONVENTION!r} ({mu1_size()} mu1 rows) but its output has "
                f"shape {out.shape}.")
        return out

    return checked_apply


def load_checkpoint(checkpoint_path):
    """
    Load model checkpoint.

    Parameters:
    -----------
    checkpoint_path : str
        Path to checkpoint file
    create_state_fn : callable, optional
        Function to create initial state if needed

    Returns:
    --------
    state : TrainState
        Loaded training state
    checkpoint_info : dict
        Additional checkpoint information
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)

    # Which mu1_bias convention does this checkpoint emit?  Absence of the key
    # means legacy 181-row: old files cannot be edited, so absence is the only
    # available signal.
    convention = checkpoint_data.get('grid_convention')
    if convention is None:
        apply_fn = _wrap_legacy_mu1_apply_fn(checkpoint_data['apply_fn'])
        print(f"Legacy checkpoint (no grid_convention): running at "
              f"{LEGACY_MU1_GRID_SIZE} mu1 rows and trimming to {mu1_size()}.")
    elif convention == GRID_CONVENTION:
        declared = checkpoint_data.get('mu1_bias_grid_size')
        if declared is not None and int(declared) != mu1_size():
            raise ValueError(
                f"Checkpoint declares mu1_bias_grid_size={declared} but the "
                f"current config axis has {mu1_size()} points.")
        apply_fn = _wrap_periodic_mu1_apply_fn(checkpoint_data['apply_fn'])
    else:
        raise ValueError(
            f"Checkpoint declares unknown grid_convention {convention!r}; "
            f"expected {GRID_CONVENTION!r} or no key at all (legacy).")

    # Reconstruct training state
    state = train_state.TrainState(
        step=checkpoint_data['step'],
        apply_fn=apply_fn,
        params=checkpoint_data['params'],
        tx=None,  # Will need to be set separately
        opt_state=checkpoint_data['opt_state']
    )

    checkpoint_info = {
        'epoch': checkpoint_data['epoch'],
        'loss': checkpoint_data['loss'],
        'timestamp': checkpoint_data['timestamp']
    }

    print(f"Checkpoint loaded from epoch {checkpoint_data['epoch']}, loss: {checkpoint_data['loss']:.4f}")
    return state, checkpoint_info


def cleanup_old_checkpoints(save_dir="checkpoints", keep_last_n=5, prefix="model"):
    """
    Remove old checkpoints, keeping only the last N.

    Parameters:
    -----------
    save_dir : str
        Directory containing checkpoints
    keep_last_n : int
        Number of recent checkpoints to keep
    prefix : str
        Prefix of checkpoint filenames to consider
    """
    checkpoint_dir = Path(save_dir)
    if not checkpoint_dir.exists():
        return

    # Find all checkpoint files
    pattern = f"{prefix}_epoch_*.pkl"
    checkpoints = list(checkpoint_dir.glob(pattern))

    if len(checkpoints) <= keep_last_n:
        return

    # Sort by epoch number (extracted from filename)
    def extract_epoch(path):
        """Extract the epoch number from a checkpoint filename."""
        stem = path.stem
        epoch_part = stem.split('_epoch_')[1]
        return int(epoch_part)

    checkpoints.sort(key=extract_epoch)

    # Remove old checkpoints
    to_remove = checkpoints[:-keep_last_n]
    for checkpoint in to_remove:
        checkpoint.unlink()
        print(f"Removed old checkpoint: {checkpoint}")


def save_training_log_smart(save_dir="checkpoints", **kwargs):
    """
    Smart training log that automatically categorizes and saves all provided metrics.

    Automatically determines what goes in training_history vs hyperparameters based on:
    - Training history: epoch, any *_loss fields, learning_rate, accuracy, etc.
    - Hyperparameters: static config like batch_size, model_type, optimizer settings

    Parameters:
    -----------
    save_dir : str
        Directory to save log
    **kwargs : dict
        Any training information - will be automatically categorized
    """
    log_path = Path(save_dir) / "training_log.json"

    # Load existing log or create new
    if log_path.exists():
        with open(log_path, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = {
            'training_history': [],
            'hyperparameters': {},
            'start_time': datetime.now().isoformat()
        }

    # Smart categorization logic
    training_fields = set()  # Fields that go in training history
    hyperparam_fields = set()  # Fields that go in hyperparameters

    # Define patterns for training metrics (things that change each epoch)
    training_patterns = [
        'epoch', 'loss', 'accuracy', 'error', 'rate', 'score', 'metric',
        'correlation', 'divergence', 'mse', 'mae', 'rmse', 'r2'
    ]

    # Define patterns for hyperparameters (static config)
    hyperparam_patterns = [
        'batch_size', 'learning_rate', 'weight_decay', 'n_epochs', 'total_steps',
        'warmup_steps', 'model_type', 'optimizer', 'architecture', 'config',
        'data_range', 'smoothing', 'use_', 'n_samples', 'seed', 'dropout'
    ]

    for key, value in kwargs.items():
        key_lower = key.lower()

        # Check if it's clearly a training metric
        is_training_metric = (
                key == 'epoch' or  # Always training metric
                any(pattern in key_lower for pattern in training_patterns) or
                key_lower.endswith('_loss') or
                key_lower.endswith('_score') or
                key_lower.endswith('_metric') or
                key_lower.endswith('_accuracy') or
                key_lower.startswith('current_') or
                key_lower.startswith('val_') or
                key_lower.startswith('test_') or
                key_lower.startswith('train_')
        )

        # Check if it's clearly a hyperparameter
        is_hyperparameter = (
                any(pattern in key_lower for pattern in hyperparam_patterns) or
                key_lower.startswith('use_') or
                key_lower.startswith('enable_') or
                key_lower.startswith('n_') or
                key_lower.endswith('_dir') or
                key_lower.endswith('_path') or
                key_lower.endswith('_type') or
                key_lower.endswith('_method') or
                key_lower.startswith('final_')  # Final evaluation metrics
        )

        if is_training_metric and not is_hyperparameter:
            training_fields.add(key)
        elif is_hyperparameter:
            hyperparam_fields.add(key)
        else:
            # Default: if we have an epoch, assume it's training data
            if 'epoch' in kwargs:
                training_fields.add(key)
            else:
                hyperparam_fields.add(key)

    # Update training history if we have epoch data
    if 'epoch' in kwargs:
        entry = {'timestamp': datetime.now().isoformat()}

        # Add all training fields to this entry
        for field in training_fields:
            if field in kwargs:
                entry[field] = kwargs[field]

        log_data['training_history'].append(entry)

    # Update hyperparameters
    for field in hyperparam_fields:
        if field in kwargs:
            log_data['hyperparameters'][field] = kwargs[field]

    # Save log
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)


def load_surface_by_params(params: Dict, surfaces_folder: str = config.surfaces_folder, smooth: bool = False) -> Dict:
    """Load a surface matching the given parameter set.

    Args:
        params: Parameter dict with `sd_feat1`, `sd_feat2`, and `sd_spat`.
        surfaces_folder: Folder containing surface pickle files.
        smooth: Whether to apply entropy-based smoothing.

    Returns:
        Loaded surface data dict.

    Raises:
        FileNotFoundError: If no matching surface is found or multiple matches exist.
    """
    regex = rf"surface_sf1_{params['sd_feat1']:.1f}_sf2_{params['sd_feat2']:.1f}_sp_{params['sd_spat']:.1f}_\w+\.pkl"
    pattern = re.compile(regex)
    # print(f'Using regex {regex}')
    matching_files = [f for f in Path(surfaces_folder).glob("surface_*.pkl") if pattern.match(f.name)]
    if len(matching_files) == 1:
        return load_surface(matching_files[0], smooth = smooth)
    elif len(matching_files) == 0:
        raise FileNotFoundError(f"Surface not found, no files matching regex '{regex}' in this folder: '{surfaces_folder}'")
    else:
        raise FileNotFoundError(
            f"Surface not found, too many files matching the pattern '{regex}' in this folder: '{surfaces_folder}")



def plot_model_preds(checkpoint_file: str = None, params=None, n_points: int = 10):
    """Plot model predictions against original surfaces for sampled parameters.

    Args:
        checkpoint_file: Optional checkpoint path; uses latest if None.
        params: Optional parameter tuple to plot; if None, samples random points.
        n_points: Number of random points to evaluate when params is None.
    """
    if checkpoint_file is None:
        all_checkpoints = [f for f in Path('checkpoints').glob("model_epoch_*.pkl")]
        checkpoint_file = max(all_checkpoints, key=lambda p: int(re.search(r'epoch_(\d+)', p.name).group(1)))
    model,_ = load_checkpoint(checkpoint_file)

    # Sample the L2 grid the surfaces are actually on; the coarse param_step lattice
    # never lands on the half-step points, so it cannot see error where they live.
    param_range = np.arange(config.param_grid_low,
                            config.param_range_high + config.param_grid_step,
                            config.param_grid_step)
    all_points = list(itertools.product(param_range, repeat=3))
    if n_points:
        all_points = random.sample(all_points, n_points)
    correlations = []
    for point in all_points:
        orig_surface = load_surface_by_params({'sd_feat1':point[0], 'sd_feat2':point[1], 'sd_spat':point[2]}, smooth=True)
        test_param = jnp.array(point)

        pred_log_likelihood = model.apply_fn(model.params, test_param)[0]
        pred_likelihood = jnp.exp(pred_log_likelihood -
                                  logsumexp(pred_log_likelihood, axis=0, keepdims=True))
        pred_surface = orig_surface.copy()
        pred_surface['surface_data'] = (pred_surface['surface_data'][0], pred_surface['surface_data'][1],
                                        pred_log_likelihood, pred_likelihood)
        #
        # # Show plots
        plot_surface_comparison(pred_surface['surface_data'], orig_surface['surface_data'],
                                ('Predicted', 'Original'), same_color_axis=False)

        plot_surface_comparison(pred_surface['surface_data'], orig_surface['surface_data'],
                                ('Predicted', 'Original'), do_exp=False)

        # Compute correlation
        _, mu1_pred = compute_expectation(pred_surface['surface_data'])
        _, mu1_orig = compute_expectation(orig_surface['surface_data'])
        corr = np.corrcoef(mu1_pred, mu1_orig)[0, 1]
        correlations.append(corr)
        print(f"Correlation at {orig_surface['parameters']['param_name']}: {corr:.3f}")
    print(f'Average correlation: {np.mean(np.array(correlations))}')
def save_variables(filename, *variables):
    """
    Save variables as a pickle file containing a dictionary.
    Automatically extracts variable names from the calling scope.

    Usage: save_variables(temp_file, sim_surf_true, sim_surf_alt, sim_data)
    """
    frame = inspect.currentframe().f_back
    all_vars = {**frame.f_locals, **frame.f_globals}

    var_dict = {}
    for var in variables:
        for name, value in all_vars.items():
            if value is var:
                var_dict[name] = var
                break

    with open(filename, 'wb') as f:
        pickle.dump(var_dict, f)


def load_variables(filename):
    """
    Load variables from a pickle file directly into the calling scope.

    Usage: load_variables(temp_file)
           # Now sim_surf_true, sim_surf_alt, sim_data are available
    """
    frame = inspect.currentframe().f_back

    with open(filename, 'rb') as f:
        var_dict = pickle.load(f)

    frame.f_globals.update(var_dict)


@dataclass
class Surface:
    """
    Surface class for storing multi-component bias likelihood surfaces.

    Parameters:
    -----------
    feat_diff_grid : jnp.ndarray - 1D array of feature difference values
    mu1_bias_grid : jnp.ndarray - 1D array of mu1 bias values
    mu2_bias_grid : jnp.ndarray - 1D array of mu2 bias values
    mu1_comp1_surface : jnp.ndarray - Log-likelihood surface for mu1 bias, component 1
    mu1_comp2_surface : jnp.ndarray - Log-likelihood surface for mu1 bias, component 2
    mu2_comp1_surface : jnp.ndarray - Log-likelihood surface for mu2 bias, component 1
    mu2_comp2_surface : jnp.ndarray - Log-likelihood surface for mu2 bias, component 2
    """

    feat_diff_grid: jnp.ndarray
    mu1_bias_grid: jnp.ndarray
    mu2_bias_grid: jnp.ndarray
    mu1_comp1_surface: jnp.ndarray
    mu1_comp2_surface: jnp.ndarray
    mu2_comp1_surface: jnp.ndarray
    mu2_comp2_surface: jnp.ndarray

    def __post_init__(self):
        """Validate surface dimensions after initialization."""
        expected_mu1 = (len(self.mu1_bias_grid), len(self.feat_diff_grid))
        expected_mu2 = (len(self.mu2_bias_grid), len(self.feat_diff_grid))

        if self.mu1_comp1_surface.shape != expected_mu1:
            raise ValueError(f"mu1_comp1_surface shape {self.mu1_comp1_surface.shape} != {expected_mu1}")
        if self.mu1_comp2_surface.shape != expected_mu1:
            raise ValueError(f"mu1_comp2_surface shape {self.mu1_comp2_surface.shape} != {expected_mu1}")
        if self.mu2_comp1_surface.shape != expected_mu2:
            raise ValueError(f"mu2_comp1_surface shape {self.mu2_comp1_surface.shape} != {expected_mu2}")
        if self.mu2_comp2_surface.shape != expected_mu2:
            raise ValueError(f"mu2_comp2_surface shape {self.mu2_comp2_surface.shape} != {expected_mu2}")

    def get_surf(self, dimension: int, component: int, log: bool = True) -> jnp.ndarray:
        """
        Get likelihood surface for specified dimension and component.

        Parameters:
        -----------
        dimension : int - 1 (mu1) or 2 (mu2)
        component : int - 1 or 2
        log : bool - True for log-likelihood, False for probability surface

        Returns: jnp.ndarray - Requested likelihood surface
        """
        if dimension not in [1, 2]: raise ValueError(f"dimension must be 1 or 2, got {dimension}")
        if component not in [1, 2]: raise ValueError(f"component must be 1 or 2, got {component}")

        # Select surface
        surface_map = {
            (1, 1): self.mu1_comp1_surface,
            (1, 2): self.mu1_comp2_surface,
            (2, 1): self.mu2_comp1_surface,
            (2, 2): self.mu2_comp2_surface
        }
        surface = surface_map[(dimension, component)]

        return surface if log else jnp.exp(surface)

    def get_bias_grid(self, dimension: int) -> jnp.ndarray:
        """Get bias grid for specified dimension."""
        if dimension == 1: return self.mu1_bias_grid
        if dimension == 2: return self.mu2_bias_grid
        raise ValueError(f"dimension must be 1 or 2, got {dimension}")

    def summary(self) -> str:
        """Generate summary string of surface properties."""
        return "\n".join([
            "Surface Summary:",
            f"  Feature difference: {len(self.feat_diff_grid)} points [{self.feat_diff_grid.min():.1f}, {self.feat_diff_grid.max():.1f}]",
            f"  Mu1 bias: {len(self.mu1_bias_grid)} points [{self.mu1_bias_grid.min():.1f}, {self.mu1_bias_grid.max():.1f}]",
            f"  Mu2 bias: {len(self.mu2_bias_grid)} points [{self.mu2_bias_grid.min():.1f}, {self.mu2_bias_grid.max():.1f}]",
            f"  Mu1 surfaces: {self.mu1_comp1_surface.shape} each (comp1, comp2)",
            f"  Mu2 surfaces: {self.mu2_comp1_surface.shape} each (comp1, comp2)"
        ])

@dataclass
class AveragedSurface(Surface):
    """
    Subclass of Surface for averaged surfaces created from samples.
    Contains higher-quality surfaces created using 2D normal weighted KDE.
    """

    n_sample_files_used: int = 0
    averaged_from_params: List[Tuple[float, float, float]] = None
    kde_parameters: Dict = None

    def __post_init__(self):
        if self.averaged_from_params is None:
            self.averaged_from_params = []
        if self.kde_parameters is None:
            self.kde_parameters = {}
        super().__post_init__()


class SurfaceUnpickler(pickle.Unpickler):
    """Remaps legacy AveragedSurface module paths to shared.utils.

    Pickle files produced before the class was centralised store the class as
    ``__main__.AveragedSurface`` or under the neural_network_optimization
    sub-module path. Both are redirected to shared.utils.AveragedSurface.
    """
    _LEGACY_MODULES = {
        '__main__',
        'neural_network_optimization.create_averaged_surfaces_from_samples',
        'create_averaged_surfaces_from_samples',
    }

    def find_class(self, module, name):
        if name == 'AveragedSurface' and module in self._LEGACY_MODULES:
            return AveragedSurface
        return super().find_class(module, name)


# Cache of per-directory {surface_filename -> bundle_path} indices, so the
# manifest scan happens at most once per surfaces dir per process.
_BUNDLE_INDEX_CACHE: Dict[str, Dict[str, Path]] = {}


def _build_bundle_index(surfaces_dir: Path) -> Dict[str, Path]:
    """Map each bundled surface filename to the bundle that contains it.

    Uses the cheap per-bundle ``*.manifest.json`` files (``surface_ids`` are
    ``"sf1|sf2|sp"``) so no ``.pkl.gz`` is decompressed during indexing.
    """
    index: Dict[str, Path] = {}
    for manifest_path in sorted(surfaces_dir.glob("surface_bundle_*.manifest.json")):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        bundle_path = manifest_path.with_name(
            manifest_path.name.removesuffix(".manifest.json") + ".pkl.gz")
        for surface_id in manifest.get("surface_ids", []):
            sf1, sf2, sp = surface_id.split("|")
            filename = f"averaged_sf1_{sf1}_sf2_{sf2}_sp_{sp}.pkl"
            index.setdefault(filename, bundle_path)
    return index


def ensure_averaged_surface_file(surfaces_dir, filename: str) -> Path:
    """Return the path to an individual averaged-surface ``.pkl``, materialising
    it from a ``surface_bundle_*.pkl.gz`` bundle if only the bundled form exists.

    This lets surface consumers request exactly the surfaces they need without a
    separately maintained list of which surfaces to pre-extract: whatever the
    caller asks for is resolved from the individual file or the bundles, and a
    surface that exists in neither raises ``FileNotFoundError`` (rather than a
    curve silently disappearing from a figure).
    """
    surfaces_dir = Path(surfaces_dir)
    file_path = surfaces_dir / filename
    if file_path.exists():
        return file_path

    key = str(surfaces_dir)
    index = _BUNDLE_INDEX_CACHE.get(key)
    if index is None:
        index = _build_bundle_index(surfaces_dir)
        _BUNDLE_INDEX_CACHE[key] = index

    bundle_path = index.get(filename)
    if bundle_path is None:
        raise FileNotFoundError(
            f"No averaged surface {filename!r} in {surfaces_dir} "
            f"(checked individual file and {len(index)} bundled surfaces).")

    with gzip.open(bundle_path, "rb") as f:
        surfaces = pickle.load(f)["surfaces"]
    content = surfaces[filename]

    # Cache atomically so later requests (and reruns) reuse the individual file.
    tmp = file_path.with_suffix(file_path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(file_path)
    return file_path


def compute_single_bias_curve(log_surfaces: jnp.ndarray, target_feat_indices: jnp.ndarray, mu1_bias_grid: jnp.ndarray) -> jnp.ndarray:
    """
    Compute bias curve from log probability surface for given feature indices.
    
    Args:
        log_surfaces: Log probability surface with shape (n_mu1_bias, n_feat_diff)
        target_feat_indices: Indices of feature difference values to compute bias for
        mu1_bias_grid: Grid of mu1 bias values
        
    Returns:
        Circular means (bias values) for the target feature indices
    """
    # Convert log probabilities to probabilities
    mu1_prob_surface = jnp.exp(log_surfaces)
    target_prob_profiles = mu1_prob_surface[:, target_feat_indices]

    # Vectorized circular mean computation for all target indices
    cos_vals = jnp.cos(jnp.radians(mu1_bias_grid)).reshape(-1, 1)
    sin_vals = jnp.sin(jnp.radians(mu1_bias_grid)).reshape(-1, 1)

    # Periodic quadrature (sum x cell width): trapezoid over x=grid spans only
    # grid[-1]-grid[0] = 358 deg of the 360 deg period and half-weights the ends.
    cos_expectations = periodic_integral(cos_vals * target_prob_profiles, axis=0)
    sin_expectations = periodic_integral(sin_vals * target_prob_profiles, axis=0)

    circular_means = jnp.degrees(jnp.arctan2(sin_expectations, cos_expectations))
    return circular_means


def compute_single_density_asymmetry(log_surfaces: jnp.ndarray, target_feat_indices: jnp.ndarray, mu1_bias_grid: jnp.ndarray, 
                                   apply_smoothing: bool = True, smoothing_sigma: float = 5.0) -> jnp.ndarray:
    """
    Compute density asymmetry for a single log probability surface with optional Gaussian smoothing.
    
    Args:
        log_surfaces: Log probability surface with shape (n_mu1_bias, n_feat_diff)
        target_feat_indices: Indices of feature difference values to compute asymmetry for
        mu1_bias_grid: Grid of mu1 bias values
        apply_smoothing: Whether to apply Gaussian smoothing to the asymmetry curve
        smoothing_sigma: Standard deviation for Gaussian smoothing kernel
        
    Returns:
        Asymmetry values for the target feature indices
    """
    # Convert to probabilities
    probs = jnp.exp(log_surfaces)
    
    # Extract probabilities for target feature indices
    target_probs = probs[:, target_feat_indices]
    
    # Compute asymmetry for each target feature difference.  Both 0 and the
    # antipode (-180) are excluded from both masks: on a circle they are the two
    # sign-ambiguous angles.  Counting -180 as negative injects a spurious
    # asymmetry at exactly the row that carries mass for broad sd_feat.
    positive_mask, negative_mask = sign_masks(mu1_bias_grid)

    # Vectorized computation across all target indices with proper discretization
    dx = mu1_cell_width()  # Periodic cell width for numerical integration

    # Use jnp.where instead of boolean indexing to avoid concreteness issues
    positive_probs = jnp.where(positive_mask[:, None], target_probs, 0.0)
    negative_probs = jnp.where(negative_mask[:, None], target_probs, 0.0)
    
    p_positive = jnp.sum(positive_probs, axis=0) * dx
    p_negative = jnp.sum(negative_probs, axis=0) * dx
    
    asymmetry = p_positive - p_negative
    
    # Apply Gaussian smoothing if requested using JAX operations
    if apply_smoothing:
        # Create Gaussian kernel
        kernel_size = int(4 * smoothing_sigma + 1)  # Kernel size based on sigma
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size
        
        # Create 1D Gaussian kernel
        x = jnp.arange(kernel_size) - kernel_size // 2
        kernel = jnp.exp(-0.5 * (x / smoothing_sigma) ** 2)
        kernel = kernel / jnp.sum(kernel)  # Normalize
        
        # Apply convolution using JAX
        # Pad the asymmetry array to handle edges
        pad_width = kernel_size // 2
        asymmetry_padded = jnp.pad(asymmetry, pad_width, mode='edge')
        
        # Convolve
        asymmetry_smoothed = jnp.convolve(asymmetry_padded, kernel, mode='valid')
        asymmetry = asymmetry_smoothed
    
    return asymmetry


KDE_BW_FLOOR = 1e-6
"""Lower bound on the bias-KDE bandwidth, purely a divide-by-zero guard.

Degenerate input (near-identical bias values) drives Silverman's rule toward 0 and
the kernel normalization divides by it, yielding NaN (MODEL_PIPELINE_FOR_AGENTS.md
D.11). Matches the guard already used in the BBZ twin. This is deliberately far
below any bandwidth real data produce -- the smallest observed is ~0.42 model
degrees -- so it never binds and never changes a target. A *resolution* floor (one
that binds, e.g. half the 2-degree bias cell) is a separate, target-changing
decision; the plotting-side twin uses 1.0 for that reason.
"""


def silverman_bandwidth(real_bias, floor=KDE_BW_FLOOR):
    """Silverman's rule-of-thumb bandwidth, the default bias-KDE bandwidth.

    Exposed separately so a caller can compute ONE bandwidth over several
    conditions and pass it back in via `kernel_bw`, rather than letting each
    condition pick its own (see `_compute_empirical_density_asymmetry_core`).
    Identical to R's `bw.nrd0`.
    """
    real_bias = jnp.asarray(real_bias)
    bias_std = jnp.std(real_bias)
    iqr = jnp.quantile(real_bias, 0.75) - jnp.quantile(real_bias, 0.25)
    bw = 0.9 * jnp.minimum(bias_std, iqr / 1.34) * (len(real_bias) ** (-1 / 5))
    return jnp.maximum(bw, floor)


_SJ_DELMAX = 1000.0  # R's cutoff: skip pair distances beyond sqrt(DELMAX) bandwidths


def _sj_pair_histogram(x, nb=1000):
    """Pair-distance counts on R's binning, so each functional evaluation is O(nb).

    R bins for the same reason -- it is what makes an O(n^2) selector tractable and
    is the whole speed story behind `stats::bw.SJ`. The convention has to be copied
    exactly or the bandwidths disagree by several percent: R assigns each VALUE a bin
    via truncation toward zero (`trunc(abs(x)/d) * sign(x)`, matching the C cast in
    band_den_bin) and then uses the difference of bin indices as the distance --
    which is NOT the same as binning the distances, because the two truncations do
    not compose. Distance counts then come from the autocorrelation of the bin
    counts, as C_bw_den_binned does.

    Returns (bin_width, counts) with counts[k] the number of unordered pairs whose
    bin indices differ by k.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.ptp(x) * 1.01
    if not np.isfinite(rng) or rng <= 0:
        return None, None
    d = rng / nb

    idx = (np.trunc(np.abs(x) / d) * np.sign(x)).astype(np.int64)
    idx -= idx.min()
    bin_counts = np.bincount(idx, minlength=nb + 1).astype(np.float64)

    ac = np.correlate(bin_counts, bin_counts, mode="full")
    counts = ac[len(bin_counts) - 1:][:nb].copy()
    counts[0] = (counts[0] - n) / 2.0   # drop the i == j diagonal, then unorder
    return d, counts


def _sj_functional(counts, d, n, h, order):
    """R's bw_phi4 / bw_phi6: the integrated squared density-derivative functionals.

    phi4(u) = exp(-u^2/2) (u^4 - 6u^2 + 3), phi6(u) = exp(-u^2/2) (u^6 - 15u^4 +
    45u^2 - 15), summed over ordered pairs (off-diagonal twice, plus the diagonal
    term phi4(0) = 3 / phi6(0) = -15).
    """
    delta = (np.arange(len(counts)) * d / h) ** 2
    keep = delta < _SJ_DELMAX
    delta = delta[keep]
    weight = counts[keep]
    if order == 4:
        term = np.exp(-delta / 2) * (delta ** 2 - 6 * delta + 3)
        total = 2 * np.sum(term * weight) + 3 * n
        power = 5.0
    else:
        term = np.exp(-delta / 2) * (delta ** 3 - 15 * delta ** 2 + 45 * delta - 15)
        total = 2 * np.sum(term * weight) - 15 * n
        power = 7.0
    return total / (n * (n - 1) * h ** power * np.sqrt(2 * np.pi))


def sheather_jones_bandwidth(real_bias, nb=1000, fallback=True):
    """Sheather-Jones solve-the-equation bandwidth, a port of `stats::bw.SJ`.

    R's own documentation recommends this over the Silverman default it ships:
    "The default, \"nrd0\", has remained the default for historical and
    compatibility reasons, rather than as a general recommendation, where e.g.
    \"SJ\" would rather fit" (?density). Silverman is the more fragile rule -- it
    takes min(sd, IQR/1.34), which misjudges heavy-tailed or multimodal error
    distributions; on one moors cell it came out 5.1x the SJ value.

    scipy has no SJ, hence the port. Constants, the DELMAX cutoff, the pair
    binning and the bracket expansion all follow R's implementation so the two
    agree; verified against `stats::bw.SJ` on real data.

    Args:
        real_bias: samples (a single condition's, or pooled across conditions).
        nb: pair-distance bins, as in R.
        fallback: if True, fall back to Silverman when the sample is too sparse for
            SJ to have a solution. R raises instead; a fitting pipeline should not
            die on one degenerate cell.
    """
    x = np.asarray(real_bias, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 data points")

    scale = min(np.std(x, ddof=1), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349)
    d, counts = _sj_pair_histogram(x, nb=nb)

    def sd_h(h):
        return _sj_functional(counts, d, n, h, order=4)

    def td_h(h):
        return _sj_functional(counts, d, n, h, order=6)

    def _fallback(reason):
        if not fallback:
            raise RuntimeError(f"bw.SJ failed: {reason}")
        warnings.warn(f"Sheather-Jones bandwidth failed ({reason}); "
                      "falling back to Silverman", RuntimeWarning)
        return float(silverman_bandwidth(x))

    if d is None or scale <= 0:
        return _fallback("degenerate sample")

    td = -td_h(1.23 * scale * n ** (-1 / 9))
    if not np.isfinite(td) or td <= 0:
        return _fallback("sample is too sparse to find TD")
    alph2 = 1.357 * (sd_h(1.24 * scale * n ** (-1 / 7)) / td) ** (1 / 7)
    if not np.isfinite(alph2):
        return _fallback("sample is too sparse to find alph2")

    c1 = 1 / (2 * np.sqrt(np.pi) * n)

    def f_sd(h):
        return (c1 / sd_h(alph2 * h ** (5 / 7))) ** (1 / 5) - h

    hmax = 1.144 * scale * n ** (-1 / 5)
    lower, upper = 0.1 * hmax, hmax
    for itry in range(1, 100):
        try:
            if f_sd(lower) * f_sd(upper) <= 0:
                break
        except (FloatingPointError, ValueError):
            return _fallback("functional evaluation failed")
        if itry % 2:
            upper *= 1.2
        else:
            lower /= 1.2
    else:
        return _fallback("no solution in the specified range of bandwidths")

    from scipy.optimize import brentq
    return float(brentq(f_sd, lower, upper, xtol=0.1 * 0.1 * hmax))


def _compute_empirical_density_asymmetry_core(real_feat_diff, real_bias, feat_diff_grid, weights_sd=20, circ_space=360,
                                              kernel_bw=None):
    """
    Core computation for empirical density asymmetry with all matrix operations.

    Args:
        real_feat_diff: Array of feature difference values, shape (n_trials,)
        real_bias: Array of bias values, shape (n_trials,)
        feat_diff_grid: Grid of feature difference values to compute asymmetry at
        weights_sd: Standard deviation for Gaussian weights in feat_diff dimension
        circ_space: Circular space size (360 for full circle)
        kernel_bw: Bias-KDE bandwidth. None (default) estimates it from THIS call's
            trials via Silverman's rule, so each condition gets its own smoothing --
            and conditions differ in error spread by construction, which is the axis
            the experiment manipulates. Pass an explicit value to share one bandwidth
            across conditions (`silverman_bandwidth` over the pooled trials, or the
            mean of the per-condition estimates).

    Returns:
        distances: feat_diff_grid values
        asymmetry_values: Asymmetry values at each grid point

    Notes:
        The bias KDE is WRAPPED (see KDE_WRAPS): the axis is a circle, so a kernel
        centred near +max_diss reappears just past -max_diss rather than being
        truncated. On orientation/colour data this is a no-op to ~1e-13 because
        errors sit near 0 and the bandwidth is a few degrees, but motion-direction
        data routinely carry a second mode of 180-degree-off reversals sitting
        exactly on the boundary, where truncation both loses mass and mis-signs it.

        Each trial's kernel carries unit mass, so the returned signed difference is
        a proportion of a unit-mass distribution. circhelp divides instead by
        (P+ + P-), excluding the sign-ambiguous 0-cell -- a different denominator
        convention, applied consistently on our model and target sides alike.
    """
    max_diss = circ_space / 2

    # Bandwidth: Silverman over this call's trials unless the caller supplies one.
    if kernel_bw is None:
        kernel_bw = silverman_bandwidth(real_bias)
    else:
        kernel_bw = jnp.maximum(jnp.asarray(kernel_bw), KDE_BW_FLOOR)


    # Use provided feat_diff_grid or create default distances
    if feat_diff_grid is not None:
        distances = feat_diff_grid
    else:
        distances = jnp.arange(4, int(max_diss * 2) + 1, 4)  # [4, 8, ..., circ_space]
    # 180 points spanning [-max_diss, +max_diss): step 2° for circ_space=360 (the
    # model-space default used by the fitter), step 1° for circ_space=180.  The
    # axis is circular, so it is half-open — +max_diss is the same angle as
    # -max_diss and must not occupy a second cell (it used to, via a 181-point
    # inclusive linspace, mirroring the model-side defect).
    n_bias_points = 180
    dx_empirical = (2 * max_diss) / n_bias_points
    bias_range = -max_diss + dx_empirical * jnp.arange(n_bias_points)
    
    # Vectorized weight matrix computation with logsumexp for numerical stability
    dist_matrix = distances[:, None] - real_feat_diff[None, :]  # Broadcasting
    log_weights = -0.5 * (dist_matrix / weights_sd) ** 2
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights, axis=1, keepdims=True)
    weight_matrix = jnp.exp(log_weights_normalized)
    
    # Vectorized kernel matrix computation: (n_bias_points, n_datapoints), summed
    # over KDE_WRAPS periodic images so the kernel closes around the circle.
    bias_matrix = bias_range[:, None] - real_bias[None, :]  # Broadcasting
    offsets = circ_space * jnp.arange(-KDE_WRAPS, KDE_WRAPS + 1, dtype=bias_matrix.dtype)
    kernel_matrix = jnp.sum(
        jnp.exp(-0.5 * ((bias_matrix[None, :, :] + offsets[:, None, None]) / kernel_bw) ** 2),
        axis=0)
    kernel_matrix = kernel_matrix / (kernel_bw * jnp.sqrt(2 * jnp.pi))  # Normalize kernel
    
    # Matrix multiplication: (n_distances, n_bias_points)
    # weight_matrix @ kernel_matrix.T gives density for each (distance, bias_point) combination
    density_matrix = weight_matrix @ kernel_matrix.T
    
    # Split into positive and negative regions for asymmetry computation.  The
    # antipode (-max_diss) is as sign-ambiguous as 0 and is excluded from both
    # masks, matching the model side (compute_single_density_asymmetry).
    pos_mask = bias_range > 0
    neg_mask = (bias_range < 0) & (bias_range > -max_diss)

    # Sum densities in positive and negative regions for each distance.
    # dx_empirical (the periodic cell width) is set with the grid above.

    # Use jnp.where instead of boolean indexing to avoid concreteness issues
    pos_densities_matrix = jnp.where(pos_mask[None, :], density_matrix, 0.0)
    neg_densities_matrix = jnp.where(neg_mask[None, :], density_matrix, 0.0)
    
    pos_densities = jnp.sum(pos_densities_matrix, axis=1) * dx_empirical  # Shape: (n_distances,)
    neg_densities = jnp.sum(neg_densities_matrix, axis=1) * dx_empirical  # Shape: (n_distances,)
    
    asymmetry_values = pos_densities - neg_densities
    return distances, asymmetry_values


def compute_target_bias_rolling_curve_core(real_feat_diff, real_bias, feat_diff_grid, weights_sd=20):
    """
    Gaussian-weighted (rolling) circular-mean bias curve over feat_diff_grid.

    Same feat_diff-space Gaussian weighting as _compute_empirical_density_asymmetry_core,
    but normalized to sum to 1 over trials (a true kernel-weighted average) and applied to
    cos/sin of bias instead of splitting into positive/negative asymmetry.

    Args:
        real_feat_diff: Array of feature difference values, shape (n_trials,)
        real_bias: Array of bias values in degrees, shape (n_trials,)
        feat_diff_grid: Grid of feature difference values to compute the curve at
        weights_sd: Standard deviation for Gaussian weights in feat_diff dimension

    Returns:
        target_bias_curve: Circular mean bias in degrees at each feat_diff_grid point
    """
    dist_matrix = feat_diff_grid[:, None] - real_feat_diff[None, :]  # (n_grid, n_trials)
    log_weights = -0.5 * (dist_matrix / weights_sd) ** 2
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights, axis=1, keepdims=True)
    weight_matrix = jnp.exp(log_weights_normalized)  # each row sums to 1

    bias_rad = jnp.radians(real_bias)
    cos_curve = weight_matrix @ jnp.cos(bias_rad)
    sin_curve = weight_matrix @ jnp.sin(bias_rad)

    return jnp.degrees(jnp.arctan2(sin_curve, cos_curve))


def filter_data_for_fitting(data, feat_diff_col=None, bias_col=None, verbose=True,
                            min_diss=4.0, max_diss=180.0):
    """
    Filter and clean data for GMM fitting by removing invalid values and applying range constraints.

    Handles both numpy arrays and pandas DataFrames:
    - For 2-column arrays: assumes first column is feat_diff, second is bias
    - For DataFrames: uses specified column names

    Args:
        data: Either a 2-column numpy array or pandas DataFrame
        feat_diff_col: Column name for feature differences (required for DataFrame)
        bias_col: Column name for bias values (required for DataFrame)
        verbose: Whether to print filtering statistics
        min_diss: Dissimilarity floor in the INPUT (raw, pre-scaling) space; values
            below it are clamped up. max_diss: dissimilarity ceiling; values above it
            are dropped. Defaults (4, 180) preserve the legacy raw-space clamp. The
            demixing fitter passes period/scale-aware bounds (feat_diff_range / scale)
            so the effective clamp is the model-space grid range for every dataset —
            see codex_audit.md report-level #3.

    Returns:
        Cleaned data in the same format as input
    """
    import pandas as pd
    
    is_dataframe = isinstance(data, pd.DataFrame)
    
    if is_dataframe:
        if feat_diff_col is None or bias_col is None:
            raise ValueError("feat_diff_col and bias_col must be specified for DataFrames")
        original_len = len(data)
        clean_data = data.copy()
        feat_diff_vals = clean_data[feat_diff_col]
        bias_vals = clean_data[bias_col]
    else:
        # Assume 2-column array: first column feat_diff, second bias
        if data.shape[1] != 2:
            raise ValueError("Array must have exactly 2 columns")
        original_len = len(data)
        clean_data = data.copy()
        feat_diff_vals = clean_data[:, 0]
        bias_vals = clean_data[:, 1]
    
    # Check for NaN values
    nan_mask = np.isnan(feat_diff_vals) | np.isnan(bias_vals)
    nan_count = nan_mask.sum()
    
    if nan_count > 0 and verbose:
        print(f"Found {nan_count} NaN values")
        
    # Remove NaN values
    if is_dataframe:
        clean_data = clean_data[~nan_mask].copy().reset_index(drop=True)
        feat_diff_vals = clean_data[feat_diff_col]
        bias_vals = clean_data[bias_col]
    else:
        clean_data = clean_data[~nan_mask]
        feat_diff_vals = clean_data[:, 0]
        bias_vals = clean_data[:, 1]
    
    # Check for infinite values
    inf_mask = ~(np.isfinite(feat_diff_vals) & np.isfinite(bias_vals))
    inf_count = inf_mask.sum()
    
    if inf_count > 0 and verbose:
        print(f"Found {inf_count} infinite values")
        
    # Remove infinite values
    if is_dataframe:
        clean_data = clean_data[~inf_mask].copy().reset_index(drop=True)
        feat_diff_vals = clean_data[feat_diff_col]
        bias_vals = clean_data[bias_col]
    else:
        clean_data = clean_data[~inf_mask]
        feat_diff_vals = clean_data[:, 0]
        bias_vals = clean_data[:, 1]
    
    # Save original values before any modifications (after all row filtering)
    if is_dataframe:
        original_feat_diff = clean_data[feat_diff_col].copy()
    else:
        original_feat_diff = clean_data[:, 0].copy()
    
    # Check feat_diff range constraints
    feat_below_min = feat_diff_vals < min_diss
    feat_above_max = feat_diff_vals > max_diss
    feat_below_count = feat_below_min.sum()
    feat_above_count = feat_above_max.sum()

    if feat_below_count > 0 and verbose:
        print(f"Found {feat_below_count} feat_diff values < {min_diss:g} - will clamp to {min_diss:g}")
        below_vals = feat_diff_vals[feat_below_min]
        print(f"  Values < {min_diss:g}: min={below_vals.min():.2f}, examples={list(below_vals.head(3) if hasattr(below_vals, 'head') else below_vals[:3].round(2))}")

    if feat_above_count > 0 and verbose:
        print(f"Found {feat_above_count} feat_diff values > {max_diss:g} - will remove")
        above_vals = feat_diff_vals[feat_above_max]
        print(f"  Values > {max_diss:g}: max={above_vals.max():.2f}, examples={list(above_vals.head(3) if hasattr(above_vals, 'head') else above_vals[:3].round(2))}")
        
    # Save clamping examples before filtering removes them
    clamp_examples_orig = None
    clamp_examples_new = None
    if feat_below_count > 0 and verbose:
        below_vals = feat_diff_vals[feat_below_min]
        orig_below_vals = original_feat_diff[feat_below_min]
        
        if hasattr(below_vals, 'head'):
            clamp_examples_new = below_vals.head(3).round(2)
            clamp_examples_orig = orig_below_vals.head(3).round(2)
        else:
            clamp_examples_new = below_vals[:3].round(2)
            clamp_examples_orig = orig_below_vals[:3].round(2)
    
    # Apply feat_diff constraints
    if is_dataframe:
        clean_data[feat_diff_col] = np.clip(clean_data[feat_diff_col], min_diss, None)  # Clamp minimum
        clean_data = clean_data[clean_data[feat_diff_col] <= max_diss].copy().reset_index(drop=True)  # Remove above max
        feat_diff_vals = clean_data[feat_diff_col]
        bias_vals = clean_data[bias_col]
    else:
        clean_data[:, 0] = np.clip(clean_data[:, 0], min_diss, None)  # Clamp minimum
        clean_data = clean_data[clean_data[:, 0] <= max_diss]  # Remove above max
        feat_diff_vals = clean_data[:, 0]
        bias_vals = clean_data[:, 1]
    
    # Show clamping examples
    if feat_below_count > 0 and verbose and clamp_examples_orig is not None:
        print(f"  Clamp examples: {list(zip(clamp_examples_orig, clamp_examples_new))}")
    
    # Check bias range and apply circular wrapping
    bias_out_of_range = (bias_vals < -180) | (bias_vals > 180)
    bias_out_count = bias_out_of_range.sum()
    
    if bias_out_count > 0 and verbose:
        print(f"Found {bias_out_count} bias values outside [-180, 180] - will wrap")
        bias_out_rows = bias_vals[bias_out_of_range]
        print(f"  Out-of-range bias: min={bias_out_rows.min():.1f}, max={bias_out_rows.max():.1f}")
    
    # Apply circular wrapping for bias
    if is_dataframe:
        original_bias = clean_data[bias_col].copy()
        clean_data[bias_col] = ((clean_data[bias_col] + 180) % 360) - 180
        bias_vals = clean_data[bias_col]
    else:
        original_bias = clean_data[:, 1].copy()
        clean_data[:, 1] = ((clean_data[:, 1] + 180) % 360) - 180
        bias_vals = clean_data[:, 1]
    
    # Show wrapping examples
    if bias_out_count > 0 and verbose:
        # Find positions of out-of-range values and take first 5
        if hasattr(bias_out_of_range, 'to_numpy'):
            out_of_range_positions = np.where(bias_out_of_range.to_numpy())[0][:5]
        else:
            out_of_range_positions = np.where(bias_out_of_range)[0][:5]
            
        if len(out_of_range_positions) > 0:
            if hasattr(original_bias, 'iloc'):
                orig_vals = original_bias.iloc[out_of_range_positions].round(1)
                new_vals = bias_vals.iloc[out_of_range_positions].round(1)
            else:
                orig_vals = original_bias[out_of_range_positions].round(1)
                new_vals = bias_vals[out_of_range_positions].round(1)
            print(f"  Wrap examples: {list(zip(orig_vals, new_vals))}")
    
    filtered_count = original_len - len(clean_data)
    if filtered_count > 0 and verbose:
        print(f"Filtered out {filtered_count}/{original_len} invalid data points")
    
    # Final statistics
    if verbose:
        if is_dataframe:
            feat_range = f"[{clean_data[feat_diff_col].min():.1f}, {clean_data[feat_diff_col].max():.1f}]"
            bias_range = f"[{clean_data[bias_col].min():.1f}, {clean_data[bias_col].max():.1f}]"
        else:
            feat_range = f"[{clean_data[:, 0].min():.1f}, {clean_data[:, 0].max():.1f}]"
            bias_range = f"[{clean_data[:, 1].min():.1f}, {clean_data[:, 1].max():.1f}]"
        
        print(f"Final: {len(clean_data)} trials, feat_diff range {feat_range}, bias range {bias_range}")
    
    return clean_data


