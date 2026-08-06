#!/usr/bin/env python3
"""
Surface Simulator for Predictions

This script reads parameters from a CSV/Arrow file, simulates surfaces and curves
using either neural network checkpoints or precomputed averaged surfaces, and
writes results for downstream R analysis.
"""

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
import sys
from pathlib import Path
import pickle
from typing import Optional, Tuple

from model_fit_to_data.grid_based_multi_condition_optimizer_jax_loops import (
    GridBasedMultiConditionOptimizer,
    generate_nn_density_asymmetry_batch,
    apply_motor_noise,
    _generate_nn_bias_curve_batch,
)
from shared.config import config
from shared.utils import (AveragedSurface, SurfaceUnpickler, resolve_input_path,
                          ensure_averaged_surface_file)

RESULTS_DIR = "results"
CHECKPOINT_PREFIX = "neural_net_checkpoints"
CHECKPOINT_EPOCH = 1500
PRETRAINED_CHECKPOINTS = {
    20: "pretrained/model_epoch1500_10ktrain_20samples.pkl",
    100: "pretrained/model_epoch1500_10ktrain_100samples.pkl",
}

def _generate_mu2_bias_curve_batch(mu2_surfaces_batch: jnp.ndarray, target_feat_indices: jnp.ndarray) -> jnp.ndarray:
    """Generate mu2 bias curves for batch of mu2 surfaces using linear integration (not circular)."""
    # Create mu2 bias grid
    mu2_bias_grid = config.create_grid('mu2_bias')

    # Vectorized computation for each surface
    def compute_single_mu2_bias_curve(log_surfaces):
        """Compute bias curve for a single mu2 surface."""
        # Convert log probabilities to probabilities
        mu2_prob_surface = jnp.exp(log_surfaces)
        target_prob_profiles = mu2_prob_surface[:, target_feat_indices]

        # Linear expectation computation for mu2 (not circular like mu1)
        # E[mu2] = integral of mu2 * p(mu2) dmu2
        expectations = jnp.trapezoid(mu2_bias_grid.reshape(-1, 1) * target_prob_profiles, x=mu2_bias_grid, axis=0)
        return expectations

    # Apply to entire batch using vmap
    vectorized_compute = jax.vmap(compute_single_mu2_bias_curve)
    return vectorized_compute(mu2_surfaces_batch)


def _generate_mu2_density_asymmetry_batch(mu2_surfaces_batch: jnp.ndarray, target_feat_indices: jnp.ndarray) -> jnp.ndarray:
    """Generate mu2 density asymmetry curves for batch of mu2 surfaces using linear bias (not circular)."""
    # Create mu2 bias grid
    mu2_bias_grid = config.create_grid('mu2_bias')

    # Vectorized computation for each surface
    def compute_single_mu2_density_asymmetry(log_surfaces, apply_smoothing: bool = True, smoothing_sigma: float = 5.0):
        """Compute density asymmetry for a single mu2 log probability surface."""
        # Convert to probabilities
        probs = jnp.exp(log_surfaces)
        
        # Extract probabilities for target feature indices
        target_probs = probs[:, target_feat_indices]
        
        # Compute asymmetry for each target feature difference
        positive_mask = mu2_bias_grid > 0
        negative_mask = mu2_bias_grid < 0
        
        # Vectorized computation across all target indices with proper discretization
        dx = config.mu2_bias_step  # Grid spacing for numerical integration
        
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
            
            # Apply convolution using JAX - pad the asymmetry values
            pad_width = kernel_size // 2
            padded_asymmetry = jnp.pad(asymmetry, pad_width, mode='edge')
            
            # Convolve using correlate (which is equivalent to convolution with flipped kernel)
            smoothed_asymmetry = jnp.correlate(padded_asymmetry, kernel, mode='valid')
            return smoothed_asymmetry
        else:
            return asymmetry

    # Apply to entire batch using vmap with smoothing enabled (sigma=5)
    vectorized_compute = jax.vmap(lambda log_surf: compute_single_mu2_density_asymmetry(
        log_surf, apply_smoothing=True, smoothing_sigma=5.0
    ))
    return vectorized_compute(mu2_surfaces_batch)


def compute_predicted_sd_curves_batch(log_surfaces_batch, feat_vals):
    """Compute predicted circular-SD curves for a batch of surfaces (fully vectorized).

    For each surface and each requested feature-difference value, integrates the
    probability distribution over the mu1_bias axis using the circular standard
    deviation formula ``sqrt(-2 * log(R))`` where ``R`` is the mean resultant
    length.

    Args:
        log_surfaces_batch: Log-probability surfaces with shape
            ``(n_surfaces, n_mu1_bias, n_feat_diff)``.
        feat_vals: Sequence of feature-difference values (in degrees) at which
            to evaluate the SD curves.  Each value is snapped to the nearest
            grid index.

    Returns:
        Array of shape ``(n_surfaces, len(feat_vals))`` containing circular
        standard deviations in degrees.
    """
    mu1_bias_grid = config.create_grid('mu1_bias')
    n_surfaces, n_mu1_bias, n_feat_diff = log_surfaces_batch.shape
    n_feat_vals = len(feat_vals)
    
    # Convert to probability space
    prob_surfaces = jnp.exp(log_surfaces_batch)  # Shape: (n_surfaces, n_mu1_bias, n_feat_diff)
    
    # Vectorized feature index mapping
    feat_indices = jnp.round((jnp.array(feat_vals) - config.feat_diff_range[0]) / config.feat_diff_step).astype(int)
    feat_indices = jnp.clip(feat_indices, 0, n_feat_diff - 1)
    
    # Extract probability profiles for all surfaces and feature values at once
    # Shape: (n_surfaces, n_mu1_bias, n_feat_vals)
    prob_profiles = prob_surfaces[:, :, feat_indices]
    
    # Precompute angular components
    angles_rad = jnp.radians(mu1_bias_grid)  # Shape: (n_mu1_bias,)
    cos_angles = jnp.cos(angles_rad)  # Shape: (n_mu1_bias,)
    sin_angles = jnp.sin(angles_rad)  # Shape: (n_mu1_bias,)
    bias_step = float(jnp.diff(mu1_bias_grid)[0])
    
    # Vectorized circular SD computation for all surfaces and feature values
    # Compute mean cos and sin using trapezoidal integration
    # prob_profiles: (n_surfaces, n_mu1_bias, n_feat_vals)
    # cos_angles: (n_mu1_bias,) -> broadcast to (1, n_mu1_bias, 1)
    mean_cos = jnp.trapezoid(prob_profiles * cos_angles[None, :, None], x=mu1_bias_grid, axis=1)  # Shape: (n_surfaces, n_feat_vals)
    mean_sin = jnp.trapezoid(prob_profiles * sin_angles[None, :, None], x=mu1_bias_grid, axis=1)  # Shape: (n_surfaces, n_feat_vals)
    
    # Compute circular standard deviations
    r = jnp.sqrt(mean_cos**2 + mean_sin**2)
    r_safe = jnp.maximum(r, 1e-10)  # Avoid log(0)
    circular_sds = jnp.degrees(jnp.sqrt(-2 * jnp.log(r_safe)))  # Shape: (n_surfaces, n_feat_vals)
    
    return circular_sds


def load_averaged_surface(sd_feat1: float, sd_feat2: float, sd_spat: float, n_samples: int,
                          surfaces_dir: str) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load averaged surfaces from file for given parameters.
    Surfaces are stored in canonical form (sf1 <= sf2) with:
    - mu1_comp1_surface: lower-noise component bias surface
    - mu1_comp2_surface: higher-noise component bias surface

    Args:
        sd_feat1, sd_feat2, sd_spat: Parameter values
        n_samples: Number of samples used for training
        surfaces_dir: Path to the averaged surfaces directory.

    Returns:
        Tuple of (mu1_surface, mu2_surface) as JAX arrays (appropriate components based on parameter ordering)

    Raises:
        FileNotFoundError: If surface file doesn't exist
    """
    # Use canonical ordering for filename; ensure .0 suffix matches saved filenames
    canonical_sf1 = float(min(sd_feat1, sd_feat2))
    canonical_sf2 = float(max(sd_feat1, sd_feat2))
    sd_spat_f = float(sd_spat)

    filename = f"averaged_sf1_{canonical_sf1}_sf2_{canonical_sf2}_sp_{sd_spat_f}.pkl"
    # Resolve the individual file, materialising it from a surface bundle when
    # only the bundled form exists (raises FileNotFoundError if truly absent).
    file_path = ensure_averaged_surface_file(surfaces_dir, filename)

    try:
        with open(file_path, 'rb') as f:
            surface_data = SurfaceUnpickler(f).load()
        
        surface_obj = surface_data['surface']
        
        # Select appropriate components based on parameter ordering
        if sd_feat1 <= sd_feat2:
            # Use lower-noise component surfaces (comp1)
            mu1_surface = jnp.array(surface_obj.mu1_comp1_surface)
            mu2_surface = jnp.array(surface_obj.mu2_comp1_surface)
        else:
            # Use higher-noise component surfaces (comp2)
            mu1_surface = jnp.array(surface_obj.mu1_comp2_surface)
            mu2_surface = jnp.array(surface_obj.mu2_comp2_surface)
            
        return mu1_surface, mu2_surface
            
    except Exception as e:
        raise RuntimeError(f"Failed to load surface from {file_path}: {e}")


def simulate_surfaces_from_file(input_path: str, n_samples: int, output_path: str,
                              skip_motor_noise: bool = False, use_nn_surfaces: bool = True,
                              averaged_surfaces_dir: Optional[str] = None,
                              explicit_checkpoint_path: Optional[str] = None):
    """
    Read parameters from CSV/Arrow file, simulate surfaces, and save results.

    Args:
        input_path: Path to CSV/Arrow file with parameters
        n_samples: Number of samples used for training (determines checkpoint path)
        output_path: Path to save results
        skip_motor_noise: Whether to skip motor noise
        use_nn_surfaces: Whether to use NN predictions (True) or averaged surfaces from files (False)
    """
    print(f"Reading parameters from {input_path}...")
    
    # Read parameters from file (support both CSV and Arrow)
    try:
        if input_path.endswith('.arrow') or input_path.endswith('.parquet'):
            params_df = pd.read_parquet(input_path)
        else:
            params_df = pd.read_csv(input_path)
        print(f"Loaded {len(params_df)} parameter combinations")
        print(f"DataFrame dtypes:\n{params_df.dtypes}")
        print(f"DataFrame head:\n{params_df.head()}")
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Validate CSV columns
    if skip_motor_noise:
        required_cols = ['sd_feat1', 'sd_feat2', 'sd_spat']
    else:
        required_cols = ['sd_feat1', 'sd_feat2', 'sd_spat', 'sd_motor']
    
    missing_cols = [col for col in required_cols if col not in params_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in CSV: {missing_cols}")
        print(f"Available columns: {list(params_df.columns)}")
        sys.exit(1)
    
    if not use_nn_surfaces and not averaged_surfaces_dir:
        raise ValueError("averaged_surfaces_dir is required when use_nn_surfaces=False")

    # Convert to JAX array and handle motor noise
    if skip_motor_noise:
        # Add sd_motor column with zeros when motor noise is skipped
        params_df['sd_motor'] = 0.0
        parameters_array = jnp.array(params_df[['sd_feat1', 'sd_feat2', 'sd_spat', 'sd_motor']].values, dtype=jnp.float32)
    else:
        parameters_array = jnp.array(params_df[required_cols].values, dtype=jnp.float32)
    
    if explicit_checkpoint_path:
        resolved_checkpoint_path = Path(explicit_checkpoint_path)
    else:
        checkpoint_path = PRETRAINED_CHECKPOINTS.get(
            n_samples,
            f'{CHECKPOINT_PREFIX}_{n_samples}samples/model_epoch_{CHECKPOINT_EPOCH:04d}.pkl',
        )
        resolved_checkpoint_path = resolve_input_path(checkpoint_path, RESULTS_DIR)

    # Initialize optimizer for simulation only (no datasets needed) - but only if using NN surfaces
    if use_nn_surfaces:
        try:
            print(f"Using checkpoint: {resolved_checkpoint_path}")
            optimizer = GridBasedMultiConditionOptimizer(
                checkpoint_path=str(resolved_checkpoint_path),
                condition_datasets=None,  # No datasets needed for simulation
                skip_motor_noise=skip_motor_noise
            )
            print("Optimizer initialized successfully")
        except Exception as e:
            print(f"Error initializing optimizer: {e}")
            sys.exit(1)
    else:
        print("Using averaged surfaces from files instead of NN predictions")
        # We still need the grid configuration, so create a minimal config object
        optimizer = type('obj', (object,), {
            'feat_diff_grid': config.create_grid('feat_diff'),
            'mu1_bias_grid': config.create_grid('mu1_bias')
        })()
    
    # Extract parameters
    sd_feat1 = parameters_array[:, 0]
    sd_feat2 = parameters_array[:, 1] 
    sd_spat = parameters_array[:, 2]
    sd_motor = parameters_array[:, 3]
    
    print(f"Simulating surfaces for {len(parameters_array)} parameter combinations...")
    
    if use_nn_surfaces:
        # Create NN input parameters (sd_feat1, sd_feat2, sd_spat)
        nn_params = jnp.column_stack([sd_feat1, sd_feat2, sd_spat])
        
        # Generate base surfaces using NN (only mu1 surfaces available)
        log_surfaces_batch = optimizer._predict_batch_fixed_size(nn_params, verbosity=1)
        mu2_surfaces_batch = None  # NN doesn't provide mu2 surfaces
        print("Note: NN surfaces only provide mu1 bias curves, mu2 bias curves not available")
    else:
        # Load averaged surfaces from files
        print("Loading averaged surfaces from files...")
        mu1_surfaces_list = []
        mu2_surfaces_list = []
        
        for i, (sf1, sf2, sp) in enumerate(zip(sd_feat1, sd_feat2, sd_spat)):
            try:
                mu1_surface, mu2_surface = load_averaged_surface(float(sf1), float(sf2), float(sp), n_samples, surfaces_dir=averaged_surfaces_dir)
                mu1_surfaces_list.append(mu1_surface)
                mu2_surfaces_list.append(mu2_surface)
            except FileNotFoundError as e:
                print(f"Error loading surface for parameter combination {i+1}: {e}")
                sys.exit(1)
            except RuntimeError as e:
                print(f"Error loading surface for parameter combination {i+1}: {e}")
                sys.exit(1)
        
        log_surfaces_batch = jnp.stack(mu1_surfaces_list, axis=0)  # Keep mu1 for existing functionality
        mu2_surfaces_batch = jnp.stack(mu2_surfaces_list, axis=0)  # Add mu2 surfaces
        print(f"Successfully loaded {len(mu1_surfaces_list)} averaged mu1 and mu2 surface pairs")
    
    # Apply motor noise if not skipped
    if not skip_motor_noise:
        print("Applying motor noise...")
        # Group by unique sd_motor values for efficiency
        unique_motors, motor_indices = jnp.unique(sd_motor, return_inverse=True)
        
        final_surfaces = jnp.empty_like(log_surfaces_batch)
        for i, unique_motor in enumerate(unique_motors):
            # Get surfaces for this motor noise level
            mask = motor_indices == i
            surfaces_subset = log_surfaces_batch[mask]
            
            if len(surfaces_subset) > 0:
                # Apply motor noise
                noisy_surfaces = apply_motor_noise(surfaces_subset, float(unique_motor))
                final_surfaces = final_surfaces.at[mask].set(noisy_surfaces)

        log_surfaces_batch = final_surfaces
    else:
        print("Skipping motor noise (sd_motor = 0)")
    
    # Generate density curves
    print("Computing density curves...")
    density_curves = generate_nn_density_asymmetry_batch(log_surfaces_batch)
    
    # Generate expectation curves (bias curves)
    print("Computing mu1 expectation curves...")
    # We need to create target feature indices for the bias curve computation
    feat_diff_grid = optimizer.feat_diff_grid
    target_feat_indices = jnp.arange(len(feat_diff_grid))  # Use all feature indices
    expectation_curves = _generate_nn_bias_curve_batch(log_surfaces_batch, target_feat_indices)
    
    # Generate mu2 expectation curves if available
    if mu2_surfaces_batch is not None:
        print("Computing mu2 expectation curves...")
        mu2_expectation_curves = _generate_mu2_bias_curve_batch(mu2_surfaces_batch, target_feat_indices)
        
        print("Computing mu2 density asymmetry curves...")
        mu2_density_curves = _generate_mu2_density_asymmetry_batch(mu2_surfaces_batch, target_feat_indices)
    else:
        mu2_expectation_curves = None
        mu2_density_curves = None
    
    # Generate SD curves
    print("Computing standard deviation curves...")
    sd_curves = compute_predicted_sd_curves_batch(log_surfaces_batch, feat_diff_grid)
    
    # Prepare results as flattened dataframe for Arrow
    n_combinations = len(parameters_array)
    n_feat_points = len(optimizer.feat_diff_grid)
    
    # Create a row for each parameter combination
    results_list = []
    for i in range(n_combinations):
        row = {
            'sd_feat1': float(parameters_array[i, 0]),
            'sd_feat2': float(parameters_array[i, 1]), 
            'sd_spat': float(parameters_array[i, 2]),
            'sd_motor': float(parameters_array[i, 3]),
            'mu1_density_curve': density_curves[i].tolist(),  # Renamed for clarity 
            'mu2_density_curve': mu2_density_curves[i].tolist() if mu2_density_curves is not None else [float('nan')] * n_feat_points,
            'mu1_expectation_curve': expectation_curves[i].tolist(),
            'mu2_expectation_curve': mu2_expectation_curves[i].tolist() if mu2_expectation_curves is not None else [float('nan')] * n_feat_points,
            'sd_curve': sd_curves[i].tolist(),  # Store SD curve as list
            # Store config info in first row only to avoid duplication
            'feat_diff_grid': optimizer.feat_diff_grid.tolist() if i == 0 else None,
            'mu1_bias_grid': optimizer.mu1_bias_grid.tolist() if i == 0 else None,
            'mu2_bias_grid': config.create_grid('mu2_bias').tolist() if i == 0 else None,
            'feat_diff_range': list(config.feat_diff_range) if i == 0 else None,
            'mu1_bias_range': list(config.mu1_bias_range) if i == 0 else None,
            'mu2_bias_range': list(config.mu2_bias_range) if i == 0 else None,
            'feat_diff_step': config.feat_diff_step if i == 0 else None,
            'mu1_bias_step': config.mu1_bias_step if i == 0 else None,
            'mu2_bias_step': config.mu2_bias_step if i == 0 else None,
            'n_samples': n_samples if i == 0 else None,
            'skip_motor_noise': skip_motor_noise if i == 0 else None,
            'has_mu2_data': mu2_expectation_curves is not None if i == 0 else None
        }
        results_list.append(row)
    
    results_df = pd.DataFrame(results_list)
    
    # Save results
    print(f"Saving results to {output_path}...")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if output_path.endswith('.arrow') or output_path.endswith('.parquet'):
            results_df.to_parquet(output_path, index=False)
        else:
            # Fallback to CSV if not arrow
            results_df.to_csv(output_path, index=False)
        print(f"Results saved successfully!")
        print(f"Generated {len(log_surfaces_batch)} surfaces and density curves.")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)


def main():
    """Parse CLI args and run surface simulation."""
    parser = argparse.ArgumentParser(description='Simulate surfaces from CSV/Arrow parameters')
    # The three required inputs may be given positionally (legacy form, used by the smoke
    # scripts) or by name. Named forms win if both are supplied.
    parser.add_argument('input_path', nargs='?',
                        help='Path to CSV or Arrow file with parameters')
    parser.add_argument('n_samples', nargs='?', type=int,
                        help='Number of samples used for training (e.g., 20, 100)')
    parser.add_argument('output_path', nargs='?',
                        help='Path to save results (Arrow/CSV file)')
    parser.add_argument('--input-path', dest='input_path_named',
                        help='Named form of the first positional argument')
    parser.add_argument('--n-samples', dest='n_samples_named', type=int,
                        help='Named form of the second positional argument')
    parser.add_argument('--output-path', dest='output_path_named',
                        help='Named form of the third positional argument')
    parser.add_argument('--skip-motor-noise', action='store_true', 
                       help='Skip motor noise computation (sd_motor = 0)')
    parser.add_argument('--surface-source', choices=['nn', 'raw'], default='nn',
                       help='nn: use NN approximation of averaged surfaces (default); raw: load averaged surfaces directly from files')
    parser.add_argument('--averaged-surfaces-dir',
                       help='Path to averaged surfaces directory (required with --surface-source raw).')
    parser.add_argument('--checkpoint-path',
                       help='Explicit path to a model checkpoint .pkl file (overrides the path '
                            'auto-derived from n_samples).')

    args = parser.parse_args()

    for name in ('input_path', 'n_samples', 'output_path'):
        named = getattr(args, f'{name}_named')
        if named is not None:
            setattr(args, name, named)
        if getattr(args, name) is None:
            parser.error(f'{name} is required (give it positionally or as '
                         f'--{name.replace("_", "-")})')

    if args.surface_source == 'raw' and not args.averaged_surfaces_dir:
        parser.error('--averaged-surfaces-dir is required when --surface-source raw')

    # Validate inputs
    if not Path(args.input_path).exists():
        print(f"Error: Input file not found: {args.input_path}")
        sys.exit(1)

    # Run simulation
    simulate_surfaces_from_file(
        args.input_path,
        args.n_samples,
        args.output_path,
        args.skip_motor_noise,
        use_nn_surfaces=args.surface_source == 'nn',
        averaged_surfaces_dir=args.averaged_surfaces_dir,
        explicit_checkpoint_path=args.checkpoint_path,
    )


if __name__ == '__main__':
    main()
