#!/usr/bin/env python3
"""
Grid-Based Multi-Condition Optimizer - Simplified Version

This optimizer implements your two-stage approach:
1. Create a coarse grid for shared parameters (sd_spat, sd_motor)
2. For each grid point, compute NN surfaces once for different (sd_feat1, sd_feat2) combinations
3. Optimize condition-specific parameters independently using shared surfaces
4. Zoom in and repeat for refinement

This avoids redundant NN evaluations by reusing surfaces across conditions.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
import pickle

from shared.config import config
from shared.mu1_axis import bin_indices, periodic_integral
from shared.utils import load_checkpoint, compute_single_density_asymmetry


def _generate_nn_bias_curve_batch(log_surfaces_batch: jnp.ndarray, target_feat_indices: jnp.ndarray) -> jnp.ndarray:
    """Generate mu1 circular-mean bias curves for a batch of surfaces.

    Args:
        log_surfaces_batch: Log-probability surfaces with shape
            ``(n_surfaces, n_mu1_bias, n_feat_diff)``.
        target_feat_indices: Integer indices into the feat_diff axis selecting
            which feature-difference values to evaluate.

    Returns:
        Circular mean bias values in degrees with shape
        ``(n_surfaces, len(target_feat_indices))``.
    """
    # Create grids
    mu1_bias_grid = config.create_grid('mu1_bias')

    # Vectorized computation for each surface
    def compute_single_bias_curve(log_surfaces):
        """

        :param log_surfaces:
        :return:
        """
        # NN always outputs 2D surfaces with shape (n_mu1_bias, n_feat_diff)
        # Convert log probabilities to probabilities
        mu1_prob_surface = jnp.exp(log_surfaces)
        target_prob_profiles = mu1_prob_surface[:, target_feat_indices]

        # Vectorized circular mean computation for all target indices
        cos_vals = jnp.cos(jnp.radians(mu1_bias_grid)).reshape(-1, 1)
        sin_vals = jnp.sin(jnp.radians(mu1_bias_grid)).reshape(-1, 1)

        # Periodic quadrature: trapezoid over x=grid spans only 358 of the 360
        # degrees and half-weights the two ends.
        cos_expectations = periodic_integral(cos_vals * target_prob_profiles, axis=0)
        sin_expectations = periodic_integral(sin_vals * target_prob_profiles, axis=0)

        circular_means = jnp.degrees(jnp.arctan2(sin_expectations, cos_expectations))
        return circular_means

    # Apply to entire batch using vmap
    vectorized_compute = jax.vmap(compute_single_bias_curve)
    return vectorized_compute(log_surfaces_batch)


@jax.jit
def compute_target_bias_curve_core(feat_diff_values: jnp.ndarray, bias_values: jnp.ndarray, n_trials: jnp.ndarray) -> \
Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute a binned target bias curve for a single condition (vmap-compatible).

    Bins trials by feature-difference into 4-degree bins, computes the circular
    mean bias per bin, and maps each bin centre to the nearest NN grid index.
    Padding beyond ``n_trials`` is masked out before any computation.

    Bin-edge detail: with the standard grid the centres are arange(2, 184, 4) =
    [2, 6, ..., 178, 182] — 46 bins whose final centre (182°) lies off-grid and
    is mapped by nearest-index onto the terminal 180° column (see
    MODEL_PIPELINE_FOR_AGENTS.md D.12).

    Args:
        feat_diff_values: Feature-difference values for each trial, zero-padded
            to the fixed maximum trial count.
        bias_values: Bias values in degrees for each trial, zero-padded to match.
        n_trials: Scalar giving the actual number of valid (unpadded) trials.

    Returns:
        Tuple of ``(feat_indices, target_bias, weights)`` where ``feat_indices``
        maps each bin to the NN feat_diff grid index, ``target_bias`` is the
        circular mean bias per bin in degrees, and ``weights`` are trial counts
        per bin (float32).
    """
    # Create mask for valid trials instead of dynamic slicing
    max_trials = len(feat_diff_values)
    valid_mask = jnp.arange(max_trials) < n_trials

    # Apply mask to get valid values
    valid_feat_diff = jnp.where(valid_mask, feat_diff_values, 0.0)
    valid_bias = jnp.where(valid_mask, bias_values, 0.0)

    # Use binning with twice the configured 2-degree step: 4-degree bins.
    bin_step = config.feat_diff_step * 2

    # Create bin centers from the configured lower bound in bin_step increments.
    min_feat = config.feat_diff_range[0]  # Standard grid starts at 2 degrees
    max_feat = config.feat_diff_range[1]  # Usually 180
    bin_centers = jnp.arange(min_feat, max_feat + bin_step, bin_step)
    n_bins = len(bin_centers)

    # Snap each trial to nearest bin center using vectorized operations
    # Only consider valid trials by masking invalid distances
    distances = jnp.abs(valid_feat_diff[:, None] - bin_centers[None, :])
    distances = jnp.where(valid_mask[:, None], distances, jnp.inf)  # Invalid trials get infinite distance
    nearest_bin_indices = jnp.argmin(distances, axis=1)

    # Count trials per bin and compute mean bias per bin
    # Use segment_sum for vectorized binning
    bin_counts = jax.ops.segment_sum(
        jnp.where(valid_mask, 1.0, 0.0),
        nearest_bin_indices,
        num_segments=n_bins
    )
    bias_rad = jnp.radians(valid_bias)
    bin_cos_sums = jax.ops.segment_sum(
        jnp.where(valid_mask, jnp.cos(bias_rad), 0.0),
        nearest_bin_indices,
        num_segments=n_bins
    )
    bin_sin_sums = jax.ops.segment_sum(
        jnp.where(valid_mask, jnp.sin(bias_rad), 0.0),
        nearest_bin_indices,
        num_segments=n_bins
    )

    # Circular mean per bin (handle empty bins)
    target_bias = jnp.where(
        bin_counts > 0,
        jnp.degrees(jnp.arctan2(bin_sin_sums / jnp.maximum(bin_counts, 1),
                                 bin_cos_sums / jnp.maximum(bin_counts, 1))),
        0.0
    )
    weights = bin_counts.astype(jnp.float32)

    # Map bin centers to NN grid indices
    feat_diff_grid = config.create_grid('feat_diff')
    grid_distances = jnp.abs(bin_centers[:, None] - feat_diff_grid[None, :])
    feat_indices = jnp.argmin(grid_distances, axis=1)

    return feat_indices, target_bias, weights


def create_shared_parameter_grid(grid_size: int = 20,
                                 sd_spat_range: Tuple[float, float] = (config.param_grid_low, config.param_range_high),
                                 sd_motor_range: Tuple[float, float] = (0.1, 50.0),
                                 skip_motor_noise: bool = False,
                                 motor_grid_size: Optional[int] = None) -> jnp.ndarray:
    """Create grid of shared parameters (sd_spat, sd_motor).

    Args:
        grid_size: Number of grid points along the sd_spat dimension
        sd_spat_range: Range for sd_spat, defaults to config bounds
        sd_motor_range: Range for sd_motor, defaults to [0.1, 50.0]
        skip_motor_noise: If True, create only sd_spat grid with sd_motor=0
        motor_grid_size: Number of grid points along the sd_motor dimension.
            Defaults to grid_size (square grid). Decoupling it lets the motor
            axis use fewer points when its data-informed range is small.

    Returns:
        Array of shape (grid_size*motor_grid_size, 2) or (grid_size, 2).
    """
    if motor_grid_size is None:
        motor_grid_size = grid_size
    sd_spat_vals = jnp.linspace(sd_spat_range[0], sd_spat_range[1], grid_size)

    if skip_motor_noise:
        # Create 1D grid with only sd_spat, sd_motor fixed to 0
        grid_points = jnp.stack([
            sd_spat_vals,
            jnp.zeros_like(sd_spat_vals)
        ], axis=1)
        
        print(f"Created shared parameter grid (motor noise SKIPPED): {grid_size} points")
        print(f"  sd_spat range: [{sd_spat_range[0]:.1f}, {sd_spat_range[1]:.1f}]")
        print(f"  sd_motor: 0.0 (fixed)")
    else:
        sd_motor_vals = jnp.linspace(sd_motor_range[0], sd_motor_range[1], motor_grid_size)

        # Create all combinations - sd_motor varies slowly (outer loop), sd_spat varies fast (inner loop)
        sd_motor_grid, sd_spat_grid = jnp.meshgrid(sd_motor_vals, sd_spat_vals, indexing='ij')

        # Flatten to get all grid points - order: sd_spat cycles through for each sd_motor
        grid_points = jnp.stack([
            sd_spat_grid.flatten(),
            sd_motor_grid.flatten()
        ], axis=1)

        print(f"Created shared parameter grid: {grid_size}×{motor_grid_size} = {len(grid_points)} points")
        print(f"  sd_spat range: [{sd_spat_range[0]:.1f}, {sd_spat_range[1]:.1f}]")
        print(f"  sd_motor range: [{sd_motor_range[0]:.1f}, {sd_motor_range[1]:.1f}]")

    return grid_points


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def generate_nn_density_asymmetry_batch(log_surfaces_batch: jnp.ndarray,
                                        feat_diff_grid=config.create_grid('feat_diff'),
                                        mu1_bias_grid=config.create_grid('mu1_bias'),
                                        weights_sd: float = 20.0,
                                        smoothing_sigma: float = None
                                        ) -> jnp.ndarray:
    """Generate density asymmetry curves for batch of surfaces using NN surface with smoothing.

    Args:
        weights_sd: Gaussian SD (degrees) used to weight empirical trials in feat_diff space.
            Used to derive smoothing_sigma when smoothing_sigma is not explicitly provided:
            smoothing_sigma = weights_sd / feat_diff_step.
        smoothing_sigma: Override for the Gaussian smoothing kernel size (grid steps).
            If None, derived from weights_sd.
    """
    target_feat_indices = jnp.arange(len(feat_diff_grid))
    if smoothing_sigma is None:
        smoothing_sigma = weights_sd / config.feat_diff_step

    vectorized_compute = jax.vmap(lambda log_surf: compute_single_density_asymmetry(
        log_surf, target_feat_indices, mu1_bias_grid, apply_smoothing=True, smoothing_sigma=smoothing_sigma
    ))
    return vectorized_compute(log_surfaces_batch)


def create_motor_noise_kernel_fft(sd_motor: float, n_mu1_bias: int) -> jnp.ndarray:
    """Create precomputed FFT kernel for motor noise convolution.
    
    Args:
        sd_motor: Motor noise standard deviation in degrees
        n_mu1_bias: Number of bias grid points
        
    Returns:
        FFT of the motor noise kernel
    """
    bias_step = config.mu1_bias_step
    sigma = jnp.maximum(sd_motor, 0.01)
    
    # Create kernel indices centered around 0
    all_kernel_indices = jnp.arange(n_mu1_bias) - n_mu1_bias // 2
    kernel_angles = all_kernel_indices * bias_step
    
    # Create weights
    weights = jnp.exp(-0.5 * (kernel_angles / sigma) ** 2)
    weights = weights / jnp.sum(weights)  # Normalize
    
    # Create full kernel array with proper positioning
    full_kernel = jnp.zeros(n_mu1_bias)
    kernel_positions = all_kernel_indices % n_mu1_bias
    full_kernel = full_kernel.at[kernel_positions].set(weights)

    # Return precomputed FFT
    return jnp.fft.fft(full_kernel)


@jax.jit
def apply_motor_noise_with_precomputed_kernel(log_surfaces_batch: jnp.ndarray, kernel_fft: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled motor noise convolution using precomputed FFT kernel.

    Args:
        log_surfaces_batch: Batch of log likelihood surfaces of shape (n_surfaces, n_mu1_bias, n_feat_diff)
        kernel_fft: Precomputed FFT of the motor noise kernel

    Returns:
        Batch of log likelihood surfaces after motor noise convolution
    """
    # Convert to probability space with numerical stability
    log_max = jnp.max(log_surfaces_batch, axis=(1, 2), keepdims=True)
    prob_surfaces_batch = jnp.exp(log_surfaces_batch - log_max)

    signal_fft = jnp.fft.fft(prob_surfaces_batch, axis=1)
    # Reshape kernel_fft for proper broadcasting: (n_mu1_bias,) -> (1, n_mu1_bias, 1)
    kernel_fft_reshaped = kernel_fft[None, :, None]
    product = signal_fft * kernel_fft_reshaped
    ifft_result = jnp.fft.ifft(product, axis=1)
    prob_convolved_batch = jnp.real(ifft_result)

    # Clamp negative values and convert back to log space
    prob_convolved_batch = jnp.clip(prob_convolved_batch, 0.0)
    log_surfaces_noisy = jnp.log(prob_convolved_batch + 1e-10) + log_max

    return log_surfaces_noisy


def apply_motor_noise(log_surfaces_batch: jnp.ndarray, sd_motor: float) -> jnp.ndarray:
    """JIT-compiled motor noise convolution using FFT with proper circular indexing.

    Args:
        log_surfaces_batch: Batch of log likelihood surfaces of shape (n_surfaces, n_mu1_bias, n_feat_diff)
        sd_motor: Motor noise standard deviation in degrees

    Returns:
        Batch of log likelihood surfaces after motor noise convolution
    """
    n_surfaces, n_mu1_bias, n_feat_diff = log_surfaces_batch.shape

    # Create full kernel exactly like test_motor_convolution.py (no size limits)
    bias_step = config.mu1_bias_step
    sigma = jnp.maximum(sd_motor, 0.01)
    
    # Create kernel indices centered around 0 (same as before)
    all_kernel_indices = jnp.arange(n_mu1_bias) - n_mu1_bias // 2
    kernel_angles = all_kernel_indices * bias_step
    
    # Create weights
    weights = jnp.exp(-0.5 * (kernel_angles / sigma) ** 2)
    weights = weights / jnp.sum(weights)  # Normalize
    
    # Create full kernel array with proper positioning (KEY FIX from test file)
    full_kernel = jnp.zeros(n_mu1_bias)
    kernel_positions = all_kernel_indices % n_mu1_bias  # Use modulo for circular indexing
    full_kernel = full_kernel.at[kernel_positions].set(weights)

    # Precompute kernel FFT
    kernel_fft = jnp.fft.fft(full_kernel)

    # Convert to probability space with numerical stability
    # Use global max per surface to preserve relative magnitudes across features
    log_max = jnp.max(log_surfaces_batch, axis=(1, 2), keepdims=True)
    prob_surfaces_batch = jnp.exp(log_surfaces_batch - log_max)

    signal_fft = jnp.fft.fft(prob_surfaces_batch, axis=1)
    # Reshape kernel_fft for proper broadcasting: (n_mu1_bias,) -> (1, n_mu1_bias, 1)
    kernel_fft_reshaped = kernel_fft[None, :, None]
    product = signal_fft * kernel_fft_reshaped  # Broadcasting: (B, N, F) * (1, N, 1) = (B, N, F)
    ifft_result = jnp.fft.ifft(product, axis=1)
    prob_convolved_batch = jnp.real(ifft_result)

    # Clamp negative values and convert back to log space
    prob_convolved_batch = jnp.clip(prob_convolved_batch, 0.0)
    log_surfaces_noisy = jnp.log(prob_convolved_batch + 1e-10) + log_max

    return log_surfaces_noisy


def _compute_curve_losses(predicted_curves: jnp.ndarray, target_curves: jnp.ndarray,
                          loss_type: str = "mse", is_angular: bool = False,
                          weights: jnp.ndarray = None, corr_weight = 0.5) -> jnp.ndarray:
    """Unified function to compute losses between predicted and target curves.

    Args:
        predicted_curves: Shape (n_combos, n_points) - predicted curve values
        target_curves: Shape (n_combos, n_points) - target curve values
        loss_type: One of "mse", "mad", "combined"
        is_angular: If True, handle angular differences properly (for bias curves)
        weights: Shape (n_combos, n_points) or (n_points,) - weights for each point

    Returns:
        losses: Shape (n_combos,) - loss for each combination
    """
    # Compute differences
    if is_angular:
        # Handle angular differences for bias curves (wrap to [-180, 180])
        diff = predicted_curves - target_curves
        diff = ((diff + 180) % 360) - 180
    else:
        # Regular differences for density curves
        diff = predicted_curves - target_curves

    loss_type = loss_type.lower()
    # Handle weights (broadcast if needed)
    if loss_type == 'mse' or loss_type == 'mad':
        if weights is not None:
            if weights.ndim == 1:
                # Broadcast weights to match predicted_curves shape
                weights = jnp.broadcast_to(weights[None, :], predicted_curves.shape)

            # Only compute loss for non-zero weights (ignore empty bins)
            valid_mask = weights > 0

            if loss_type == "mse":
                weighted_sq_errors = jnp.where(valid_mask, (diff ** 2) * weights, 0.0)
                total_weights = jnp.sum(jnp.where(valid_mask, weights, 0.0), axis=1)
                losses = jnp.sum(weighted_sq_errors, axis=1) / jnp.maximum(total_weights, 1.0)
            elif loss_type == "mad":
                weighted_abs_errors = jnp.where(valid_mask, jnp.abs(diff) * weights, 0.0)
                total_weights = jnp.sum(jnp.where(valid_mask, weights, 0.0), axis=1)
                losses = jnp.sum(weighted_abs_errors, axis=1) / jnp.maximum(total_weights, 1.0)
        else:
            # No weights - use regular mean
            if loss_type == "mse":
                losses = jnp.mean(diff ** 2, axis=1)
            elif loss_type == "mad":
                losses = jnp.mean(jnp.abs(diff), axis=1)

    elif loss_type == "combined":
        # Combined MSE + correlation loss (like in two_stage_sgd_class)
        mse_losses = jnp.mean(diff ** 2, axis=1)

        # Compute correlation component (1 - correlation) for each combo
        def compute_correlation_loss(pred, target):
            """Compute correlation-based loss for a single pair of curves."""
            # Compute correlation and convert to loss
            corr_matrix = jnp.corrcoef(pred, target)
            corr = corr_matrix[0, 1]
            corr_loss = 1 - jnp.where(jnp.isnan(corr), 0.0, corr)  # Handle NaN correlation
            return corr_loss

        corr_losses = jax.vmap(compute_correlation_loss)(predicted_curves, target_curves)

        # Scale MSE by target range and combine
        target_range = jnp.abs(jnp.max(target_curves, axis=1) - jnp.min(target_curves, axis=1))
        mse_scaled = mse_losses / jnp.maximum(target_range, 1e-6)

        # Combined loss with weights
        losses = (1 - corr_weight) * mse_scaled + corr_weight * corr_losses
    elif loss_type == "ccc":
        pred_mean = jnp.mean(predicted_curves, axis=1)
        target_mean = jnp.mean(target_curves, axis=1)
        pred_centered = predicted_curves - pred_mean[:, None]
        target_centered = target_curves - target_mean[:, None]

        covariance = jnp.mean(pred_centered * target_centered, axis=1)
        pred_var = jnp.mean(pred_centered ** 2, axis=1)
        target_var = jnp.mean(target_centered ** 2, axis=1)
        mean_diff_sq = (pred_mean - target_mean) ** 2

        ccc = (2 * covariance) / jnp.maximum(pred_var + target_var + mean_diff_sq, 1e-10)
        losses = 1 - ccc
    elif loss_type == "nrmse":
        rmse = jnp.sqrt(jnp.mean(diff ** 2, axis=1))
        target_range = jnp.abs(jnp.max(target_curves, axis=1) - jnp.min(target_curves, axis=1))
        losses = rmse / jnp.maximum(target_range, 1e-6)
    else:
        raise ValueError("Unknown loss type {}. Should be one of mse, mad, combined, ccc, or nrmse.".format(loss_type))

    return losses


class GridBasedMultiConditionOptimizer:
    """Simplified grid-based multi-condition optimizer with direct NN loading."""

    def __init__(self, checkpoint_path: str, condition_datasets: Optional[Dict[str, jax.Array]] = None, corr_weight = 0.25, skip_motor_noise: bool = False, emp_density_weights_sd: float = 20.0, density_smoothing_sigma: float = None):
        """Initialize the grid-based multi-condition optimizer.

        Args:
            checkpoint_path: Path to neural network checkpoint
            condition_datasets: Optional dict of {condition_name: jnp.Array}. If None, optimizer can be used for simulation only.
            corr_weight: Weight for correlation loss in combined fitting
            skip_motor_noise: If True, skip all motor noise computations (sd_motor fixed to 0)
        """
        self.checkpoint_path = checkpoint_path
        self.condition_datasets = condition_datasets
        self.condition_names = list(condition_datasets.keys()) if condition_datasets else []
        self.n_conditions = len(self.condition_names)
        self.corr_weight = corr_weight
        self.skip_motor_noise = skip_motor_noise
        self.emp_density_weights_sd = emp_density_weights_sd
        self.density_smoothing_sigma = density_smoothing_sigma  # None → derived from emp_density_weights_sd

        if condition_datasets:
            print(f"Initializing GridBasedMultiConditionOptimizer with {self.n_conditions} conditions:")
            for i, name in enumerate(self.condition_names):
                n_trials = condition_datasets[name].shape[0]
                print(f"  {i + 1}. {name}: {n_trials} trials")
        else:
            print("Initializing GridBasedMultiConditionOptimizer for simulation only (no datasets)")
        
        if self.skip_motor_noise:
            print("Motor noise computations will be SKIPPED (sd_motor = 0)")

        # Load NN model directly
        self._load_nn_model()

        # Cache grids (needed before preparing data)
        self._setup_grids()

        # Prepare data indices for all conditions (only if datasets provided)
        if condition_datasets:
            self._prepare_all_condition_data()

    def _load_nn_model(self):
        """Load neural network model directly from checkpoint."""
        import sys as _sys
        _nn_opt_dir = str(Path(__file__).resolve().parents[1] / "neural_network_optimization")
        if _nn_opt_dir not in _sys.path:
            _sys.path.insert(0, _nn_opt_dir)

        print(f"Loading NN model from {self.checkpoint_path}...")

        try:
            self.model_state, checkpoint_info = load_checkpoint(self.checkpoint_path)
            print(f"Loaded model from {self.checkpoint_path}")
            print(f"Model trained to epoch {checkpoint_info['epoch']}, loss: {checkpoint_info['loss']:.4f}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.checkpoint_path}: {e}")

        # Create batched prediction functions with fixed batch sizes
        @jax.jit
        def predict_batch_64(params_batch):
            """Predict log likelihood surfaces for batch of 64 parameters."""
            return self.model_state.apply_fn(self.model_state.params, params_batch)

        @jax.jit
        def predict_batch_256(params_batch):
            """Predict log likelihood surfaces for batch of 256 parameters."""
            return self.model_state.apply_fn(self.model_state.params, params_batch)

        self.predict_batch_64 = predict_batch_64
        self.predict_batch_256 = predict_batch_256

        # Precompile both batch sizes to avoid recompilation overhead
        print("Precompiling batch prediction functions...")
        dummy_params_64 = jnp.ones((64, 3))  # [sd_feat1, sd_feat2, sd_spat]
        dummy_params_256 = jnp.ones((256, 3))

        # Trigger compilation
        _ = self.predict_batch_64(dummy_params_64)
        _ = self.predict_batch_256(dummy_params_256)

        print("NN model loaded and compiled successfully (batch sizes: 64, 256)")

    def _prepare_all_condition_data(self):
        """Prepare data indices for all conditions with padded arrays for efficient processing."""
        print("Preparing data indices for all conditions...")

        condition_data = []
        max_trials = 0

        for condition_name, dataset in self.condition_datasets.items():
            # Convert data points to grid indices
            feat_diff_vals = dataset[:, 0]
            bias_vals = dataset[:, 1]

            # Convert to grid indices
            feat_indices = jnp.round((feat_diff_vals - config.feat_diff_range[0]) / config.feat_diff_step).astype(int)
            feat_indices = jnp.clip(feat_indices, 0, len(config.create_grid('feat_diff')) - 1)

            # Bias is circular: wrap, never clip.  Clipping sends (179, 180) onto
            # the last row (+178) instead of round-tripping it to bin 0.
            bias_indices = bin_indices(bias_vals)

            condition_data.append({
                'name': condition_name,
                'bias_indices': bias_indices,
                'feat_indices': feat_indices,
                'n_trials': dataset.shape[0]
            })

            max_trials = max(max_trials, dataset.shape[0])
            print(f"  {condition_name}: {dataset.shape[0]} trials → {len(bias_indices)} data points")

        # Create padded arrays for efficient tensor operations
        print(f"Padding all conditions to {max_trials} trials for efficient processing...")

        self.padded_bias_indices = jnp.zeros((self.n_conditions, max_trials), dtype=int)
        self.padded_feat_indices = jnp.zeros((self.n_conditions, max_trials), dtype=int)
        self.trial_masks = jnp.zeros((self.n_conditions, max_trials), dtype=bool)

        for i, cond_data in enumerate(condition_data):
            n_trials = cond_data['n_trials']

            # Fill actual data
            self.padded_bias_indices = self.padded_bias_indices.at[i, :n_trials].set(cond_data['bias_indices'])
            self.padded_feat_indices = self.padded_feat_indices.at[i, :n_trials].set(cond_data['feat_indices'])

            # Create mask for valid trials
            self.trial_masks = self.trial_masks.at[i, :n_trials].set(True)

            # Pad with last valid indices (to avoid indexing errors)
            if n_trials < max_trials:
                last_bias = cond_data['bias_indices'][-1]
                last_feat = cond_data['feat_indices'][-1]
                self.padded_bias_indices = self.padded_bias_indices.at[i, n_trials:].set(last_bias)
                self.padded_feat_indices = self.padded_feat_indices.at[i, n_trials:].set(last_feat)

        self.max_trials = max_trials
        total_trials = sum(cond_data['n_trials'] for cond_data in condition_data)
        print(f"Total data points: {total_trials}")
        print(f"Padded shape: ({self.n_conditions}, {max_trials})")

        # Precompute trial indices for JIT optimization
        self._precompute_trial_indices()

        # JIT compile the optimization function with static fitting_method
        self._optimize_conditions_jit = jax.jit(self._optimize_all_conditions_hierarchical_jit,
                                                static_argnames=['fitting_method'])

        # JIT compile grid creation functions
        # self._create_shared_grid_jit = jax.jit(self._create_shared_parameter_grid_jit)
        # self._create_condition_grid_jit = jax.jit(self._create_condition_parameter_grid_jit)

        # JIT compile motor noise convolution
        # self._apply_motor_noise_jit = jax.jit(_apply_motor_noise_jit_impl)

        # Pre-compile motor noise kernels for common sd_motor values
        # self._precompile_motor_noise_kernels()

        # Pre-compile shared parameter evaluation function
        self._setup_shared_param_evaluation()

        # Precompute all target curves for curve-based fitting
        self._precompute_target_curves()

    def update_dataset(self, condition_datasets: Dict[str, Dict]):
        """Update optimizer with new condition datasets (e.g. the next subject in a
        batch loop).

        Each subject's padded trial-array shape differs, so
        _prepare_all_condition_data below always retraces/recompiles
        _optimize_conditions_jit (and, via _setup_shared_param_evaluation's dummy
        warm-up call, the NN predict_batch_64/predict_batch_256 functions too) —
        this does NOT skip recompilation despite what an earlier version of this
        docstring claimed. Left unmanaged, a long-running batch loop accumulates
        one compiled executable per distinct shape for the life of the process,
        which grows host memory until it OOMs partway through a large batch (see
        codex_audit.md investigation, 2026-07-17: crashed 38/51 subjects into a
        csh2026 run). jax.clear_caches() releases the previous subject's compiled
        executables before this subject's replacements get built.

        Args:
            condition_datasets: Dict of {condition_name: {'data_df': DataFrame}}
        """
        jax.clear_caches()

        self.condition_datasets = condition_datasets
        self.condition_names = list(condition_datasets.keys())
        self.n_conditions = len(self.condition_names)

        # Re-prepare data indices for new conditions. Also re-precomputes target
        # curves internally (see _prepare_all_condition_data) — do not call
        # _precompute_target_curves() again here, it would just redo the same work.
        self._prepare_all_condition_data()

        print(f"Updated optimizer with {self.n_conditions} conditions")

    def _setup_grids(self):
        """Setup grid parameters and cache them."""
        self.feat_diff_grid = config.create_grid('feat_diff')
        self.mu1_bias_grid = config.create_grid('mu1_bias')
        self.n_feat_diff = len(self.feat_diff_grid)
        self.n_mu1_bias = len(self.mu1_bias_grid)

        # Circular distance matrix for CRPS: D[k, l] = min(|grid[k]-grid[l]|, 360-|grid[k]-grid[l]|)
        diff = jnp.abs(self.mu1_bias_grid[:, None] - self.mu1_bias_grid[None, :])
        self.D_circ_matrix = jnp.minimum(diff, 360.0 - diff)

        print(
            f"Grid setup: {self.n_feat_diff} feat_diff × {self.n_mu1_bias} mu1_bias = {self.n_feat_diff * self.n_mu1_bias} points per surface")

    def _precompute_trial_indices(self):
        """Precompute flat trial indices for JIT optimization."""
        print("Precomputing trial indices for JIT optimization...")

        # Create flat trial indices for ALL trials across ALL conditions
        all_trial_indices = []
        trial_to_condition = []

        for cond_idx in range(self.n_conditions):
            condition_bias = self.padded_bias_indices[cond_idx, :]
            condition_feat = self.padded_feat_indices[cond_idx, :]
            condition_mask = self.trial_masks[cond_idx, :]

            # Convert 2D indices to flat indices: bias_idx * n_feat + feat_idx
            flat_indices = condition_bias * self.n_feat_diff + condition_feat

            # Only keep valid trials
            valid_indices = flat_indices[condition_mask]
            all_trial_indices.extend(valid_indices)
            trial_to_condition.extend([cond_idx] * len(valid_indices))

        self.all_trial_indices = jnp.array(all_trial_indices)
        self.trial_to_condition = jnp.array(trial_to_condition)

        print(f"Precomputed {len(self.all_trial_indices)} trial indices")

    def _predict_batch_fixed_size(self, full_params: jnp.ndarray, verbosity: int = 1) -> jnp.ndarray:
        """Predict surfaces using fixed batch sizes to avoid recompilation.

        Args:
            full_params: Array of parameters with shape (n_surfaces, 3)
            verbosity: Verbosity level

        Returns:
            Log likelihood surfaces with shape (n_surfaces, n_mu1_bias, n_feat_diff)
        """
        n_surfaces = len(full_params)

        # Choose batch size based on total number of surfaces
        if n_surfaces <= 128:
            batch_size = 64
            predict_fn = self.predict_batch_64
        else:
            batch_size = 256
            predict_fn = self.predict_batch_256

        if verbosity > 2:
            print(f"        Using batch size {batch_size} for {n_surfaces} surfaces")

        all_surfaces = []

        for i in range(0, n_surfaces, batch_size):
            end_idx = min(i + batch_size, n_surfaces)
            actual_batch_size = end_idx - i

            if actual_batch_size == batch_size:
                # Full batch - use directly
                batch_params = full_params[i:end_idx]
                surfaces = predict_fn(batch_params)
            else:
                # Partial batch - pad to required size
                batch_params = full_params[i:end_idx]

                # Pad with the last parameter to reach required batch size
                pad_size = batch_size - actual_batch_size
                last_param = batch_params[-1:]
                padded_params = jnp.concatenate([
                    batch_params,
                    jnp.repeat(last_param, pad_size, axis=0)
                ], axis=0)

                # Predict and take only the non-padded results
                padded_surfaces = predict_fn(padded_params)
                surfaces = padded_surfaces[:actual_batch_size]

            all_surfaces.append(surfaces)

        return jnp.concatenate(all_surfaces, axis=0)

    # def _precompile_motor_noise_kernels(self):
    #     """Pre-compile motor noise kernels for common sd_motor values to avoid JIT recompilation."""
    #     print("Pre-compiling motor noise kernels...")
    #
    #     # Common sd_motor values from 0.1 to 50.0 (grid boundaries)
    #     common_sd_motors = [0.1, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
    #
    #     # Pre-compile kernels by calling the JIT function
    #     dummy_batch = jnp.ones((1, self.n_mu1_bias, self.n_feat_diff))
    #
    #     for sd_motor in common_sd_motors:
    #         # Trigger compilation for this sd_motor value
    #         _ = apply_motor_noise(dummy_batch, sd_motor)
    #
    #     print(f"Pre-compiled motor noise kernels for {len(common_sd_motors)} sd_motor values")
    #
    def _precompile_stage_motor_kernels(self, shared_grid: jnp.ndarray):
        """Precompile motor noise kernels for all sd_motor values in a given stage.
        
        Args:
            shared_grid: Array of shape (n_points, 2) with [sd_spat, sd_motor] values
            
        Returns:
            Tuple of (kernel_ffts, motor_kernel_indices) where:
            - kernel_ffts: Array of shape (n_unique_motors, n_mu1_bias) with precomputed FFT kernels
            - motor_kernel_indices: Array of shape (n_grid_points,) mapping each grid point to its kernel index
        """
        print("Pre-compiling motor noise kernels for current stage...")
        
        # Extract unique sd_motor values and get inverse indices
        unique_sd_motors, motor_kernel_indices = jnp.unique(shared_grid[:, 1], return_inverse=True)
        n_kernels = len(unique_sd_motors)
        
        # Create array of precomputed kernel FFTs
        kernel_ffts = jnp.zeros((n_kernels, self.n_mu1_bias), dtype=jnp.complex64)
        
        for i, sd_motor in enumerate(unique_sd_motors):
            kernel_fft = create_motor_noise_kernel_fft(float(sd_motor), self.n_mu1_bias)
            kernel_ffts = kernel_ffts.at[i].set(kernel_fft)
            
        print(f"Pre-compiled {n_kernels} motor noise kernels for stage")
        
        # Pre-compile the JIT function with a dummy batch 
        dummy_batch = jnp.ones((1, self.n_mu1_bias, self.n_feat_diff))
        dummy_kernel = kernel_ffts[0]
        _ = apply_motor_noise_with_precomputed_kernel(dummy_batch, dummy_kernel)
        
        print("Pre-compiled motor noise convolution function")
        
        return kernel_ffts, motor_kernel_indices

    def _setup_shared_param_evaluation(self):
        """Setup optimized shared parameter evaluation."""
        print("Setting up optimized shared parameter evaluation...")

        # Pre-compile with dummy data to avoid first-call overhead
        dummy_cond_params = jnp.array([[0, 10.0, 10.0], [1, 15.0, 15.0]])
        dummy_surface_data = {
            'surfaces': jnp.ones((2, self.n_mu1_bias, self.n_feat_diff)),
            'cond_params_with_idx': dummy_cond_params,
            'surface_indices': jnp.array([0, 1])
        }

        # Trigger compilation
        _ = self._optimize_conditions_jit(
            dummy_surface_data['surfaces'],
            dummy_surface_data['cond_params_with_idx'],
            dummy_surface_data['surface_indices']
        )

        print("Shared parameter evaluation setup complete")

    def _precompute_target_curves(self):
        """Precompute all target curves for all conditions using vmap (no loops)."""
        print("Precomputing target curves for all conditions...")

        # Prepare data for vectorized computation
        condition_dataframes = [self.condition_datasets[cond_name] for cond_name in self.condition_names]

        # Extract feat_diff and bias arrays
        all_feat_diff = [df[:, 0] for df in condition_dataframes]
        all_bias_values = [df[:, 1] for df in condition_dataframes]

        # Pad arrays to same length for vmap
        max_trials = max(len(vals) for vals in all_feat_diff)

        padded_feat_diff = jnp.stack([
            jnp.pad(vals, (0, max_trials - len(vals)), constant_values=vals[-1] if len(vals) > 0 else 0)
            for vals in all_feat_diff
        ])
        padded_bias_values = jnp.stack([
            jnp.pad(vals, (0, max_trials - len(vals)), constant_values=vals[-1] if len(vals) > 0 else 0)
            for vals in all_bias_values
        ])

        # Create trial counts for bias computation
        trial_counts = jnp.array([len(vals) for vals in all_feat_diff])

        # Vectorized bias curve computation using vmap
        vectorized_bias_compute = jax.vmap(compute_target_bias_curve_core, in_axes=(0, 0, 0))
        feat_indices, target_bias, weights = vectorized_bias_compute(padded_feat_diff, padded_bias_values, trial_counts)

        # Density curve computation over the REAL (unpadded) per-condition data.
        # Unlike the bias core, _compute_empirical_density_asymmetry_core has no
        # trial-count mask, so vmapping it over the rectangular padded arrays would
        # feed the repeated last-observation pad rows into the empirical density —
        # both as a mass clump at the last trial's coordinates and via the
        # std/quantile/n bandwidth terms. This is a one-time precompute (not in the
        # JIT objective), so we loop over the real condition arrays instead, exactly
        # as the balanced-CRPS block below does. See codex_audit.md #3.
        from shared.utils import _compute_empirical_density_asymmetry_core, compute_target_bias_rolling_curve_core
        feat_diff_grid = self.feat_diff_grid

        def density_curve_wrapper(feat_diff_vals, bias_vals, grid):
            """Wrapper to extract only asymmetry values from density computation."""
            distances, asymmetry_values = _compute_empirical_density_asymmetry_core(feat_diff_vals, bias_vals, grid, weights_sd=self.emp_density_weights_sd)
            return asymmetry_values

        density_curves = jnp.stack([
            density_curve_wrapper(condition_dataframes[i][:, 0],
                                  condition_dataframes[i][:, 1],
                                  feat_diff_grid)
            for i in range(len(self.condition_names))
        ])

        # Smoothed (rolling-mean) circular bias curve, for the smoothed_exp method:
        # same curve-vs-curve idea as the density asymmetry curve above, but for the
        # circular-mean bias itself instead of hard 8-degree bins.
        smoothed_bias_curves = jnp.stack([
            compute_target_bias_rolling_curve_core(condition_dataframes[i][:, 0],
                                                    condition_dataframes[i][:, 1],
                                                    feat_diff_grid,
                                                    weights_sd=self.emp_density_weights_sd)
            for i in range(len(self.condition_names))
        ])

        # Store unified results
        self.unified_feat_indices = feat_indices[0]  # Same for all conditions
        self.unified_target_bias = target_bias  # Shape: (n_conditions, n_bins)
        self.unified_bias_weights = weights  # Shape: (n_conditions, n_bins)
        self.unified_target_density = density_curves  # Shape: (n_conditions, n_feat_points)
        self.unified_target_bias_curve = smoothed_bias_curves  # Shape: (n_conditions, n_feat_points)

        print(f"Target curves precomputed for {len(self.condition_names)} conditions (vmap)")
        print(f"Bias curves shape: {self.unified_target_bias.shape}")
        print(f"Bias weights shape: {self.unified_bias_weights.shape}")
        print(f"Density curves shape: {self.unified_target_density.shape}")
        print(f"Smoothed bias curves shape: {self.unified_target_bias_curve.shape}")

        # Precompute empirical bias distributions shared by the balanced_crps and
        # bias_weighted_crps methods (unified_target_d for both; unified_fd_weights
        # for balanced_crps, unified_bias_fd_weights for bias_weighted_crps).
        # Q[c, j, k] = Gaussian-weighted empirical probability of bias bin k at feat_diff grid point j.
        # target_d[c, j, k] = sum_l D_circ[k,l] * Q[c,j,l]  — expected circular distance from bin k
        #                       to the empirical distribution.  Precomputed so the JIT branch is cheap.
        print("Precomputing empirical bias distributions for balanced_crps/bias_weighted_crps...")
        D_np = np.array(self.D_circ_matrix)        # (n_bias, n_bias)
        fd_grid_np = np.array(self.feat_diff_grid)  # (n_feat,)
        bias_low = config.mu1_bias_range[0]
        bias_step_val = config.mu1_bias_step
        n_feat_pts = len(fd_grid_np)
        n_bias_pts = self.n_mu1_bias

        target_d_list = []
        fd_weights_list = []
        fd_bias_weights_list = []
        for cond_name in self.condition_names:
            dataset_np = np.array(self.condition_datasets[cond_name])
            fd_vals = dataset_np[:, 0]
            bias_vals = dataset_np[:, 1]

            # Gaussian weights over feat_diff distance: shape (n_feat, n_trials)
            gw = np.exp(-0.5 * ((fd_vals[None, :] - fd_grid_np[:, None]) / self.emp_density_weights_sd) ** 2)
            w_c = gw.sum(axis=1)  # (n_feat,) — effective support weight per fd point

            # Weighted histogram: Q_c[j, k] = sum_i gw[j,i] * I(bias_bin[i]==k) / w_c[j]
            # Circular binning: wrap rather than clip (see _prepare_all_condition_data).
            bias_bin = np.mod(
                np.round((bias_vals - bias_low) / bias_step_val).astype(int), n_bias_pts
            )
            one_hot = np.zeros((len(bias_vals), n_bias_pts))
            one_hot[np.arange(len(bias_vals)), bias_bin] = 1.0
            Q_c = (gw @ one_hot) / np.maximum(w_c[:, None], 1e-10)  # (n_feat, n_bias)

            # target_d[j, k] = (Q_c @ D_circ)[j, k]  (D_circ is symmetric)
            target_d_list.append(Q_c @ D_np)   # (n_feat, n_bias)

            # Loss weights: binary mask — 1 where there is data support, 0 elsewhere.
            # Threshold at 1% of the median weight so edge fd bins with marginal
            # support are excluded, but weighting is otherwise uniform (not trial-count-proportional).
            support_threshold = np.median(w_c) * 0.01
            support_mask = (w_c > support_threshold).astype(np.float32)
            fd_weights_list.append(support_mask)

            # Bias-weighted CRPS: weight each fd point by the squared smoothed
            # signed mean bias. Squaring discards the sign, so feat_diff ranges
            # with large-magnitude observed mean bias contribute more to the loss.
            mean_bias = (gw @ bias_vals) / np.maximum(w_c, 1e-10)   # (n_feat,)
            fd_bias_weights_list.append(mean_bias ** 2 * support_mask)

        self.unified_target_d = jnp.array(np.stack(target_d_list))         # (n_cond, n_feat, n_bias)
        self.unified_fd_weights = jnp.array(np.stack(fd_weights_list))      # (n_cond, n_feat) — binary mask
        self.unified_bias_fd_weights = jnp.array(np.stack(fd_bias_weights_list))  # (n_cond, n_feat)
        print(f"  target_d shape: {self.unified_target_d.shape}, "
              f"fd_weights shape: {self.unified_fd_weights.shape}, "
              f"mean support: {self.unified_fd_weights.mean():.2f}")

    def _optimize_all_conditions_hierarchical_jit(self, surfaces: jnp.ndarray,
                                                  cond_params_with_idx: jnp.ndarray,
                                                  surface_indices: jnp.ndarray,
                                                  fitting_method: str = "likelihood") -> jnp.ndarray:
        """JIT-able function to optimize all conditions using different fitting methods.

        Args:
            surfaces: Precomputed surfaces, shape (n_unique_surfaces, n_mu1_bias, n_feat_diff)
            cond_params_with_idx: Parameter combinations, shape (n_total_combinations, 3)
            surface_indices: Surface index for each parameter combination, shape (n_total_combinations,)
            fitting_method: One of "likelihood", "expectation", "smoothed_exp", "density"

        Returns:
            Array of shape (n_conditions, 3) with [sd_feat1, sd_feat2, loss] for each condition
        """
        # Reshape surfaces for flat indexing: (n_unique_surfaces, n_bias * n_feat)
        n_unique_surfaces, n_bias, n_feat = surfaces.shape
        surfaces_flat = surfaces.reshape(n_unique_surfaces, -1)

        # Extract condition indices
        condition_indices = cond_params_with_idx[:, 0].astype(int)

        # Compute losses based on fitting method
        if fitting_method == "likelihood":
            # SINGLE MEGA EXTRACTION: All surfaces × All trials
            all_log_probs = surfaces_flat[:, self.all_trial_indices]  # Shape: (n_unique_surfaces, total_valid_trials)

            # Use segment_sum to compute total log likelihood for each surface × condition combination
            def sum_by_condition(surface_log_probs):
                """Sum log likelihoods per condition using segment indices."""
                return jax.ops.segment_sum(surface_log_probs, self.trial_to_condition, num_segments=self.n_conditions)

            # Apply to all surfaces at once: (n_unique_surfaces, n_conditions)
            surface_condition_log_likelihoods = jax.vmap(sum_by_condition)(all_log_probs)

            # Get log likelihood for each parameter combination
            param_log_likelihoods = surface_condition_log_likelihoods[surface_indices, condition_indices]
            losses = -param_log_likelihoods

        elif fitting_method == "expectation":
            # Use precomputed target bias curves (no function calls inside JIT)
            # Generate bias curves for ALL surfaces at once: shape (n_surfaces, n_bins)
            all_predicted_bias = _generate_nn_bias_curve_batch(surfaces, self.unified_feat_indices)

            # Get predicted and target bias for each parameter combination
            predicted_bias_per_combo = all_predicted_bias[surface_indices]  # Shape: (n_combos, n_bins)
            target_bias_per_combo = self.unified_target_bias[condition_indices]  # Shape: (n_combos, n_bins)
            weights_per_combo = self.unified_bias_weights[condition_indices]  # Shape: (n_combos, n_bins)

            # Compute losses using unified function with weights (MSE for expectation, angular=True)
            losses = _compute_curve_losses(predicted_bias_per_combo, target_bias_per_combo,
                                           loss_type="mse", is_angular=True, weights=weights_per_combo)

        elif fitting_method == "smoothed_exp":
            # Curve-vs-curve version of "expectation": same MSE-only, angular loss, but
            # the target is a smoothed (Gaussian rolling-mean) circular bias curve over
            # the full feat_diff grid instead of hard 8-degree bins, and the model curve
            # is evaluated at every grid point instead of only the bin-center indices.
            # No correlation term and no support weighting, matching how "density" fits.
            all_predicted_bias_curve = _generate_nn_bias_curve_batch(surfaces, jnp.arange(n_feat))

            predicted_bias_curve_per_combo = all_predicted_bias_curve[surface_indices]  # Shape: (n_combos, n_feat_points)
            target_bias_curve_per_combo = self.unified_target_bias_curve[condition_indices]  # Shape: (n_combos, n_feat_points)

            losses = _compute_curve_losses(predicted_bias_curve_per_combo, target_bias_curve_per_combo,
                                           loss_type="mse", is_angular=True)

        elif fitting_method == "density":
            # Use precomputed target density curves (no function calls inside JIT)
            # Generate density asymmetry for ALL surfaces at once: shape (n_surfaces, n_feat_points)
            all_predicted_asymmetry = generate_nn_density_asymmetry_batch(surfaces, weights_sd=self.emp_density_weights_sd, smoothing_sigma=self.density_smoothing_sigma)

            # Get predicted and target asymmetry for each parameter combination
            predicted_asymmetry_per_combo = all_predicted_asymmetry[surface_indices]  # Shape: (n_combos, n_feat_points)
            target_asymmetry_per_combo = self.unified_target_density[
                condition_indices]  # Shape: (n_combos, n_feat_points)

            # Compute losses using unified function (combined for density, angular=False)
            losses = _compute_curve_losses(predicted_asymmetry_per_combo, target_asymmetry_per_combo,
                                           loss_type="combined", is_angular=False, corr_weight=self.corr_weight)

        elif fitting_method == "crps":
            # Energy score CRPS with circular distance.
            # crps_surface[u, k, j] = E_X[d_circ(X, mu1_grid[k]) | feat_diff[j]] - 0.5 * E_{X,X'}[d_circ | feat_diff[j]]
            # Indexed with all_trial_indices exactly like likelihood.
            log_prob_norm = surfaces - jax.scipy.special.logsumexp(surfaces, axis=1, keepdims=True)
            prob_surfaces = jnp.exp(log_prob_norm)  # (n_unique, n_mu1, n_feat), sums to 1 per column

            # Reshape for batch matmul: (n_unique * n_feat, n_mu1)
            prob_flat = prob_surfaces.transpose(0, 2, 1).reshape(-1, n_bias)
            Dp = prob_flat @ self.D_circ_matrix          # (n_unique * n_feat, n_mu1)

            # E2: quadratic form prob^T D_circ prob per (surface, feat_diff)
            E2 = (prob_flat * Dp).sum(axis=1).reshape(n_unique_surfaces, n_feat)

            # E1 surface: (D_circ @ prob)^T at each (k, j) = Dp reshaped back
            E1 = Dp.reshape(n_unique_surfaces, n_feat, n_bias).transpose(0, 2, 1)  # (n_unique, n_mu1, n_feat)

            crps_surface = E1 - 0.5 * E2[:, None, :]    # (n_unique, n_mu1, n_feat)
            crps_flat = crps_surface.reshape(n_unique_surfaces, -1)

            all_crps_vals = crps_flat[:, self.all_trial_indices]  # (n_unique, total_trials)

            def sum_crps_by_condition(surface_crps):
                return jax.ops.segment_sum(surface_crps, self.trial_to_condition, num_segments=self.n_conditions)

            surface_condition_crps = jax.vmap(sum_crps_by_condition)(all_crps_vals)
            param_crps = surface_condition_crps[surface_indices, condition_indices]
            losses = param_crps

        elif fitting_method == "balanced_crps":
            # Balanced energy score: equal weight to every feat_diff grid point regardless of
            # how many trials fall there.  At each fd point j we compare the model distribution
            # p[u,:,j] to the Gaussian-weighted empirical distribution Q[c,j,:] (precomputed).
            #
            # loss[u,c] = sum_j w[c,j] * (2 * p[u,:,j] @ target_d[c,j] - E2_model[u,j])
            #             / sum_j w[c,j]
            # where target_d[c,j] = D_circ @ Q[c,j]  and  E2_model[u,j] = p^T D_circ p.

            log_prob_norm = surfaces - jax.scipy.special.logsumexp(surfaces, axis=1, keepdims=True)
            prob_surfaces = jnp.exp(log_prob_norm)  # (n_unique, n_bias, n_feat)

            # E2_model[u, j]: self-energy of the model distribution at each fd point
            prob_fk = prob_surfaces.transpose(0, 2, 1).reshape(-1, n_bias)   # (n_unique*n_feat, n_bias)
            Dp = prob_fk @ self.D_circ_matrix                                  # (n_unique*n_feat, n_bias)
            E2 = (prob_fk * Dp).sum(axis=1).reshape(n_unique_surfaces, n_feat) # (n_unique, n_feat)

            # Cross term: sum_{j,k} w[c,j] * p[u,k,j] * target_d[c,j,k]
            # T_w[c,j,k] = w[c,j] * target_d[c,j,k]
            T_w = self.unified_target_d * self.unified_fd_weights[:, :, None]  # (n_cond, n_feat, n_bias)
            # einsum 'ukj,cjk->uc' — contract over bias (k) and feat (j)
            cross_uc = jnp.einsum('ukj,cjk->uc', prob_surfaces, T_w)           # (n_unique, n_cond)

            # Weighted E2: sum_j w[c,j] * E2[u,j]
            E2_uc = E2 @ self.unified_fd_weights.T   # (n_unique, n_cond)

            norm_c = self.unified_fd_weights.sum(axis=1)                        # (n_cond,)
            loss_uc = (2.0 * cross_uc - E2_uc) / norm_c[None, :]               # (n_unique, n_cond)

            losses = loss_uc[surface_indices, condition_indices]

        elif fitting_method == "bias_weighted_crps":
            # Same energy-score formula as balanced_crps but fd points are weighted
            # by squared smoothed mean bias instead of a binary support mask, so
            # feat_diff ranges with larger-magnitude observed bias contribute more.

            log_prob_norm = surfaces - jax.scipy.special.logsumexp(surfaces, axis=1, keepdims=True)
            prob_surfaces = jnp.exp(log_prob_norm)  # (n_unique, n_bias, n_feat)

            prob_fk = prob_surfaces.transpose(0, 2, 1).reshape(-1, n_bias)
            Dp = prob_fk @ self.D_circ_matrix
            E2 = (prob_fk * Dp).sum(axis=1).reshape(n_unique_surfaces, n_feat)

            T_w = self.unified_target_d * self.unified_bias_fd_weights[:, :, None]
            cross_uc = jnp.einsum('ukj,cjk->uc', prob_surfaces, T_w)

            E2_uc = E2 @ self.unified_bias_fd_weights.T

            norm_c = self.unified_bias_fd_weights.sum(axis=1)
            # Avoid division by zero for conditions where all bias weights are 0
            loss_uc = (2.0 * cross_uc - E2_uc) / jnp.maximum(norm_c[None, :], 1e-10)

            losses = loss_uc[surface_indices, condition_indices]

        else:
            raise ValueError(
                f"Unknown fitting method: {fitting_method}. Must be one of "
                f"'likelihood', 'expectation', 'smoothed_exp', 'density', 'crps', 'balanced_crps', 'bias_weighted_crps'")

        # Find best parameter for each condition
        min_losses = jax.ops.segment_min(losses, condition_indices, num_segments=self.n_conditions)

        # Find indices of minimum values using vectorized operations
        # Create condition masks and find argmin for each condition simultaneously
        condition_masks = condition_indices[:, None] == jnp.arange(self.n_conditions)[None,
                                                        :]  # Shape: (n_combos, n_conditions)
        masked_losses = jnp.where(condition_masks, losses[:, None], jnp.inf)  # Shape: (n_combos, n_conditions)
        min_indices = jnp.argmin(masked_losses, axis=0)  # Shape: (n_conditions,)

        best_params = cond_params_with_idx[min_indices, 1:]  # Remove condition index

        # Return shape: (n_conditions, 3) = [sd_feat1, sd_feat2, loss]
        return jnp.column_stack([best_params, min_losses])

    def fit_hierarchical_grid(self, shared_grid_size: int = 20, feat_grid_size: int = 10,
                              min_grid_step: float = 1.0, zoom_factor: float = 0.5,
                              fitting_method: str = "likelihood", verbosity: int = 2,
                              stop_after_first_stage: bool = False,
                              sd_motor_max: float = 50.0) -> Dict:
        """Run hierarchical grid-based multi-condition optimization.

        Args:
            shared_grid_size: Size of initial shared parameter grid
            feat_grid_size: Size of initial condition-specific parameter grid
            min_grid_step: Minimum grid step size (stop criterion)
            zoom_factor: Factor to reduce grid size when zooming in
            fitting_method: One of "likelihood", "expectation", "smoothed_exp", "density"
            verbosity: Verbosity level

        Returns:
            Optimization results
        """
        print(f"=== HIERARCHICAL GRID-BASED MULTI-CONDITION OPTIMIZATION ===")
        if self.skip_motor_noise:
            print(f"Initial shared parameter grid (motor noise SKIPPED): {shared_grid_size} points")
        else:
            print(f"Initial shared parameter grid: {shared_grid_size} (sd_spat) × motor axis sized from empirical cap (see below)")
        print(
            f"Initial feature grid, specific by condition: {feat_grid_size}×{feat_grid_size} = {feat_grid_size ** 2} per condition")
        print(f"Minimum grid step: {min_grid_step} degrees")
        print(f"Zoom factor: {zoom_factor}")
        print(f"Total conditions: {self.n_conditions}")

        start_time = time.time()
        total_nn_evaluations = 0

        # Initialize bounds for shared parameters
        sd_spat_range = (config.param_grid_low, config.param_range_high)
        # sd_motor is one component of the total response error, so it can never
        # exceed the empirical error SD; sd_motor_max carries that data-informed
        # cap (min-over-conditions error SD). Capping the range and sizing the
        # motor axis to ~2*min_grid_step resolution shrinks the grid from
        # shared_grid_size**2 toward shared_grid_size*motor_grid_size.
        sd_motor_low = 0.1
        if self.skip_motor_noise:
            # Motor axis unused (sd_motor fixed to 0). Keep the legacy phantom range
            # and full width so staging/termination is byte-identical to before.
            sd_motor_hi = 50.0
            motor_grid_size = shared_grid_size
        else:
            # sd_motor is one component of the total response error, so it can never
            # exceed the empirical error SD; sd_motor_max carries that data-informed
            # cap. Capping the range and sizing the motor axis to ~2*min_grid_step
            # resolution shrinks the grid from shared_grid_size**2 toward
            # shared_grid_size*motor_grid_size.
            sd_motor_hi = float(min(sd_motor_max, 50.0))
            motor_span = max(sd_motor_hi - sd_motor_low, 0.0)
            motor_grid_size = int(min(shared_grid_size,
                                      max(4, int(np.ceil(motor_span / (2.0 * min_grid_step))) + 1)))
            print(f"Motor-noise range capped at sd_motor <= {sd_motor_hi:.1f} "
                  f"(empirical error SD); motor axis: {motor_grid_size} points")
        sd_motor_range = (sd_motor_low, sd_motor_hi)
        # The first feature pass must cover the whole supported domain, otherwise the
        # search can never reach parameter values it never looks at. Centre on the domain
        # midpoint and, when the fixed schedule below is too fine to span it with
        # feat_grid_size points, prepend the exact spanning step. Previously the centre
        # used param_range_low (the Level-1 floor, 10) and the prepended step was
        # int()-truncated, so the first pass covered [10, 200] at feat_grid_size=20 and
        # [10.5, 199.5] at 10 — never reaching param_grid_low = 5 at all.
        center = (config.param_grid_low + config.param_range_high) / 2
        feat_step_schedule = [10, 6, 4, 2, 1] # Decreasing step sizes
        full_span_step = (config.param_range_high - config.param_grid_low) / (feat_grid_size - 1)
        if full_span_step > feat_step_schedule[0]:
            feat_step_schedule = [full_span_step] + feat_step_schedule
        feat_step_schedule = jnp.array(feat_step_schedule, dtype=jnp.float32)
        print(f"feat_step_schedule: {feat_step_schedule}")

        best_overall_loss = float('inf')
        best_overall_result = None
        stage_times = []

        # Stage 1: Coarse shared parameter search
        stage = 1

        while True:
            stage_start_time = time.time()

            if verbosity > 0:
                print(f"\n--- STAGE {stage}: SHARED PARAMETER SEARCH ---")
                if self.skip_motor_noise:
                    print(f"Search ranges: sd_spat=[{sd_spat_range[0]:.1f}, {sd_spat_range[1]:.1f}], sd_motor=0.0 (fixed)")
                    print(f"Grid size (motor noise SKIPPED): {shared_grid_size} points")
                else:
                    print(f"Search ranges: sd_spat=[{sd_spat_range[0]:.1f}, {sd_spat_range[1]:.1f}], "
                          f"sd_motor=[{sd_motor_range[0]:.1f}, {sd_motor_range[1]:.1f}]")
                    print(f"Grid size: {shared_grid_size}×{motor_grid_size} = {shared_grid_size * motor_grid_size} points")

            # Create shared parameter grid for this stage
            shared_grid = create_shared_parameter_grid(shared_grid_size, sd_spat_range, sd_motor_range, self.skip_motor_noise, motor_grid_size=motor_grid_size)
            
            # Precompile motor noise kernels for this stage (only if motor noise enabled)
            if not self.skip_motor_noise:
                stage_kernel_ffts, motor_kernel_indices = self._precompile_stage_motor_kernels(shared_grid)

            init_best_params = jnp.full((self.n_conditions, 3), jnp.array([center, center, jnp.inf]))

            # Define scan function for shared parameter evaluation
            def evaluate_shared_params(carry, params_with_kernel_idx):
                """Evaluate shared parameters for all conditions at one grid point."""
                if self.skip_motor_noise:
                    sd_spat, sd_motor = params_with_kernel_idx
                    kernel_fft = None
                else:
                    sd_spat, sd_motor, kernel_idx = params_with_kernel_idx
                    # Get the precompiled kernel for this grid point
                    kernel_fft = stage_kernel_ffts[kernel_idx.astype(int)]
                
                # jax.debug.print("    Processing shared params: sd_spat={:.1f}, sd_motor={:.1f}", sd_spat, sd_motor)
                # start_param_evals = time.time()

                def scan_body_fn(current_best_params, current_step):
                    """Refine condition-specific parameters for one step size."""
                    # Each scan iteration corresponds to one refinement step
                    # jax.debug.print("      Refinement step: step_size={}", current_step)

                    cond_data = jnp.column_stack([
                        jnp.arange(self.n_conditions),
                        current_best_params[:, 0],  # sd_feat1
                        current_best_params[:, 1]  # sd_feat2
                    ])

                    all_grids = jax.vmap(create_condition_grid, in_axes=(0, None, None))(cond_data, feat_grid_size, current_step)
                    param_grid = all_grids.reshape(-1, 3)

                    nn_params = jnp.column_stack([
                        param_grid[:, 1],  # sd_feat1
                        param_grid[:, 2],  # sd_feat2
                        jnp.full(len(param_grid), sd_spat)  # sd_spat
                    ])

                    log_surfaces_batch = predict_nn(self, nn_params)
                    if self.skip_motor_noise:
                        log_surfaces_batch = log_surfaces_batch
                    else:
                        log_surfaces_batch = apply_motor_noise_with_precomputed_kernel(log_surfaces_batch, kernel_fft)

                    results_array = self._optimize_conditions_jit(
                        log_surfaces_batch,
                        param_grid,
                        jnp.arange(len(param_grid)),
                        fitting_method
                    )

                    improved_params = jnp.where(
                        results_array[:, 2:3] < current_best_params[:, 2:3],
                        results_array,
                        current_best_params
                    )

                    # Print updated parameters after each step
                    # for i in range(4):
                    #     jax.debug.print("        {}: feat1={:.2f}, feat2={:.2f}, loss={:.6f}",
                    #                    self.condition_names[i], improved_params[i, 0], improved_params[i, 1], improved_params[i, 2])

                    
                    #jax.lax.fori_loop(0, self.n_conditions, lambda i, _: print_condition_params(i), None)

                    return improved_params, None

                # Run the refinement loop using lax.scan
                final_best_params, _ = jax.lax.scan(scan_body_fn, init_best_params, feat_step_schedule)
                # Calculate total loss for this shared parameter combination
                total_loss = jnp.sum(final_best_params[:, 2])
                

                # Create result for this shared parameter combination
                current_result = jnp.hstack([jnp.tile(jnp.array([sd_spat, sd_motor, total_loss]), (final_best_params.shape[0], 1)),
                                                     final_best_params])

                updated_best = jnp.where(
                    current_result[:, 2:3] < carry[:, 2:3],
                    current_result,
                    carry
                )

                # jax.debug.print("  Final loss: {}, computed in {}", total_loss, time.time()-start_param_evals)
                return updated_best, current_result
            
            # Run scan over all shared parameter combinations
            init_carry = jnp.tile(jnp.array([0.0, 0.0, jnp.inf]), (self.n_conditions, 1))
            init_carry = jnp.hstack([
                init_carry,
                jnp.zeros((self.n_conditions, 3))])

            # Create scan inputs
            if self.skip_motor_noise:
                # Only sd_spat and sd_motor (which is 0)
                scan_inputs = shared_grid  # Shape: (N, 2)
            else:
                # Include kernel indices for motor noise
                grid_indices = jnp.arange(len(shared_grid))
                kernel_indices_expanded = motor_kernel_indices.reshape(-1, 1)  # Convert to column vector
                scan_inputs = jnp.hstack([shared_grid, kernel_indices_expanded])  # Shape: (N, 3)
            # print(jax.make_jaxpr(evaluate_shared_params)(init_carry, scan_inputs[0,:]))
            # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True, create_perfetto_trace=True):
            if verbosity > 0:
                print('Fitting the model...')

            best_shared_result, all_results = jax.lax.scan(
                evaluate_shared_params,
                init_carry,
                scan_inputs
            )
            

            # Extract best shared parameters 
            sd_spat = best_shared_result[0,0]
            sd_motor = best_shared_result[0,1]
            stage_best_loss = best_shared_result[0,2]
            stage_best_result = best_shared_result.copy()
            # stage_best_result['condition_results'] = condition_results.copy()

            # jax.lax.scan dispatches asynchronously; without forcing completion
            # here, stage_time below would only measure dispatch latency rather
            # than actual scan execution. This also fixes timing in callers (e.g.
            # fit_model_to_data.py's `duration`/"done in Xs"), since they wrap this
            # synchronous function call and can't return early once it blocks here.
            jax.block_until_ready(best_shared_result)

            stage_time = time.time() - stage_start_time
            stage_times.append(stage_time)

            if verbosity > 0:
                print(f"\nStage {stage} completed in {stage_time:.1f}s")
                print(f"Best shared params: sd_spat={sd_spat:.1f}, "
                      f"sd_motor={sd_motor:.1f}, loss={stage_best_loss:.3f}")
                nn_evals_this_stage = len(shared_grid) * feat_grid_size ** 2
                total_nn_evaluations += nn_evals_this_stage
                print(f"NN evaluations this stage: {nn_evals_this_stage:,}")

            # Boundary-hit detection
            spat_step = (sd_spat_range[1] - sd_spat_range[0]) / (shared_grid_size - 1)
            if float(sd_spat) <= sd_spat_range[0] + spat_step:
                print(f"  BOUNDARY: sd_spat={float(sd_spat):.1f} near lower bound {sd_spat_range[0]}")
            if float(sd_spat) >= config.param_range_high - spat_step:
                print(f"  BOUNDARY: sd_spat={float(sd_spat):.1f} near upper bound {config.param_range_high}")
            feat1_best = best_shared_result[:, 3]
            feat2_best = best_shared_result[:, 4]
            current_feat_step = float(feat_step_schedule[-1])  # finest step reached this stage
            for i in range(self.n_conditions):
                cname = self.condition_names[i]
                f1, f2 = float(feat1_best[i]), float(feat2_best[i])
                feat_low = config.param_grid_low
                if f1 <= feat_low + current_feat_step:
                    print(f"  BOUNDARY: {cname} sd_feat1={f1:.1f} near lower bound {feat_low}")
                if f1 >= config.param_range_high - current_feat_step:
                    print(f"  BOUNDARY: {cname} sd_feat1={f1:.1f} near upper bound {config.param_range_high}")
                if f2 <= feat_low + current_feat_step:
                    print(f"  BOUNDARY: {cname} sd_feat2={f2:.1f} near lower bound {feat_low}")
                if f2 >= config.param_range_high - current_feat_step:
                    print(f"  BOUNDARY: {cname} sd_feat2={f2:.1f} near upper bound {config.param_range_high}")

            # Update overall best
            if stage_best_loss < best_overall_loss:
                best_overall_loss = stage_best_loss
                best_overall_result = stage_best_result.copy()
                # best_overall_result['stage'] = stage

            # Check if we should zoom in on shared parameters
            zoom_check_start_time = time.time()
            # Stop once every dimension actually being searched is resolved to
            # min_grid_step. Previously this compared sd_spat against a FICTIONAL motor
            # schedule -- the uncapped [0.1, 50] range, computed even when motor noise was
            # skipped and no motor axis existed -- and stopped as soon as EITHER reached
            # tolerance. Since that phantom value fell below tolerance first, sd_spat was
            # left far coarser than min_grid_step implies (2.5 deg at the production
            # shared_grid_size=40, not 1.0). Comparing the real spacings, and requiring all
            # of them, also removes the original motivation for the phantom schedule: a
            # small empirical motor cap can no longer end the search early, because
            # sd_spat must reach tolerance on its own regardless.
            current_spat_step = (sd_spat_range[1] - sd_spat_range[0]) / (shared_grid_size - 1)
            current_motor_step = ((sd_motor_range[1] - sd_motor_range[0]) / (motor_grid_size - 1)
                                  if not self.skip_motor_noise else 0.0)
            active_steps = ([current_spat_step] if self.skip_motor_noise
                            else [current_spat_step, current_motor_step])

            if verbosity > 0:
                motor_desc = "skipped" if self.skip_motor_noise else f"{current_motor_step:.2f}"
                print(f"Current grid steps: spat={current_spat_step:.2f}, motor={motor_desc}")

            if max(active_steps) <= min_grid_step:
                if verbosity > 0:
                    print(f"Optimization stopped: minimum grid step {min_grid_step} reached")
                break
                
            if stop_after_first_stage and stage >= 1:
                if verbosity > 0:
                    print(f"Optimization stopped: stop_after_first_stage=True")
                break

            # Zoom in around the best point
            best_sd_spat = sd_spat
            best_sd_motor = sd_motor

            # Calculate zoom ranges
            current_spat_range = sd_spat_range[1] - sd_spat_range[0]
            current_motor_range = sd_motor_range[1] - sd_motor_range[0]

            new_spat_half_range = current_spat_range * zoom_factor / 2
            new_motor_half_range = current_motor_range * zoom_factor / 2

            # Update ranges, ensuring they stay within bounds
            sd_spat_range = (
                max(config.param_grid_low, best_sd_spat - new_spat_half_range),
                min(config.param_range_high, best_sd_spat + new_spat_half_range)
            )
            sd_motor_range = (
                max(sd_motor_low, best_sd_motor - new_motor_half_range),
                min(sd_motor_hi, best_sd_motor + new_motor_half_range)
            )

            zoom_check_time = time.time() - zoom_check_start_time

            if verbosity > 0:
                print(f"Zoom calculation completed in {zoom_check_time:.4f}s")
                print(f"Next stage ranges: spat=[{sd_spat_range[0]:.1f}, {sd_spat_range[1]:.1f}], "
                      f"motor=[{sd_motor_range[0]:.1f}, {sd_motor_range[1]:.1f}]")

            stage += 1

        total_time = time.time() - start_time

        # Prepare final result
        # Convert best result back to expected format
        condition_results = {}
        for i, condition_name in enumerate(self.condition_names):
            best_params = best_overall_result[:,3:6]
            condition_results[condition_name] = {
                'condition_name': condition_name,
                'sd_feat1': float(best_params[i, 0]),
                'sd_feat2': float(best_params[i, 1]),
                'loss': float(best_params[i, 2]),
                'surface_idx': None
            }

        final_result = {
            'best_loss': best_overall_loss,
            'shared_params': {
                'sd_spat': best_overall_result[0,0],
                'sd_motor': best_overall_result[0,1]
            },
            'condition_results': condition_results,
            'total_time': total_time,
            'stage_times': stage_times,
            'initial_grid_size': shared_grid_size,
            'initial_cond_grid': feat_grid_size,
            'min_grid_step': min_grid_step,
            'stages_completed': stage,
            'n_conditions': self.n_conditions,
            'condition_names': self.condition_names,
            'total_nn_evaluations': total_nn_evaluations
        }

        if verbosity > 0:
            print(f"\n=== HIERARCHICAL OPTIMIZATION COMPLETED ===")
            print(f"Total time: {total_time:.1f}s")
            print(f"Total NN evaluations: {total_nn_evaluations:,}")
            print(f"Stages completed: {stage}")
            print(f"Best loss: {best_overall_loss:.3f}")
            print(f"Best shared params: sd_spat={final_result['shared_params']['sd_spat']:.1f}, "
                  f"sd_motor={final_result['shared_params']['sd_motor']:.1f}")
            print(f"Condition-specific results:")
            for condition_name, result in final_result['condition_results'].items():
                print(f"  {condition_name}: sd_feat1={result['sd_feat1']:.1f}, "
                      f"sd_feat2={result['sd_feat2']:.1f}, loss={result['loss']:.3f}")

        return final_result

@partial(jax.jit, static_argnums=1)
def centered_grid(center, num_points, step_size, lower, upper):
    """Create a centered 1D grid clamped to bounds.

    Args:
        center: Center point for the grid.
        num_points: Number of grid points.
        step_size: Spacing between points.
        lower: Lower bound for clamping.
        upper: Upper bound for clamping.

    Returns:
        1D JAX array of grid values.
    """
    half = step_size * (num_points - 1) / 2
    shift = jnp.maximum(lower - (center - half), 0.0) + jnp.minimum(upper - (center + half), 0.0)
    shifted = center + shift + jnp.linspace(-half, half, num_points)
    # When the window is wider than the domain both shift terms are non-zero and cancel,
    # leaving the grid unclamped and out of bounds (e.g. num_points=21, step=10 gave
    # [2.5, 202.5]). In that case the only sensible grid spans the domain exactly.
    return jnp.where(2 * half >= (upper - lower),
                     jnp.linspace(lower, upper, num_points), shifted)


def create_condition_grid(cond_idx_and_params, feat_grid_size, current_step):
    """Create a condition-specific parameter grid for feature stds.

    Args:
        cond_idx_and_params: Tuple of (condition index, sd_feat1, sd_feat2).
        feat_grid_size: Number of points per feature dimension.
        current_step: Step size for the grid.

    Returns:
        Array of shape (feat_grid_size^2, 3) with condition index and params.
    """
    cond_idx, best_feat1, best_feat2 = cond_idx_and_params

    # Clamp to the grid the surfaces actually cover, matching the coarse stage's
    # feat_low. Using param_range_low here floored refined sd_feat at 10 while the
    # coarse sweep could already reach 5, so low-noise fits railed at 10.
    feat1_vals = centered_grid(best_feat1, feat_grid_size, current_step,
                               config.param_grid_low, config.param_range_high)
    feat2_vals = centered_grid(best_feat2, feat_grid_size, current_step,
                               config.param_grid_low, config.param_range_high)

    feat1_grid, feat2_grid = jnp.meshgrid(feat1_vals, feat2_vals, indexing='ij')
    n_points = feat_grid_size * feat_grid_size

    return jnp.stack([
        jnp.full(n_points, cond_idx),
        feat1_grid.flatten(),
        feat2_grid.flatten()
    ], axis=-1)

import functools
@functools.partial(jax.jit, static_argnums=(0,))
def predict_nn(self, nn_params):
    """Predict log surfaces from NN parameters with JIT caching."""
    return self._predict_batch_fixed_size(nn_params, 0)
