#!/usr/bin/env python3
"""
Create Unified Subject Plots

This script creates unified plots for each subject, combining all conditions
in a single plot with parameter information displayed.

For each subject, it creates a plot showing:
- All conditions (noise levels) in subplots
- Different optimizers (likelihood, expectation, density, CRPS) as different colors
- Parameter values displayed for each optimizer
- Both bias curves and density asymmetry
"""

import pandas as pd
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
import jax

warnings.filterwarnings('ignore')

from model_fit_to_data.grid_based_multi_condition_optimizer_jax_loops import (
    GridBasedMultiConditionOptimizer,
    _compute_curve_losses,
)
from shared.config import config
from shared import seed_manager
from shared.utils import resolve_input_path, resolve_results_path

# Initialize seed manager
seed = seed_manager.SeedManager(quiet=True)

# Global motor noise kernel cache to avoid recomputing kernels across subjects
global_motor_kernel_cache = {}

RESULTS_DIR = "results"
MODEL_FIT_RESULTS_PREFIX = "model_fit_to_data_results_v2"
CHECKPOINT_PREFIX = "neural_net_checkpoints"
CHECKPOINT_EPOCH = 1500

# Colorblind-friendly method palette. Keep likelihood and CRPS visually separated
# because they are often compared directly in the summary plots.
OPTIMIZER_COLORS = {
    'likelihood': '#0072B2',   # blue
    'crps': '#E69F00',         # orange
    'expectation': '#CC79A7',  # reddish purple
    'density': '#009E73',      # green
    'balanced_crps': '#D55E00',  # vermillion
}
OPTIMIZER_LABELS = {
    'likelihood': 'Likelihood',
    'crps': 'CRPS',
    'expectation': 'Expectation',
    'density': 'Density',
    'balanced_crps': 'Balanced CRPS',
}
LOSS_EVALUATION_METHODS = ['density', 'expectation', 'likelihood', 'crps', 'balanced_crps']


def _angle_display_scale(circ_space: int = 360) -> float:
    """Scale angular model-space values back to the data circular space."""
    return circ_space / (2 * config.feat_diff_range[1])


def compute_empirical_sd_curve(feat_diff_values, bias_values, n_bins=18):
    """Compute empirical standard deviation curve binned by feature difference - fully vectorized."""
    # Create bins for feature difference
    feat_min, feat_max = config.feat_diff_range[0], config.feat_diff_range[1]
    bin_edges = np.linspace(feat_min, feat_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Vectorized binning - assign each trial to a bin
    bin_indices = np.digitize(feat_diff_values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Ensure within bounds

    # Initialize output array
    empirical_sds = np.full(n_bins, np.nan)

    # Process all bins at once using unique bin indices
    unique_bins = np.unique(bin_indices)

    for bin_idx in unique_bins:
        mask = bin_indices == bin_idx
        bin_bias_values = bias_values[mask]

        if len(bin_bias_values) > 1:
            # Vectorized circular standard deviation computation
            angles_rad = np.radians(bin_bias_values)
            mean_cos = np.mean(np.cos(angles_rad))
            mean_sin = np.mean(np.sin(angles_rad))
            r = np.sqrt(mean_cos**2 + mean_sin**2)
            circular_sd = np.degrees(np.sqrt(-2 * np.log(np.maximum(r, 1e-10))))
            empirical_sds[bin_idx] = circular_sd

    return bin_centers, empirical_sds




def compute_predicted_sd_curves_batch(log_surfaces_batch, feat_vals):
    """Batch version: Compute predicted SD curves for multiple surfaces at once - fully vectorized."""
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

    # Vectorized circular SD computation for all surfaces and feature values
    # NN normalizes via trapezoid integral = 1 (continuous), so use trapezoid not sum.
    # prob_profiles: (n_surfaces, n_mu1_bias, n_feat_vals)
    # cos_angles: (n_mu1_bias,) -> broadcast to (1, n_mu1_bias, 1)
    mean_cos = jnp.trapezoid(prob_profiles * cos_angles[None, :, None], x=mu1_bias_grid, axis=1)
    mean_sin = jnp.trapezoid(prob_profiles * sin_angles[None, :, None], x=mu1_bias_grid, axis=1)

    # Compute circular standard deviations
    r = jnp.sqrt(mean_cos**2 + mean_sin**2)
    r_safe = jnp.minimum(jnp.maximum(r, 1e-10), 1.0 - 1e-10)  # clamp to (0, 1)
    circular_sds = jnp.degrees(jnp.sqrt(-2 * jnp.log(r_safe)))  # Shape: (n_surfaces, n_feat_vals)

    return circular_sds


def load_extended_results(results_path: str) -> Dict:
    """Load extended batch fitting results."""
    print(f"Loading extended results from {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    print(f"Loaded {len(results)} extended results")
    return results


def organize_results_by_subject(extended_results: Dict) -> Dict:
    """Organize results by subject and experiment."""
    subjects = defaultdict(lambda: defaultdict(list))

    for condition_name, result in extended_results.items():
        if result is None:
            continue

        # Handle both new naming pattern (S12#color_1#high) and old pattern (S12.color.1_high)
        if '#' in condition_name:
            # New naming pattern: "S12#color_1#high - low" -> subject="S12", exp="color_1", noise="high - low"
            parts = condition_name.split('#')
            if len(parts) < 3:
                continue

            subject_id = parts[0]  # "S12"
            experiment = parts[1]  # "color_1" or "color_2" or "color_2_first" or "color_2_second"
            noise_condition = parts[2]  # "high - low"

        else:
            # Old naming pattern: "S1.color.1_low - high" -> subject="S1", exp="color.1", noise="low - high"
            parts = condition_name.split('.')
            if len(parts) < 3:
                continue

            subject_id = parts[0]  # "S1"
            exp_part = parts[1]    # "color"

            # Extract experiment and noise condition
            noise_part = parts[2]  # "1_low - high"
            if '_' in noise_part:
                exp_num = noise_part.split('_')[0]  # "1"
                noise_condition = noise_part.split('_', 1)[1]  # "low - high"
            else:
                exp_num = noise_part
                noise_condition = "unknown"

            experiment = f"{exp_part}.{exp_num}"  # "color.1"

        subjects[subject_id][experiment].append({
            'condition_name': condition_name,
            'noise_condition': noise_condition,
            'result': result
        })

    return subjects


def prepare_all_subjects_data(
    subjects_data: Dict,
    global_optimizer: GridBasedMultiConditionOptimizer
) -> Dict:
    """
    Batch data preparation routine - processes all subjects at once for maximum efficiency.

    Returns:
        Dictionary with all precomputed data for all subjects
    """

    print("Preparing data for all subjects in batch...")

    # Collect all parameters and data across ALL subjects
    all_params_3d = []
    all_motor_noise = []
    param_mapping = {}  # Maps (subject_id, experiment, condition, optimizer) -> index in batch
    batch_idx = 0

    # Use full feature grid from config (model space 0–180°)
    feat_vals = jnp.arange(config.feat_diff_range[0], config.feat_diff_range[1]+1, 2)

    # First pass: collect all parameters
    subjects_structure = {}

    for subject_id, subject_data in subjects_data.items():
        subjects_structure[subject_id] = {}

        for experiment, conditions_list in subject_data.items():
            if len(conditions_list) == 0:
                continue

            # Determine available optimizers
            available_optimizers = []
            sample_result = conditions_list[0]['result']
            for key in sample_result.keys():
                if key.endswith('_fitted_params'):
                    opt_type = key.replace('_fitted_params', '')
                    available_optimizers.append(opt_type)

            # Organize conditions by noise level
            noise_conditions = {}
            for cond_data in conditions_list:
                noise = cond_data['noise_condition']
                if noise not in noise_conditions:
                    noise_conditions[noise] = []
                noise_conditions[noise].append(cond_data)

            subjects_structure[subject_id][experiment] = {
                'noise_conditions': noise_conditions,
                'available_optimizers': available_optimizers
            }

            # Collect parameters for all condition×optimizer combinations
            for condition_name, cond_data_list in noise_conditions.items():
                cond_data = cond_data_list[0]
                result = cond_data['result']

                for opt in available_optimizers:
                    param_key = f'{opt}_fitted_params'
                    if param_key in result:
                        params = result[param_key]
                        params_3d = params[:3] if len(params) >= 3 else params
                        motor_noise = params[3] if len(params) >= 4 else 0.0

                        all_params_3d.append(params_3d)
                        all_motor_noise.append(motor_noise)
                        param_mapping[(subject_id, experiment, condition_name, opt)] = batch_idx
                        batch_idx += 1

    print(f"Collected {len(all_params_3d)} parameter combinations from all subjects")

    # MASSIVE BATCH COMPUTATION FOR ALL SUBJECTS AT ONCE
    if all_params_3d:
        print(f"Batch computing {len(all_params_3d)} parameter combinations for ALL subjects...")

        # Single NN prediction for ALL subjects×experiments×conditions×optimizers
        params_batch = jnp.array(all_params_3d)
        log_surfaces_batch = global_optimizer._predict_batch_fixed_size(params_batch, verbosity=0)

        # Apply motor noise in batch grouped by motor noise value
        from grid_based_multi_condition_optimizer_jax_loops import apply_motor_noise_with_precomputed_kernel, create_motor_noise_kernel_fft, _generate_nn_bias_curve_batch, generate_nn_density_asymmetry_batch

        # Group surfaces by motor noise values using unique with indices
        all_motor_noise_array = jnp.array(all_motor_noise)
        unique_motor_noise, inverse_indices = jnp.unique(all_motor_noise_array, return_inverse=True)
        n_mu1_bias = log_surfaces_batch.shape[1]

        print(f"Found {len(unique_motor_noise)} unique motor noise values: {unique_motor_noise}")

        # Initialize output array
        surfaces_with_noise = jnp.zeros_like(log_surfaces_batch)

        # Process each unique motor noise value in batches
        for i, sd_motor in enumerate(unique_motor_noise):
            # Find all surfaces with this motor noise value
            mask = inverse_indices == i
            n_surfaces_with_this_noise = jnp.sum(mask)

            print(f"  Processing motor noise {sd_motor:.1f}: {n_surfaces_with_this_noise} surfaces")

            if sd_motor > 0:
                # Get or create precomputed kernel
                key = (float(sd_motor), n_mu1_bias)
                if key not in global_motor_kernel_cache:
                    global_motor_kernel_cache[key] = create_motor_noise_kernel_fft(sd_motor, n_mu1_bias)

                kernel_fft = global_motor_kernel_cache[key]

                # Apply motor noise to all surfaces with this motor noise value in one batch
                batch_surfaces = log_surfaces_batch[mask]
                batch_surfaces_with_noise = apply_motor_noise_with_precomputed_kernel(batch_surfaces, kernel_fft)
                surfaces_with_noise = surfaces_with_noise.at[mask].set(batch_surfaces_with_noise)
            else:
                # No motor noise - keep original surfaces
                surfaces_with_noise = surfaces_with_noise.at[mask].set(log_surfaces_batch[mask])

        log_surfaces_batch = surfaces_with_noise

        # Batch compute ALL curves at once for all subjects
        print("Computing bias curves for all surfaces...")
        feat_indices = jnp.arange(len(feat_vals))
        all_bias_curves = _generate_nn_bias_curve_batch(log_surfaces_batch, feat_indices)

        print("Computing density asymmetry curves for all surfaces...")
        all_asymm_curves = generate_nn_density_asymmetry_batch(log_surfaces_batch)

        print("Computing standard deviation curves for all surfaces...")
        all_predicted_sd = compute_predicted_sd_curves_batch(log_surfaces_batch, feat_vals)

        print("Mapping results back to subject structure...")

    # Build the complete prepared data structure
    prepared_all_subjects = {}

    for subject_id, subject_structure in subjects_structure.items():
        prepared_subject = {
            'subject_id': subject_id,
            'experiments': {}
        }

        for experiment, exp_structure in subject_structure.items():
            noise_conditions = exp_structure['noise_conditions']
            available_optimizers = exp_structure['available_optimizers']

            # Store optimizer curves
            experiment_optimizer_curves = {}
            experiment_empirical_curves = {}
            experiment_parameters = {}

            # Map results back to (condition, optimizer) pairs
            if all_params_3d:
                for condition_name, cond_data_list in noise_conditions.items():
                    for opt in available_optimizers:
                        key = (subject_id, experiment, condition_name, opt)
                        if key in param_mapping:
                            idx = param_mapping[key]

                            if condition_name not in experiment_optimizer_curves:
                                experiment_optimizer_curves[condition_name] = {}

                            experiment_optimizer_curves[condition_name][opt] = {
                                'bias': all_bias_curves[idx],
                                'asymmetry': all_asymm_curves[idx],
                                'predicted_sd': all_predicted_sd[idx]
                            }

            # Collect all condition datasets for this subject at once
            all_condition_datasets = {}
            condition_data_mapping = {}

            # Check if empirical curves already exist in the results (skip expensive recomputation)
            sample_result = noise_conditions[list(noise_conditions.keys())[0]][0]['result']
            empirical_curves_exist = 'empirical_curves' in sample_result and sample_result['empirical_curves']

            # First pass: collect parameters and prepare datasets
            for noise_cond, cond_data_list in noise_conditions.items():
                cond_data = cond_data_list[0]
                result = cond_data['result']

                optimizer_params = {}
                optimizer_losses = {}
                optimizer_eval_losses = {}
                for opt in available_optimizers:
                    param_key = f'{opt}_fitted_params'
                    loss_key = f'{opt}_loss'
                    eval_key = f'{opt}_evaluation_losses'
                    if param_key in result:
                        optimizer_params[opt] = result[param_key]
                    if loss_key in result:
                        optimizer_losses[opt] = result[loss_key]
                    if eval_key in result:
                        optimizer_eval_losses[opt] = result[eval_key]

                experiment_parameters[noise_cond] = {
                    'params': optimizer_params,
                    'losses': optimizer_losses,
                    'eval_losses': optimizer_eval_losses,
                }

                # Collect condition data (data_df is already cleaned)
                condition_raw_data = []
                for i, cond_data in enumerate(cond_data_list):
                    if 'data_df' in cond_data['result']:
                        data_array = cond_data['result']['data_df']
                        dataset_name = f"{noise_cond}_dataset_{i}"
                        all_condition_datasets[dataset_name] = data_array
                        condition_raw_data.extend(data_array.tolist())

                condition_data_mapping[noise_cond] = {
                    'dataset_names': [f"{noise_cond}_dataset_{i}" for i in range(len(cond_data_list)) if 'data_df' in cond_data_list[i]['result']],
                    'raw_data': condition_raw_data
                }

            # Single optimizer update for all conditions of this subject (skip if empirical curves exist)
            if empirical_curves_exist:
                print(f"  Using pre-saved empirical curves for {subject_id}#{experiment}")
                # Extract empirical curves directly from saved results
                for noise_cond in noise_conditions.keys():
                    result = noise_conditions[noise_cond][0]['result']
                    saved_curves = result['empirical_curves']

                    # Compute empirical SD from stored data_df (cols: feat_diff, bias)
                    empirical_sd = None
                    if 'data_df' in result:
                        data_array = np.array(result['data_df'])
                        empirical_sd = compute_empirical_sd_curve(
                            data_array[:, 0], data_array[:, 1], n_bins=18)

                    experiment_empirical_curves[noise_cond] = {
                        'bias': saved_curves['target_bias'],
                        'asymmetry': saved_curves['target_density'],
                        'sd': empirical_sd,
                        'bias_grid': saved_curves['density_feat_grid'][saved_curves['bias_feat_indices']],
                        'asymm_grid': saved_curves['density_feat_grid']
                    }

            elif all_condition_datasets:
                global_optimizer.update_dataset(all_condition_datasets)
                dataset_names_list = list(all_condition_datasets.keys())

                # Process each condition using precomputed curves and individual empirical SD
                for noise_cond in noise_conditions.keys():
                    if noise_cond in condition_data_mapping:
                        mapping = condition_data_mapping[noise_cond]

                        # Get optimizer curves for this condition's datasets
                        condition_indices = [dataset_names_list.index(name) for name in mapping['dataset_names'] if name in dataset_names_list]

                        if condition_indices:
                            condition_indices_array = jnp.array(condition_indices)
                            empirical_bias = jnp.mean(global_optimizer.unified_target_bias[condition_indices_array], axis=0)
                            empirical_asymm = jnp.mean(global_optimizer.unified_target_density[condition_indices_array], axis=0)
                        else:
                            empirical_bias = None
                            empirical_asymm = None

                        # Compute empirical SD for this specific condition
                        empirical_sd = None
                        if mapping['raw_data']:
                            condition_array = np.array(mapping['raw_data'])
                            empirical_sd = compute_empirical_sd_curve(condition_array[:, 0], condition_array[:, 1], n_bins=18)

                        # Get the separate empirical feature grids from optimizer
                        empirical_bias_grid = None
                        empirical_asymm_grid = None

                        if hasattr(global_optimizer, 'unified_feat_indices') and hasattr(global_optimizer, 'feat_diff_grid'):
                            # Bias uses binned grid (via unified_feat_indices)
                            empirical_bias_grid = global_optimizer.feat_diff_grid[global_optimizer.unified_feat_indices]
                            # Asymmetry uses full grid
                            empirical_asymm_grid = global_optimizer.feat_diff_grid

                        experiment_empirical_curves[noise_cond] = {
                            'bias': empirical_bias,
                            'asymmetry': empirical_asymm,
                            'sd': empirical_sd,
                            'bias_grid': empirical_bias_grid,
                            'asymm_grid': empirical_asymm_grid
                        }

            # Store experiment data
            prepared_subject['experiments'][experiment] = {
                'optimizer_curves': experiment_optimizer_curves,
                'empirical_curves': experiment_empirical_curves,
                'parameters': experiment_parameters,
                'available_optimizers': available_optimizers,
                'feat_vals': feat_vals,
                'noise_conditions': list(noise_conditions.keys())
            }

        prepared_all_subjects[subject_id] = prepared_subject

    print(f"Batch data preparation completed for {len(prepared_all_subjects)} subjects")
    return prepared_all_subjects



def create_unified_subject_plot(
    prepared_data: Dict,
    output_dir: str = 'model_fit_to_data_results_v2',
    circ_space: int = 360,
) -> None:
    """
    Create unified plot for a single subject using precomputed data.

    Args:
        prepared_data: Dictionary with all precomputed data from prepare_subject_data()
        output_dir: Output directory for plots
    """

    subject_id = prepared_data['subject_id']
    print(f"Creating unified plot for {subject_id}...")

    # Create output directory
    plots_dir = Path(output_dir) / 'unified_subject_plots'
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Process each experiment for this subject
    for experiment, experiment_data in prepared_data['experiments'].items():
        available_optimizers = experiment_data['available_optimizers']
        noise_conditions = experiment_data['noise_conditions']
        feat_vals = experiment_data['feat_vals']
        optimizer_curves = experiment_data['optimizer_curves']
        empirical_curves = experiment_data['empirical_curves']
        parameters = experiment_data['parameters']

        angle_display_scale = _angle_display_scale(circ_space)
        display_feat_vals = np.array(feat_vals) * angle_display_scale

        n_conditions = len(noise_conditions)
        if n_conditions == 0:
            continue

        # Create figure: 3 rows × n_conditions columns
        # Row 1: bias curves, Row 2: density asymmetry, Row 3: standard deviation
        fig, axes = plt.subplots(3, n_conditions, figsize=(6*n_conditions, 15))
        if n_conditions == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'Subject {subject_id} - Experiment {experiment}', fontsize=16, y=0.98)

        # Define colors for optimizers
        opt_colors = OPTIMIZER_COLORS
        opt_labels = OPTIMIZER_LABELS

        # NOW JUST PLOT THE PRECOMPUTED RESULTS
        for col, noise_cond in enumerate(sorted(noise_conditions)):
            print(f"    Plotting condition: {noise_cond}")

            # Get precomputed curves for this condition
            condition_optimizer_curves = optimizer_curves.get(noise_cond, {})
            condition_empirical_curves = empirical_curves.get(noise_cond, {})
            condition_parameters = parameters.get(noise_cond, {})

            # Plot bias curves (top row)
            ax1 = axes[0, col]

            # Plot optimizer curves
            legend_entries = []
            for opt in available_optimizers:
                if opt not in condition_optimizer_curves:
                    continue

                color = opt_colors.get(opt, 'black')
                label = opt_labels.get(opt, opt.capitalize())
                params = condition_parameters.get('params', {}).get(opt, condition_parameters.get(opt))

                # Format parameter string
                if len(params) >= 4:
                    param_str = f'[{params[0]:.1f}, {params[1]:.1f}, {params[2]:.1f}, {params[3]:.1f}]'
                else:
                    param_str = f'[{params[0]:.1f}, {params[1]:.1f}, {params[2]:.1f}]'

                ax1.plot(display_feat_vals, condition_optimizer_curves[opt]['bias'] * angle_display_scale,
                        color=color, linewidth=2, label=f'{label}')

                legend_entries.append(f'{label}: {param_str}')

            # Plot empirical data
            empirical_bias = condition_empirical_curves.get('bias')
            empirical_bias_grid = condition_empirical_curves.get('bias_grid')
            if empirical_bias is not None and empirical_bias_grid is not None:
                ax1.plot(np.array(empirical_bias_grid) * angle_display_scale,
                        empirical_bias * angle_display_scale,
                        'k--', linewidth=2, alpha=0.7, label='Data')
                legend_entries.append('Data')

            ax1.set_xlabel('Feature Difference (°)')
            ax1.set_ylabel('Mu1 Bias (degrees)')
            ax1.set_title(f'{noise_cond}')
            ax1.grid(True, alpha=0.3)

            # Add parameter info as text box
            param_text = '\n'.join(legend_entries[:-1])  # Exclude 'Data' entry
            ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Plot density asymmetry (bottom row)
            ax2 = axes[1, col]

            for opt in available_optimizers:
                if opt not in condition_optimizer_curves:
                    continue

                color = opt_colors.get(opt, 'black')
                label = opt_labels.get(opt, opt.capitalize())

                ax2.plot(display_feat_vals, condition_optimizer_curves[opt]['asymmetry'],
                        color=color, linewidth=2, label=label)

            # Plot empirical asymmetry
            empirical_asymm = condition_empirical_curves.get('asymmetry')
            empirical_asymm_grid = condition_empirical_curves.get('asymm_grid')
            if empirical_asymm is not None and empirical_asymm_grid is not None:
                valid_mask = ~jnp.isnan(empirical_asymm)
                if jnp.any(valid_mask):
                    ax2.plot(np.array(empirical_asymm_grid)[valid_mask] * angle_display_scale,
                            empirical_asymm[valid_mask], 'k--', linewidth=2, alpha=0.7, label='Data')

            ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
            ax2.set_xlabel('Feature Difference (°)')
            ax2.set_ylabel('Density Asymmetry')
            if col == 0:
                ax2.set_title('Density Asymmetry')
            ax2.grid(True, alpha=0.3)

            # Plot standard deviation curves (third row)
            ax3 = axes[2, col]

            for opt in available_optimizers:
                if opt not in condition_optimizer_curves:
                    continue

                color = opt_colors.get(opt, 'black')
                label = opt_labels.get(opt, opt.capitalize())

                ax3.plot(display_feat_vals,
                        condition_optimizer_curves[opt]['predicted_sd'] * angle_display_scale,
                        color=color, linewidth=2, label=f'{label}', linestyle='-')

            # Plot empirical standard deviation
            empirical_sd = condition_empirical_curves.get('sd')
            if empirical_sd is not None:
                emp_bin_centers, emp_sd_values = empirical_sd
                valid_mask = ~np.isnan(emp_sd_values)
                if np.any(valid_mask):
                    ax3.plot(emp_bin_centers[valid_mask] * angle_display_scale,
                            emp_sd_values[valid_mask] * angle_display_scale,
                            'ko-', linewidth=2, alpha=0.7, markersize=4, label='Data')

            ax3.set_xlabel('Feature Difference (°)')
            ax3.set_ylabel('Standard Deviation (degrees)')
            if col == 0:
                ax3.set_title('Response Variability')
            ax3.grid(True, alpha=0.3)

            # Add legend only to first column
            if col == 0:
                ax1.legend(loc='upper right')
                ax2.legend(loc='upper right')
                ax3.legend(loc='upper right')

        plt.tight_layout()

        # Save plot
        safe_exp_name = experiment.replace('.', '_')
        plot_path = plots_dir / f"{subject_id}_{safe_exp_name}_unified.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    Saved: {plot_path}")

def organize_preprocessed_results_by_experiment(prepared_all_subjects: Dict) -> Dict:
    """Organize preprocessed results by experiment and condition."""
    experiments = {}

    for subject_id, prepared_data in prepared_all_subjects.items():
        for experiment, experiment_data in prepared_data['experiments'].items():
            # Map experiment names for consistency with old function
            if experiment == 'color.1':
                exp = 'color_1'
            elif experiment == 'color_hv.1':
                exp = 'color_hv_1'
            elif experiment == 'color.2':
                exp = 'color_2'
            else:
                exp = experiment.replace('.', '_')

            if exp not in experiments:
                experiments[exp] = {}

            # Process each noise condition for this experiment
            for noise_cond in experiment_data['noise_conditions']:
                if noise_cond not in experiments[exp]:
                    experiments[exp][noise_cond] = []

                # Store the prepared data for this subject+experiment+condition
                experiments[exp][noise_cond].append({
                    'subject_id': subject_id,
                    'experiment_data': experiment_data,
                    'noise_condition': noise_cond
                })

    return experiments


def create_extended_summary_plots(prepared_all_subjects: Dict,
                                 output_dir: str = 'model_fit_to_data_results_v2',
                                 create_individual_plots: bool = True,
                                 corr_weight: float = 0.25,
                                 circ_space: int = 360) -> Tuple[List, List]:
    """Create extended summary plots using preprocessed data.

    Args:
        prepared_all_subjects: Dictionary from prepare_all_subjects_data() with all precomputed curves
        output_dir: Output directory for plots
        create_individual_plots: Whether to create individual plots
        corr_weight: Weight for correlation loss in combined fitting (for density optimizer loss components)

    Returns:
        Tuple of (curve_data_for_csv, parameter_data_for_csv) for CSV export
    """

    print("Creating extended summary plots using preprocessed data...")

    # Data collection for CSV export
    curve_data_for_csv = []
    parameter_data_for_csv = []

    # Organize preprocessed data by experiment and condition
    experiments = organize_preprocessed_results_by_experiment(prepared_all_subjects)

    if len(experiments) == 0:
        print("No experiments found with sufficient data for summary plots")
        return [], []

    # Create plots directory
    plots_dir = Path(output_dir) / 'summary_plots'
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Create plots for each experiment
    for exp_name, exp_conditions in experiments.items():
        print(f"Creating extended summary plot for experiment: {exp_name}")

        noise_conditions = sorted(exp_conditions.keys())
        n_conditions = len(noise_conditions)

        # Create figure: 3 rows (bias, asymmetry, SD) × n_conditions columns.
        # constrained_layout reserves space for the suptitle so it cannot overlap
        # the condition title or first plot row.
        fig, axes = plt.subplots(3, n_conditions, figsize=(5*n_conditions, 14),
                                 constrained_layout=True)
        if n_conditions == 1:
            axes = axes.reshape(3, 1)

        fig.suptitle(f'Extended Analysis: {exp_name}', fontsize=16, fontweight='bold')

        for col, noise_cond in enumerate(noise_conditions):
            prepared_results_list = exp_conditions[noise_cond]

            print(f"  Processing {noise_cond}: {len(prepared_results_list)} subjects")

            # Get available optimizers and precomputed curves from first subject
            if not prepared_results_list:
                continue

            first_subject_data = prepared_results_list[0]['experiment_data']
            available_optimizers = first_subject_data['available_optimizers']
            feat_vals = first_subject_data['feat_vals']
            angle_display_scale = _angle_display_scale(circ_space)
            display_feat_vals = np.array(feat_vals) * angle_display_scale

            if not available_optimizers:
                continue

            # Collect precomputed curves from all subjects for this experiment+condition
            stage_bias_curves = {opt: [] for opt in available_optimizers}
            stage_asymm_curves = {opt: [] for opt in available_optimizers}
            stage_sd_curves = {opt: [] for opt in available_optimizers}

            # Collect parameters and empirical curves
            all_parameters = {opt: [] for opt in available_optimizers}
            all_empirical_bias = []
            all_empirical_asymm = []
            all_empirical_sd = []
            empirical_bias_grid = None
            empirical_asymm_grid = None

            for prepared_result in prepared_results_list:
                subject_id = prepared_result['subject_id']
                experiment_data = prepared_result['experiment_data']
                noise_condition = prepared_result['noise_condition']

                # Skip the "combined" pseudo-subject from summary statistics so
                # group means/SDs reflect individual observers only.
                if subject_id == "combined":
                    continue

                # Get precomputed optimizer curves for this subject+condition
                optimizer_curves = experiment_data['optimizer_curves'].get(noise_condition, {})
                empirical_curves = experiment_data['empirical_curves'].get(noise_condition, {})
                parameters = experiment_data['parameters'].get(noise_condition, {})

                for opt in available_optimizers:
                    if opt in optimizer_curves:
                        stage_bias_curves[opt].append(optimizer_curves[opt]['bias'])
                        stage_asymm_curves[opt].append(optimizer_curves[opt]['asymmetry'])
                        if 'predicted_sd' in optimizer_curves[opt]:
                            stage_sd_curves[opt].append(optimizer_curves[opt]['predicted_sd'])
                        params = parameters.get('params', {}).get(opt, parameters.get(opt))
                        if params is not None:
                            all_parameters[opt].append(params)

                # Collect empirical curves and separate feature grids
                if empirical_curves.get('bias') is not None:
                    all_empirical_bias.append(empirical_curves['bias'])
                    if empirical_bias_grid is None:
                        empirical_bias_grid = empirical_curves.get('bias_grid')
                if empirical_curves.get('asymmetry') is not None:
                    all_empirical_asymm.append(empirical_curves['asymmetry'])
                    if empirical_asymm_grid is None:
                        empirical_asymm_grid = empirical_curves.get('asymm_grid')
                if empirical_curves.get('sd') is not None:
                    all_empirical_sd.append(empirical_curves['sd'])

            # Convert to arrays for statistics
            for opt in available_optimizers:
                if stage_bias_curves[opt]:
                    stage_bias_curves[opt] = jnp.array(stage_bias_curves[opt])
                    stage_asymm_curves[opt] = jnp.array(stage_asymm_curves[opt])
                    all_parameters[opt] = jnp.array(all_parameters[opt])
                if len(stage_sd_curves[opt]) > 0:
                    stage_sd_curves[opt] = jnp.array(stage_sd_curves[opt])

            # Drop optimizers that had no subjects with data for this condition;
            # stage_bias_curves[opt] stays a plain list when conversion was skipped.
            available_optimizers = [opt for opt in available_optimizers
                                    if not isinstance(stage_bias_curves[opt], list)]

            # Collect data for CSV export using preprocessed data
            valid_results = []
            for prepared_result in prepared_results_list:
                subject_id = prepared_result['subject_id']
                if subject_id == "combined":
                    continue
                experiment_data = prepared_result['experiment_data']
                noise_condition = prepared_result['noise_condition']

                valid_results.append({
                    'subject': subject_id,
                    'experiment': exp_name,
                    'condition': noise_condition,
                    'experiment_data': experiment_data
                })

            if len(valid_results) > 0:
                # Create curve and parameter data for CSV export
                n_subjects = len(valid_results)
                n_feat_points = len(feat_vals)

                subjects = np.array([r['subject'] for r in valid_results])
                experiments = np.array([r['experiment'] for r in valid_results])
                conditions = np.array([r['condition'] for r in valid_results])

                # Process all available optimizers
                for opt in available_optimizers:
                    # Curve data
                    opt_bias_data = stage_bias_curves[opt][:n_subjects]
                    opt_asymm_data = stage_asymm_curves[opt][:n_subjects]
                    if opt in stage_sd_curves and len(stage_sd_curves[opt]) > 0:
                        opt_sd_data = stage_sd_curves[opt][:n_subjects]
                    else:
                        opt_sd_data = np.full((n_subjects, n_feat_points), np.nan)

                    opt_curve_df = pd.DataFrame({
                        'subject': np.repeat(subjects, n_feat_points),
                        'experiment': np.repeat(experiments, n_feat_points),
                        'condition': np.repeat(conditions, n_feat_points),
                        'optimizer': opt,
                        'feat_diff': np.tile(display_feat_vals, n_subjects),
                        'mu_bias': (opt_bias_data * angle_display_scale).flatten(),
                        'sd_deg': (opt_sd_data * angle_display_scale).flatten(),
                        'density_asymmetry': opt_asymm_data.flatten()
                    })

                    curve_data_for_csv.extend(opt_curve_df.to_dict('records'))

                    # Parameter data from preprocessed results
                    opt_params_list = []
                    opt_losses = []
                    opt_eval_losses = {method: [] for method in LOSS_EVALUATION_METHODS}
                    opt_mse_losses = []
                    opt_corr_losses = []

                    for result in valid_results:
                        experiment_data = result['experiment_data']
                        noise_condition = result['condition']
                        parameters = experiment_data['parameters'].get(noise_condition, {})

                        # Handle both old format (parameters[opt]) and new format (parameters['params'][opt])
                        opt_params = parameters.get('params', {}).get(opt, parameters.get(opt))
                        opt_loss = parameters.get('losses', {}).get(opt, 0.0)
                        eval_losses = parameters.get('eval_losses', {}).get(opt, {})

                        if opt_params is not None:
                            opt_params_list.append(opt_params)
                            opt_losses.append(opt_loss)
                            for loss_method in LOSS_EVALUATION_METHODS:
                                opt_eval_losses[loss_method].append(eval_losses.get(loss_method, np.nan))

                            # Compute density loss components for density optimizer
                            if opt == 'density':
                                # Get target and predicted curves for this subject/condition
                                optimizer_curves = experiment_data['optimizer_curves'].get(noise_condition, {})
                                empirical_curves = experiment_data['empirical_curves'].get(noise_condition, {})

                                if opt in optimizer_curves and empirical_curves.get('asymmetry') is not None:
                                    predicted_asymm = jnp.array(optimizer_curves[opt]['asymmetry']).reshape(1, -1)
                                    target_asymm = jnp.array(empirical_curves['asymmetry']).reshape(1, -1)

                                    # Use the same function as the optimizer with exact same parameters
                                    # Use the corr_weight passed to this function

                                    # Compute total combined loss using optimizer's function
                                    total_loss = _compute_curve_losses(predicted_asymm, target_asymm,
                                                                     loss_type="combined", is_angular=False,
                                                                     corr_weight=corr_weight)[0]

                                    # Compute individual components manually (same as in _compute_curve_losses)
                                    diff = predicted_asymm - target_asymm
                                    mse_loss = jnp.mean(diff ** 2)

                                    # Correlation component
                                    corr_matrix = jnp.corrcoef(predicted_asymm.flatten(), target_asymm.flatten())
                                    corr = corr_matrix[0, 1]
                                    corr_loss = 1 - jnp.where(jnp.isnan(corr), 0.0, corr)

                                    # Scale MSE by target range (same as in optimizer)
                                    target_range = jnp.abs(jnp.max(target_asymm) - jnp.min(target_asymm))
                                    mse_scaled = mse_loss / jnp.maximum(target_range, 1e-6)

                                    opt_mse_losses.append(float(mse_scaled ))
                                    opt_corr_losses.append(float(corr_loss))
                                else:
                                    opt_mse_losses.append(np.nan)
                                    opt_corr_losses.append(np.nan)
                            else:
                                opt_mse_losses.append(np.nan)
                                opt_corr_losses.append(np.nan)

                    opt_params_array = np.array(opt_params_list)

                    param_df_data = {
                        'subject': subjects,
                        'experiment': experiments,
                        'condition': conditions,
                        'optimizer': opt,
                        'sd_feat1': opt_params_array[:, 0],
                        'sd_feat2': opt_params_array[:, 1],
                        'sd_spat': opt_params_array[:, 2],
                        f'{opt}_loss': opt_losses
                    }
                    for loss_method, values in opt_eval_losses.items():
                        param_df_data[f'eval_{loss_method}_loss'] = values

                    # Add density loss components for density optimizer
                    if opt == 'density':
                        param_df_data['density_mse_loss'] = opt_mse_losses
                        param_df_data['density_corr_loss'] = opt_corr_losses

                    # Add sd_motor if available
                    if opt_params_array.shape[1] > 3:
                        param_df_data['sd_motor'] = opt_params_array[:, 3]

                    opt_param_df = pd.DataFrame(param_df_data)
                    parameter_data_for_csv.extend(opt_param_df.to_dict('records'))

            # Compute statistics for plotting
            avg_stage_bias = {}
            sem_stage_bias = {}
            avg_stage_asymm = {}
            sem_stage_asymm = {}
            avg_stage_sd = {}
            sem_stage_sd = {}

            for opt in available_optimizers:
                avg_stage_bias[opt] = jnp.mean(stage_bias_curves[opt], axis=0)
                sem_stage_bias[opt] = jnp.std(stage_bias_curves[opt], axis=0) / jnp.sqrt(len(stage_bias_curves[opt]))
                avg_stage_asymm[opt] = jnp.mean(stage_asymm_curves[opt], axis=0)
                sem_stage_asymm[opt] = jnp.std(stage_asymm_curves[opt], axis=0) / jnp.sqrt(len(stage_asymm_curves[opt]))
                if isinstance(stage_sd_curves[opt], jnp.ndarray):
                    avg_stage_sd[opt] = jnp.mean(stage_sd_curves[opt], axis=0)
                    sem_stage_sd[opt] = jnp.std(stage_sd_curves[opt], axis=0) / jnp.sqrt(len(stage_sd_curves[opt]))

            # Use precomputed empirical curves from prepared data
            if all_empirical_bias and all_empirical_asymm:
                # Average empirical curves across all subjects for this condition
                empirical_bias_smoothed = jnp.mean(jnp.array(all_empirical_bias), axis=0)
                emp_asymm = jnp.mean(jnp.array(all_empirical_asymm), axis=0)
                # Use the separate empirical feature grids
                emp_bias_feat_vals = empirical_bias_grid
                emp_asymm_feat_vals = empirical_asymm_grid
            else:
                empirical_bias_smoothed = jnp.zeros_like(feat_vals)
                emp_bias_feat_vals, emp_asymm_feat_vals, emp_asymm = feat_vals, feat_vals, jnp.zeros_like(feat_vals)

            # Average empirical SD curves across subjects
            emp_sd_bin_centers = None
            emp_sd_mean = None
            emp_sd_sem = None
            if all_empirical_sd:
                # each entry is (bin_centers, sd_values); stack sd_values, keep one bin_centers
                emp_sd_bin_centers = all_empirical_sd[0][0]
                sd_matrix = np.array([s[1] for s in all_empirical_sd])  # (n_subjects, n_bins)
                emp_sd_mean = np.nanmean(sd_matrix, axis=0)
                emp_sd_sem  = np.nanstd(sd_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(sd_matrix), axis=0).clip(1))

            # === PLOT 1: Bias curves ===
            ax1 = axes[0, col]

            # Plot optimizer bias curves
            opt_colors = OPTIMIZER_COLORS
            opt_labels = OPTIMIZER_LABELS

            for opt in available_optimizers:
                color = opt_colors.get(opt, 'black')
                label = opt_labels.get(opt, opt.capitalize())

                avg_bias_display = avg_stage_bias[opt] * angle_display_scale
                sem_bias_display = sem_stage_bias[opt] * angle_display_scale
                ax1.plot(display_feat_vals, avg_bias_display, color=color, linestyle='-', linewidth=2,
                        label=f'{label} n={len(stage_bias_curves[opt])}')
                ax1.fill_between(display_feat_vals, avg_bias_display - sem_bias_display,
                               avg_bias_display + sem_bias_display, alpha=0.3, color=color)

            # Plot empirical data
            if emp_bias_feat_vals is not None:
                ax1.plot(np.array(emp_bias_feat_vals) * angle_display_scale,
                         empirical_bias_smoothed * angle_display_scale,
                         'k--', linewidth=2, alpha=0.7, label='Data (smooth)')

            ax1.set_xlabel('Feature Difference (°)')
            ax1.set_ylabel('Mu1 Bias (degrees)')
            ax1.set_title(f'{noise_cond}')
            ax1.legend()
            ax1.grid(True)

            # === PLOT 2: Asymmetry curves ===
            ax2 = axes[1, col]

            # Plot optimizer asymmetry curves
            for opt in available_optimizers:
                color = opt_colors.get(opt, 'black')
                label = opt_labels.get(opt, opt.capitalize())

                ax2.plot(display_feat_vals, avg_stage_asymm[opt], color=color, linestyle='-', linewidth=2, label=label)
                ax2.fill_between(display_feat_vals, avg_stage_asymm[opt] - sem_stage_asymm[opt],
                               avg_stage_asymm[opt] + sem_stage_asymm[opt], alpha=0.3, color=color)

            # Plot empirical asymmetry
            if emp_asymm_feat_vals is not None:
                valid_emp_asymm = ~jnp.isnan(emp_asymm)
                ax2.plot(np.array(emp_asymm_feat_vals)[valid_emp_asymm] * angle_display_scale,
                        emp_asymm[valid_emp_asymm],
                        'ko--', linewidth=2, markersize=4, alpha=0.7, label='Data')

            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Feature Difference (°)')
            ax2.set_ylabel('Density Asymmetry')
            if col == 0:
                ax2.set_title('Density Asymmetry')
            ax2.legend()
            ax2.grid(True)

            # === PLOT 3: Response variability (circular SD vs feat_diff) ===
            ax3 = axes[2, col]

            for opt in available_optimizers:
                if opt not in avg_stage_sd:
                    continue
                color = opt_colors.get(opt, 'black')
                label = opt_labels.get(opt, opt.capitalize())
                avg_sd_display = avg_stage_sd[opt] * angle_display_scale
                sem_sd_display = sem_stage_sd[opt] * angle_display_scale
                ax3.plot(display_feat_vals, avg_sd_display, color=color, linestyle='-', linewidth=2, label=label)
                ax3.fill_between(display_feat_vals,
                                 avg_sd_display - sem_sd_display,
                                 avg_sd_display + sem_sd_display,
                                 alpha=0.3, color=color)

            if emp_sd_bin_centers is not None:
                valid = ~np.isnan(emp_sd_mean)
                display_sd_centers = emp_sd_bin_centers * angle_display_scale
                emp_sd_mean_display = emp_sd_mean * angle_display_scale
                emp_sd_sem_display = emp_sd_sem * angle_display_scale
                ax3.plot(display_sd_centers[valid], emp_sd_mean_display[valid],
                         'ko-', linewidth=2, markersize=4, alpha=0.8, label='Data')
                ax3.fill_between(display_sd_centers[valid],
                                 emp_sd_mean_display[valid] - emp_sd_sem_display[valid],
                                 emp_sd_mean_display[valid] + emp_sd_sem_display[valid],
                                 alpha=0.2, color='black')

            ax3.set_xlabel('Feature Difference (°)')
            ax3.set_ylabel('Circular SD (degrees)')
            if col == 0:
                ax3.set_title('Response Variability')
            ax3.legend()
            ax3.grid(True)

        # Save plot
        safe_name = exp_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        plot_path = plots_dir / f"extended_summary_{safe_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved extended summary: {plot_path}")

    return curve_data_for_csv, parameter_data_for_csv


def _model_slice(log_surf: np.ndarray, feat_grid: np.ndarray, mu1_grid: np.ndarray,
                 target_fd: float, weights_sd: float):
    """Return (prob, E, asym) for one Gaussian-weighted model surface slice."""
    w = np.exp(-0.5 * ((feat_grid - target_fd) / weights_sd) ** 2)
    w /= w.sum()
    surf = np.exp(log_surf - log_surf.max(axis=0, keepdims=True))
    surf /= surf.sum(axis=0, keepdims=True) * config.mu1_bias_step
    prob = surf @ w
    E    = float(np.sum(mu1_grid * prob) * config.mu1_bias_step)
    asym = float(np.sum(prob[mu1_grid > 0]) * config.mu1_bias_step
                 - np.sum(prob[mu1_grid < 0]) * config.mu1_bias_step)
    return prob, E, asym


def _empirical_slice(fd_vals: np.ndarray, bias_vals: np.ndarray,
                     target_fd: float, bias_grid: np.ndarray, weights_sd: float):
    """Return Gaussian-weighted KDE of bias values around target_fd."""
    w = np.exp(-0.5 * ((fd_vals - target_fd) / weights_sd) ** 2)
    if w.sum() < 1e-10:
        return np.zeros_like(bias_grid)
    w /= w.sum()
    bias_std = bias_vals.std()
    iqr = np.percentile(bias_vals, 75) - np.percentile(bias_vals, 25)
    bw  = max(0.9 * min(bias_std, iqr / 1.34) * len(bias_vals) ** (-0.2), 1.0)
    diff    = bias_grid[:, None] - bias_vals[None, :]
    kernels = np.exp(-0.5 * (diff / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    return (kernels * w[None, :]).sum(axis=1)


def create_pdf_slice_plots(
    extended_results: Dict,
    global_optimizer,
    output_dir: str,
    circ_space: int = 360,
    optimizer_names=None,
    n_subjects: int = 3,
    feat_diffs_data: Optional[List[float]] = None,
    weights_sd: float = 20.0,
) -> None:
    """For each experiment × condition × fitting method, plot p(mu1_bias | feat_diff) slices.

    Rows = subjects, columns = feat_diff values.  Selects subjects that show a
    bias/asymmetry dissociation first; falls back to the first n_subjects.
    One file is produced per fitting method.
    optimizer_names: list of method names, or None to auto-detect from results.
    """
    plots_dir = Path(output_dir) / 'pdf_slice_plots'
    plots_dir.mkdir(exist_ok=True, parents=True)

    angle_display_scale = _angle_display_scale(circ_space)
    mu1_grid  = np.array(config.create_grid('mu1_bias'))
    feat_grid = np.array(config.create_grid('feat_diff'))  # model space
    bias_limit_data = circ_space / 2
    bias_plot_grid = np.linspace(-bias_limit_data, bias_limit_data, 361)
    bias_plot_grid_model = bias_plot_grid / angle_display_scale

    # Default feat_diffs in data space: four values spanning ~5–50 % of range
    if feat_diffs_data is None:
        fd_max = circ_space / 2
        feat_diffs_data = [round(fd / 2) * 2
                           for fd in np.linspace(fd_max * 0.05, fd_max * 0.50, 4)]

    # Convert to model space for NN lookup
    feat_diffs_model = [fd / angle_display_scale for fd in feat_diffs_data]
    weights_sd_model = weights_sd / angle_display_scale  # scale sigma too

    # Auto-detect available methods if not specified
    if optimizer_names is None:
        seen = []
        for result in extended_results.values():
            for key in result:
                if key.endswith('_fitted_params'):
                    name = key[:-len('_fitted_params')]
                    if name not in seen:
                        seen.append(name)
        optimizer_names = seen

    for optimizer_name in optimizer_names:
        # Group by (experiment, noise_condition)
        by_exp_cond: Dict = {}
        for cond_key, result in extended_results.items():
            if f'{optimizer_name}_fitted_params' not in result or 'data_df' not in result:
                continue
            parts = cond_key.split('#')
            if len(parts) < 3:
                continue
            subject_id, experiment, noise_cond = parts[0], parts[1], '#'.join(parts[2:])
            key = (experiment, noise_cond)
            by_exp_cond.setdefault(key, []).append((subject_id, result))

        for (exp_name, noise_cond), subject_list in sorted(by_exp_cond.items()):
            print(f"  PDF slices: {exp_name} / {noise_cond}  ({len(subject_list)} subjects)"
                  f"  [{optimizer_name}]")

            # Score each subject by |E| at the smallest feat_diff to prefer informative ones;
            # break ties by picking dissociation cases (E and asym have opposite signs).
            scored = []
            for subject_id, result in subject_list:
                params = np.array(result[f'{optimizer_name}_fitted_params'])
                log_surf = global_optimizer._predict_batch_fixed_size(
                    jnp.array([params[:3]]), verbosity=0)[0]
                sd_motor = float(params[3]) if len(params) >= 4 else 0.0
                if sd_motor > 0:
                    from grid_based_multi_condition_optimizer_jax_loops import (
                        apply_motor_noise_with_precomputed_kernel, create_motor_noise_kernel_fft)
                    n_mu1_bias = log_surf.shape[0]
                    key = (sd_motor, n_mu1_bias)
                    if key not in global_motor_kernel_cache:
                        global_motor_kernel_cache[key] = create_motor_noise_kernel_fft(sd_motor, n_mu1_bias)
                    log_surf = apply_motor_noise_with_precomputed_kernel(
                        log_surf[None], global_motor_kernel_cache[key])[0]
                log_surf = np.array(log_surf)
                fd0 = feat_diffs_model[0]
                _, E0, asym0 = _model_slice(log_surf, feat_grid, mu1_grid, fd0, weights_sd_model)
                dissoc = int((E0 < 0 and asym0 > 0) or (E0 > 0 and asym0 < 0))
                scored.append((dissoc, abs(E0), subject_id, result, log_surf, params))

            scored.sort(key=lambda x: (-x[0], -x[1]))  # dissociation first, then |E|
            selected = scored[:n_subjects]

            n_rows = len(selected)
            n_cols = len(feat_diffs_data)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            if n_cols == 1:
                axes = axes.reshape(-1, 1)

            for row, (dissoc, _, subject_id, result, log_surf, params) in enumerate(selected):
                fd_vals  = np.array(result['data_df'])[:, 0]  # model space
                bias_vals = np.array(result['data_df'])[:, 1]

                for col, (fd_data, fd_model) in enumerate(zip(feat_diffs_data, feat_diffs_model)):
                    ax = axes[row, col]
                    prob, E, asym = _model_slice(log_surf, feat_grid, mu1_grid,
                                                 fd_model, weights_sd_model)
                    emp = _empirical_slice(fd_vals, bias_vals, fd_model,
                                           bias_plot_grid_model, weights_sd_model)
                    prob_display = prob / angle_display_scale
                    E_display = E * angle_display_scale

                    ax.fill_between(mu1_grid * angle_display_scale, prob_display, alpha=0.55, color='#6baed6')
                    ax.plot(mu1_grid * angle_display_scale, prob_display, color='#2171b5', lw=1.2)
                    ax.fill_between(mu1_grid * angle_display_scale, prob_display, where=(mu1_grid > 0),
                                    alpha=0.35, color='forestgreen')
                    ax.fill_between(mu1_grid * angle_display_scale, prob_display, where=(mu1_grid < 0),
                                    alpha=0.35, color='tomato')

                    ax.plot(bias_plot_grid, emp / angle_display_scale, color='darkorange', lw=1.5, alpha=0.85,
                            label='data (KDE)')

                    ax.axvline(E_display, color='black', lw=1.5, ls='--')
                    ax.axvline(0, color='gray',  lw=0.8, ls=':')
                    ax.set_xlim(-bias_limit_data, bias_limit_data)
                    ax.tick_params(labelsize=8)
                    ax.set_xlabel('mu1 bias (°)', fontsize=9)

                    dissoc_here = (E < 0 and asym > 0) or (E > 0 and asym < 0)
                    sign  = '+' if asym >= 0 else ''
                    title = (f"fd ≈ {fd_data:.0f}°\n"
                             f"E={E_display:.1f}°  asym={sign}{asym:.3f}"
                             + ('  ★' if dissoc_here else ''))
                    ax.set_title(title, fontsize=8.5,
                                 color='darkred' if dissoc_here else 'black')

                    if col == 0:
                        ax.set_ylabel('Density', fontsize=9)
                        p = params
                        lbl = (f"Subj {subject_id}\n"
                               f"f1={p[0]:.0f} f2={p[1]:.0f} sp={p[2]:.0f}")
                        ax.annotate(lbl, xy=(-0.42, 0.5), xycoords='axes fraction',
                                    fontsize=8, ha='center', va='center', rotation=90,
                                    annotation_clip=False)
                    if col == 0 and row == 0:
                        ax.legend(fontsize=7, loc='upper right', framealpha=0.7)

            safe = f"{exp_name}_{noise_cond}".replace(' ', '_').replace('-', '_')
            fig.suptitle(
                f"PDF slices — {exp_name} / {noise_cond} — {optimizer_name} fit\n"
                "Green = p>0, Red = p<0, dashed = E[bias]  |  ★ = dissociation",
                fontsize=10, y=1.01,
            )
            plt.tight_layout()
            out = plots_dir / f"pdf_slices_{optimizer_name}_{safe}.png"
            plt.savefig(out, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {out}")


def export_fitted_parameters_csv(parameter_data: List[Dict], output_dir: str = 'model_fit_to_data_results_v2') -> None:
    """Export fitted parameters to CSV in long format using pre-collected data."""

    print("Exporting fitted parameters to CSV...")

    # Save to CSV
    df = pd.DataFrame(parameter_data)

    # Create output directory
    csv_dir = Path(output_dir) / 'csv_exports'
    csv_dir.mkdir(exist_ok=True)

    # Save parameters CSV
    params_path = csv_dir / 'fitted_parameters.csv'
    df.to_csv(params_path, index=False)

    print(f"  Saved fitted parameters: {params_path}")
    print(f"  Shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})")


def export_fitted_curves_csv(curve_data: List[Dict], output_dir: str = 'model_fit_to_data_results_v2') -> None:
    """Export fitted bias, SD, and density asymmetry curves to CSV in long format."""

    print("Exporting fitted curves to CSV...")

    # Save to CSV
    df = pd.DataFrame(curve_data)

    # Create output directory
    csv_dir = Path(output_dir) / 'csv_exports'
    csv_dir.mkdir(exist_ok=True)

    # Save curves CSV
    curves_path = csv_dir / 'fitted_curves.csv'
    df.to_csv(curves_path, index=False)

    print(f"  Saved fitted curves: {curves_path}")
    print(f"  Shape: {df.shape} (rows: {df.shape[0]}, columns: {df.shape[1]})")


def create_unified_plots_with_summaries(
        n_samples: int = 20,
        outliers: bool = False,
        data_path: str = 'dasta_color_comb.csv',
        max_subjects: Optional[int] = None,
        create_summary_plots: bool = True,
        create_individual_plots: bool = True,
        create_csv_exports: bool = True,
        create_pdf_slices: bool = True,
        pdf_slice_optimizers=None,
        pdf_slice_n_subjects: int = 3,
        skip_motor_noise: bool = True,
        corr_weight: float = 0.25,
        results_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        results_dir: str = RESULTS_DIR,
        circ_space: int = 360,
) -> None:
    """Create unified subject plots and summary exports.

    Args:
        n_samples: Training sample count used for checkpoints.
        outliers: Whether the results include outliers.
        data_path: Unused legacy path (kept for signature compatibility).
        max_subjects: Optional limit on subjects to process.
        create_summary_plots: Whether to generate summary plots.
        create_individual_plots: Whether to generate per-subject plots.
        create_csv_exports: Whether to write CSV exports.
        skip_motor_noise: Whether motor noise was skipped in fitting.
        corr_weight: Correlation weight used during fitting.
        results_path: Optional path to extended_fit_results.pkl.
        checkpoint_path: Optional checkpoint path for loading the model.
        output_dir: Optional output directory for plots/exports.
        results_dir: Base directory for resolving relative paths.
    """
    corr_str = f'_cw_{corr_weight:.2f}' if corr_weight != 0.25 else ''

    outliers_str = 'no_outliers' if outliers == False else 'with_outliers'
    motor_str = '' if not skip_motor_noise else "_no_motor_noise"
    default_results_path = (
        f"{MODEL_FIT_RESULTS_PREFIX}{motor_str}{corr_str}/"
        f"{n_samples}samples/{outliers_str}/extended_fit_results.pkl"
    )
    default_checkpoint_path = (
        f"{CHECKPOINT_PREFIX}_{n_samples}samples/model_epoch_{CHECKPOINT_EPOCH:04d}.pkl"
    )
    resolved_results_path = resolve_input_path(results_path or default_results_path, results_dir)
    resolved_checkpoint_path = resolve_input_path(
        checkpoint_path or default_checkpoint_path, results_dir
    )

    if output_dir is None:
        resolved_output_dir = Path(resolved_results_path).parent
    else:
        resolved_output_dir = Path(resolve_results_path(output_dir, results_dir))

    # print(f'Reading {resolved_results_path}')
    """Create both unified plots and summary plots with CSV exports."""

    # Load results and organize by subject
    extended_results = load_extended_results(str(resolved_results_path))

    # Initialize optimizer
    print("Initializing optimizer...")
    # df_clean, dummy_dataset = get_data(data_path)
    # Create dummy condition dataset for GridBasedMultiConditionOptimizer
    dummy_condition_datasets = {'dummy': jnp.asarray(np.random.uniform(low=-180, high=180.0, size=(100,2)))}
    global_optimizer = GridBasedMultiConditionOptimizer(
        str(resolved_checkpoint_path), dummy_condition_datasets
    )
    print("Optimizer initialized.")
    print()

    # Create unified subject plots
    print("=== Creating Unified Subject Plots ===")
    subjects_data = organize_results_by_subject(extended_results)
    print(f"Found {len(subjects_data)} subjects")

    # Prepare data for all subjects in batch
    if max_subjects:
        limited_subjects_data = dict(list(subjects_data.items())[:max_subjects])
    else:
        limited_subjects_data = subjects_data

    prepared_all_subjects = prepare_all_subjects_data(limited_subjects_data, global_optimizer)

    # Create plots for each subject using prepared data
    if create_individual_plots:
        subjects_processed = 0
        for subject_id, prepared_data in prepared_all_subjects.items():
            create_unified_subject_plot(prepared_data, resolved_output_dir, circ_space=circ_space)
            subjects_processed += 1

        print(f"\\nCompleted unified plots for {subjects_processed} subjects")
        print(f"Plots saved to: {output_dir}/unified_subject_plots/")

    # Create summary plots if requested
    if create_summary_plots or create_csv_exports:
        print("\n=== Creating Summary Plots and Exports ===")
        curve_data_for_csv, parameter_data_for_csv = create_extended_summary_plots(
            prepared_all_subjects, resolved_output_dir, create_individual_plots=False,
            corr_weight=corr_weight, circ_space=circ_space,
        )

        if create_csv_exports:
            export_fitted_parameters_csv(parameter_data_for_csv, resolved_output_dir)
            export_fitted_curves_csv(curve_data_for_csv, resolved_output_dir)
            print(f"CSV files saved to: {resolved_output_dir}/csv_exports/")

    if create_pdf_slices:
        print("\n=== Creating PDF Slice Plots ===")
        create_pdf_slice_plots(
            extended_results, global_optimizer, resolved_output_dir,
            circ_space=circ_space,
            optimizer_names=pdf_slice_optimizers,
            n_subjects=pdf_slice_n_subjects,
        )


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create unified subject plots and summary exports from model-fit results."
    )
    parser.add_argument("--results-path", default=None,
                        help="Path to extended_fit_results.pkl (overrides defaults).")
    parser.add_argument("--checkpoint-path", default=None,
                        help="Path to model checkpoint (overrides defaults).")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write plots/exports (defaults to results parent).")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Sample count used for training (default: 20).")
    parser.add_argument("--include-outliers", action=argparse.BooleanOptionalAction, default=True,
                        help="Include outlier trials (default: true).")
    parser.add_argument("--skip-motor-noise", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip motor noise in optimizer (default: true).")
    parser.add_argument("--corr-weight", type=float, default=0.25,
                        help="Correlation weight for optimizer (default: 0.25).")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Limit number of subjects to process.")
    parser.add_argument("--individual-plots", action=argparse.BooleanOptionalAction, default=False,
                        help="Create per-subject plots (default: false).")
    parser.add_argument("--summary-plots", action=argparse.BooleanOptionalAction, default=True,
                        help="Create summary plots (default: true).")
    parser.add_argument("--csv-exports", action=argparse.BooleanOptionalAction, default=True,
                        help="Create CSV exports (default: true).")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help="Base directory for outputs (default: results).")
    parser.add_argument("--circ-space", type=int, default=360, choices=[180, 360],
                        help="Circular space of the fitted data. Use 180 for axial orientation "
                             "data fitted after doubling into model space; plots/CSVs are shown "
                             "back in the original data space. Use 360 when data already match "
                             "model space.")
    parser.add_argument("--pdf-slices", action=argparse.BooleanOptionalAction, default=True,
                        help="Create PDF slice plots (default: true).")
    parser.add_argument("--pdf-slice-optimizer", nargs='*', default=None,
                        choices=['density', 'expectation', 'likelihood', 'crps', 'balanced_crps'],
                        help="Which optimizer(s) to use for PDF slices (default: all available).")
    parser.add_argument("--pdf-slice-n-subjects", type=int, default=3,
                        help="Number of subjects to show per PDF slice plot (default: 3).")

    args = parser.parse_args()

    create_unified_plots_with_summaries(
        n_samples=args.n_samples,
        outliers=args.include_outliers,
        max_subjects=args.max_subjects,
        create_individual_plots=args.individual_plots,
        create_summary_plots=args.summary_plots,
        create_csv_exports=args.csv_exports,
        create_pdf_slices=args.pdf_slices,
        pdf_slice_optimizers=args.pdf_slice_optimizer,
        pdf_slice_n_subjects=args.pdf_slice_n_subjects,
        skip_motor_noise=args.skip_motor_noise,
        corr_weight=args.corr_weight,
        results_path=args.results_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        circ_space=args.circ_space,
    )


if __name__ == "__main__":
    main()
