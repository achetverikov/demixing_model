#!/usr/bin/env python3
"""
Demixing Model Prediction Generator

This script reads parameter combinations from CSV/Arrow, evaluates the packaged
production surrogate (WNM by default), or explicitly uses the historical surface
network / stored averaged surfaces, and writes prediction curves for Python or R
analysis.
"""

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import argparse
import sys
import re
from pathlib import Path
from typing import Optional, Tuple

# Support direct script execution from a checkout without PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_fit_to_data.grid_based_multi_condition_optimizer_jax_loops import (
    GridBasedMultiConditionOptimizer,
    generate_nn_density_asymmetry_batch,
    apply_motor_noise,
    _generate_nn_bias_curve_batch,
)
from shared.config import DENSITY_CURVE_SPEC, config
from shared.mu1_axis import guard_surface_mu1_axis, periodic_integral
from shared import surrogate
from shared.prediction import predictor_from_surrogate
from shared.utils import (SurfaceUnpickler, ensure_averaged_surface_file,
                          gaussian_curve_smoother)

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

        # Same smoother as the mu1 path, from the one shared implementation.
        # This used to be a second inline copy that convolved via `correlate`;
        # the kernel is symmetric, so the two agree exactly (checked in
        # tests/test_curve_smoother.py) and the duplicate bought nothing.
        if apply_smoothing:
            return gaussian_curve_smoother(asymmetry, smoothing_sigma)
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
    
    # Vectorized circular SD computation for all surfaces and feature values
    # Compute mean cos and sin using periodic quadrature (sum x cell width)
    # prob_profiles: (n_surfaces, n_mu1_bias, n_feat_vals)
    # cos_angles: (n_mu1_bias,) -> broadcast to (1, n_mu1_bias, 1)
    mass = periodic_integral(prob_profiles, axis=1)  # Shape: (n_surfaces, n_feat_vals)
    mean_cos = periodic_integral(prob_profiles * cos_angles[None, :, None], axis=1)  # Shape: (n_surfaces, n_feat_vals)
    mean_sin = periodic_integral(prob_profiles * sin_angles[None, :, None], axis=1)  # Shape: (n_surfaces, n_feat_vals)

    # Compute circular standard deviations. Dividing by the column's own mass
    # keeps the resultant a true mean resultant length rather than relying on
    # the columns being exactly rectangle-normalized (MODEL_PIPELINE_FOR_AGENTS
    # D.10); numerically a no-op for normalized surfaces.
    r = jnp.sqrt(mean_cos**2 + mean_sin**2) / jnp.where(mass > 0, mass, jnp.nan)
    r_safe = jnp.minimum(jnp.maximum(r, 1e-10), 1.0 - 1e-10)  # clamp to (0, 1)
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
    # The directory carries the observer model -- the filename holds only the SDs
    # -- so n_samples cannot select a file here and must not be taken as evidence
    # of one. It is checked against the directory instead: passing 20 while
    # reading a 100-sample directory used to return that directory's arrays
    # labelled 20, byte-identical to the same call with 100.
    # An exact token, not containment: "120samples" contains "20samples", and a
    # directory renamed "..._100samples_copy_20samples" would pass a containment
    # check and export n=100 arrays under an n=20 label.
    directory_name = Path(surfaces_dir).name
    tokens = re.findall(r"(?<![0-9])(\d+)samples(?![0-9])", directory_name)
    distinct = sorted(set(tokens))
    if len(distinct) > 1:
        raise ValueError(
            f"the averaged-surface directory {directory_name!r} names more than one observer "
            f"model ({distinct}), so it cannot identify which produced its surfaces. Rename "
            "it, or pass a directory whose name is unambiguous.")
    if str(n_samples) not in tokens:
        raise ValueError(
            f"n_samples={n_samples} does not match the averaged-surface directory "
            f"{directory_name!r}, whose sample-count tokens are {tokens or 'none'}. The "
            "directory is what identifies the observer model here -- the surfaces are keyed "
            "on disk by it, so the argument cannot select them and would only mislabel the "
            "output.")

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
        guard_surface_mu1_axis(surface_obj, source=str(file_path))

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
                              skip_motor_noise: bool = False,
                              use_nn_surfaces: Optional[bool] = None,
                              averaged_surfaces_dir: Optional[str] = None,
                              explicit_checkpoint_path: Optional[str] = None,
                              surface_source: str = "model"):
    """Generate prediction curves from the packaged model or stored surfaces.

    surface_source="model" uses the current production surrogate, which is WNM.
    "nn" explicitly selects the historical surface network and "raw" loads
    averaged simulation surfaces, retaining the separate mu2 outputs.
    use_nn_surfaces is accepted only as a compatibility alias: True means the
    current packaged model, False means raw surfaces.
    """
    if use_nn_surfaces is not None:
        compatibility_source = "model" if use_nn_surfaces else "raw"
        if surface_source != "model" and surface_source != compatibility_source:
            raise ValueError(
                "use_nn_surfaces and surface_source request different prediction sources")
        surface_source = compatibility_source
    if surface_source not in {"model", "nn", "raw"}:
        raise ValueError(f"unknown surface_source {surface_source!r}")

    print(f"Reading parameters from {input_path}...")
    try:
        if input_path.endswith('.arrow') or input_path.endswith('.parquet'):
            params_df = pd.read_parquet(input_path)
        else:
            params_df = pd.read_csv(input_path)
        print(f"Loaded {len(params_df)} parameter combinations")
    except Exception as exc:
        raise RuntimeError(f"could not read parameter file {input_path}: {exc}") from exc

    required_cols = ['sd_feat1', 'sd_feat2', 'sd_spat']
    if not skip_motor_noise:
        required_cols.append('sd_motor')
    missing_cols = [col for col in required_cols if col not in params_df.columns]
    if missing_cols:
        raise ValueError(
            f"missing required parameter columns {missing_cols}; "
            f"available columns are {list(params_df.columns)}")

    if skip_motor_noise:
        params_df = params_df.copy()
        params_df['sd_motor'] = 0.0

    try:
        motor_values = np.asarray(params_df['sd_motor'], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("sd_motor values must be numeric, finite, and non-negative") from exc
    if not np.all(np.isfinite(motor_values)) or np.any(motor_values < 0):
        raise ValueError(
            f"sd_motor values must be finite and non-negative, got {motor_values.tolist()}")
    params_df = params_df.copy()
    params_df['sd_motor'] = motor_values

    parameters_array = jnp.asarray(
        params_df[['sd_feat1', 'sd_feat2', 'sd_spat', 'sd_motor']].values,
        dtype=jnp.float32)

    feat_diff_grid = jnp.asarray(config.create_grid('feat_diff'), dtype=jnp.float32)
    mu1_bias_grid = jnp.asarray(config.create_grid('mu1_bias'), dtype=jnp.float32)
    n_feat_points = len(feat_diff_grid)
    resolved_checkpoint_path = None
    surrogate_family = None
    resolved_n_samples = n_samples
    mu2_surfaces_batch = None

    density_curves = []
    expectation_curves = []
    sd_curves = []

    if surface_source == "raw":
        if not averaged_surfaces_dir:
            raise ValueError("averaged_surfaces_dir is required with surface_source=raw")
        surrogate_family = "averaged_surfaces"
        mu1_surfaces_list = []
        mu2_surfaces_list = []
        for sf1, sf2, sp in parameters_array[:, :3]:
            mu1_surface, mu2_surface = load_averaged_surface(
                float(sf1), float(sf2), float(sp), n_samples,
                surfaces_dir=averaged_surfaces_dir)
            mu1_surfaces_list.append(mu1_surface)
            mu2_surfaces_list.append(mu2_surface)
        log_surfaces_batch = jnp.stack(mu1_surfaces_list)
        mu2_surfaces_batch = jnp.stack(mu2_surfaces_list)

        if not skip_motor_noise:
            noisy = []
            for surface, motor in zip(log_surfaces_batch, parameters_array[:, 3]):
                noisy.append(apply_motor_noise(surface[None], float(motor))[0]
                             if float(motor) > 0 else surface)
            log_surfaces_batch = jnp.stack(noisy)

        density_curves = generate_nn_density_asymmetry_batch(log_surfaces_batch)
        target_feat_indices = jnp.arange(n_feat_points)
        expectation_curves = _generate_nn_bias_curve_batch(
            log_surfaces_batch, target_feat_indices)
        sd_curves = compute_predicted_sd_curves_batch(
            log_surfaces_batch, feat_diff_grid)
    else:
        if explicit_checkpoint_path:
            resolved_checkpoint_path = Path(explicit_checkpoint_path)
        elif surface_source == "nn":
            resolved_checkpoint_path = surrogate.SURFACE_DEFAULTS[n_samples]
        else:
            resolved_checkpoint_path = surrogate.production_checkpoint(n_samples)

        loaded = surrogate.load_surrogate(
            checkpoint_path=resolved_checkpoint_path, n_samples=n_samples)
        resolved_n_samples = loaded.n_samples
        surrogate_family = loaded.family

        if surface_source == "nn" and loaded.family != surrogate.FAMILY_SURFACE_NN:
            raise ValueError(
                f"surface_source=nn requires a historical surface checkpoint, got "
                f"{loaded.family!r}")

        if loaded.family == surrogate.FAMILY_WNM:
            predictor = predictor_from_surrogate(loaded)
            smoothing_sigma = DENSITY_CURVE_SPEC["density_smoothing_sigma"]
            if smoothing_sigma is None:
                smoothing_sigma = (
                    DENSITY_CURVE_SPEC["emp_density_weights_sd"] / config.feat_diff_step)
            for sf1, sf2, sp, motor in np.asarray(parameters_array):
                rows = jnp.column_stack([
                    jnp.full(feat_diff_grid.shape, sf1),
                    jnp.full(feat_diff_grid.shape, sf2),
                    jnp.full(feat_diff_grid.shape, sp),
                    feat_diff_grid,
                ])
                effective_motor = 0.0 if skip_motor_noise else float(motor)
                mean, _ = predictor.mean_and_resultant(
                    rows, sd_motor=effective_motor)
                density = predictor.smoothed_asymmetry_curve(
                    rows, float(smoothing_sigma), sd_motor=effective_motor)
                sd = predictor.circular_sd(rows, sd_motor=effective_motor)
                expectation_curves.append(np.asarray(mean))
                density_curves.append(np.asarray(density))
                sd_curves.append(np.asarray(sd))
            expectation_curves = np.stack(expectation_curves)
            density_curves = np.stack(density_curves)
            sd_curves = np.stack(sd_curves)
        else:
            optimizer = GridBasedMultiConditionOptimizer(
                checkpoint_path=str(resolved_checkpoint_path),
                condition_datasets=None,
                skip_motor_noise=skip_motor_noise)
            nn_params = parameters_array[:, :3]
            log_surfaces_batch = optimizer._predict_batch_fixed_size(
                nn_params, verbosity=1)
            if not skip_motor_noise:
                noisy = []
                for surface, motor in zip(log_surfaces_batch, parameters_array[:, 3]):
                    noisy.append(apply_motor_noise(surface[None], float(motor))[0]
                                 if float(motor) > 0 else surface)
                log_surfaces_batch = jnp.stack(noisy)
            density_curves = generate_nn_density_asymmetry_batch(log_surfaces_batch)
            target_feat_indices = jnp.arange(n_feat_points)
            expectation_curves = _generate_nn_bias_curve_batch(
                log_surfaces_batch, target_feat_indices)
            sd_curves = compute_predicted_sd_curves_batch(
                log_surfaces_batch, feat_diff_grid)

    if mu2_surfaces_batch is not None:
        target_feat_indices = jnp.arange(n_feat_points)
        mu2_expectation_curves = _generate_mu2_bias_curve_batch(
            mu2_surfaces_batch, target_feat_indices)
        mu2_density_curves = _generate_mu2_density_asymmetry_batch(
            mu2_surfaces_batch, target_feat_indices)
    else:
        mu2_expectation_curves = None
        mu2_density_curves = None

    results_list = []
    for i in range(len(parameters_array)):
        row = {
            'sd_feat1': float(parameters_array[i, 0]),
            'sd_feat2': float(parameters_array[i, 1]),
            'sd_spat': float(parameters_array[i, 2]),
            'sd_motor': float(parameters_array[i, 3]),
            'mu1_density_curve': np.asarray(density_curves[i]).tolist(),
            'mu2_density_curve': (
                np.asarray(mu2_density_curves[i]).tolist()
                if mu2_density_curves is not None
                else [float('nan')] * n_feat_points),
            'mu1_expectation_curve': np.asarray(expectation_curves[i]).tolist(),
            'mu2_expectation_curve': (
                np.asarray(mu2_expectation_curves[i]).tolist()
                if mu2_expectation_curves is not None
                else [float('nan')] * n_feat_points),
            'sd_curve': np.asarray(sd_curves[i]).tolist(),
            'feat_diff_grid': np.asarray(feat_diff_grid).tolist() if i == 0 else None,
            'mu1_bias_grid': np.asarray(mu1_bias_grid).tolist() if i == 0 else None,
            'mu2_bias_grid': config.create_grid('mu2_bias').tolist() if i == 0 else None,
            'feat_diff_range': list(config.feat_diff_range) if i == 0 else None,
            'mu1_bias_range': list(config.mu1_bias_range) if i == 0 else None,
            'mu2_bias_range': list(config.mu2_bias_range) if i == 0 else None,
            'feat_diff_step': config.feat_diff_step if i == 0 else None,
            'mu1_bias_step': config.mu1_bias_step if i == 0 else None,
            'mu2_bias_step': config.mu2_bias_step if i == 0 else None,
            'n_samples': resolved_n_samples if i == 0 else None,
            'surrogate_family': surrogate_family if i == 0 else None,
            'surrogate_artifact': (
                Path(resolved_checkpoint_path).name
                if i == 0 and resolved_checkpoint_path is not None else None),
            'skip_motor_noise': skip_motor_noise if i == 0 else None,
            'has_mu2_data': mu2_expectation_curves is not None if i == 0 else None,
        }
        results_list.append(row)

    results_df = pd.DataFrame(results_list)
    print(f"Saving results to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if output_path.endswith('.arrow') or output_path.endswith('.parquet'):
        results_df.to_parquet(output_path, index=False)
    else:
        results_df.to_csv(output_path, index=False)
    print(
        f"Generated predictions for {len(parameters_array)} parameter combinations "
        f"with {surrogate_family}.")

def main():
    """Parse CLI args and run surface simulation."""
    parser = argparse.ArgumentParser(description='Generate Demixing Model predictions from CSV/Arrow parameters')
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
    parser.add_argument('--surface-source', choices=['model', 'nn', 'raw'], default='model',
                       help='model: use the current packaged production surrogate (default, WNM); '
                            'nn: explicitly use the historical surface network; '
                            'raw: load averaged simulation surfaces and include mu2 outputs')
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
        averaged_surfaces_dir=args.averaged_surfaces_dir,
        explicit_checkpoint_path=args.checkpoint_path,
        surface_source=args.surface_source,
    )


if __name__ == '__main__':
    main()
