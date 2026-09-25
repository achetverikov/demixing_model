#!/usr/bin/env python3
"""
Create Unified Subject and Summary Plots

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
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
import jax

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_fit_to_data.grid_based_multi_condition_optimizer_jax_loops import (
    GridBasedMultiConditionOptimizer,
)
from model_fit_to_data.run_fingerprint import read_fingerprint_sidecar
from shared import surrogate
from shared.config import config
from shared.mu1_axis import mu1_cell_width, periodic_integral, sign_masks
from shared.prediction import (mixture_plot_curves, pooled_bias_weighted_crps,
                               predictor_from_surrogate)
from shared import seed_manager
from shared.utils import (
    KDE_WRAPS,
    compute_target_bias_rolling_curve_core,
    resolve_input_path,
    resolve_results_path,
)

# Initialize seed manager
seed = seed_manager.SeedManager(quiet=True)

# Global motor noise kernel cache to avoid recomputing kernels across subjects
global_motor_kernel_cache = {}

RESULTS_DIR = "results"
MODEL_FIT_RESULTS_PREFIX = "model_fit_to_data_results_v2"

# Colorblind-friendly method palette. Keep likelihood and CRPS visually separated
# because they are often compared directly in the summary plots.
OPTIMIZER_COLORS = {
    'likelihood': '#0072B2',   # blue
    'crps': '#E69F00',         # orange
    'expectation': '#CC79A7',  # reddish purple
    'density': '#009E73',      # green
    'balanced_crps': '#D55E00',  # vermillion
    'bias_weighted_crps': '#56B4E9',  # sky blue
    'smoothed_exp': '#F0E442',  # yellow
}
OPTIMIZER_LABELS = {
    'likelihood': 'Likelihood',
    'crps': 'CRPS',
    'expectation': 'Expectation',
    'density': 'Density',
    'balanced_crps': 'Balanced CRPS',
    'bias_weighted_crps': 'Bias-weighted CRPS',
    'smoothed_exp': 'Smoothed Exp',
}
def _angle_display_scale(circ_space: int = 360) -> float:
    """Scale angular model-space values back to the data circular space."""
    return circ_space / (2 * config.feat_diff_range[1])


def _pooled_bias_weighted_crps(log_surfaces, datasets, feat_grid, distance_matrix,
                               weights_sd):
    """Distribution-level BWCRPS for separately fitted report-order surfaces.

    The pooling and scoring live in ``shared.prediction`` so both surrogate
    families share one definition; this wrapper supplies the surface backend's
    own probability convention -- renormalise the sampled grid over the bias axis
    -- and is pinned unchanged against a pre-routing reference in
    ``tests/test_pooled_bwcrps.py``.
    """
    from shared.prediction import pooled_bias_weighted_crps

    log_surfaces = np.asarray(log_surfaces, dtype=float)
    probabilities = np.exp(log_surfaces - log_surfaces.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return pooled_bias_weighted_crps(probabilities, datasets, feat_grid,
                                     distance_matrix, weights_sd)


SD_N_BINS = 18


def _feat_diff_bin_edges(n_bins=SD_N_BINS):
    """Edges and centers of the feature-difference bins used by both SD curves."""
    feat_min, feat_max = config.feat_diff_range[0], config.feat_diff_range[1]
    bin_edges = np.linspace(feat_min, feat_max, n_bins + 1)
    return bin_edges, (bin_edges[:-1] + bin_edges[1:]) / 2


def _feat_diff_bin_indices(feat_diff_values, n_bins=SD_N_BINS):
    """Assign trials to SD bins.

    Shared by the empirical curve and by the model-pooling weights so the two SD
    estimators aggregate over exactly the same partition of trials.
    """
    bin_edges, _ = _feat_diff_bin_edges(n_bins)
    indices = np.digitize(np.asarray(feat_diff_values, dtype=float), bin_edges) - 1
    return np.clip(indices, 0, n_bins - 1)


def compute_empirical_sd_curve(feat_diff_values, bias_values, n_bins=SD_N_BINS):
    """Empirical circular SD of the report bias, binned by feature difference.

    Small-sample correction: the sample resultant length is biased *upward*
    (``E[r̄²] = ρ² + (1−ρ²)/n`` — a handful of unit vectors cannot cancel), and
    because the SD is ``sqrt(-2 ln r)`` that propagates to an *under*-estimate of
    the SD: roughly −27% at n=3, −8% at n=10, −3% at n=25, −1% at n=50, and
    close to scale-free in true SD over 10–60°. The model-side estimator is an
    integral with no sampling error, so uncorrected this reads on the plot as
    "the model overestimates variability". We therefore substitute the unbiased
    resultant ``ρ̂² = (n·r̄² − 1)/(n − 1)`` for the raw ``r̄²``: Kutil (2012),
    Statistics 46(4) 549–561, eq. (5), derived from iid alone (no von Mises
    assumption). Note it is unbiased in ρ², NOT in ρ — (Eρ̂')² = E(ρ̂'²) − V(ρ̂'),
    so a small downward residual survives at very low n (~6% at n=3, vs ~27%
    uncorrected). Kutil's eq. (28) removes more of it at the cost of variance;
    not used here. Bias is traded for variance, which is the right trade only
    because these curves are averaged across subjects before being read.

    Bins whose ``ρ̂²`` is non-positive are consistent with a uniform bias
    distribution, for which this parameterization of circular SD is unbounded;
    they are returned as NaN rather than clamped to a large finite value (the
    old ``maximum(r, 1e-10)`` floor would have plotted them at ~389°).

    The correction removes bias, not variance — low-count bins remain noisy,
    which is why the per-bin trial counts are returned alongside.

    Returns:
        (bin_centers, sds, counts), all length ``n_bins``.
    """
    _, bin_centers = _feat_diff_bin_edges(n_bins)
    bin_indices = _feat_diff_bin_indices(feat_diff_values, n_bins)
    bias_values = np.asarray(bias_values, dtype=float)

    empirical_sds = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for bin_idx in np.unique(bin_indices):
        bin_bias_values = bias_values[bin_indices == bin_idx]
        n = len(bin_bias_values)
        bin_counts[bin_idx] = n

        if n < 2:
            continue

        angles_rad = np.radians(bin_bias_values)
        r2 = np.mean(np.cos(angles_rad)) ** 2 + np.mean(np.sin(angles_rad)) ** 2
        r2_unbiased = (n * r2 - 1.0) / (n - 1.0)
        if r2_unbiased <= 0:
            continue

        r = np.sqrt(min(r2_unbiased, 1.0))  # ==1 for identical values -> SD 0
        empirical_sds[bin_idx] = np.degrees(np.sqrt(-2 * np.log(r)))

    return bin_centers, empirical_sds, bin_counts


def compute_feat_bin_weights(feat_diff_values, feat_vals, n_bins=SD_N_BINS):
    """Per-bin mixture weights over model feature columns, taken from the data.

    The empirical SD in a bin is the spread of *every trial in that bin pooled
    together*, so the comparable model quantity is the spread of the MIXTURE of
    the per-column bias densities, mixed in the proportion the trials actually
    occupy. Reading those proportions off the data — rather than assuming trials
    fill the bin uniformly — is what makes this safe for designs where feature
    difference takes a handful of discrete values instead of spanning the bin
    (Moors): there the weight collapses onto the single column carrying the
    trials, the pooled model SD reduces to the unpooled one, and the model is
    not inflated by columns no trial ever visited. For continuously sampled
    designs the weight spreads over the ~5 columns a bin covers and reproduces
    the same law-of-total-variance broadening the empirical estimator has.

    Returns:
        (n_bins, len(feat_vals)); rows sum to 1, or to 0 for bins with no trials.
    """
    feat_vals = np.asarray(feat_vals, dtype=float)
    feat_diff_values = np.asarray(feat_diff_values, dtype=float)
    bin_indices = _feat_diff_bin_indices(feat_diff_values, n_bins)

    # Same nearest-column mapping the model SD curve uses on its feat_vals.
    columns = np.round(
        (feat_diff_values - config.feat_diff_range[0]) / config.feat_diff_step
    ).astype(int)
    columns = np.clip(columns, 0, len(feat_vals) - 1)

    weights = np.zeros((n_bins, len(feat_vals)), dtype=float)
    np.add.at(weights, (bin_indices, columns), 1.0)
    totals = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)




def compute_predicted_sd_curves_batch(log_surfaces_batch, feat_vals):
    """Batch version: Compute predicted SD curves for multiple surfaces at once - fully vectorized.

    This is the FINE-GRID model SD, one value per 2° feature column, and it is
    what `fitted_curves.csv` exports as `sd_deg`. It is NOT the curve plotted
    against the empirical SD — see compute_pooled_predicted_sd_curves_batch,
    which pools columns into the empirical bins so the two are comparable.

    The sine/cosine moments are periodic integrals (plain sum × cell width),
    matching the rectangle rule the NN columns are normalized under, and are
    divided by the mixture mass explicitly so the estimator does not depend on
    that normalization holding (MODEL_PIPELINE_FOR_AGENTS.md D.10; before the
    circularity fix these were trapezoidal integrals over x=grid, which
    half-weighted the endpoint bins and spanned 358° of the 360° period, and the
    missing division then biased the model SD upward).
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

    # Vectorized circular SD computation for all surfaces and feature values.
    # prob_profiles: (n_surfaces, n_mu1_bias, n_feat_vals)
    # cos_angles: (n_mu1_bias,) -> broadcast to (1, n_mu1_bias, 1)
    mass = periodic_integral(prob_profiles, axis=1)
    mean_cos = periodic_integral(prob_profiles * cos_angles[None, :, None], axis=1)
    mean_sin = periodic_integral(prob_profiles * sin_angles[None, :, None], axis=1)

    # Compute circular standard deviations
    r = jnp.sqrt(mean_cos**2 + mean_sin**2) / jnp.where(mass > 0, mass, jnp.nan)
    r_safe = jnp.minimum(jnp.maximum(r, 1e-10), 1.0 - 1e-10)  # clamp to (0, 1)
    circular_sds = jnp.degrees(jnp.sqrt(-2 * jnp.log(r_safe)))  # Shape: (n_surfaces, n_feat_vals)

    return circular_sds


def compute_predicted_sd_curves_batch_pooled(log_surfaces_batch, bin_weights_batch):
    """Model SD pooled into the empirical SD's feature-difference bins.

    Row 3 of the plots compares a model SD against an empirical SD computed over
    ~10 model-degree bins of trials. Evaluating the model at a single 2° column
    is not the same quantity: pooling trials whose bias means differ adds a
    between-column term to the empirical spread that the single-column model
    value has none of, inflating the data curve wherever the bias curve is
    steep. Mixing the model columns with the bin's own trial distribution
    (compute_feat_bin_weights) puts that term on both sides.

    Because the weights come from the data, discrete-level designs (Moors)
    degenerate correctly: a bin whose trials all sit at one feature difference
    mixes exactly one column and the pooled value equals the unpooled one.

    The computation itself lives in ``shared.prediction.SurfacePredictor`` so the
    two surrogate families share one definition of this estimator rather than
    two that can drift. Pinned unchanged against a pre-routing reference in
    ``tests/test_plot_estimators.py``.

    Args:
        log_surfaces_batch: (n_surfaces, n_mu1_bias, n_feat_diff) log densities.
        bin_weights_batch: (n_surfaces, n_bins, n_feat_vals) mixture weights,
            rows summing to 1 (or 0 for bins with no trials → NaN output).

    Returns:
        (n_surfaces, n_bins) circular SDs in model degrees.
    """
    from shared.prediction import SurfacePredictor

    predictor = SurfacePredictor(log_surfaces_batch, n_samples=0, artifact="plots")
    return predictor.pooled_circular_sd(bin_weights_batch)


def _sanitize_result_key_part(value: str) -> str:
    """Match fit_model_to_data.py's result-key sanitization."""
    return re.sub(r'[^\w]', '_', str(value)).strip('_')


def _canonical_condition_key(key: str) -> str:
    parts = str(key).split('#')
    if len(parts) < 3:
        return str(key)
    subject, experiment = parts[0], parts[1]
    condition = "#".join(parts[2:])
    return (
        f"{_sanitize_result_key_part(subject)}#"
        f"{_sanitize_result_key_part(experiment)}#"
        f"{_sanitize_result_key_part(condition)}"
    )


def canonicalize_result_keys(results: Dict) -> Dict:
    canonical = {}
    source_keys = {}
    for key, entry in results.items():
        ckey = _canonical_condition_key(key)
        if ckey in canonical and source_keys[ckey] != str(key):
            raise ValueError(
                "Result-key collision after sanitization: "
                f"{source_keys[ckey]!r} and {str(key)!r} both map to {ckey!r}"
            )
        canonical[ckey] = entry
        source_keys[ckey] = str(key)
    return canonical


def _surface_plot_bundle(optimizer, params_batch, motor_noise, feat_vals,
                         bin_weights_batch):
    """Run the historical surface curve path without changing its arithmetic."""
    from grid_based_multi_condition_optimizer_jax_loops import (
        _generate_nn_bias_curve_batch,
        apply_motor_noise_with_precomputed_kernel,
        create_motor_noise_kernel_fft,
        generate_nn_density_asymmetry_batch,
    )

    log_surfaces = optimizer._predict_batch_fixed_size(params_batch, verbosity=0)
    motors = jnp.asarray(motor_noise)
    unique_motor_noise, inverse_indices = jnp.unique(motors, return_inverse=True)
    n_mu1_bias = log_surfaces.shape[1]
    print(f"Found {len(unique_motor_noise)} unique motor noise values: {unique_motor_noise}")
    surfaces_with_noise = jnp.zeros_like(log_surfaces)
    for index, sd_motor in enumerate(unique_motor_noise):
        mask = inverse_indices == index
        print(f"  Processing motor noise {sd_motor:.1f}: {jnp.sum(mask)} surfaces")
        if sd_motor > 0:
            key = (float(sd_motor), n_mu1_bias)
            if key not in global_motor_kernel_cache:
                global_motor_kernel_cache[key] = create_motor_noise_kernel_fft(
                    sd_motor, n_mu1_bias)
            noisy = apply_motor_noise_with_precomputed_kernel(
                log_surfaces[mask], global_motor_kernel_cache[key])
            surfaces_with_noise = surfaces_with_noise.at[mask].set(noisy)
        else:
            surfaces_with_noise = surfaces_with_noise.at[mask].set(log_surfaces[mask])
    log_surfaces = surfaces_with_noise

    print("Computing bias curves for all surfaces...")
    bias = _generate_nn_bias_curve_batch(log_surfaces, jnp.arange(len(feat_vals)))
    print("Computing density asymmetry curves for all surfaces...")
    asymmetry = generate_nn_density_asymmetry_batch(log_surfaces)
    print("Computing standard deviation curves for all surfaces...")
    sd = compute_predicted_sd_curves_batch(log_surfaces, feat_vals)
    pooled_sd = compute_predicted_sd_curves_batch_pooled(log_surfaces, bin_weights_batch)
    return {
        "bias": bias, "asymmetry": asymmetry, "sd": sd, "pooled_sd": pooled_sd,
        "distributions": log_surfaces, "distribution_kind": "log_density",
    }


def _mixture_probability_surfaces(predictor, params_batch, motor_noise, feat_vals):
    """Exact reporting-cell masses, shaped like a surface for shared pooling."""
    outputs = []
    feat_vals = jnp.asarray(feat_vals, dtype=jnp.float32)
    for params, sd_motor in zip(np.asarray(params_batch), np.asarray(motor_noise)):
        rows = jnp.column_stack([
            jnp.full(feat_vals.shape, params[0]),
            jnp.full(feat_vals.shape, params[1]),
            jnp.full(feat_vals.shape, params[2]),
            feat_vals,
        ])
        # cell_probabilities is (feature, bias); the shared report-order scorer
        # consumes (bias, feature), matching the historical surface orientation.
        outputs.append(np.asarray(predictor.cell_probabilities(
            rows, sd_motor=float(sd_motor))).T)
    return np.stack(outputs)


def _mixture_plot_bundle(predictor, params_batch, motor_noise, feat_vals,
                         bin_weights_batch, feature_operators,
                         density_bandwidths, density_curve_spec):
    """Direct fitted WNM curves in bounded observed-operator batches."""
    chunks = []
    for start in range(0, len(params_batch), 16):
        stop = min(start + 16, len(params_batch))
        chunks.append(mixture_plot_curves(
            predictor, params_batch[start:stop], feat_vals,
            bin_weights=bin_weights_batch[start:stop],
            sd_motor_by_row=motor_noise[start:stop],
            emp_density_weights_sd=density_curve_spec["emp_density_weights_sd"],
            density_smoothing_sigma=density_curve_spec["density_smoothing_sigma"],
            feature_operators=np.stack(feature_operators[start:stop]),
            density_bandwidths=density_bandwidths[start:stop],
        ))
    return {
        name: np.concatenate([chunk[name] for chunk in chunks])
        for name in ("bias", "asymmetry", "sd", "pooled_sd")
    }


def load_extended_results(results_path: str) -> Dict:
    """Load extended batch fitting results."""
    print(f"Loading extended results from {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    results = canonicalize_result_keys(results)
    print(f"Loaded {len(results)} extended results")
    return results


def _resolve_plot_circ_space(extended_results: Dict,
                             requested: Optional[int] = None) -> int:
    """Use the circular period recorded by the fit, validating any CLI override."""
    stored = {
        float(result['circ_space'])
        for result in extended_results.values()
        if result is not None and result.get('circ_space') is not None
    }
    if len(stored) > 1:
        raise ValueError(
            "plotting requires one circular period per result set, but the fitted "
            f"results contain {sorted(stored)}")
    if stored:
        recorded = stored.pop()
        if requested is not None and not np.isclose(float(requested), recorded):
            raise ValueError(
                f"--circ-space={requested} disagrees with the fitted result period "
                f"{recorded:g}; plotting in a different physical angular scale would "
                "mislabel curves and fitted SDs")
        if not np.isclose(recorded, round(recorded)):
            raise ValueError(f"unsupported non-integer circular period {recorded:g}")
        return int(round(recorded))
    return 360 if requested is None else int(requested)


def display_condition_label(value: object) -> str:
    text = str(value)
    if "___" in text:
        return text.replace("___", " - ")
    return text


def _result_plot_identity(condition_name: str, result: Dict) -> Tuple[str, str, str]:
    """Return subject, experiment, and condition labels for plotting.

    Bundle-native WNM results use opaque analysis-cell IDs, so their scientific
    labels must come from ``analysis_cell_values``. Legacy CSV/surface results
    predate that metadata and retain their historical key parsing as a fallback.
    Report-order cells are mapped to the established ``*_first``/``*_second``
    experiment labels so the existing pooled report-order comparison still pairs
    the two fitted distributions.
    """
    values = result.get('analysis_cell_values') or {}
    if 'subject_id' in values and 'experiment_id' in values:
        subject_id = str(values['subject_id'])
        experiment = str(values['experiment_id'])
        report_order = values.get('report_order')
        if report_order is not None and str(report_order).lower() not in {'', 'nan', 'none'}:
            try:
                order = int(report_order)
            except (TypeError, ValueError):
                order = None
            if order == 1:
                experiment = f"{experiment}_first"
            elif order == 2:
                experiment = f"{experiment}_second"
            else:
                experiment = f"{experiment}_report_{report_order}"
        noise_condition = str(
            values.get('condition_id', result.get('condition', condition_name))
        )
        return subject_id, experiment, noise_condition

    # Legacy result keys: "S12#color_1#high - low".
    if '#' in condition_name:
        parts = condition_name.split('#')
        if len(parts) >= 3:
            return parts[0], parts[1], '#'.join(parts[2:])

    # Older result keys: "S1.color.1_low - high".
    parts = condition_name.split('.')
    if len(parts) < 3:
        return str(condition_name), 'unknown', 'unknown'
    subject_id = parts[0]
    exp_part = parts[1]
    noise_part = parts[2]
    if '_' in noise_part:
        exp_num, noise_condition = noise_part.split('_', 1)
    else:
        exp_num, noise_condition = noise_part, 'unknown'
    return subject_id, f"{exp_part}.{exp_num}", noise_condition


def organize_results_by_subject(extended_results: Dict) -> Dict:
    """Organize results by subject and experiment using bundle metadata when available."""
    subjects = defaultdict(lambda: defaultdict(list))

    for condition_name, result in extended_results.items():
        if result is None:
            continue
        subject_id, experiment, noise_condition = _result_plot_identity(
            condition_name, result)
        subjects[subject_id][experiment].append({
            'condition_name': condition_name,
            'noise_condition': noise_condition,
            'result': result
        })

    return subjects


def prepare_all_subjects_data(
    subjects_data: Dict,
    prediction_backend,
    density_curve_spec: Optional[Dict] = None,
) -> Dict:
    """
    Batch data preparation routine - processes all subjects at once for maximum efficiency.

    One selection caveat (see MODEL_PIPELINE_FOR_AGENTS.md S10.1):
    - The available-optimizer list per (subject, experiment) is read from the
      FIRST condition's result only and reused for every condition; a method
      fitted only in a later condition is never predicted or plotted.

    Surface fits retain their historical grid computations. WNM fits use direct
    analytic curves and exact reporting-cell masses, with the observed-design
    operator and KDE bandwidth stored by the fit.

    Returns:
        Dictionary with all precomputed data for all subjects
    """

    print("Preparing data for all subjects in batch...")
    backend_family = getattr(prediction_backend, "family", surrogate.FAMILY_SURFACE_NN)
    if hasattr(prediction_backend, "identity"):
        surrogate_identity = prediction_backend.identity().as_dict()
    else:
        surrogate_identity = {
            "dm_version": surrogate.dm_version(
                surrogate.FAMILY_SURFACE_NN,
                Path(prediction_backend.checkpoint_path).name,
            ),
            "surrogate_family": surrogate.FAMILY_SURFACE_NN,
            "surrogate_artifact": Path(prediction_backend.checkpoint_path).name,
            "surrogate_n_samples": np.nan,
        }
    if density_curve_spec is None:
        density_curve_spec = {
            "emp_density_weights_sd": float(
                getattr(prediction_backend, "emp_density_weights_sd", 20.0)),
            "density_smoothing_sigma": getattr(
                prediction_backend, "density_smoothing_sigma", None),
        }

    # Collect all parameters and data across ALL subjects
    all_params_3d = []
    all_motor_noise = []
    all_feature_operators = []
    all_density_bandwidths = []
    param_mapping = {}  # Maps (subject_id, experiment, condition, optimizer) -> index in batch
    # Maps (subject_id, experiment, condition) -> feature differences of the
    # trials the empirical SD curve will be built from, used to pool the model
    # SD over the same bins.
    condition_feat_diff = {}
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

            # Which trials the empirical SD will use depends on the branch taken
            # further down (saved curves -> first stored result only; recompute
            # -> every dataset of the condition). Mirror that choice here so the
            # pooling weights are built from exactly the same trials.
            saved_curves_branch = bool(sample_result.get('empirical_curves'))

            # Collect parameters for all condition×optimizer combinations
            for condition_name, cond_data_list in noise_conditions.items():
                cond_data = cond_data_list[0]
                result = cond_data['result']

                sd_sources = ([cond_data_list[0]['result']] if saved_curves_branch
                              else [cd['result'] for cd in cond_data_list])
                feat_diff_arrays = [np.asarray(src['data_df'])[:, 0]
                                    for src in sd_sources if 'data_df' in src]
                if feat_diff_arrays:
                    condition_feat_diff[(subject_id, experiment, condition_name)] = (
                        np.concatenate(feat_diff_arrays))

                for opt in available_optimizers:
                    param_key = f'{opt}_fitted_params'
                    if param_key in result:
                        params = result[param_key]
                        params_3d = params[:3] if len(params) >= 3 else params
                        motor_noise = params[3] if len(params) >= 4 else 0.0

                        all_params_3d.append(params_3d)
                        all_motor_noise.append(motor_noise)
                        if backend_family == surrogate.FAMILY_WNM:
                            empirical = result.get("empirical_curves", {})
                            if "feature_operator" not in empirical or "density_bandwidth" not in empirical:
                                raise ValueError(
                                    f"WNM fit {result.get('condition', condition_name)!r} lacks "
                                    "its stored feature_operator or density_bandwidth; direct curves "
                                    "cannot reproduce the fitted objective without both")
                            all_feature_operators.append(empirical["feature_operator"])
                            all_density_bandwidths.append(empirical["density_bandwidth"])
                        param_mapping[(subject_id, experiment, condition_name, opt)] = batch_idx
                        batch_idx += 1

    print(f"Collected {len(all_params_3d)} parameter combinations from all subjects")

    zero_weights = np.zeros((SD_N_BINS, len(feat_vals)))
    unique_weight_rows = [zero_weights]
    weight_row_of_condition = {}
    for condition_key, feat_diff_vals in condition_feat_diff.items():
        weight_row_of_condition[condition_key] = len(unique_weight_rows)
        unique_weight_rows.append(compute_feat_bin_weights(feat_diff_vals, feat_vals))
    weight_rows = jnp.asarray(np.stack(unique_weight_rows))
    weight_index = jnp.asarray([
        weight_row_of_condition.get(key[:3], 0)
        for key, _ in sorted(param_mapping.items(), key=lambda item: item[1])
    ])

    # One batch over every subject/condition/objective, through its own family.
    if all_params_3d:
        print(f"Batch computing {len(all_params_3d)} parameter combinations for ALL subjects...")
        params_batch = jnp.array(all_params_3d)
        if backend_family == surrogate.FAMILY_WNM:
            bundle = _mixture_plot_bundle(
                prediction_backend, params_batch, all_motor_noise, feat_vals,
                weight_rows[weight_index], all_feature_operators,
                np.asarray(all_density_bandwidths), density_curve_spec)
        else:
            bundle = _surface_plot_bundle(
                prediction_backend, params_batch, all_motor_noise, feat_vals,
                weight_rows[weight_index])

        all_bias_curves = bundle["bias"]
        all_asymm_curves = bundle["asymmetry"]
        all_predicted_sd = bundle["sd"]
        all_predicted_sd_pooled = bundle["pooled_sd"]
        if backend_family == surrogate.FAMILY_SURFACE_NN:
            distributions = bundle["distributions"]
            distance_matrix = prediction_backend.D_circ_matrix
        else:
            bias_grid = np.asarray(config.create_grid("mu1_bias"))
            difference = np.abs(bias_grid[:, None] - bias_grid[None, :])
            distance_matrix = np.minimum(difference, 360.0 - difference)

        # Report-order fits have separate parameters but enter the comparison as
        # one color_2 distribution. Mix their predicted surfaces with the same
        # feature-distance kernel support used for the empirical target, then apply
        # the nonlinear energy score once. Only the scalar is retained.
        report_order_bwcrps = {}
        for key, first_idx in param_mapping.items():
            subject_id, experiment, condition_name, opt = key
            if experiment != "color_2_first":
                continue
            second_key = (subject_id, "color_2_second", condition_name, opt)
            if second_key not in param_mapping:
                continue
            first_result = subjects_structure[subject_id]["color_2_first"][
                "noise_conditions"][condition_name][0]["result"]
            second_result = subjects_structure[subject_id]["color_2_second"][
                "noise_conditions"][condition_name][0]["result"]
            if "data_df" not in first_result or "data_df" not in second_result:
                continue
            indices = [first_idx, param_mapping[second_key]]
            if backend_family == surrogate.FAMILY_WNM:
                selected = _mixture_probability_surfaces(
                    prediction_backend, np.asarray(params_batch)[indices],
                    np.asarray(all_motor_noise)[indices], feat_vals)
                scorer = pooled_bias_weighted_crps
            else:
                selected = np.take(distributions, indices, axis=0)
                scorer = _pooled_bias_weighted_crps
            score = scorer(
                selected, [first_result["data_df"], second_result["data_df"]],
                feat_vals, distance_matrix,
                density_curve_spec["emp_density_weights_sd"])
            report_order_bwcrps[key] = score
            report_order_bwcrps[second_key] = score

        print("Mapping results back to subject structure...")
    else:
        report_order_bwcrps = {}

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
                                'predicted_sd': all_predicted_sd[idx],
                                'predicted_sd_pooled': all_predicted_sd_pooled[idx],
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
                optimizer_pooled_bwcrps = {}
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
                    pooled_key = (subject_id, experiment, noise_cond, opt)
                    if pooled_key in report_order_bwcrps:
                        optimizer_pooled_bwcrps[opt] = report_order_bwcrps[pooled_key]

                experiment_parameters[noise_cond] = {
                    'params': optimizer_params,
                    'losses': optimizer_losses,
                    'eval_losses': optimizer_eval_losses,
                    'pooled_bwcrps': optimizer_pooled_bwcrps,
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
                            data_array[:, 0], data_array[:, 1], n_bins=SD_N_BINS)

                    bias_curve_smoothed = saved_curves.get('target_bias_curve')
                    if bias_curve_smoothed is None and 'data_df' in result:
                        data_array = np.array(result['data_df'])
                        bias_curve_smoothed = compute_target_bias_rolling_curve_core(
                            jnp.asarray(data_array[:, 0]),
                            jnp.asarray(data_array[:, 1]),
                            jnp.asarray(saved_curves['density_feat_grid']),
                        )

                    experiment_empirical_curves[noise_cond] = {
                        'bias': saved_curves['target_bias'],
                        'bias_weights': saved_curves.get('bias_weights'),
                        'asymmetry': (
                            saved_curves.get('matched_density_target', saved_curves['target_density'])
                            if backend_family == surrogate.FAMILY_WNM
                            else saved_curves['target_density']),
                        'sd': empirical_sd,
                        'bias_grid': saved_curves['density_feat_grid'][saved_curves['bias_feat_indices']],
                        'asymm_grid': saved_curves['density_feat_grid'],
                        # Rolling-mean (Gaussian-weighted) bias curve, same grid as density's
                        # asymmetry curve. For older results that predate this saved field,
                        # it is reconstructed above from the stored trial data.
                        'bias_smoothed': bias_curve_smoothed,
                        'bias_smoothed_grid': saved_curves['density_feat_grid'] if bias_curve_smoothed is not None else None,
                    }

            elif all_condition_datasets:
                prediction_backend.update_dataset(all_condition_datasets)
                dataset_names_list = list(all_condition_datasets.keys())

                # Process each condition using precomputed curves and individual empirical SD
                for noise_cond in noise_conditions.keys():
                    if noise_cond in condition_data_mapping:
                        mapping = condition_data_mapping[noise_cond]

                        # Get optimizer curves for this condition's datasets
                        condition_indices = [dataset_names_list.index(name) for name in mapping['dataset_names'] if name in dataset_names_list]

                        if condition_indices:
                            condition_indices_array = jnp.array(condition_indices)
                            empirical_bias = jnp.mean(prediction_backend.unified_target_bias[condition_indices_array], axis=0)
                            empirical_bias_weights = jnp.sum(prediction_backend.unified_bias_weights[condition_indices_array], axis=0)
                            empirical_asymm = jnp.mean(prediction_backend.unified_target_density[condition_indices_array], axis=0)
                            empirical_bias_smoothed = jnp.mean(prediction_backend.unified_target_bias_curve[condition_indices_array], axis=0)
                        else:
                            empirical_bias = None
                            empirical_bias_weights = None
                            empirical_asymm = None
                            empirical_bias_smoothed = None

                        # Compute empirical SD for this specific condition
                        empirical_sd = None
                        if mapping['raw_data']:
                            condition_array = np.array(mapping['raw_data'])
                            empirical_sd = compute_empirical_sd_curve(condition_array[:, 0], condition_array[:, 1], n_bins=SD_N_BINS)

                        # Get the separate empirical feature grids from optimizer
                        empirical_bias_grid = None
                        empirical_asymm_grid = None

                        if hasattr(prediction_backend, 'unified_feat_indices') and hasattr(prediction_backend, 'feat_diff_grid'):
                            # Bias uses binned grid (via unified_feat_indices)
                            empirical_bias_grid = prediction_backend.feat_diff_grid[prediction_backend.unified_feat_indices]
                            # Asymmetry uses full grid
                            empirical_asymm_grid = prediction_backend.feat_diff_grid

                        experiment_empirical_curves[noise_cond] = {
                            'bias': empirical_bias,
                            'bias_weights': empirical_bias_weights,
                            'asymmetry': empirical_asymm,
                            'sd': empirical_sd,
                            'bias_grid': empirical_bias_grid,
                            'asymm_grid': empirical_asymm_grid,
                            # Smoothed bias curve shares the full feat_diff grid with asymmetry.
                            'bias_smoothed': empirical_bias_smoothed,
                            'bias_smoothed_grid': empirical_asymm_grid if empirical_bias_smoothed is not None else None,
                        }

            # Store experiment data
            prepared_subject['experiments'][experiment] = {
                'optimizer_curves': experiment_optimizer_curves,
                'empirical_curves': experiment_empirical_curves,
                'parameters': experiment_parameters,
                'available_optimizers': available_optimizers,
                'feat_vals': feat_vals,
                'noise_conditions': list(noise_conditions.keys()),
                'surrogate_identity': surrogate_identity,
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
            param_lines = []
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

                param_lines.append(f'{label}: {param_str}')

            # Plot empirical data: raw 4°-binned circular mean (what "expectation" fits
            # against) and the Gaussian-smoothed rolling-mean curve (what "smoothed_exp"
            # fits against), so the two targets can be compared directly.
            empirical_bias = condition_empirical_curves.get('bias')
            empirical_bias_grid = condition_empirical_curves.get('bias_grid')
            if empirical_bias is not None and empirical_bias_grid is not None:
                empirical_bias = np.asarray(empirical_bias)
                bias_weights = condition_empirical_curves.get('bias_weights')
                valid = (np.asarray(bias_weights) > 0 if bias_weights is not None
                         else np.isfinite(empirical_bias))
                ax1.plot(np.asarray(empirical_bias_grid)[valid] * angle_display_scale,
                        empirical_bias[valid] * angle_display_scale,
                        color='black', marker='o', linestyle='none', markersize=5,
                        alpha=0.8, label='Data (binned)')

            empirical_bias_smoothed = condition_empirical_curves.get('bias_smoothed')
            empirical_bias_smoothed_grid = condition_empirical_curves.get('bias_smoothed_grid')
            if empirical_bias_smoothed is not None and empirical_bias_smoothed_grid is not None:
                ax1.plot(np.array(empirical_bias_smoothed_grid) * angle_display_scale,
                        np.array(empirical_bias_smoothed) * angle_display_scale,
                        color='black', linestyle=':', linewidth=2, alpha=0.9, label='Data (smoothed)')

            ax1.set_xlabel('Feature Difference (°)')
            ax1.set_ylabel('Mu1 Bias (degrees)')
            ax1.set_title(f'{noise_cond}')
            ax1.grid(True, alpha=0.3)

            # Add parameter info as text box
            param_text = '\n'.join(param_lines)
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

            # Model SD is shown bin-pooled, on the empirical curve's own bins, so
            # both sides carry the same between-column mixing (see
            # compute_predicted_sd_curves_batch_pooled). Results predating that
            # field fall back to the fine-grid curve.
            _, sd_bin_centers = _feat_diff_bin_edges()
            display_sd_bin_centers = sd_bin_centers * angle_display_scale

            for opt in available_optimizers:
                if opt not in condition_optimizer_curves:
                    continue

                color = opt_colors.get(opt, 'black')
                label = opt_labels.get(opt, opt.capitalize())

                pooled_sd = condition_optimizer_curves[opt].get('predicted_sd_pooled')
                pooled_sd = None if pooled_sd is None else np.asarray(pooled_sd)
                if pooled_sd is not None and np.any(~np.isnan(pooled_sd)):
                    pooled_display = pooled_sd * angle_display_scale
                    pooled_valid = ~np.isnan(pooled_display)
                    ax3.plot(display_sd_bin_centers[pooled_valid],
                             pooled_display[pooled_valid],
                             color=color, linewidth=2, marker='o', markersize=4,
                             label=f'{label}', linestyle='-')
                else:
                    ax3.plot(display_feat_vals,
                            condition_optimizer_curves[opt]['predicted_sd'] * angle_display_scale,
                            color=color, linewidth=2, label=f'{label}', linestyle='-')

            # Plot empirical standard deviation
            empirical_sd = condition_empirical_curves.get('sd')
            if empirical_sd is not None:
                emp_bin_centers, emp_sd_values = empirical_sd[0], empirical_sd[1]
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
                                 circ_space: int = 360) -> None:
    """Create extended summary plots using preprocessed data.

    Only optimizers present in every prepared result of an
    experiment+condition are aggregated. The intersection is computed before
    the "combined" pseudo-subject is excluded from the statistics, preserving
    the historical plotting behavior without coupling plots to tabular exports.

    Args:
        prepared_all_subjects: Dictionary from prepare_all_subjects_data() with all precomputed curves
        output_dir: Output directory for plots
    """

    print("Creating extended summary plots using preprocessed data...")

    # Organize preprocessed data by experiment and condition
    experiments = organize_preprocessed_results_by_experiment(prepared_all_subjects)

    if len(experiments) == 0:
        print("No experiments found with sufficient data for summary plots")
        return

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
            feat_vals = first_subject_data['feat_vals']
            angle_display_scale = _angle_display_scale(circ_space)
            display_feat_vals = np.array(feat_vals) * angle_display_scale

            # Only aggregate optimizers every subject in this condition actually
            # has. Subjects can be a mix of vintages (e.g. legacy fits from
            # before motor-noise runs excluded "expectation", alongside subjects
            # fit later under the current --include-methods set); trusting just
            # the first subject's optimizer list produces ragged per-optimizer
            # arrays below and corrupts the group summaries.
            common_optimizers = set(first_subject_data['available_optimizers'])
            for prepared_result in prepared_results_list[1:]:
                common_optimizers &= set(prepared_result['experiment_data']['available_optimizers'])
            available_optimizers = [
                opt for opt in first_subject_data['available_optimizers']
                if opt in common_optimizers
            ]

            if not available_optimizers:
                continue

            # Collect precomputed curves from all subjects for this experiment+condition
            stage_bias_curves = {opt: [] for opt in available_optimizers}
            stage_asymm_curves = {opt: [] for opt in available_optimizers}
            # Fine-grid SD remains as a fallback for older prepared results;
            # current results use the bin-pooled twin for comparison with data.
            stage_sd_curves = {opt: [] for opt in available_optimizers}
            stage_sd_pooled_curves = {opt: [] for opt in available_optimizers}

            # Collect empirical curves
            all_empirical_bias = []
            all_empirical_bias_smoothed = []
            all_empirical_asymm = []
            all_empirical_sd = []
            empirical_bias_grid = None
            empirical_bias_smoothed_grid = None
            empirical_asymm_grid = None

            for prepared_result in prepared_results_list:
                subject_id = prepared_result['subject_id']
                experiment_data = prepared_result['experiment_data']
                noise_condition_key = prepared_result['noise_condition']
                noise_condition = display_condition_label(noise_condition_key)

                # Skip the "combined" pseudo-subject from summary statistics so
                # group means/SDs reflect individual observers only.
                if subject_id == "combined":
                    continue

                # Get precomputed optimizer curves for this subject+condition
                optimizer_curves = experiment_data['optimizer_curves'].get(noise_condition_key, {})
                empirical_curves = experiment_data['empirical_curves'].get(noise_condition_key, {})
                for opt in available_optimizers:
                    if opt in optimizer_curves:
                        stage_bias_curves[opt].append(optimizer_curves[opt]['bias'])
                        stage_asymm_curves[opt].append(optimizer_curves[opt]['asymmetry'])
                        if 'predicted_sd' in optimizer_curves[opt]:
                            stage_sd_curves[opt].append(optimizer_curves[opt]['predicted_sd'])
                        if optimizer_curves[opt].get('predicted_sd_pooled') is not None:
                            stage_sd_pooled_curves[opt].append(
                                optimizer_curves[opt]['predicted_sd_pooled'])

                # Collect empirical curves and separate feature grids
                if empirical_curves.get('bias') is not None:
                    empirical_bias = np.asarray(empirical_curves['bias'], dtype=float)
                    bias_weights = empirical_curves.get('bias_weights')
                    if bias_weights is not None:
                        empirical_bias = np.where(np.asarray(bias_weights) > 0,
                                                  empirical_bias, np.nan)
                    all_empirical_bias.append(empirical_bias)
                    if empirical_bias_grid is None:
                        empirical_bias_grid = empirical_curves.get('bias_grid')
                if empirical_curves.get('bias_smoothed') is not None:
                    all_empirical_bias_smoothed.append(empirical_curves['bias_smoothed'])
                    if empirical_bias_smoothed_grid is None:
                        empirical_bias_smoothed_grid = empirical_curves.get('bias_smoothed_grid')
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
                if len(stage_sd_curves[opt]) > 0:
                    stage_sd_curves[opt] = jnp.array(stage_sd_curves[opt])
                if len(stage_sd_pooled_curves[opt]) > 0:
                    stage_sd_pooled_curves[opt] = np.array(stage_sd_pooled_curves[opt])

            # Drop optimizers that had no subjects with data for this condition;
            # stage_bias_curves[opt] stays a plain list when conversion was skipped.
            available_optimizers = [opt for opt in available_optimizers
                                    if not isinstance(stage_bias_curves[opt], list)]

            # Compute statistics for plotting
            avg_stage_bias = {}
            sem_stage_bias = {}
            avg_stage_asymm = {}
            sem_stage_asymm = {}
            avg_stage_sd = {}
            sem_stage_sd = {}
            avg_stage_sd_pooled = {}
            sem_stage_sd_pooled = {}

            for opt in available_optimizers:
                avg_stage_bias[opt] = jnp.mean(stage_bias_curves[opt], axis=0)
                sem_stage_bias[opt] = jnp.std(stage_bias_curves[opt], axis=0) / jnp.sqrt(len(stage_bias_curves[opt]))
                avg_stage_asymm[opt] = jnp.mean(stage_asymm_curves[opt], axis=0)
                sem_stage_asymm[opt] = jnp.std(stage_asymm_curves[opt], axis=0) / jnp.sqrt(len(stage_asymm_curves[opt]))
                if isinstance(stage_sd_curves[opt], jnp.ndarray):
                    avg_stage_sd[opt] = jnp.mean(stage_sd_curves[opt], axis=0)
                    sem_stage_sd[opt] = jnp.std(stage_sd_curves[opt], axis=0) / jnp.sqrt(len(stage_sd_curves[opt]))
                # Bins empty for a subject are NaN, so aggregate them the same
                # way the empirical SD is aggregated below.
                if isinstance(stage_sd_pooled_curves[opt], np.ndarray):
                    pooled = stage_sd_pooled_curves[opt]
                    n_valid = np.sum(~np.isnan(pooled), axis=0).clip(1)
                    avg_stage_sd_pooled[opt] = np.nanmean(pooled, axis=0)
                    sem_stage_sd_pooled[opt] = np.nanstd(pooled, axis=0) / np.sqrt(n_valid)

            # Use the separate binned and rolling-smoothed empirical curves from
            # prepared data. Empty subject-level bins were converted to NaN above,
            # so they do not pull the group binned mean toward zero.
            empirical_bias_binned = None
            empirical_bias_smoothed = None
            emp_asymm = None
            emp_bias_feat_vals = empirical_bias_grid
            emp_bias_smoothed_feat_vals = empirical_bias_smoothed_grid
            emp_asymm_feat_vals = empirical_asymm_grid
            if all_empirical_bias:
                empirical_bias_binned = np.nanmean(np.array(all_empirical_bias), axis=0)
            if all_empirical_bias_smoothed:
                empirical_bias_smoothed = jnp.mean(jnp.array(all_empirical_bias_smoothed), axis=0)
            if all_empirical_asymm:
                emp_asymm = jnp.mean(jnp.array(all_empirical_asymm), axis=0)

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

            # Plot the binned expectation target as points and the actual rolling
            # smoothed-exp target as a line.
            if empirical_bias_binned is not None and emp_bias_feat_vals is not None:
                valid = np.isfinite(empirical_bias_binned)
                ax1.plot(np.asarray(emp_bias_feat_vals)[valid] * angle_display_scale,
                         np.asarray(empirical_bias_binned)[valid] * angle_display_scale,
                         color='black', marker='o', linestyle='none', markersize=5,
                         alpha=0.8, label='Data (binned)')
            if empirical_bias_smoothed is not None and emp_bias_smoothed_feat_vals is not None:
                valid = np.isfinite(np.asarray(empirical_bias_smoothed))
                ax1.plot(np.asarray(emp_bias_smoothed_feat_vals)[valid] * angle_display_scale,
                         np.asarray(empirical_bias_smoothed)[valid] * angle_display_scale,
                         'k--', linewidth=2, alpha=0.8, label='Data (smoothed)')

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

            # Model curves are bin-pooled onto the empirical bins where that
            # field exists; older prepared results fall back to the fine grid.
            _, sd_bin_centers = _feat_diff_bin_edges()
            display_sd_bin_centers = sd_bin_centers * angle_display_scale

            for opt in available_optimizers:
                color = opt_colors.get(opt, 'black')
                label = opt_labels.get(opt, opt.capitalize())
                if opt in avg_stage_sd_pooled:
                    avg_sd_display = avg_stage_sd_pooled[opt] * angle_display_scale
                    sem_sd_display = sem_stage_sd_pooled[opt] * angle_display_scale
                    pooled_valid = ~np.isnan(avg_sd_display)
                    ax3.plot(display_sd_bin_centers[pooled_valid],
                             avg_sd_display[pooled_valid], color=color,
                             linestyle='-', linewidth=2, marker='o', markersize=4,
                             label=label)
                    ax3.fill_between(display_sd_bin_centers[pooled_valid],
                                     (avg_sd_display - sem_sd_display)[pooled_valid],
                                     (avg_sd_display + sem_sd_display)[pooled_valid],
                                     alpha=0.3, color=color)
                    continue
                if opt not in avg_stage_sd:
                    continue
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

    return None


def _model_slice(log_surf: np.ndarray, feat_grid: np.ndarray, mu1_grid: np.ndarray,
                 target_fd: float, weights_sd: float):
    """Return (prob, E, asym) for one Gaussian-weighted model surface slice."""
    w = np.exp(-0.5 * ((feat_grid - target_fd) / weights_sd) ** 2)
    w /= w.sum()
    surf = np.exp(log_surf - log_surf.max(axis=0, keepdims=True))
    dx = mu1_cell_width()
    surf /= surf.sum(axis=0, keepdims=True) * dx
    prob = surf @ w
    E    = float(np.sum(mu1_grid * prob) * dx)
    # Signed split excludes both 0 and the antipode (-180) — see sign_masks.
    pos_mask, neg_mask = (np.asarray(m) for m in sign_masks(mu1_grid))
    asym = float(np.sum(prob[pos_mask]) * dx - np.sum(prob[neg_mask]) * dx)
    return prob, E, asym


def _empirical_slice(fd_vals: np.ndarray, bias_vals: np.ndarray,
                     target_fd: float, bias_grid: np.ndarray, weights_sd: float,
                     period: float = 360.0):
    """Return Gaussian-weighted KDE of bias values around target_fd.

    The bias kernel is wrapped over `period` (model space, so 360 by default),
    matching the fit-side target in shared/utils.py. Note this twin still floors
    the bandwidth at 1.0 while the fit side has no floor — the two empirical KDEs
    remain inconsistent in that one respect (MODEL_PIPELINE_FOR_AGENTS.md D.11).
    """
    w = np.exp(-0.5 * ((fd_vals - target_fd) / weights_sd) ** 2)
    if w.sum() < 1e-10:
        return np.zeros_like(bias_grid)
    w /= w.sum()
    bias_std = bias_vals.std()
    iqr = np.percentile(bias_vals, 75) - np.percentile(bias_vals, 25)
    bw  = max(0.9 * min(bias_std, iqr / 1.34) * len(bias_vals) ** (-0.2), 1.0)
    diff    = bias_grid[:, None] - bias_vals[None, :]
    offsets = period * np.arange(-KDE_WRAPS, KDE_WRAPS + 1)
    kernels = sum(np.exp(-0.5 * ((diff + o) / bw) ** 2) for o in offsets)
    kernels = kernels / (bw * np.sqrt(2 * np.pi))
    return (kernels * w[None, :]).sum(axis=1)


def _plot_log_surface(prediction_backend, params, feat_grid, mu1_grid):
    """Display-grid density from either backend, with fitted motor noise."""
    params = np.asarray(params, dtype=float)
    sd_motor = float(params[3]) if len(params) >= 4 else 0.0
    if getattr(prediction_backend, "family", None) == surrogate.FAMILY_WNM:
        feat = jnp.asarray(feat_grid, dtype=jnp.float32)
        rows = jnp.column_stack([
            jnp.full(feat.shape, params[0]),
            jnp.full(feat.shape, params[1]),
            jnp.full(feat.shape, params[2]),
            feat,
        ])
        # Direct analytic WNM evaluation on the display grid; no NN surface is
        # loaded or reconstructed. Transpose to the plotting helper's
        # historical (bias, feature) convention.
        return np.asarray(prediction_backend.grid_log_density(
            rows, grid=jnp.asarray(mu1_grid), sd_motor=sd_motor)).T

    log_surf = prediction_backend._predict_batch_fixed_size(
        jnp.array([params[:3]]), verbosity=0)[0]
    if sd_motor > 0:
        from grid_based_multi_condition_optimizer_jax_loops import (
            apply_motor_noise_with_precomputed_kernel,
            create_motor_noise_kernel_fft,
        )
        n_mu1_bias = log_surf.shape[0]
        key = (sd_motor, n_mu1_bias)
        if key not in global_motor_kernel_cache:
            global_motor_kernel_cache[key] = create_motor_noise_kernel_fft(
                sd_motor, n_mu1_bias)
        log_surf = apply_motor_noise_with_precomputed_kernel(
            log_surf[None], global_motor_kernel_cache[key])[0]
    return np.asarray(log_surf)


def create_pdf_slice_plots(
    extended_results: Dict,
    prediction_backend,
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

    # Convert to model space for surrogate evaluation
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
            subject_id, experiment, noise_cond = _result_plot_identity(cond_key, result)
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
                log_surf = _plot_log_surface(
                    prediction_backend, params, feat_grid, mu1_grid)
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
                    shade_pos, shade_neg = (np.asarray(m) for m in sign_masks(mu1_grid))
                    ax.fill_between(mu1_grid * angle_display_scale, prob_display, where=shade_pos,
                                    alpha=0.35, color='forestgreen')
                    ax.fill_between(mu1_grid * angle_display_scale, prob_display, where=shade_neg,
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


def create_unified_plots_with_summaries(
        n_samples: int = 20,
        outliers: bool = False,
        data_path: str = 'dasta_color_comb.csv',
        max_subjects: Optional[int] = None,
        create_summary_plots: bool = True,
        create_individual_plots: bool = True,
        create_pdf_slices: bool = True,
        pdf_slice_optimizers=None,
        pdf_slice_n_subjects: int = 3,
        skip_motor_noise: bool = True,
        corr_weight: float = 0.25,
        results_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        results_dir: str = RESULTS_DIR,
        circ_space: Optional[int] = None,
) -> None:
    """Create unified subject plots and summary exports.

    Args:
        n_samples: Training sample count used for checkpoints.
        outliers: Whether the results include outliers.
        data_path: Unused legacy path (kept for signature compatibility).
        max_subjects: Optional limit on subjects to process.
        create_summary_plots: Whether to generate summary plots.
        create_individual_plots: Whether to generate per-subject plots.
        skip_motor_noise: Whether motor noise was skipped in fitting.
        corr_weight: Correlation weight used during fitting. Only affects the
            output path name here, and only reaches the `density_legacy`
            objective during fitting; `density` is 1 - CCC and has no such term.
        results_path: Optional path to extended_fit_results.pkl.
        checkpoint_path: Optional checkpoint path for loading the model.
        output_dir: Optional output directory for plots/exports.
        results_dir: Base directory for resolving relative paths.
        circ_space: Optional display-period override. Normally inferred from
            stored fit metadata; a conflicting override raises.
    """
    corr_str = f'_cw_{corr_weight:.2f}' if corr_weight != 0.25 else ''

    outliers_str = 'no_outliers' if outliers == False else 'with_outliers'
    motor_str = '' if not skip_motor_noise else "_no_motor_noise"
    default_results_path = (
        f"{MODEL_FIT_RESULTS_PREFIX}{motor_str}{corr_str}/"
        f"{n_samples}samples/{outliers_str}/extended_fit_results.pkl"
    )
    resolved_results_path = resolve_input_path(results_path or default_results_path, results_dir)
    # These plots recompute curves, moments and SDs at *stored* parameters, so
    # they must use the surrogate that produced those parameters, not whatever
    # is production today. The run's fingerprint says which by SHA-256. A run
    # without that identity must supply an explicit checkpoint; it never falls
    # back to today's production artifact. An explicit --checkpoint-path is
    # verified against the recorded digest when one exists.
    #
    # Resolving by n_samples alone was wrong twice over: the old default named
    # epoch 1500 while the fitter's named epoch 1425 -- different architectures,
    # not just different epochs -- and after a promotion it would recompute a
    # historical fit's curves from the new model.
    resolved_checkpoint_path = surrogate.checkpoint_for_run(
        resolved_results_path,
        explicit=resolve_input_path(checkpoint_path, results_dir) if checkpoint_path else None,
        n_samples=n_samples)
    _family = surrogate.detect_family(resolved_checkpoint_path)
    sidecar = read_fingerprint_sidecar(Path(resolved_results_path).parent)
    fingerprint_payload = (sidecar or {}).get("payload", {})
    density_curve_spec = (fingerprint_payload.get("density_curve_spec") or {
        "emp_density_weights_sd": 20.0,
        "density_smoothing_sigma": None,
    })
    matmul_precision = fingerprint_payload.get(
        "continuous_spec", {}).get("matmul_precision", "default")

    if output_dir is None:
        resolved_output_dir = Path(resolved_results_path).parent
    else:
        resolved_output_dir = Path(resolve_results_path(output_dir, results_dir))

    # print(f'Reading {resolved_results_path}')
    """Create unified subject, group-summary, and optional PDF-slice plots."""

    # Load results and recover the physical circular period recorded by the fit.
    extended_results = load_extended_results(str(resolved_results_path))
    circ_space = _resolve_plot_circ_space(extended_results, circ_space)

    print("Initializing prediction backend...")
    if _family == surrogate.FAMILY_WNM:
        prediction_backend = predictor_from_surrogate(
            surrogate.load_surrogate(checkpoint_path=resolved_checkpoint_path))
    else:
        rng = np.random.default_rng(0)
        dummy_condition_datasets = {'dummy': jnp.asarray(np.column_stack([
            rng.uniform(config.feat_diff_range[0], config.feat_diff_range[1], 100),
            rng.uniform(config.mu1_bias_range[0], config.mu1_bias_range[1], 100),
        ]))}
        prediction_backend = GridBasedMultiConditionOptimizer(
            str(resolved_checkpoint_path), dummy_condition_datasets)
    print(f"{_family} prediction backend initialized.")
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

    with jax.default_matmul_precision(matmul_precision):
        prepared_all_subjects = prepare_all_subjects_data(
            limited_subjects_data, prediction_backend, density_curve_spec)

    # Create plots for each subject using prepared data
    if create_individual_plots:
        subjects_processed = 0
        for subject_id, prepared_data in prepared_all_subjects.items():
            create_unified_subject_plot(prepared_data, resolved_output_dir, circ_space=circ_space)
            subjects_processed += 1

        print(f"\\nCompleted unified plots for {subjects_processed} subjects")
        print(f"Plots saved to: {output_dir}/unified_subject_plots/")

    # Create summary plots if requested
    if create_summary_plots:
        print("\n=== Creating Summary Plots ===")
        create_extended_summary_plots(
            prepared_all_subjects, resolved_output_dir, circ_space=circ_space,
        )

    if create_pdf_slices:
        print("\n=== Creating PDF Slice Plots ===")
        with jax.default_matmul_precision(matmul_precision):
            create_pdf_slice_plots(
                extended_results, prediction_backend, resolved_output_dir,
                circ_space=circ_space,
                optimizer_names=pdf_slice_optimizers,
                n_subjects=pdf_slice_n_subjects,
            )


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create unified subject, group-summary, and PDF-slice plots from model-fit results."
    )
    parser.add_argument("--results-path", default=None,
                        help="Path to extended_fit_results.pkl (overrides defaults).")
    parser.add_argument("--checkpoint-path", default=None,
                        help="Path to model checkpoint (overrides defaults).")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write plots (defaults to results parent).")
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
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help="Base directory for outputs (default: results).")
    parser.add_argument("--circ-space", type=int, default=None, choices=[180, 360],
                        help="Optional circular-space override. By default it is inferred from "
                             "the fitted result metadata; a conflicting override raises.")
    parser.add_argument("--pdf-slices", action=argparse.BooleanOptionalAction, default=True,
                        help="Create PDF slice plots (default: true).")
    parser.add_argument("--pdf-slice-optimizer", nargs='*', default=None,
                        choices=['density', 'expectation', 'smoothed_exp', 'likelihood',
                                 'crps', 'balanced_crps', 'bias_weighted_crps'],
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
