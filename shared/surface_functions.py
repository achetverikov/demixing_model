"""Core surface math utilities for normalization and smoothing."""

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter1d
from jax.scipy.special import logsumexp
from typing import Dict
import jax

def normalize_to_density(log_probs):
    """
    Normalize log probabilities to proper probability densities that integrate to 1.
    Uses logsumexp for numerical stability with dx correction.

    Args:
        log_probs: Shape (batch, mu1_error, feat_diff) - uses config for dimensions

    Returns:
        Normalized log densities with same shape
    """
    from shared.config import config
    
    # Grid spacing for mu1_error dimension from config
    mu1_error_min, mu1_error_max = config.mu1_bias_range
    n_mu1_points = log_probs.shape[1]
    dmu1 = (mu1_error_max - mu1_error_min) / (n_mu1_points - 1)

    # Normalize using logsumexp (ensures discrete probabilities sum to 1)
    log_probs_discrete = log_probs - jax.scipy.special.logsumexp(log_probs, axis=1, keepdims=True)

    # Convert to continuous density by dividing by dx (in log space: subtract log(dx))
    log_density = log_probs_discrete - jnp.log(dmu1)

    return log_density

def entropy_smooth_columns_with_mask(log_surface, sigma=1.0, p_thresh=1e-5, dx = 2, eps=1e-12):
    """
    Entropy-respecting smoothing along columns with masking of low-probability regions.

    Parameters
    ----------
    log_surface : np.ndarray, shape (N_rows, N_cols)
        Input log-likelihood surface.
    sigma : float
        Standard deviation of Gaussian kernel along columns.
    p_thresh : float
        Probability threshold below which values are masked out.
    renorm : {"row", "global", "none"}
        Normalization scheme after smoothing.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    np.ndarray, shape (N_rows, N_cols)
        Smoothed log-likelihood surface.
    """
    # 1. Normalize to log-probabilities per row
    log_probs = log_surface - logsumexp(log_surface, axis=0, keepdims=True)

    # 2. Create mask of values above threshold
    mask = log_probs > jnp.log(p_thresh)

    # 3. Mask out low-probability values by setting them to -inf (log 0)
    # masked_log_probs = np.where(mask, log_probs, -np.inf)

    # 4. Smooth in log-space along columns; -inf treated as very low values, won't affect neighbors much
    smoothed_log_probs = gaussian_filter1d(log_probs, sigma=sigma, axis=1, mode='reflect')

    # 5. Renormalize to get a valid distribution
    probs = np.exp(smoothed_log_probs)
    probs /= np.sum(probs * dx, axis=0, keepdims=True)
    smoothed_log_norm = np.log(probs+eps)

    return smoothed_log_norm

def smooth_surface(surface_obj: Dict):
    """Smooth a surface object in-place using entropy-aware column smoothing.

    Args:
        surface_obj: Surface dict with `surface_data` tuple.

    Returns:
        The same surface dict with updated smoothed log/probability surfaces.
    """
    orig_surface_ll = entropy_smooth_columns_with_mask(surface_obj['surface_data'][2])
    orig_surface_l = jnp.exp(orig_surface_ll)
    surface_obj['surface_data'] = (surface_obj['surface_data'][0], surface_obj['surface_data'][1],
                                    orig_surface_ll, orig_surface_l)
    return surface_obj

def compute_expectation(surface_data):
    """Compute E[mu1_error] as function of feat_diff."""
    feat_diff_grid, mu1_error_grid, log_likelihood_surface, likelihood_surface = surface_data
    feat_diff_values = feat_diff_grid[0, :]
    mu1_error_values = mu1_error_grid[:, 0]

    log_probs = log_likelihood_surface - logsumexp(log_likelihood_surface, axis=0, keepdims=True)

    # Step 2: Convert log-probs to probs (still safe now)
    probs = jnp.exp(log_probs)

    # Step 3: Compute expectations
    expected_mu1_error = jnp.dot(mu1_error_values, probs)  # shape: (num_feat_diff,)
    # Compute the expected mu1_error for each feat_diff column

    return feat_diff_values, expected_mu1_error
