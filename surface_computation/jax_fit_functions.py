"""JAX utilities for fitting and simulating mixture models."""

import matplotlib
import time, timeit
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import pandas as pd
import itertools
from typing import Optional
from jax.sharding import PartitionSpec as P, NamedSharding

# Set pandas display options for better readability
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)

# Set matplotlib style
plt.style.use('fivethirtyeight')

# Initialize JAX random number generator
jrnd = jax.random

# Set data type based on user preference
use_float64 = False  # @param {type:"boolean"}
dtype = np.float64 if use_float64 else np.float32

# Print the number of available JAX devices
print(jax.device_count())

# Configuration flags
wrap_feat = True
wrap_idf = False
enable_sharding = False   # @param {type:"boolean"}

import sys
from enum import IntEnum

for d in jax.local_devices():
    print(d)

# ============================================================================
# Result array column definitions (single source of truth)
# ============================================================================

# Column names for jax_generate_and_fit output arrays
# Used for CSV export and programmatic access via ResCol enum
RESULT_COLUMNS = [
    'comp_id',
    'orig_comp_id',
    'weight',
    'mu_feat_est',
    'mu_idf_est',
    'sigma_feat_est',
    'sigma_idf_est',
    'r_est',              # Always present, 0.0 for diagonal covariance
    'mu_feat_true',
    'mu_idf_true',
    'sigma_feat_true',
    'sigma_idf_true',
    'sigma_feat_true_flipped',
    'n_samples',
    'mu_feat_sample',
    'mu_idf_sample',
    'sigma_feat_sample',
    'sigma_idf_sample',
    'weight_mix',
    'mu_feat_mix',
    'mu_idf_mix',
]

# Auto-generate enum for zero-overhead column indexing
# Usage: results[:, :, ResCol.mu_feat_est] instead of results[:, :, 3]
ResCol = IntEnum('ResCol', {name: idx for idx, name in enumerate(RESULT_COLUMNS)})

# ============================================================================

@jax.jit
def rearrange_components(res, mu_idf_true):
    """
    Rearranges the components of the result based on the second mean (mu_idf).

    :param res: A tuple containing (weights, mu_feat, mu_idf, sigma_feat, sigma_idf).
    :return: A rearranged array with component indices and parameters.
    """
    (weights, mu_feat, mu_idf, sigma_feat, sigma_idf) = res

    def true_fun(res):
        """Return parameters ordered as-is for component assignment."""
        ids = jnp.array([[0, 1], [0, 1]])
        return jnp.concatenate([ids, jnp.vstack(res)])

    def false_fun(res):
        """Return parameters with flipped component ordering."""
        res_new = [None] * 5
        for i in range(5):
            res_new[i] = jnp.flip(res[i])
        ids = jnp.array([[0, 1], [1, 0]])
        return jnp.concatenate([ids, jnp.vstack(res_new)])
    if wrap_idf:
        mu_diff1 = jnp.power(angular_difference_jit(mu_idf[0], mu_idf_true[0]),2)+jnp.power(angular_difference_jit(mu_idf[1], mu_idf_true[1]),2)
        mu_diff2 = jnp.power(angular_difference_jit(mu_idf[0], mu_idf_true[1]),2)+jnp.power(angular_difference_jit(mu_idf[1], mu_idf_true[0]),2)
    else:
        mu_diff1 = jnp.power(mu_idf[0]- mu_idf_true[0],2)+jnp.power(mu_idf[1]- mu_idf_true[1],2)
        mu_diff2 = jnp.power(mu_idf[0]-mu_idf_true[1],2)+jnp.power(mu_idf[1]- mu_idf_true[0],2)
    res = jax.lax.cond(mu_idf[0]<mu_idf[1], true_fun, false_fun, res).T

    return res

@partial(jax.jit)
def compute_log_prob_circ(X, mu_feat, mu_idf, sigma_feat, sigma_idf, n_wraps=3):
    """
    Computes the log probability for a wrapped Gaussian mixture model.

    :param X: Input data, shape (N, D).
    :param mu_feat: Means of the first dimension, shape (K,).
    :param mu_idf: Means of the second dimension, shape (K,).
    :param sigma_feat: Standard deviations of the first dimension, shape (K,).
    :param sigma_idf: Standard deviations of the second dimension, shape (K,).
    :param n_wraps: Number of periodic shifts for the wrapped dimension.
    :return: Log probabilities, shape (N, K).
    """
    n_samples, num_comp = X.shape[0], mu_feat.shape[0]

    # Standard Gaussian logpdf for non-wrapped dimensions
    # logpdf_standard = jsp.stats.norm.logpdf(X[:, None, 1], loc=mu_idf, scale=sigma_idf)

    if wrap_idf:
        X_idf = X[:, None, 1]
        X_idf = angular_difference_jit(X_idf, mu_idf)

        # Create an array of shifts from -n_wraps to n_wraps
        shifts = jnp.arange(-n_wraps, n_wraps + 1) * 360.0

        # Expand X_idf to apply shifts
        X_shifted = X_idf[:, :, None] + shifts

        # Expand sigma_feat to match the broadcasting requirements
        sigma_idf_expanded = sigma_idf[None, :, None]

        # Compute log PDF using broadcasting
        logpdfs = -0.5 * jnp.log(2 * jnp.pi * sigma_idf_expanded ** 2) - (X_shifted ** 2) / (2 * sigma_idf_expanded ** 2)

        # Apply log-sum-exp trick along the shift dimension
        logpdf_2nd = jsp.special.logsumexp(logpdfs, axis=2)
    else:
        logpdf_2nd = jsp.stats.norm.logpdf(X[:, None, 1], loc=mu_idf, scale=sigma_idf)

    if wrap_feat:
        X_feat = X[:, None, 0]
        X_feat = angular_difference_jit(X_feat, mu_feat)

        # Create an array of shifts from -n_wraps to n_wraps
        shifts = jnp.arange(-n_wraps, n_wraps + 1) * 360.0

        # Expand X_feat to apply shifts
        X_shifted = X_feat[:, :, None] + shifts

        # Expand sigma_feat to match the broadcasting requirements
        sigma_feat_expanded = sigma_feat[None, :, None]

        # Compute log PDF using broadcasting
        logpdfs = -0.5 * jnp.log(2 * jnp.pi * sigma_feat_expanded ** 2) - (X_shifted ** 2) / (2 * sigma_feat_expanded ** 2)

        # Apply log-sum-exp trick along the shift dimension
        logpdf_1st = jsp.special.logsumexp(logpdfs, axis=2)
    else:
        logpdf_1st = jsp.stats.norm.logpdf(X[:, None, 0], loc=mu_feat, scale=sigma_feat)

    return logpdf_1st + logpdf_2nd

@jax.jit
def e_step_circ(X, pi, mu_feat, mu_idf, sigma_feat, sigma_idf):
    """
    Perform the E-step of the EM algorithm for a wrapped Gaussian mixture model.

    :param X: Input data, shape (N, D).
    :param pi: Mixture weights, shape (K,).
    :param mu_feat: Means of the first dimension, shape (K,).
    :param mu_idf: Means of the second dimension, shape (K,).
    :param sigma_feat: Standard deviations of the first dimension, shape (K,).
    :param sigma_idf: Standard deviations of the second dimension, shape (K,).
    :return: Membership weights, shape (N, K).
    """
    log_prob = compute_log_prob_circ(X, mu_feat, mu_idf, sigma_feat, sigma_idf) + jnp.log(pi)
    log_membership_weight = log_prob - jsp.special.logsumexp(log_prob, axis=-1, keepdims=True)
    return jnp.exp(log_membership_weight)

@jax.jit
def compute_vlb_circ(X, pi, mu_feat, mu_idf, sigma_feat, sigma_idf, membership_weight):
    """
    Compute the Variational Lower Bound (VLB) for the wrapped Gaussian mixture model.

    :param X: Input data, shape (N, D).
    :param pi: Mixture weights, shape (K,).
    :param mu_feat: Means of the first dimension, shape (K,).
    :param mu_idf: Means of the second dimension, shape (K,).
    :param sigma_feat: Standard deviations of the first dimension, shape (K,).
    :param sigma_idf: Standard deviations of the second dimension, shape (K,).
    :param membership_weight: Membership weights, shape (N, K).
    :return: Variational Lower Bound (VLB).
    """
    log_prob = compute_log_prob_circ(X, mu_feat, mu_idf, sigma_feat, sigma_idf) + jnp.log(pi)
    vlb = membership_weight * (
        log_prob - jnp.log(jnp.clip(membership_weight, min=jnp.finfo(X.dtype).eps))
    )
    return jnp.sum(vlb)

@jax.jit
def wn_mean_var(X_wrapped, membership_weight=None):
    """
    Compute the mean and variance for wrapped dimensions.

    :param X_wrapped: Wrapped input data, shape (N,).
    :param membership_weight: Membership weights, shape (N, K).
    :return: Tuple (mu_wrapped, sigma_wrapped).
    """
    N = X_wrapped.shape[0]
    effect_number = membership_weight.sum(0)  # Effective count of points per component

    X_rad = jnp.deg2rad(X_wrapped)
    if membership_weight is None:
        membership_weight = 1 / N
    else:
        effect_number = jnp.clip(effect_number, min=1)
        membership_weight_norm = membership_weight / effect_number

    sin_term = jnp.sum(jnp.sin(X_rad)[:, None] * membership_weight_norm, axis=0)
    cos_term = jnp.sum(jnp.cos(X_rad)[:, None] * membership_weight_norm, axis=0)

    mu_wrapped_rad = jnp.arctan2(sin_term, cos_term)
    mu_wrapped = jnp.rad2deg(mu_wrapped_rad)

    R_sq = (sin_term ** 2 + cos_term ** 2)
    R_sq_e = N / (N - 1) * (R_sq - 1 / N)
    R_sq_e = jnp.clip(R_sq_e, jnp.finfo(jnp.float32).eps, 1.0)

    sigma_sq_wrapped = jnp.log(1 / R_sq_e)
    sigma_wrapped = jnp.rad2deg(jnp.sqrt(sigma_sq_wrapped))
    sigma_wrapped = jnp.clip(sigma_wrapped, min=1)

    return mu_wrapped, sigma_wrapped

@jax.jit
def stn_mean_var(X_standard, membership_weight=None):
    """
    Compute the mean and variance for standard dimensions.

    :param X_standard: Standard input data, shape (N,).
    :param membership_weight: Membership weights, shape (N, K).
    :return: Tuple (mu_standard, sigma_standard).
    """
    effect_number = membership_weight.sum(0)  # Effective count of points per component
    effect_number = jnp.clip(effect_number, min=1)

    mu_standard = jnp.sum(X_standard[:, None] * membership_weight, axis=0) / effect_number

    sq_dev = (X_standard[:, None] - mu_standard) ** 2
    var_diag = jnp.sum(sq_dev * membership_weight, axis=0) / effect_number
    sd_standard = jnp.sqrt(var_diag)
    sd_standard = jnp.clip(sd_standard, min=1)

    return mu_standard, sd_standard

@jax.jit
def m_step_circ(X, membership_weight, debug = False):
    """
    Perform the M-step of the EM algorithm, updating mixture parameters.

    :param X: Input data, shape (n_samples, dims).
    :param membership_weight: Membership weights, shape (n_samples, num_components).
    :return: Tuple (pi_upd, mu_wrapped, mu_standard, sigma_wrapped, sigma_standard).
    """
    effect_number = membership_weight.sum(0)  # Effective count of points per component
    pi_upd = effect_number / X.shape[0]  # Updated mixture weights

    X_feat, X_idf = X[:, 0], X[:, 1]

    if wrap_feat:
        mu_featst, sigma_1st = wn_mean_var(X_feat, membership_weight)
    else:
        mu_featst, sigma_1st = stn_mean_var(X_feat, membership_weight)

    if wrap_idf:
        mu_idfnd, sigma_2nd = wn_mean_var(X_idf, membership_weight)
    else:
        mu_idfnd, sigma_2nd = stn_mean_var(X_idf, membership_weight)

    return pi_upd, mu_featst, mu_idfnd, sigma_1st, sigma_2nd


def format_state_change(state, new_state):
    """
    Generate a formatted string showing the change in parameters and loss
    from the previous state to the updated state.

    :param state: Tuple containing the previous iteration state.
    :param new_state: Tuple containing the updated state.
    :return: Formatted string showing the change in parameters.
    """
    i, (pi, mu_feat, mu_idf, sigma_feat, sigma_idf), loss_i, loss_diff_i = state
    i_upd, (pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd), loss_upd, loss_diff_upd = new_state

    print(f"Iteration {i} -> {i_upd}:\r\n"
            f"  π:      {pi} -> {pi_upd}\n"
            f"  μ1:     {mu_feat} -> {mu_feat_upd}\n"
            f"  μ2:     {mu_idf} -> {mu_idf_upd}\n"
            f"  σ1:     {sigma_feat} -> {sigma_feat_upd}\n"
            f"  σ2:     {sigma_idf} -> {sigma_idf_upd}\n"
            f"  Loss:   {loss_i:.6f} -> {loss_upd:.6f}\n"
            f"  ΔLoss:  {loss_diff_i:.6f} -> {loss_diff_upd:.6f}")

@jax.jit
def compute_log_prob(X,  mu_feat, mu_idf, sigma_feat, sigma_idf):
    """
    Compute the log probability for a Gaussian mixture model.

    :param X: Input data, shape (N, D).
    :param mu_feat: Means of the first dimension, shape (K,).
    :param mu_idf: Means of the second dimension, shape (K,).
    :param sigma_feat: Standard deviations of the first dimension, shape (K,).
    :param sigma_idf: Standard deviations of the second dimension, shape (K,).
    :return: Log probabilities, shape (N, K).
    """
    mu = jnp.vstack([mu_feat, mu_idf]).T
    comp1_cov = jnp.diag(jnp.array([sigma_feat[0], sigma_idf[0]]))
    comp2_cov = jnp.diag(jnp.array([sigma_feat[1], sigma_idf[1]]))
    sigma = jnp.stack((comp1_cov, comp2_cov), axis=0)
    return jax.scipy.stats.multivariate_normal.logpdf(
        X[:, None, ...], mean=mu, cov=sigma
    )

@jax.jit
def e_step(X, pi, mu_feat, mu_idf, sigma_feat, sigma_idf):
    """
    Perform the E-step of the EM algorithm for a Gaussian mixture model.

    :param X: Input data, shape (N, D).
    :param pi: Mixture weights, shape (K,).
    :param mu_feat: Means of the first dimension, shape (K,).
    :param mu_idf: Means of the second dimension, shape (K,).
    :param sigma_feat: Standard deviations of the first dimension, shape (K,).
    :param sigma_idf: Standard deviations of the second dimension, shape (K,).
    :return: Membership weights, shape (N, K).
    """
    mixture_log_prob = compute_log_prob(X, mu_feat, mu_idf, sigma_feat, sigma_idf) + jnp.log(pi)
    log_membership_weight = mixture_log_prob - jsp.special.logsumexp(mixture_log_prob, axis=-1, keepdims=True)
    return jnp.exp(log_membership_weight)

@jax.jit
def compute_vlb(X, pi, mu_feat, mu_idf, sigma_feat, sigma_idf, membership_weight):
    """
    Compute the Variational Lower Bound (VLB) for the Gaussian mixture model.

    :param X: Input data, shape (N, D).
    :param pi: Mixture weights, shape (K,).
    :param mu_feat: Means of the first dimension, shape (K,).
    :param mu_idf: Means of the second dimension, shape (K,).
    :param sigma_feat: Standard deviations of the first dimension, shape (K,).
    :param sigma_idf: Standard deviations of the second dimension, shape (K,).
    :param membership_weight: Membership weights, shape (N, K).
    :return: Variational Lower Bound (VLB).
    """
    # Compute the log probability for each component
    component_log_prob = compute_log_prob(X, mu_feat, mu_idf, sigma_feat, sigma_idf)

    # Compute the Variational Lower Bound (VLB)
    vlb = membership_weight * (
        jnp.log(pi) + component_log_prob - jnp.log(jnp.clip(membership_weight, min=jnp.finfo(X.dtype).eps))
    )

    return jnp.sum(vlb)

@jax.jit
def m_step(X, membership_weight):
    """
    Perform the M-step of the EM algorithm, updating mixture parameters.

    :param X: Input data, shape (N, D).
    :param membership_weight: Membership weights, shape (N, K).
    :return: Tuple (pi_updated, mu_feat_updated, mu_idf_updated, sigma_feat_updated, sigma_idf_updated).
    """
    # Compute the effective number of points per component
    effect_number = jnp.clip(membership_weight.sum(0), 1e-10)
    pi_updated = effect_number / X.shape[0]

    # Compute the updated means
    mu_updated = jnp.sum(X[:, None, ...] * membership_weight[..., None], axis=0) / effect_number[..., None]
    mu_feat_updated = mu_updated[:, 0]
    mu_idf_updated = mu_updated[:, 1]

    # Compute the centered data
    centered_x = X[:, None, ...] - mu_updated

    # Compute the updated diagonal covariances
    sigma_updated_diag = jnp.clip(
        jnp.sum(centered_x ** 2 * membership_weight[..., None], axis=0) / effect_number[..., None],
        1e-6
    )
    sigma_feat_updated = sigma_updated_diag[:, 0]
    sigma_idf_updated = sigma_updated_diag[:, 1]

    return pi_updated, mu_feat_updated, mu_idf_updated, sigma_feat_updated, sigma_idf_updated


@jax.jit
def compute_log_prob_full_common(X, mu_feat, mu_idf, sigma_feat, sigma_idf, rho):
    """Log probability for a two-component Gaussian with one shared full covariance."""
    dx = X[:, None, 0] - mu_feat[None, :]
    dy = X[:, None, 1] - mu_idf[None, :]
    sigma_feat = jnp.clip(sigma_feat, min=1e-3)
    sigma_idf = jnp.clip(sigma_idf, min=1e-3)
    rho = jnp.clip(rho, -0.98, 0.98)
    one_minus_r2 = jnp.clip(1.0 - rho ** 2, min=1e-6)
    z = (dx / sigma_feat) ** 2 - 2.0 * rho * dx * dy / (sigma_feat * sigma_idf) + (dy / sigma_idf) ** 2
    log_norm = -jnp.log(2.0 * jnp.pi) - jnp.log(sigma_feat) - jnp.log(sigma_idf) - 0.5 * jnp.log(one_minus_r2)
    return log_norm - 0.5 * z / one_minus_r2


@jax.jit
def e_step_full_common(X, pi, mu_feat, mu_idf, sigma_feat, sigma_idf, rho):
    mixture_log_prob = compute_log_prob_full_common(X, mu_feat, mu_idf, sigma_feat, sigma_idf, rho) + jnp.log(pi)
    log_membership_weight = mixture_log_prob - jsp.special.logsumexp(mixture_log_prob, axis=-1, keepdims=True)
    return jnp.exp(log_membership_weight)


@jax.jit
def compute_loglike_full_common(X, pi, mu_feat, mu_idf, sigma_feat, sigma_idf, rho):
    mixture_log_prob = compute_log_prob_full_common(X, mu_feat, mu_idf, sigma_feat, sigma_idf, rho) + jnp.log(pi)
    return jnp.sum(jsp.special.logsumexp(mixture_log_prob, axis=-1))


@jax.jit
def m_step_full_common(X, membership_weight):
    """M-step for free weights and a single full covariance shared by both components."""
    effect_number = jnp.clip(membership_weight.sum(0), min=1.0)
    pi_updated = effect_number / X.shape[0]

    mu_updated = jnp.sum(X[:, None, :] * membership_weight[:, :, None], axis=0) / effect_number[:, None]
    mu_feat_updated = mu_updated[:, 0]
    mu_idf_updated = mu_updated[:, 1]

    centered = X[:, None, :] - mu_updated[None, :, :]
    weighted_outer = membership_weight[:, :, None, None] * (
        centered[:, :, :, None] * centered[:, :, None, :]
    )
    cov = jnp.sum(weighted_outer, axis=(0, 1)) / X.shape[0]
    cov_00 = jnp.clip(cov[0, 0], min=1.0)
    cov_11 = jnp.clip(cov[1, 1], min=1.0)
    sigma_feat_updated = jnp.sqrt(cov_00)
    sigma_idf_updated = jnp.sqrt(cov_11)
    rho_updated = jnp.clip(cov[0, 1] / (sigma_feat_updated * sigma_idf_updated), -0.98, 0.98)

    return pi_updated, mu_feat_updated, mu_idf_updated, sigma_feat_updated, sigma_idf_updated, rho_updated


@jax.jit
def compute_log_prob_diag_common(X, mu_feat, mu_idf, sigma_feat, sigma_idf):
    dx = X[:, None, 0] - mu_feat[None, :]
    dy = X[:, None, 1] - mu_idf[None, :]
    sigma_feat = jnp.clip(sigma_feat, min=1e-3)
    sigma_idf = jnp.clip(sigma_idf, min=1e-3)
    return (
        -jnp.log(2.0 * jnp.pi)
        - jnp.log(sigma_feat)
        - jnp.log(sigma_idf)
        - 0.5 * ((dx / sigma_feat) ** 2 + (dy / sigma_idf) ** 2)
    )


@jax.jit
def e_step_diag_common(X, pi, mu_feat, mu_idf, sigma_feat, sigma_idf):
    mixture_log_prob = compute_log_prob_diag_common(X, mu_feat, mu_idf, sigma_feat, sigma_idf) + jnp.log(pi)
    log_membership_weight = mixture_log_prob - jsp.special.logsumexp(mixture_log_prob, axis=-1, keepdims=True)
    return jnp.exp(log_membership_weight)


@jax.jit
def compute_loglike_diag_common(X, pi, mu_feat, mu_idf, sigma_feat, sigma_idf):
    mixture_log_prob = compute_log_prob_diag_common(X, mu_feat, mu_idf, sigma_feat, sigma_idf) + jnp.log(pi)
    return jnp.sum(jsp.special.logsumexp(mixture_log_prob, axis=-1))


@jax.jit
def m_step_diag_common(X, membership_weight):
    effect_number = jnp.clip(membership_weight.sum(0), min=1.0)
    pi_updated = effect_number / X.shape[0]
    mu_updated = jnp.sum(X[:, None, :] * membership_weight[:, :, None], axis=0) / effect_number[:, None]
    mu_feat_updated = mu_updated[:, 0]
    mu_idf_updated = mu_updated[:, 1]
    centered = X[:, None, :] - mu_updated[None, :, :]
    weighted_sq = membership_weight[:, :, None] * centered ** 2
    var = jnp.sum(weighted_sq, axis=(0, 1)) / X.shape[0]
    sigma_feat_updated = jnp.sqrt(jnp.clip(var[0], min=1.0))
    sigma_idf_updated = jnp.sqrt(jnp.clip(var[1], min=1.0))
    return pi_updated, mu_feat_updated, mu_idf_updated, sigma_feat_updated, sigma_idf_updated


@jax.jit
def m_step_Lk_B(X, membership_weight):
    """M-step for Gaussian_pk_Lk_B: per-component volume, shared normalised diagonal shape.

    Each component gets its own overall scale (lambda_k) but both share the same
    sigma_feat/sigma_idf ratio.  Implements the Celeux-Govaert [lambda_k B] update:
      lambda_k = sqrt(var_feat_k * var_idf_k)
      B (pooled) = weighted mean of (Sigma_k / lambda_k) across components, normalised |B|=1
      sigma_feat_k = sqrt(lambda_k * b_feat),  sigma_idf_k = sqrt(lambda_k * b_idf)
    """
    effect_number = jnp.clip(membership_weight.sum(0), min=1.0)  # (K,)
    pi_updated = effect_number / X.shape[0]

    mu_updated = jnp.sum(X[:, None, :] * membership_weight[:, :, None], axis=0) / effect_number[:, None]
    mu_feat_updated = mu_updated[:, 0]
    mu_idf_updated = mu_updated[:, 1]

    centered = X[:, None, :] - mu_updated[None, :, :]        # (N, K, 2)
    weighted_sq = membership_weight[:, :, None] * centered ** 2
    var = jnp.sum(weighted_sq, axis=0) / effect_number[:, None]   # (K, 2)

    var_feat = var[:, 0]
    var_idf = var[:, 1]

    lam = jnp.sqrt(jnp.clip(var_feat * var_idf, min=1e-6))  # per-component volume

    # Pooled normalised shape, weighted by soft counts
    b_feat_unnorm = jnp.sum(effect_number * var_feat / lam) / X.shape[0]
    b_idf_unnorm = jnp.sum(effect_number * var_idf / lam) / X.shape[0]
    norm = jnp.sqrt(jnp.clip(b_feat_unnorm * b_idf_unnorm, min=1e-12))
    b_feat = b_feat_unnorm / norm
    b_idf = b_idf_unnorm / norm

    sigma_feat_updated = jnp.sqrt(jnp.clip(lam * b_feat, min=1.0))
    sigma_idf_updated = jnp.sqrt(jnp.clip(lam * b_idf, min=1.0))
    return pi_updated, mu_feat_updated, mu_idf_updated, sigma_feat_updated, sigma_idf_updated


@partial(jax.jit, static_argnames=['num_comp', 'n_init', 'rtol', 'max_iter'])
def train_em_jax_Lk_B(key, observed, num_comp=2, n_init=400, rtol=1e-6, max_iter=5000):
    """EM for Gaussian_pk_Lk_B: per-component volume, shared diagonal shape (ratio).

    Equivalent to Rmixmod Gaussian_pk_Lk_B (diagonal family).
    Both components share the same sigma_feat/sigma_idf ratio but can differ in
    overall scale.  Uses the same initialisation grid as train_em_jax_diag_common.
    """

    def cond_fn(state):
        i, pi, mu_feat, mu_idf, sigma_feat, sigma_idf, loss, loss_diff = state
        return (i < max_iter) & (loss_diff > rtol)

    def project_to_simplex_with_min(x, min_val=0.01):
        x = jnp.maximum(x, min_val)
        excess = jnp.sum(x) - 1.0
        x -= excess * (x - min_val) / jnp.sum(x - min_val + 1e-8)
        return x

    def one_step(state):
        i, pi, mu_feat, mu_idf, sigma_feat, sigma_idf, loss_i, loss_diff_i = state
        membership_weight = e_step_diag_common(observed, pi, mu_feat, mu_idf, sigma_feat, sigma_idf)
        pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd = m_step_Lk_B(observed, membership_weight)
        pi_upd = project_to_simplex_with_min(pi_upd, 0.1)
        loss_upd = compute_loglike_diag_common(observed, pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd)
        loss_diff_upd = jnp.abs(loss_upd - loss_i) / (1e-12 + jnp.abs(loss_i))
        return (i + 1, pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd, loss_upd, loss_diff_upd)

    def run_one(init_state):
        return jax.lax.while_loop(cond_fn, one_step, init_state)

    key, idx_key, sigma_key = jax.random.split(key, 3)
    pi_init = jnp.full((n_init, num_comp), 1.0 / num_comp)
    idxs = jax.random.randint(idx_key, (n_init, num_comp), 0, observed.shape[0])
    mu_init = observed[idxs]          # (n_init, K, D)
    data_sd = observed.std(0)         # (D,)
    sigma_init = jax.random.uniform(
        sigma_key,
        minval=data_sd[None, None, :] * 0.3,
        maxval=data_sd[None, None, :] * 2.0,
        shape=(n_init, num_comp, observed.shape[-1]),
    )
    sigma_feat_init = sigma_init[..., 0]  # (n_init, K)
    sigma_idf_init = sigma_init[..., 1]

    init_states = (
        jnp.zeros((n_init,), dtype=jnp.int32),
        pi_init,
        mu_init[:, :, 0],
        mu_init[:, :, 1],
        sigma_feat_init,
        sigma_idf_init,
        -jnp.ones((n_init,)) * jnp.inf,
        jnp.ones((n_init,)) * jnp.inf,
    )

    _, pi_est, mu_feat_est, mu_idf_est, sigma_feat_est, sigma_idf_est, loss, _ = jax.vmap(run_one)(init_states)
    index = jnp.argmax(jnp.nan_to_num(loss, nan=-jnp.inf))
    weights, mu_feat_best, mu_idf_best, sigma_feat_best, sigma_idf_best, loss_best = jax.tree.map(
        lambda x: x[index],
        (pi_est, mu_feat_est, mu_idf_est, sigma_feat_est, sigma_idf_est, loss),
    )
    return weights, mu_feat_best, mu_idf_best, sigma_feat_best, sigma_idf_best, loss_best


@partial(jax.jit, static_argnames=['num_comp', 'n_init', 'rtol', 'max_iter'])
def train_em_jax_diag_common(key, observed, num_comp=2, n_init=400, rtol=1e-6, max_iter=5000):
    """EM for Gaussian_pk_L_B / mclust EEI-like common diagonal covariance."""

    def cond_fn(state):
        i, pi, mu_feat, mu_idf, sigma_feat, sigma_idf, loss, loss_diff = state
        return (i < max_iter) & (loss_diff > rtol)

    def project_to_simplex_with_min(x, min_val=0.01):
        x = jnp.maximum(x, min_val)
        excess = jnp.sum(x) - 1.0
        x -= excess * (x - min_val) / jnp.sum(x - min_val + 1e-8)
        return x

    def one_step(state):
        i, pi, mu_feat, mu_idf, sigma_feat, sigma_idf, loss_i, loss_diff_i = state
        membership_weight = e_step_diag_common(observed, pi, mu_feat, mu_idf, sigma_feat, sigma_idf)
        pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd = m_step_diag_common(observed, membership_weight)
        pi_upd = project_to_simplex_with_min(pi_upd, 0.1)
        loss_upd = compute_loglike_diag_common(observed, pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd)
        loss_diff_upd = jnp.abs(loss_upd - loss_i) / (1e-12 + jnp.abs(loss_i))
        return (i + 1, pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd, loss_upd, loss_diff_upd)

    def run_one(init_state):
        return jax.lax.while_loop(cond_fn, one_step, init_state)

    key, idx_key, sigma_key = jax.random.split(key, 3)
    pi_init = jnp.full((n_init, num_comp), 1.0 / num_comp)
    idxs = jax.random.randint(idx_key, (n_init, num_comp), 0, observed.shape[0])
    mu_init = observed[idxs]          # (n_init, K, D)
    data_sd = observed.std(0)         # (D,)
    sigma_init = jax.random.uniform(
        sigma_key,
        minval=data_sd[None, :] * 0.3,
        maxval=data_sd[None, :] * 2.0,
        shape=(n_init, 2),
    )

    init_states = (
        jnp.zeros((n_init,), dtype=jnp.int32),
        pi_init,
        mu_init[:, :, 0],
        mu_init[:, :, 1],
        sigma_init[:, 0],
        sigma_init[:, 1],
        -jnp.ones((n_init,)) * jnp.inf,
        jnp.ones((n_init,)) * jnp.inf,
    )

    _, pi_est, mu_feat_est, mu_idf_est, sigma_feat_est, sigma_idf_est, loss, _ = jax.vmap(run_one)(init_states)
    index = jnp.argmax(jnp.nan_to_num(loss, nan=-jnp.inf))
    weights, mu_feat_best, mu_idf_best, sigma_feat_best, sigma_idf_best, loss_best = jax.tree.map(
        lambda x: x[index],
        (pi_est, mu_feat_est, mu_idf_est, sigma_feat_est, sigma_idf_est, loss),
    )
    sigma_feat_arr = jnp.full((num_comp,), sigma_feat_best)
    sigma_idf_arr = jnp.full((num_comp,), sigma_idf_best)
    return weights, mu_feat_best, mu_idf_best, sigma_feat_arr, sigma_idf_arr, loss_best


@partial(jax.jit, static_argnames=['num_comp', 'n_init', 'rtol', 'max_iter', 'fix_weights'])
def train_em_jax(key, observed, num_comp=2, n_init=225, rtol=1e-6, max_iter=5000,
                 fix_weights=False):
    """
    Trains a mixture model using the Expectation-Maximization algorithm in JAX.

    :param key: PRNG key for random number generation.
    :param observed: The observed data.
    :param num_comp: The number of components in the mixture model.
    :param n_init: The number of initializations for the algorithm.
    :param rtol: The relative tolerance for convergence.
    :param max_iter: The maximum number of iterations for the algorithm.
    :param fix_weights: If True, hold the mixture weights fixed at 1/num_comp (equal
        proportions) throughout — the M-step weight update is skipped. If False (default),
        weights are estimated and projected onto the simplex with a 0.1 floor each step.
    :return: A tuple containing the estimated parameters: pi_best, mu_best, sigma_best.
    """

    def cond_fn(state):
        """
        Condition function for the EM loop. Checks if the maximum number of iterations
        is reached or if the relative change in loss is below the tolerance.

        :param state: Current state of the EM algorithm.
        :return: Boolean indicating whether to continue the loop.
        """
        i, thetas, loss, loss_diff = state
        return jnp.any((i < max_iter) & (loss_diff > rtol))

    @partial(jax.jit, static_argnames=['min_val'])
    def project_to_simplex_with_min(x, min_val=0.01):
        """
        Projects a 2-element array onto the probability simplex while ensuring
        each element is at least `min_val` and the sum remains 1.

        :param x: Input array.
        :param min_val: Minimum value for each element.
        :return: Projected array.
        """
        x = jnp.maximum(x, min_val)  # Ensure minimum value
        excess = jnp.sum(x) - 1.0  # Compute excess amount
        x -= excess * (x - min_val) / jnp.sum(x - min_val + 1e-8)  # Adjust
        return x

    def generate_inits(key):
        pi_init = jnp.full((n_init, num_comp), 1.0 / num_comp)

        key, idx_key = jax.random.split(key)
        idxs = jax.random.randint(idx_key, (n_init, num_comp), 0, observed.shape[0])
        mu_init = observed[idxs]  # (n_init, K, D)
        mu_feat_init = mu_init[..., 0]
        mu_idf_init = mu_init[..., 1]

        if wrap_feat:
            mu_feat_init = jnp.mod(mu_feat_init + 180, 360.0) - 180
        if wrap_idf:
            mu_idf_init = jnp.mod(mu_idf_init + 180, 360.0) - 180

        key, subkey = jax.random.split(key)
        data_sd = observed.std(0)
        sigma_init = jax.random.uniform(
            subkey,
            minval=data_sd[None, None, :] * 0.3,
            maxval=data_sd[None, None, :] * 2.0,
            shape=(n_init, num_comp, observed.shape[-1])
        )
        sigma_feat_init = sigma_init[..., 0]
        sigma_idf_init = sigma_init[..., 1]

        loss_init = -jnp.ones([n_init]) * jnp.inf
        loss_upd_init = jnp.ones([n_init]) * jnp.inf

        init_val = (jnp.zeros([n_init], jnp.int32),
                    (pi_init, mu_feat_init, mu_idf_init, sigma_feat_init, sigma_idf_init),
                    loss_init,
                    loss_upd_init)
        return key, init_val

    def one_step(state, debug=False):
        """
        Performs one step of the EM algorithm.

        :param state: Current state of the EM algorithm.
        :param debug: Boolean indicating whether to plot debug information.
        :return: Updated state of the EM algorithm.
        """
        i, (pi, mu_feat, mu_idf, sigma_feat, sigma_idf), loss_i, loss_diff_i = state
        membership_weight = e_step_circ(observed, pi, mu_feat, mu_idf, sigma_feat, sigma_idf)

        if debug:
            plot_X(observed, jnp.vstack([mu_feat, mu_idf]).T, sigma_true=jnp.vstack([sigma_feat, sigma_idf]),
                   est_weights=membership_weight, title=f'step {i}')

        pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd = m_step_circ(observed, membership_weight, debug = debug)

        if fix_weights:
            # Keep weights pinned at their initial equal value (1/num_comp); skip the update.
            pi_upd = pi
        else:
            pi_upd = project_to_simplex_with_min(pi_upd, 0.1)

        membership_weight_upd = e_step_circ(
            observed, pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd
            )

        loss_upd = compute_vlb_circ(observed, pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd, membership_weight_upd)
        loss_diff_upd = jnp.abs(loss_upd - loss_i) / (1e-12 + jnp.abs(loss_i))
        new_state = (i + 1,
                     (pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd),
                     loss_upd,
                     loss_diff_upd)
        if debug:
            format_state_change(state, new_state)

        return new_state

    def one_step_norm(state, debug=False):
        """
        Performs one step of the EM algorithm for the normal distribution.

        :param state: Current state of the EM algorithm.
        :param debug: Boolean indicating whether to plot debug information.
        :return: Updated state of the EM algorithm.
        """
        i, (pi, mu_feat, mu_idf, sigma_feat, sigma_idf), loss_i, loss_diff_i = state
        membership_weight = e_step(observed, pi, mu_feat, mu_idf, sigma_feat, sigma_idf)

        if debug:
            plot_X(observed, jnp.vstack([mu_feat, mu_idf]).T, sigma_true=jnp.vstack([sigma_feat, sigma_idf]),
                   est_weights=membership_weight, title=f'step {i}')

        pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd = m_step(observed, membership_weight)

        pi_upd = project_to_simplex_with_min(pi_upd, 0.1)
        membership_weight_upd = e_step(
            observed, pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd
        )
        loss_upd = compute_vlb(observed, pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd, membership_weight_upd)
        loss_diff_upd = jnp.abs(loss_upd - loss_i) / (1e-12 + jnp.abs(loss_i))
        new_state = (i + 1,
                     (pi_upd, mu_feat_upd, mu_idf_upd, sigma_feat_upd, sigma_idf_upd),
                     loss_upd,
                     loss_diff_upd)
        if debug:
            format_state_change(state, new_state)

        return new_state

    def em_loop(state):
        """
        Runs the EM loop until convergence.

        :param state: Initial state of the EM algorithm.
        :return: Final state of the EM algorithm.
        """
        return jax.lax.while_loop(cond_fn, one_step, state)

    one_step_parallel = jax.vmap(one_step)

    key, init_val = generate_inits(key)
    em_loop_parallel = jax.vmap(em_loop)

    num_iter, est_pars, loss, loss_diff = em_loop_parallel(init_val)
    weights, mu_feat_est, mu_idf_est, sigma_feat_est, sigma_idf_est = est_pars

    index = jnp.argmax(jnp.nan_to_num(loss, nan=-jnp.inf))

    debug = False
    if debug:
        pi_init, mu_feat_init, mu_idf_init, sigma_feat_init, sigma_idf_init = init_val[1]
        state = (1,
                 (pi_init[index], mu_feat_init[index], mu_idf_init[index], sigma_feat_init[index], sigma_idf_init[index]),
                 -jnp.inf,
                 jnp.inf)
        # state_norm = (1,
        #               (pi_init[index], mu_feat_init[index], mu_idf_init[index], sigma_feat_init[index] ** 2, sigma_idf_init[index] ** 2),
        #               -jnp.inf, jnp.inf)
        for _ in range(max_iter):
            new_state = one_step(state, debug=True)
            # new_state_norm = one_step_norm(state_norm, debug=True)

            i, pars, loss_i, loss_diff_i = new_state
            state = new_state
            # state_norm = new_state_norm
            if not cond_fn(state):
                i, thetas, loss, loss_diff = state
                print(i < max_iter)
                print(loss_diff > rtol)

                break

        i, (pi, mu_feat, mu_idf, sigma_feat, sigma_idf), loss_i, loss_diff_i = state
        membership_weight = e_step_circ(observed, pi, mu_feat, mu_idf, sigma_feat, sigma_idf)
        plot_X(observed, jnp.vstack([mu_feat, mu_idf]).T, sigma_true=jnp.vstack([sigma_feat, sigma_idf]), est_weights=membership_weight)

    weights, mu_feat_est, mu_idf_est, sigma_feat_est, sigma_idf_est, loss = jax.tree.map(
        lambda x: x[index], (weights, mu_feat_est, mu_idf_est, sigma_feat_est, sigma_idf_est, loss))

    return weights, mu_feat_est, mu_idf_est, sigma_feat_est, sigma_idf_est, loss


@partial(jax.jit, static_argnames=['n_samples', 'weights'])
def jax_jax_mixture(key, mu_feat_true, mu_idf_true, sigma_feat_true, sigma_idf_true, weights=0.5, n_samples=100):
    """
    Generate samples from a Gaussian mixture model using JAX.

    :param key: PRNG key for random number generation.
    :param mu_feat_true: Means of the first dimension for each component.
    :param mu_idf_true: Means of the second dimension for each component.
    :param sigma_feat_true: Standard deviations of the first dimension for each component.
    :param sigma_idf_true: Standard deviations of the second dimension for each component.
    :param weights: Mixture weights for the components.
    :param n_samples: Total number of samples to generate.
    :return: Generated samples.
    """
    key, *subkeys = jrnd.split(key, 4)
    grp1_size = jnp.int32(jrnd.binomial(subkeys[0], n=n_samples, p=weights, shape=1))

    # Create covariance matrices for each component
    mv_cov1 = jnp.eye(2) * jnp.array([sigma_feat_true[0], sigma_idf_true[0]])**2
    mv_cov2 = jnp.eye(2) * jnp.array([sigma_feat_true[1], sigma_idf_true[1]])**2

    # Create mean vectors for each component
    mv_mu_feat = jnp.array([mu_feat_true[0], mu_idf_true[0]])
    mv_mu_idf = jnp.array([mu_feat_true[1], mu_idf_true[1]])

    # Generate samples for each component
    sample1 = jax.random.multivariate_normal(subkeys[1], mv_mu_feat, mv_cov1, (n_samples,))
    sample2 = jax.random.multivariate_normal(subkeys[2], mv_mu_idf, mv_cov2, (n_samples,))

    # Combine samples and select the required number of samples
    full_sample = jnp.vstack((sample1, sample2))
    full_sample = jax.lax.dynamic_slice(full_sample, (n_samples - grp1_size.min(), 0), (n_samples, 2))

    # Wrap the first dimension (x values) in 360 degrees if required
    if wrap_feat:
        full_sample = full_sample.at[:, 0].set(jnp.mod(full_sample[:, 0]+180, 360.0)-180)
    # Wrap the 2nd dimension (x values) in 360 degrees if required
    if wrap_idf:
        full_sample = full_sample.at[:, 1].set(jnp.mod(full_sample[:, 1]+180, 360.0)-180)

    return full_sample

@partial(jax.jit, static_argnames=['weights', 'n_samples', 'algorithm', 'fix_weights', 'diagonal_covariance'])
def jax_generate_and_fit(key, mu_feat_true, mu_idf_true, sigma_feat_true, sigma_idf_true,
                         weights=0.5, n_samples=100, algorithm='EM',
                         fix_weights=False, diagonal_covariance: bool = True):
    """
    Generate samples from a Gaussian mixture model and fit the model using EM algorithm.

    :param key: PRNG key for random number generation.
    :param mu_feat_true: Means of the first dimension for each component.
    :param mu_idf_true: Means of the second dimension for each component.
    :param sigma_feat_true: Standard deviations of the first dimension for each component.
    :param sigma_idf_true: Standard deviations of the second dimension for each component.
    :param weights: Mixture weights for the components.
    :param n_samples: Total number of samples to generate.
    :return: Fitted parameters and true parameters.
    """
    # Honor the requested fitting mode instead of silently ignoring these flags.
    # Only the diagonal, circular EM is implemented in this path; anything else must fail
    # loudly rather than return a mislabeled diagonal fit. (algorithm/diagonal_covariance
    # are static under jit, so these checks resolve at trace time.)
    if algorithm.upper() != 'EM':
        raise NotImplementedError(
            f"jax_generate_and_fit only implements EM in this path; got algorithm={algorithm!r}. "
            "VBEM/VBEM_MIX have no inference code in this repo."
        )
    if not diagonal_covariance:
        raise NotImplementedError(
            "Full covariance is not implemented for the circular demixing EM; only diagonal "
            "(independent circular-feature + linear-identifiability axes) is supported."
        )

    key, subkey = jrnd.split(key, 2)

    # Generate observed samples
    observed = jax_jax_mixture(key, mu_feat_true, mu_idf_true, sigma_feat_true, sigma_idf_true, weights, n_samples)

    out = train_em_jax(subkey, observed, num_comp=2, fix_weights=fix_weights)

    # Rearrange components to match true component order based on mu_idf
    res = rearrange_components(out[:5], mu_idf_true)

    # Build output array with stable column order (pads missing values with zeros)
    result = jnp.zeros((res.shape[0], len(RESULT_COLUMNS)))
    result = result.at[:, ResCol.comp_id].set(res[:, 0])
    result = result.at[:, ResCol.orig_comp_id].set(res[:, 1])
    result = result.at[:, ResCol.weight].set(res[:, 2])
    result = result.at[:, ResCol.mu_feat_est].set(res[:, 3])
    result = result.at[:, ResCol.mu_idf_est].set(res[:, 4])
    result = result.at[:, ResCol.sigma_feat_est].set(res[:, 5])
    result = result.at[:, ResCol.sigma_idf_est].set(res[:, 6])
    # True parameters
    result = result.at[:, ResCol.mu_feat_true].set(mu_feat_true)
    result = result.at[:, ResCol.mu_idf_true].set(mu_idf_true)
    result = result.at[:, ResCol.sigma_feat_true].set(sigma_feat_true)
    result = result.at[:, ResCol.sigma_idf_true].set(sigma_idf_true)
    result = result.at[:, ResCol.sigma_feat_true_flipped].set(jnp.flip(sigma_feat_true))
    result = result.at[:, ResCol.n_samples].set(n_samples)
    # Sample statistics / VBEM_MIX fields are unavailable in this routine; remain zero-filled

    return result

def plot_X(X, mu_true=None, mu_est=None, sigma_true=None, est_weights=None, title=None):
    """
    Plots the data points in X, coloring the first and last 50 samples differently.

    - First 50 samples: blue
    - Last 50 samples: red
    - True means (mu_true): large filled circles
    - Estimated means (mu_est): large hollow circles
    - If est_weights is provided, the border color of each dot varies between blue and red.

    :param X: (100, 2) array of data points.
    :param mu_true: (2, 2) array of true means (optional).
    :param mu_est: (2, 2) array of estimated means (optional).
    :param sigma_true: (2, 2) array of standard deviations for the components (optional).
    :param est_weights: (100, 2) array of estimated responsibilities for the two components (optional).
    :param title: Title for the plot (optional).
    """
    # Wrap the first dimension of X to be within [-180, 180] degrees
    X = X.at[:, 0].set(angular_difference(X[:, 0], 0))

    # Split the data into the first and last 50 samples
    X_first = X[:50]
    X_last = X[50:]

    # Create a plot with a specific size and layout
    plt.subplots(figsize=(7, 7), constrained_layout=True)

    # Determine border colors based on responsibilities if provided
    if est_weights is not None:
        colors = [(r, 0, 1 - r) for r in est_weights[:, 1]]  # Red intensity based on second component responsibility
    else:
        colors = ['black'] * 100  # Default black border

    # Plot the first 50 samples in blue and the last 50 samples in red
    plt.scatter(X_first[:, 0], X_first[:, 1], color='blue', edgecolors=colors[:50], label='First 50 samples')
    plt.scatter(X_last[:, 0], X_last[:, 1], color='red', edgecolors=colors[50:], label='Last 50 samples')

    def plot_ellipse(mu, sigma, color):
        """
        Plots an ellipse representing the covariance matrix.

        :param mu: Mean of the component.
        :param sigma: Standard deviation of the component.
        :param color: Color of the ellipse.
        """
        width, height = sigma
        ellipse = Ellipse(mu, width=width, height=height, edgecolor=color, fill=False, linestyle='--', linewidth=2)
        plt.gca().add_patch(ellipse)

    # Plot true means and their corresponding ellipses if provided
    if mu_true is not None:
        plt.scatter(mu_true[0, 0], mu_true[0, 1], color='blue', s=200, marker='o', edgecolors='black', label='True Mean (First)')
        plt.scatter(mu_true[1, 0], mu_true[1, 1], color='red', s=200, marker='o', edgecolors='black', label='True Mean (Last)')
        if sigma_true is not None:
            plot_ellipse(mu_true[0, :], sigma_true[0, :] * 2, 'blue')
            plot_ellipse(mu_true[1, :], sigma_true[1, :] * 2, 'red')

    # Plot estimated means if provided
    if mu_est is not None:
        plt.scatter(mu_est[0, 0], mu_est[0, 1], color='blue', s=200, marker='o', facecolors='none', edgecolors='black', label='Estimated Mean (First)')
        plt.scatter(mu_est[1, 0], mu_est[1, 1], color='red', s=200, marker='o', facecolors='none', edgecolors='black', label='Estimated Mean (Last)')

    # Set plot labels and title
    plt.xlabel('X[:, 0]')
    plt.ylabel('X[:, 1]')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(title if title else 'Scatter Plot of X with True and Estimated Means')
    plt.show()

@jax.jit
def angular_difference(angle1, angle2):
    """
    Computes the angular difference between two angles in 360-degree space.

    :param angle1: First angle(s) in degrees.
    :param angle2: Second angle(s) in degrees.
    :return: Minimum angular difference in [-180, 180] degrees.
    """
    return jnp.mod(angle1 - angle2 + 180, 360) - 180

# JIT compile the angular_difference function for performance
angular_difference_jit = jax.jit(angular_difference)

def circular_std(angles_deg):
    """
    Calculate circular standard deviation of angles in degrees (0-360 range)
    using JAX and complex number representation.

    Parameters:
    -----------
    angles_deg : array-like
        Angles in degrees (0-360 range)

    Returns:
    --------
    float
        Circular standard deviation in degrees
    """
    # Convert degrees to radians
    angles_rad = jnp.radians(angles_deg)

    # Convert to unit vectors in complex plane (exp(i*theta))
    complex_vectors = jnp.exp(1j * angles_rad)

    # Calculate mean resultant vector
    mean_vector = jnp.mean(complex_vectors)

    # Calculate resultant vector length (R)
    R = jnp.abs(mean_vector)

    # Clip R to avoid log(0) issues
    R_clipped = jnp.clip(R, 1e-10, 1.0)

    # Circular standard deviation
    # Using formula: std = sqrt(-2 * ln(R))
    circular_std_rad = jnp.sqrt(-2 * jnp.log(R_clipped))

    # Convert back to degrees
    circular_std_deg = jnp.degrees(circular_std_rad)

    return circular_std_deg
circular_std_jit = jax.jit(circular_std)
#%%
