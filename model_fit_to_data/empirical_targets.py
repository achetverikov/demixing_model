"""Surrogate-neutral empirical target construction primitives."""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from shared.config import config


@jax.jit
def compute_target_bias_curve_core(
        feat_diff_values: jnp.ndarray, bias_values: jnp.ndarray,
        n_trials: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the historical binned circular-mean target for one condition."""
    max_trials = len(feat_diff_values)
    valid_mask = jnp.arange(max_trials) < n_trials
    valid_feat_diff = jnp.where(valid_mask, feat_diff_values, 0.0)
    valid_bias = jnp.where(valid_mask, bias_values, 0.0)

    bin_step = config.feat_diff_step * 2
    min_feat = config.feat_diff_range[0]
    max_feat = config.feat_diff_range[1]
    bin_centers = jnp.arange(min_feat, max_feat + bin_step, bin_step)
    n_bins = len(bin_centers)

    distances = jnp.abs(valid_feat_diff[:, None] - bin_centers[None, :])
    distances = jnp.where(valid_mask[:, None], distances, jnp.inf)
    nearest = jnp.argmin(distances, axis=1)

    counts = jax.ops.segment_sum(
        jnp.where(valid_mask, 1.0, 0.0), nearest, num_segments=n_bins)
    bias_rad = jnp.radians(valid_bias)
    cos_sums = jax.ops.segment_sum(
        jnp.where(valid_mask, jnp.cos(bias_rad), 0.0), nearest, num_segments=n_bins)
    sin_sums = jax.ops.segment_sum(
        jnp.where(valid_mask, jnp.sin(bias_rad), 0.0), nearest, num_segments=n_bins)

    target_bias = jnp.where(
        counts > 0,
        jnp.degrees(jnp.arctan2(
            sin_sums / jnp.maximum(counts, 1),
            cos_sums / jnp.maximum(counts, 1))),
        0.0)
    weights = counts.astype(jnp.float32)

    feat_grid = config.create_grid("feat_diff")
    grid_distances = jnp.abs(bin_centers[:, None] - feat_grid[None, :])
    feat_indices = jnp.argmin(grid_distances, axis=1)
    return feat_indices, target_bias, weights


def compute_bwcrps_condition_targets(fd_vals, bias_vals, fd_grid, d_circ, weights_sd,
                                     bias_low, bias_step, n_bias):
    """Build empirical targets for balanced and bias-weighted circular CRPS."""
    fd_vals = np.asarray(fd_vals)
    bias_vals = np.asarray(bias_vals)

    gaussian = np.exp(
        -0.5 * ((fd_vals[None, :] - fd_grid[:, None]) / weights_sd) ** 2)
    support = gaussian.sum(axis=1)

    bias_bin = np.mod(
        np.round((bias_vals - bias_low) / bias_step).astype(int), n_bias)
    one_hot = np.zeros((len(bias_vals), n_bias))
    one_hot[np.arange(len(bias_vals)), bias_bin] = 1.0
    empirical = (gaussian @ one_hot) / np.maximum(support[:, None], 1e-10)

    target_d = empirical @ d_circ
    support_threshold = np.median(support) * 0.01
    support_mask = (support > support_threshold).astype(np.float32)

    bias_rad = np.deg2rad(bias_vals)
    mean_bias = np.rad2deg(np.arctan2(
        gaussian @ np.sin(bias_rad), gaussian @ np.cos(bias_rad)))
    return target_d, support_mask, mean_bias ** 2 * support_mask
