"""Surrogate-neutral fitting objectives.

These functions define losses that are shared by the maintained wrapped-normal
mixture path and the historical surface path. They intentionally know nothing
about surrogate loading, surface generation, or optimizer state.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from model_fit_to_data.density_objective import DEGENERATE_TARGET_EPS


def compute_curve_losses(predicted_curves: jnp.ndarray, target_curves: jnp.ndarray,
                         loss_type: str = "mse", is_angular: bool = False,
                         weights: jnp.ndarray = None, corr_weight=0.5) -> jnp.ndarray:
    """Compute per-curve losses for the shared curve objectives."""
    if is_angular:
        diff = ((predicted_curves - target_curves + 180) % 360) - 180
    else:
        diff = predicted_curves - target_curves

    loss_type = loss_type.lower()
    if loss_type in ("mse", "mad"):
        if weights is not None:
            if weights.ndim == 1:
                weights = jnp.broadcast_to(weights[None, :], predicted_curves.shape)
            valid_mask = weights > 0
            total_weights = jnp.sum(jnp.where(valid_mask, weights, 0.0), axis=1)
            if loss_type == "mse":
                error = jnp.where(valid_mask, (diff ** 2) * weights, 0.0)
            else:
                error = jnp.where(valid_mask, jnp.abs(diff) * weights, 0.0)
            return jnp.sum(error, axis=1) / jnp.maximum(total_weights, 1.0)
        if loss_type == "mse":
            return jnp.mean(diff ** 2, axis=1)
        return jnp.mean(jnp.abs(diff), axis=1)

    if loss_type == "combined":
        mse_losses = jnp.mean(diff ** 2, axis=1)

        def correlation_loss(pred, target):
            corr = jnp.corrcoef(pred, target)[0, 1]
            return 1 - jnp.where(jnp.isnan(corr), 0.0, corr)

        corr_losses = jax.vmap(correlation_loss)(predicted_curves, target_curves)
        target_range = jnp.abs(
            jnp.max(target_curves, axis=1) - jnp.min(target_curves, axis=1))
        mse_scaled = mse_losses / jnp.maximum(target_range, 1e-6)
        return (1 - corr_weight) * mse_scaled + corr_weight * corr_losses

    if loss_type == "ccc":
        pred_mean = jnp.mean(predicted_curves, axis=1)
        target_mean = jnp.mean(target_curves, axis=1)
        pred_centered = predicted_curves - pred_mean[:, None]
        target_centered = target_curves - target_mean[:, None]
        covariance = jnp.mean(pred_centered * target_centered, axis=1)
        pred_var = jnp.mean(pred_centered ** 2, axis=1)
        target_var = jnp.mean(target_centered ** 2, axis=1)
        denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
        safe = jnp.where(denominator < DEGENERATE_TARGET_EPS, 1.0, denominator)
        return 1 - (2 * covariance) / safe

    if loss_type == "nrmse":
        rmse = jnp.sqrt(jnp.mean(diff ** 2, axis=1))
        target_range = jnp.abs(
            jnp.max(target_curves, axis=1) - jnp.min(target_curves, axis=1))
        return rmse / jnp.maximum(target_range, 1e-6)

    raise ValueError(
        f"Unknown loss type {loss_type}. Should be one of mse, mad, combined, ccc, or nrmse.")


def bwcrps_energy_score(prob_surfaces, target_d, fd_weights, d_circ, norm_floor=None):
    """Weighted energy score shared by balanced and bias-weighted CRPS."""
    n_unique, n_bias, n_feat = prob_surfaces.shape
    prob_fk = prob_surfaces.transpose(0, 2, 1).reshape(-1, n_bias)
    d_prob = prob_fk @ d_circ
    self_energy = (prob_fk * d_prob).sum(axis=1).reshape(n_unique, n_feat)

    weighted_target = target_d * fd_weights[:, :, None]
    cross = jnp.einsum("ukj,cjk->uc", prob_surfaces, weighted_target)
    self_energy_by_condition = self_energy @ fd_weights.T

    normalizer = fd_weights.sum(axis=1)
    if norm_floor is not None:
        normalizer = jnp.maximum(normalizer, norm_floor)
    return (2.0 * cross - self_energy_by_condition) / normalizer[None, :]
