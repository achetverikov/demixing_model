"""Loss functions for neural network surface optimization."""

import jax
import jax.numpy as jnp
from shared.surface_functions import normalize_to_density


def kl_divergence_loss(pred_log_density, target_log_density):
    """KL divergence between predicted and target log-probability densities"""

    # KL divergence: sum over mu1_error (axis=1), mean over batch and feat_diff
    # No dx factor needed since normalize_to_density already accounts for grid spacing
    kl_integrand = jnp.exp(target_log_density) * (target_log_density - pred_log_density)
    kl_div = jnp.sum(kl_integrand, axis=1)  # Sum over mu1_error dimension (axis=1)

    return jnp.mean(kl_div)  # Mean over batch and feat_diff


def expectation_loss(pred_log_density, target_log_density):
    """Loss based on expected mu1_error values"""
    from shared.config import config
    
    # Normalize to proper densities
    pred_density = jnp.exp(pred_log_density)
    target_density = jnp.exp(target_log_density)

    # Grid setup - mu1_error values for expectation calculation from config
    mu1_error_min, mu1_error_max = config.mu1_bias_range
    n_mu1_points = pred_log_density.shape[1]  # Get actual number of points from data
    mu1_error_values = jnp.linspace(mu1_error_min, mu1_error_max, n_mu1_points)

    # Compute expectations: sum over mu1_error (axis=1)
    # No dx factor needed since densities are already properly normalized
    pred_expectation = jnp.sum(mu1_error_values[None, :, None] * pred_density, axis=1)  # Shape: (batch, feat_diff)
    target_expectation = jnp.sum(mu1_error_values[None, :, None] * target_density, axis=1)  # Shape: (batch, feat_diff)

    return jnp.mean((pred_expectation - target_expectation) ** 2)


def smoothness_regularization(log_density):
    """Penalize rapid changes in the likelihood surface"""
    # Gradients along both spatial dimensions
    grad_mu1 = jnp.diff(log_density, axis=1)  # Along mu1_error
    grad_feat = jnp.diff(log_density, axis=2)  # Along feat_diff

    return jnp.mean(grad_mu1 ** 2) + jnp.mean(grad_feat ** 2)


def combined_probabilistic_loss(pred_log_probs, target_log_probs,
                                kl_weight=0.4, expectation_weight=0.3,
                                mse_weight=0.2, smooth_weight=0.1):
    """Combined loss respecting probabilistic nature"""

    # Traditional MSE loss
    mse = jnp.mean((pred_log_probs - target_log_probs) ** 2)

    pred_log_density, target_log_density = normalize_to_density(pred_log_probs), normalize_to_density(target_log_probs)
    # Probabilistic losses
    kl_loss = kl_divergence_loss(pred_log_density, target_log_density)
    exp_loss = expectation_loss(pred_log_density, target_log_density)
    smooth_loss = smoothness_regularization(pred_log_density)

    total_loss = (mse_weight * mse +
                  kl_weight * kl_loss +
                  expectation_weight * exp_loss +
                  smooth_weight * smooth_loss)

    return total_loss, {
        'mse': mse,
        'kl': kl_loss,
        'expectation': exp_loss,
        'smoothness': smooth_loss,
        'total': total_loss
    }
