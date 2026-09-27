"""Curves and motor convolution for stored simulation surfaces."""

import jax
import jax.numpy as jnp

from shared.config import config
from shared.utils import compute_single_bias_curve, compute_single_density_asymmetry


def bias_curves(log_surfaces):
    indices = jnp.arange(log_surfaces.shape[-1])
    grid = config.create_grid("mu_feat_bias")
    return jax.vmap(lambda surface: compute_single_bias_curve(surface, indices, grid))(
        log_surfaces)


def density_asymmetry_curves(log_surfaces, smoothing_sigma=None):
    indices = jnp.arange(log_surfaces.shape[-1])
    grid = config.create_grid("mu_feat_bias")
    sigma = 20.0 / config.feat_diff_step if smoothing_sigma is None else smoothing_sigma
    return jax.vmap(lambda surface: compute_single_density_asymmetry(
        surface, indices, grid, smoothing_sigma=sigma))(log_surfaces)


def apply_motor_noise(log_surfaces, sd_motor):
    """Convolve stored mu_feat densities on their periodic bias grid."""
    n_bias = log_surfaces.shape[1]
    offsets = jnp.arange(n_bias) - n_bias // 2
    weights = jnp.exp(-0.5 * (offsets * config.mu_feat_bias_step /
                              jnp.maximum(sd_motor, 0.01)) ** 2)
    weights /= jnp.sum(weights)
    kernel = jnp.zeros(n_bias).at[offsets % n_bias].set(weights)
    log_max = jnp.max(log_surfaces, axis=(1, 2), keepdims=True)
    probabilities = jnp.exp(log_surfaces - log_max)
    convolved = jnp.real(jnp.fft.ifft(
        jnp.fft.fft(probabilities, axis=1) * jnp.fft.fft(kernel)[None, :, None],
        axis=1))
    return jnp.log(jnp.clip(convolved, 0.0) + 1e-10) + log_max
