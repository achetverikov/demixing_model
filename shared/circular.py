"""Lightweight circular-curve utilities used by maintained prediction code."""
import jax.numpy as jnp

def gaussian_curve_smoother(curve: jnp.ndarray, smoothing_sigma: float) -> jnp.ndarray:
    """Smooth a curve with a normalized Gaussian kernel over edge-padded input.

    The one implementation of the model-side curve smoother.  It is deliberately
    not a general filter: ``smoothing_sigma`` is in *grid steps*, the kernel spans
    ``4*sigma`` rounded up to an odd length, and the input is edge-padded so the
    output keeps its length.  Those three choices are part of the density target's
    definition -- the empirical side smooths at actual trial locations while the
    model side smooths a finite grid with edge padding, and that asymmetry is
    established, not accidental (MODEL_PIPELINE_FOR_AGENTS.md S7.4).  Changing any
    of them changes the estimator, not just its numerics.

    Args:
        curve: 1-D array over a uniform grid.
        smoothing_sigma: Gaussian SD in grid steps.

    Returns:
        Smoothed curve, same length as ``curve``.
    """
    kernel_size = int(4 * smoothing_sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size

    x = jnp.arange(kernel_size) - kernel_size // 2
    kernel = jnp.exp(-0.5 * (x / smoothing_sigma) ** 2)
    kernel = kernel / jnp.sum(kernel)

    pad_width = kernel_size // 2
    padded = jnp.pad(curve, pad_width, mode='edge')
    # The kernel is symmetric, so convolve and correlate agree here; convolve is
    # what the surface path has always used and is kept for that reason.
    return jnp.convolve(padded, kernel, mode='valid')
