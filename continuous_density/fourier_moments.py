"""Fourier diagnostics and grid distances for circular bias distributions.

Angles are in degrees and empirical coefficients follow
``c_k = E[exp(i k b)]``.  Dense grids are evaluation tools only; fitting data
remain raw simulator outcomes.
"""

from __future__ import annotations

import numpy as np


def empirical_coefficients(samples: np.ndarray, kmax: int,
                           chunk: int = 32) -> np.ndarray:
    """Return finite-aware ``c_0, ..., c_kmax`` for each row of samples.

    ``samples`` may be one- or two-dimensional.  Harmonics are chunked to avoid
    constructing a ``rows x samples x kmax`` tensor for the 100k corpora.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or kmax < 0:
        raise ValueError("samples must be (rows, outcomes) and kmax must be non-negative")
    finite = np.isfinite(x)
    count = finite.sum(axis=1)
    radians = np.radians(np.where(finite, x, 0.0))
    out = np.empty((len(x), kmax + 1), dtype=np.complex128)
    out[:, 0] = np.where(count > 0, 1.0, np.nan)
    for first in range(1, kmax + 1, chunk):
        k = np.arange(first, min(first + chunk, kmax + 1))
        z = np.exp(1j * radians[..., None] * k)
        z *= finite[..., None]
        out[:, k] = z.sum(axis=1) / np.maximum(count[:, None], 1)
    out[count == 0] = np.nan
    return out


def direct_asymmetry(samples: np.ndarray) -> np.ndarray:
    """Return ``P(b > 0) - P(b < 0)``, excluding zero and the antipode."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    finite = np.isfinite(x)
    antipode = np.isclose(np.abs(x), 180.0, atol=1e-6)
    count = finite.sum(axis=1)
    signed = ((finite & ~antipode & (x > 0)).sum(axis=1)
              - (finite & ~antipode & (x < 0)).sum(axis=1))
    return np.where(count > 0, signed / np.maximum(count, 1), np.nan)


def asymmetry_from_coefficients(coefficients: np.ndarray,
                                kmax: int | None = None) -> np.ndarray:
    """Odd-sine partial sum for density asymmetry."""
    c = np.asarray(coefficients)
    if c.ndim == 1:
        c = c[None, :]
    available = c.shape[1] - 1
    kmax = available if kmax is None else min(int(kmax), available)
    k = np.arange(1, kmax + 1)
    odd = k % 2 == 1
    return (4.0 / np.pi) * np.sum(c[:, 1:kmax + 1].imag[:, odd]
                                  / k[odd], axis=1)


def linear_fourier_density(coefficients: np.ndarray, grid_deg: np.ndarray,
                           kmax: int | None = None) -> np.ndarray:
    """Evaluate the linear Fourier reconstruction as a per-degree density.

    This diagnostic reconstruction may be negative.  It must not be used as a
    fitted likelihood without a positivity-preserving parameterization.
    """
    c = np.asarray(coefficients)
    if c.ndim == 1:
        c = c[None, :]
    available = c.shape[1] - 1
    kmax = available if kmax is None else min(int(kmax), available)
    k = np.arange(1, kmax + 1)
    phase = np.exp(-1j * np.radians(np.asarray(grid_deg))[None, :, None] * k)
    series = 1.0 + 2.0 * np.real(np.sum(c[:, None, 1:kmax + 1] * phase, axis=-1))
    return series / 360.0


def fejer_density(coefficients: np.ndarray, grid_deg: np.ndarray,
                  kmax: int | None = None) -> np.ndarray:
    """Positive Cesaro/Fejer reconstruction of empirical circular moments."""
    c = np.asarray(coefficients)
    if c.ndim == 1:
        c = c[None, :]
    available = c.shape[1] - 1
    kmax = available if kmax is None else min(int(kmax), available)
    weighted = c.copy()
    k = np.arange(1, kmax + 1)
    weighted[:, 1:kmax + 1] *= 1.0 - k / (kmax + 1.0)
    density = linear_fourier_density(weighted, grid_deg, kmax)
    return np.maximum(density, 0.0)


def circular_wasserstein_grid(p: np.ndarray, q: np.ndarray,
                              cell_width_deg: float) -> np.ndarray:
    """Exact circular W1 for equal-spaced probability masses, in degrees.

    For a circle the arbitrary transport flow across the cut is optimized by
    subtracting the median cumulative imbalance.
    """
    p = np.atleast_2d(np.asarray(p, dtype=np.float64))
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("circular Wasserstein requires non-negative masses")
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    flow = np.cumsum(p - q, axis=1)
    offset = np.median(flow, axis=1, keepdims=True)
    return float(cell_width_deg) * np.sum(np.abs(flow - offset), axis=1)


def required_asymmetry_bandwidth(coefficients: np.ndarray, reference: np.ndarray,
                                 margin: float) -> np.ndarray:
    """Smallest K after which every partial-sum error stays within ``margin``."""
    c = np.asarray(coefficients)
    ref = np.asarray(reference, dtype=np.float64)
    errors = []
    for k in range(1, c.shape[1]):
        errors.append(np.abs(asymmetry_from_coefficients(c, k) - ref))
    errors = np.stack(errors, axis=1)
    stable_error = np.maximum.accumulate(errors[:, ::-1], axis=1)[:, ::-1]
    within = stable_error <= margin
    answer = np.where(within.any(axis=1), np.argmax(within, axis=1) + 1, -1)
    return answer
