"""Positive maximum-entropy Fourier log densities on the 360-degree circle."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from continuous_density.fourier_moments import direct_asymmetry


def periodic_grid(size: int = 4096) -> np.ndarray:
    if size < 8:
        raise ValueError("quadrature size must be at least 8")
    return np.linspace(-180.0, 180.0, size, endpoint=False)


def _features(grid_deg: np.ndarray, k: int) -> np.ndarray:
    angle = np.radians(np.asarray(grid_deg, dtype=np.float64))[:, None]
    harmonic = np.arange(1, k + 1)[None, :]
    return np.concatenate([np.cos(angle * harmonic), np.sin(angle * harmonic)], axis=1)


@dataclass(frozen=True)
class MaxentFourierDensity:
    """``log p = theta . [cos(kb), sin(kb)] - log Z`` per degree."""

    theta: np.ndarray
    quadrature_size: int = 4096

    @property
    def n_harmonics(self) -> int:
        return len(self.theta) // 2

    @cached_property
    def _default_log_normalizer(self) -> float:
        grid = periodic_grid(self.quadrature_size)
        eta = _features(grid, self.n_harmonics) @ self.theta
        return float(logsumexp(eta) + np.log(360.0 / len(grid)))

    def log_normalizer(self, quadrature_size: int | None = None) -> float:
        if quadrature_size is None:
            return self._default_log_normalizer
        grid = periodic_grid(quadrature_size or self.quadrature_size)
        eta = _features(grid, self.n_harmonics) @ self.theta
        return float(logsumexp(eta) + np.log(360.0 / len(grid)))

    def logpdf(self, bias_deg: np.ndarray) -> np.ndarray:
        x = np.asarray(bias_deg, dtype=np.float64)
        return (_features(x.ravel(), self.n_harmonics) @ self.theta
                - self.log_normalizer()).reshape(x.shape)

    def density_grid(self, grid_deg: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(grid_deg))

    def moments(self, kmax: int | None = None) -> np.ndarray:
        kmax = self.n_harmonics if kmax is None else int(kmax)
        grid = periodic_grid(self.quadrature_size)
        mass = self.density_grid(grid) * (360.0 / len(grid))
        angle = np.radians(grid)
        k = np.arange(kmax + 1)
        return np.sum(mass[:, None] * np.exp(1j * angle[:, None] * k), axis=0)

    def circular_stats(self) -> dict[str, float]:
        m1 = self.moments(1)[1]
        resultant = float(abs(m1))
        return {"mean_bias": float(np.degrees(np.angle(m1))),
                "resultant": resultant,
                "response_sd": float(np.degrees(np.sqrt(
                    -2.0 * np.log(np.clip(resultant, 1e-12, 1.0)))))}

    def asymmetry(self) -> float:
        grid = periodic_grid(self.quadrature_size)
        mass = self.density_grid(grid) * (360.0 / len(grid))
        return float(mass[(grid > 0) & (grid < 180)].sum()
                     - mass[(grid < 0) & (grid > -180)].sum())


@dataclass(frozen=True)
class FitResult:
    density: MaxentFourierDensity
    success: bool
    message: str
    n_iterations: int
    train_nll: float
    gradient_norm: float


def fit_maxent_fourier(samples: np.ndarray, n_harmonics: int,
                       quadrature_size: int = 4096, maxiter: int = 1000,
                       initial: np.ndarray | None = None) -> FitResult:
    """Fit one log-Fourier density by raw-sample maximum likelihood.

    The sufficient statistics are empirical sine/cosine moments.  No density
    grid or KDE is fitted to the samples; the grid is periodic quadrature for Z.
    """
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0 or n_harmonics < 1:
        raise ValueError("finite samples and at least one harmonic are required")
    grid = periodic_grid(quadrature_size)
    grid_features = _features(grid, n_harmonics)
    empirical = _features(x, n_harmonics).mean(axis=0)
    log_dx = np.log(360.0 / quadrature_size)

    def objective(theta):
        eta = grid_features @ theta
        log_z = logsumexp(eta) + log_dx
        mass = np.exp(eta - logsumexp(eta))
        model_moments = mass @ grid_features
        value = log_z - theta @ empirical
        return value, model_moments - empirical

    theta0 = (np.zeros(2 * n_harmonics) if initial is None
              else np.asarray(initial, dtype=np.float64))
    if theta0.shape != (2 * n_harmonics,):
        raise ValueError("initial must contain cosine then sine coefficients")
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B",
                      options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8,
                               "maxls": 50})
    density = MaxentFourierDensity(np.asarray(result.x), quadrature_size)
    return FitResult(density, bool(result.success), str(result.message), int(result.nit),
                     float(-density.logpdf(x).mean()),
                     float(np.linalg.norm(np.asarray(result.jac))))


def quadrature_convergence(density: MaxentFourierDensity,
                           sizes: tuple[int, ...] = (2048, 4096, 8192)) -> dict:
    """Normalization and log-Z changes across nested periodic quadratures."""
    log_z = np.array([density.log_normalizer(size) for size in sizes])
    mass = []
    for size in sizes:
        grid = periodic_grid(size)
        mass.append(float(density.density_grid(grid).sum() * 360.0 / size))
    return {"sizes": list(sizes), "log_normalizer": log_z.tolist(), "mass": mass,
            "max_logz_step": float(np.max(np.abs(np.diff(log_z))))}


def heldout_metrics(density: MaxentFourierDensity, samples: np.ndarray) -> dict[str, float]:
    """Core local metrics from untouched raw samples."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]
    raw_angle = np.radians(x)
    raw_m1 = np.mean(np.exp(1j * raw_angle))
    model = density.circular_stats()
    raw_r = float(abs(raw_m1))
    return {
        "nll": float(-density.logpdf(x).mean()),
        "raw_mean_bias": float(np.degrees(np.angle(raw_m1))),
        "pred_mean_bias": model["mean_bias"],
        "raw_resultant": raw_r, "pred_resultant": model["resultant"],
        "raw_response_sd": float(np.degrees(np.sqrt(
            -2.0 * np.log(np.clip(raw_r, 1e-12, 1.0))))),
        "pred_response_sd": model["response_sd"],
        "raw_density_asymmetry": float(direct_asymmetry(x)[0]),
        "pred_density_asymmetry": density.asymmetry(),
    }
