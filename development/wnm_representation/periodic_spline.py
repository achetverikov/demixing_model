"""Maximum-likelihood periodic cubic-spline log densities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def wrap_deg(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def basis(x: np.ndarray, knots: int) -> np.ndarray:
    """Uniform cyclic cardinal-cubic B-spline basis on the circle."""
    centers = np.linspace(-180.0, 180.0, knots, endpoint=False)
    u = np.abs(wrap_deg(np.asarray(x)[..., None] - centers)) / (360.0 / knots)
    values = np.where(u < 1.0, 2.0 / 3.0 - u ** 2 + u ** 3 / 2.0,
                      np.where(u < 2.0, (2.0 - u) ** 3 / 6.0, 0.0))
    return values


@dataclass(frozen=True)
class PeriodicSplineDensity:
    theta: np.ndarray
    quadrature_size: int = 2880

    @property
    def knots(self) -> int:
        return len(self.theta) + 1

    def _full_theta(self) -> np.ndarray:
        return np.r_[self.theta, 0.0]

    @cached_property
    def log_normalizer(self) -> float:
        grid = np.linspace(-180.0, 180.0, self.quadrature_size, endpoint=False)
        return float(logsumexp(basis(grid, self.knots) @ self._full_theta())
                     + np.log(360 / len(grid)))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return basis(x, self.knots) @ self._full_theta() - self.log_normalizer

    def density_grid(self, grid: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(grid))

    def circular_stats(self) -> dict[str, float]:
        grid = np.linspace(-180.0, 180.0, self.quadrature_size, endpoint=False)
        mass = self.density_grid(grid) * (360.0 / len(grid))
        moment = np.sum(mass * np.exp(1j * np.radians(grid)))
        resultant = float(abs(moment))
        return {"mean_bias": float(np.degrees(np.angle(moment))),
                "resultant": resultant,
                "response_sd": float(np.degrees(np.sqrt(
                    -2 * np.log(np.clip(resultant, 1e-12, 1.0)))))}

    def asymmetry(self) -> float:
        grid = np.linspace(-180.0, 180.0, self.quadrature_size, endpoint=False)
        mass = self.density_grid(grid) * (360.0 / len(grid))
        valid = ~np.isclose(np.abs(grid), 180.0)
        return float(np.sum(mass[valid] * np.sign(grid[valid])))


@dataclass(frozen=True)
class SplineFit:
    density: PeriodicSplineDensity
    success: bool
    n_iterations: int
    gradient_norm: float


def fit_periodic_spline(samples: np.ndarray, knots: int, quadrature_size: int = 2880,
                        maxiter: int = 1000, initial: np.ndarray | None = None) -> SplineFit:
    finite = np.asarray(samples)[np.isfinite(samples)]
    empirical = basis(finite, knots).mean(axis=0)
    grid = np.linspace(-180.0, 180.0, quadrature_size, endpoint=False)
    grid_basis = basis(grid, knots)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        full = np.r_[theta, 0.0]
        logits = grid_basis @ full
        probability = np.exp(logits - logsumexp(logits))
        value = logsumexp(logits) + np.log(360 / quadrature_size) - empirical @ full
        gradient = (probability @ grid_basis - empirical)[:-1]
        return float(value), gradient

    start = np.zeros(knots - 1) if initial is None else np.asarray(initial)
    result = minimize(objective, start, jac=True, method="L-BFGS-B",
                      options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8})
    return SplineFit(PeriodicSplineDensity(result.x, quadrature_size), bool(result.success),
                     int(result.nit), float(np.linalg.norm(result.jac)))
