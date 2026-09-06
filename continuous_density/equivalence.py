"""Curve-level summaries and uncertainty-aware practical equivalence tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


DEFAULT_MARGINS = {
    "density_asymmetry": {"coherent": 0.002, "pointwise": 0.005},
    "mean_bias_r_ge_0.8_deg": {"coherent": 0.10, "pointwise": 0.25},
    "mean_bias_r_0.5_0.8_deg": {"coherent": 0.25, "pointwise": 0.50},
    "response_sd_deg": {"coherent": 0.10, "pointwise": 0.25},
    "excess_nll_nat": {"coherent": 0.0005, "pointwise": 0.002},
}


def wrap_deg(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) + 180.0) % 360.0 - 180.0


def longest_sign_run(error: np.ndarray) -> tuple[int, float, int, int]:
    """Length, signed mean, start, and inclusive end of the longest nonzero run."""
    e = np.asarray(error, dtype=np.float64)
    valid = np.isfinite(e) & (e != 0)
    best = (0, np.nan, -1, -1)
    start = 0
    while start < len(e):
        if not valid[start]:
            start += 1
            continue
        sign = np.sign(e[start])
        end = start + 1
        while end < len(e) and valid[end] and np.sign(e[end]) == sign:
            end += 1
        if end - start > best[0]:
            best = (end - start, float(e[start:end].mean()), start, end - 1)
        start = end
    return best


def curve_summary(predicted: np.ndarray, reference: np.ndarray,
                  circular: bool = False) -> dict[str, float | int]:
    """Mandatory dissimilarity-aware diagnostics for one ordered trajectory."""
    pred = np.asarray(predicted, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    error = wrap_deg(pred - ref) if circular else pred - ref
    finite = np.isfinite(error) & np.isfinite(ref) & np.isfinite(pred)
    if not finite.any():
        raise ValueError("trajectory contains no finite comparisons")
    e = error[finite]
    p = pred[finite]
    r = ref[finite]
    max_local = int(np.argmax(np.abs(e)))
    finite_indices = np.flatnonzero(finite)
    run = longest_sign_run(error)
    denom = np.sum((r - r.mean()) ** 2)
    slope = np.sum((r - r.mean()) * (p - p.mean())) / denom if denom > 0 else np.nan
    return {
        "n": len(e), "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "max_abs_error": float(np.abs(e[max_local])),
        "max_error": float(e[max_local]),
        "max_index": int(finite_indices[max_local]),
        "mean_signed_error": float(e.mean()), "amplitude_slope": float(slope),
        "longest_sign_run": run[0], "longest_run_mean_error": run[1],
        "longest_run_start": run[2], "longest_run_end": run[3],
    }


def bootstrap_trajectory(samples: np.ndarray, statistic: Callable[[np.ndarray], np.ndarray],
                         predicted: np.ndarray, n_boot: int = 1000, seed: int = 0,
                         shared_indices: bool = False,
                         circular: bool = False) -> np.ndarray:
    """Bootstrap prediction-minus-reference errors over a complete trajectory.

    ``statistic`` accepts a ``(points, outcomes)`` sample matrix and returns one
    value per point.  With ``shared_indices=True``, one resampled outcome index
    is applied to every point, preserving common-random-number covariance.
    """
    x = np.asarray(samples)
    pred = np.asarray(predicted, dtype=np.float64)
    if x.ndim != 2 or pred.shape != (x.shape[0],):
        raise ValueError("samples must be (points, outcomes) and predicted one per point")
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, x.shape[0]))
    for b in range(n_boot):
        if shared_indices:
            index = rng.integers(0, x.shape[1], x.shape[1])
            draw = x[:, index]
        else:
            index = rng.integers(0, x.shape[1], x.shape)
            draw = np.take_along_axis(x, index, axis=1)
        error = pred - np.asarray(statistic(draw))
        out[b] = wrap_deg(error) if circular else error
    return out


def bootstrap_circular_statistics_shared(samples: np.ndarray, n_boot: int = 1000,
                                         seed: int = 0,
                                         bootstrap_chunk: int = 25
                                         ) -> dict[str, np.ndarray]:
    """Efficient shared-index bootstrap of circular mean/SD/asymmetry.

    Multinomial resampling counts are multiplied by per-outcome sufficient
    statistics, preserving cross-point covariance without repeatedly copying the
    full sample tensor.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("samples must be (trajectory points, outcomes)")
    finite = np.isfinite(x)
    angle = np.radians(np.where(finite, x, 0.0))
    z = np.where(finite, np.exp(1j * angle), 0.0)
    antipode = np.isclose(np.abs(x), 180.0, atol=1e-6)
    sign = np.where(finite & ~antipode, np.sign(x), 0.0)
    rng = np.random.default_rng(seed)
    mean = np.empty((n_boot, len(x)))
    resultant = np.empty_like(mean)
    response_sd = np.empty_like(mean)
    asymmetry = np.empty_like(mean)
    probability = np.full(x.shape[1], 1.0 / x.shape[1])
    for first in range(0, n_boot, bootstrap_chunk):
        stop = min(first + bootstrap_chunk, n_boot)
        counts = rng.multinomial(x.shape[1], probability, size=stop - first)
        denominator = finite @ counts.T
        m1 = (z @ counts.T) / np.maximum(denominator, 1)
        asym = (sign @ counts.T) / np.maximum(denominator, 1)
        r = np.abs(m1)
        mean[first:stop] = np.degrees(np.angle(m1)).T
        resultant[first:stop] = r.T
        response_sd[first:stop] = np.degrees(np.sqrt(
            -2.0 * np.log(np.clip(r, 1e-12, 1.0)))).T
        asymmetry[first:stop] = asym.T
    return {"mean_bias": mean, "resultant": resultant,
            "response_sd": response_sd, "density_asymmetry": asymmetry}


def bootstrap_circular_statistics_independent(samples: np.ndarray, n_boot: int = 1000,
                                              seed: int = 0,
                                              bootstrap_chunk: int = 25
                                              ) -> dict[str, np.ndarray]:
    """Row-independent bootstrap for references generated without CRN."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("samples must be (trajectory points, outcomes)")
    output = {name: np.empty((n_boot, len(x))) for name in
              ("mean_bias", "resultant", "response_sd", "density_asymmetry")}
    rng = np.random.default_rng(seed)
    probability = np.full(x.shape[1], 1.0 / x.shape[1])
    for row, values in enumerate(x):
        finite = np.isfinite(values)
        angle = np.radians(np.where(finite, values, 0.0))
        z = np.where(finite, np.exp(1j * angle), 0.0)
        antipode = np.isclose(np.abs(values), 180.0, atol=1e-6)
        sign = np.where(finite & ~antipode, np.sign(values), 0.0)
        if len(values) > 20_000:
            # Resample many iid outcome blocks rather than allocate an
            # n_boot x multi-million-outcome multinomial matrix.
            n_units = 2000
            stop_at = (len(values) // n_units) * n_units
            block = stop_at // n_units
            finite = finite[:stop_at].reshape(n_units, block).sum(axis=1)
            z = z[:stop_at].reshape(n_units, block).sum(axis=1)
            sign = sign[:stop_at].reshape(n_units, block).sum(axis=1)
            probability_row = np.full(n_units, 1.0 / n_units)
        else:
            probability_row = probability
        for first in range(0, n_boot, bootstrap_chunk):
            stop = min(first + bootstrap_chunk, n_boot)
            counts = rng.multinomial(len(probability_row), probability_row,
                                     size=stop - first)
            denominator = counts @ finite
            m1 = (counts @ z) / np.maximum(denominator, 1)
            asymmetry = (counts @ sign) / np.maximum(denominator, 1)
            resultant = np.abs(m1)
            output["mean_bias"][first:stop, row] = np.degrees(np.angle(m1))
            output["resultant"][first:stop, row] = resultant
            output["response_sd"][first:stop, row] = np.degrees(np.sqrt(
                -2.0 * np.log(np.clip(resultant, 1e-12, 1.0))))
            output["density_asymmetry"][first:stop, row] = asymmetry
    return output


@dataclass(frozen=True)
class EquivalenceResult:
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    pointwise_pass: bool
    coherent_estimate: float
    coherent_lower: float
    coherent_upper: float
    coherent_pass: bool
    simultaneous_radius: float


def equivalence_from_bootstrap(error_draws: np.ndarray, pointwise_margin: float,
                               coherent_margin: float,
                               alpha: float = 0.05) -> EquivalenceResult:
    """Simultaneous pointwise and coherent equivalence from bootstrap errors.

    A max-deviation bootstrap radius gives family-wise pointwise intervals.  The
    coherent interval uses bootstrap quantiles of the trajectory mean.
    """
    draws = np.asarray(error_draws, dtype=np.float64)
    if draws.ndim != 2 or len(draws) < 2:
        raise ValueError("error_draws must be (bootstrap draws, trajectory points)")
    estimate = draws.mean(axis=0)
    centered = draws - estimate
    radius = float(np.quantile(np.max(np.abs(centered), axis=1), 1.0 - alpha))
    lower, upper = estimate - radius, estimate + radius
    coherent_draws = draws.mean(axis=1)
    coherent_estimate = float(coherent_draws.mean())
    coherent_lower, coherent_upper = np.quantile(
        coherent_draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return EquivalenceResult(
        estimate, lower, upper,
        bool(np.all(lower > -pointwise_margin) and np.all(upper < pointwise_margin)),
        coherent_estimate, float(coherent_lower), float(coherent_upper),
        bool(coherent_lower > -coherent_margin and coherent_upper < coherent_margin),
        radius,
    )
