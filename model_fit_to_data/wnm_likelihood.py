"""Canonical WNM likelihood evaluation for fitted behavioral trials.

The WNM path reports a continuous density at each observation and, separately,
the exact probability mass of the reporting cell containing that observation.
All exporters and postprocessors should delegate here so the observation
convention and column transformations cannot drift.
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from model_fit_to_data.wnm_scoring import trial_log_density
from shared.config import config
from shared.mu1_axis import bin_indices, mu1_cell_width, mu1_grid


def _parameters(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size < 3:
        raise ValueError("WNM fit parameters need sd_feat1, sd_feat2, and sd_spat")
    if values.size == 3:
        values = np.concatenate([values, [0.0]])
    return values[:4]


def exact_cell_log_probability(predictor, parameters, feat_diff_deg, bias_deg):
    """Float64 log probability of each observation's reporting cell."""
    sd_feat1, sd_feat2, sd_spat, sd_motor = _parameters(parameters)
    bias = np.asarray(bias_deg)
    feat = np.asarray(feat_diff_deg)
    indices = np.asarray(
        bin_indices(jnp.asarray(bias, dtype=jnp.float32)))

    rows = jnp.stack([
        jnp.full(len(indices), sd_feat1, jnp.float32),
        jnp.full(len(indices), sd_feat2, jnp.float32),
        jnp.full(len(indices), sd_spat, jnp.float32),
        jnp.asarray(feat, dtype=jnp.float32),
    ], axis=-1)
    dist = predictor.distribution(
        rows, validate=False, sd_motor=float(sd_motor))

    mu = np.asarray(dist["mu"], dtype=np.float64)
    sigma = np.asarray(dist["sigma"], dtype=np.float64)
    weights = np.asarray(jnp.exp(dist["log_pi"]), dtype=np.float64)

    centres = np.asarray(mu1_grid())
    half = mu1_cell_width() / 2.0
    lows = (centres[indices] - half)[:, None]
    highs = (centres[indices] + half)[:, None]
    shifts = np.arange(-8, 9, dtype=np.float64) * 360.0

    mass = np.zeros(len(indices), dtype=np.float64)
    for shift in shifts:
        upper = (highs + shift - mu) / sigma
        lower = (lows + shift - mu) / sigma
        mass += np.sum(weights * (norm.cdf(upper) - norm.cdf(lower)), axis=-1)

    floored = int(np.sum(mass <= 0.0))
    if floored:
        warnings.warn(
            f"{floored} of {mass.size} cell masses are below the smallest positive "
            "normal double and were floored; these are genuine tail zeros, not lost "
            "float32 precision.",
            RuntimeWarning,
            stacklevel=2,
        )
    return np.log(np.maximum(mass, np.finfo(np.float64).tiny))


def evaluate_trial_likelihoods(
        predictor, parameters, feat_diff_deg, bias_deg, *,
        physical_bin_width_deg=None, include_cell_probability=False):
    """Evaluate the canonical WNM likelihood columns for one fitted condition."""
    sd_feat1, sd_feat2, sd_spat, sd_motor = _parameters(parameters)
    feat = jnp.asarray(np.asarray(feat_diff_deg), dtype=jnp.float32)
    bias = jnp.asarray(np.asarray(bias_deg), dtype=jnp.float32)
    log_density = np.asarray(trial_log_density(
        predictor, sd_feat1, sd_feat2, sd_spat, feat, bias,
        sd_motor=float(sd_motor)))

    model_bin_width = float(config.mu1_bias_step)
    if physical_bin_width_deg is None:
        physical_bin_width_deg = model_bin_width
    physical_bin_width_deg = float(physical_bin_width_deg)

    log_mass = log_density + np.log(model_bin_width)
    values = {
        "loglik_density_model_deg": log_density,
        "nll_density_model_deg": -log_density,
        "loglik_mass": log_mass,
        "nll_mass": -log_mass,
        "loglik_density_deg": log_mass - np.log(physical_bin_width_deg),
        "nll_density_deg": -log_mass + np.log(physical_bin_width_deg),
        "bin_width_deg": physical_bin_width_deg,
        "loglik_convention": "continuous_at_observation",
    }
    if include_cell_probability:
        values["loglik_cell_probability"] = exact_cell_log_probability(
            predictor, parameters, feat_diff_deg, bias_deg)
    return values
