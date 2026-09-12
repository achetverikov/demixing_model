"""On-demand WNM evaluation for the surface browser.

This is intentionally independent of Streamlit: the browser is a consumer of
the production prediction API, not a second implementation of mixture maths.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from shared import surrogate
from shared.config import config
from shared.mu1_axis import mu1_grid_np
from shared.prediction import mirror_params, predictor_from_surrogate


@dataclass(frozen=True)
class WNMView:
    identity: dict
    parameters: tuple[float, float, float]
    feat_diff: np.ndarray
    bias_grid: np.ndarray
    log_density: np.ndarray
    mean_bias: np.ndarray
    circular_sd: np.ndarray
    density_asymmetry: np.ndarray


def load_wnm_predictor(n_samples: int, checkpoint: Path | None = None):
    loaded = surrogate.load_surrogate(
        family=surrogate.FAMILY_WNM, n_samples=n_samples,
        checkpoint_path=checkpoint)
    return predictor_from_surrogate(loaded)


def evaluate_predictor(predictor, parameters, feat_diff=None, bias_grid=None) -> WNMView:
    """Evaluate both components continuously at one SD triple."""
    parameters = tuple(float(value) for value in parameters)
    if len(parameters) != 3:
        raise ValueError("parameters must be (sd_feat1, sd_feat2, sd_spat)")
    feat_diff = np.asarray(
        config.create_grid("feat_diff") if feat_diff is None else feat_diff,
        dtype=np.float32)
    bias_grid = np.asarray(mu1_grid_np() if bias_grid is None else bias_grid,
                           dtype=np.float32)
    if feat_diff.ndim != 1 or bias_grid.ndim != 1:
        raise ValueError("feature-difference and bias grids must be one-dimensional")

    rows = jnp.column_stack([
        jnp.full(len(feat_diff), parameters[0]),
        jnp.full(len(feat_diff), parameters[1]),
        jnp.full(len(feat_diff), parameters[2]),
        jnp.asarray(feat_diff),
    ])
    component_rows = (rows, mirror_params(rows))
    log_density, means, sds, asymmetries = [], [], [], []
    for component in component_rows:
        mean, _ = predictor.mean_and_resultant(component, validate=True)
        log_density.append(np.asarray(
            predictor.grid_log_density(component, jnp.asarray(bias_grid), validate=False)))
        means.append(np.asarray(mean))
        sds.append(np.asarray(predictor.circular_sd(component, validate=False)))
        asymmetries.append(np.asarray(
            predictor.signed_arc_asymmetry(component, validate=False)))

    return WNMView(
        identity=predictor.identity().as_dict(), parameters=parameters,
        feat_diff=feat_diff, bias_grid=bias_grid,
        log_density=np.stack(log_density), mean_bias=np.stack(means),
        circular_sd=np.stack(sds), density_asymmetry=np.stack(asymmetries))


def evaluate_wnm(n_samples: int, parameters, *, checkpoint: Path | None = None,
                 feat_diff=None, bias_grid=None) -> WNMView:
    return evaluate_predictor(
        load_wnm_predictor(n_samples, checkpoint), parameters,
        feat_diff=feat_diff, bias_grid=bias_grid)
