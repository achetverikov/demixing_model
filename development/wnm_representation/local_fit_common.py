"""Shared data selection and metrics for independent local density fits."""

from __future__ import annotations

import numpy as np

from development.wnm_transition import evaluate
from development.wnm_representation import fourier_moments as fm


GRID = np.linspace(-180.0, 180.0, 720, endpoint=False)
EDGES = np.linspace(-180.0, 180.0, 721)
CELL_WIDTH = 0.5


def select_trajectory(design: np.ndarray, bias: np.ndarray, sd_feat1: float,
                      sd_feat2: float, sd_ident: float, component: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Select and feature-difference-sort one complete raw trajectory."""
    mask = (np.isclose(design[:, 0], sd_feat1)
            & np.isclose(design[:, 1], sd_feat2)
            & np.isclose(design[:, 2], sd_ident))
    rows = np.flatnonzero(mask)
    rows = rows[np.argsort(design[rows, 3])]
    if len(rows) == 0:
        raise ValueError("no rows match the requested trajectory")
    return np.asarray(design[rows]), np.asarray(bias[rows, :, component - 1])


def split_outcomes(samples: np.ndarray, fraction: float = 0.5, seed: int = 0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Split by a shared simulation-index permutation, preserving CRN structure."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    order = rng.permutation(samples.shape[1])
    cut = int(round(fraction * len(order)))
    return samples[:, order[:cut]], samples[:, order[cut:]]


def empirical_core_metrics(samples: np.ndarray) -> dict[str, np.ndarray]:
    moments = evaluate.empirical_moments(samples)
    return {"mean_bias": moments["mean_bias"], "resultant": moments["resultant"],
            "response_sd": moments["circ_sd"],
            "density_asymmetry": evaluate.empirical_density_asymmetry(samples)}


def histogram_mass(samples: np.ndarray) -> np.ndarray:
    rows = []
    for row in np.asarray(samples):
        finite = row[np.isfinite(row)]
        counts = np.histogram(finite, bins=EDGES)[0].astype(np.float64)
        rows.append(counts / counts.sum())
    return np.stack(rows)


def circular_wasserstein_to_density(samples: np.ndarray,
                                    density_per_degree: np.ndarray) -> np.ndarray:
    predicted_mass = np.asarray(density_per_degree) * CELL_WIDTH
    return fm.circular_wasserstein_grid(histogram_mass(samples), predicted_mass, CELL_WIDTH)

