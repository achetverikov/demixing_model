import numpy as np

from model_fit_to_data.create_unified_subject_plots import _pooled_bias_weighted_crps
from shared.mu1_axis import mu1_grid, mu1_size


def test_report_order_pooling_matches_concatenation_for_identical_models():
    rng = np.random.default_rng(4)
    log_surface = rng.normal(size=(mu1_size(), 90))
    first = np.column_stack([rng.uniform(2, 180, 40), rng.normal(0, 15, 40)])
    second = np.column_stack([rng.uniform(2, 180, 35), rng.normal(0, 15, 35)])
    feat_grid = np.arange(2, 181, 2)
    # The mu1_bias axis is circular and half-open: read it, never rebuild it.
    bias_grid = np.asarray(mu1_grid())
    difference = np.abs(bias_grid[:, None] - bias_grid[None, :])
    distance = np.minimum(difference, 360 - difference)

    pooled = _pooled_bias_weighted_crps(
        np.stack([log_surface, log_surface]), [first, second],
        feat_grid, distance, 20,
    )
    concatenated = _pooled_bias_weighted_crps(
        log_surface[None], [np.vstack([first, second])],
        feat_grid, distance, 20,
    )
    assert np.isfinite(pooled)
    assert np.isclose(pooled, concatenated, atol=1e-10)
