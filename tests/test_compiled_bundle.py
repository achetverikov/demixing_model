from types import SimpleNamespace
from pathlib import Path

import pytest

import numpy as np

from model_fit_to_data.compiled_bundle import fitting_targets_from_compiled, load_compiled_wnm_bundle


def test_compiled_targets_select_the_group_cells_without_reconstruction():
    group = SimpleNamespace(
        analysis_cell_index=np.array([1], np.int32),
        analysis_cell_ids=("cell-b",),
        feature_operator=np.ones((1, 2, 3), np.float32),
        prediction_coordinates_deg=np.array([3, 7, 90], np.float32),
        prediction_condition_index=np.zeros(3, np.int32),
        coordinate_count=2,
        prediction_capacity=3,
    )
    shared = {
        "feature_grid_model_deg": np.array([2, 4], np.float32),
        "density_asymmetry": np.array([[1, 2], [3, 4]], np.float32),
        "smoothed_bias_deg": np.array([[5, 6], [7, 8]], np.float32),
        "effective_support": np.array([[9, 10], [11, 12]], np.float32),
        "bwcrps_target_distance": np.zeros((2, 2, 4), np.float32),
        "bwcrps_support_weight": np.ones((2, 2), np.float32),
        "bwcrps_bias_weight": np.full((2, 2), 2, np.float32),
        "density_bandwidth_model_deg": np.array([13, 14], np.float32),
    }
    targets = fitting_targets_from_compiled(group, shared)
    assert targets.condition_names == ("cell-b",)
    np.testing.assert_array_equal(targets.matched_density_target, [[3, 4]])
    np.testing.assert_array_equal(targets.target_bias_curve, [[7, 8]])
    np.testing.assert_array_equal(targets.smoothed_support, [[11, 12]])
    assert targets.density_bandwidth == (14.0,)
    assert targets.prediction_coordinate_count == 2
    assert targets.prediction_capacity == 3


# Cross-repository contract: consumers must accept both production populations.


@pytest.mark.integration
@pytest.mark.parametrize('variant', ['united', 'report_separate'])
def test_inclusive_bundle_supplies_every_scored_row_and_target(variant):
    database = Path(__file__).resolve().parents[2] / 'contextual_biases_database'
    path = database / 'data/bundles/chr2026' / f'chr2026_{variant}_with_outliers_signed_v1'
    groups, shared, assignments, manifest = load_compiled_wnm_bundle(path)
    assert manifest['population'] == 'signed_bias_with_outliers'
    assert sum(len(group.row_id) for group in groups) == len(assignments)
    for group in groups:
        targets = fitting_targets_from_compiled(group, shared)
        np.testing.assert_array_equal(targets.target_bias_curve,
                                      shared['smoothed_bias_deg'][group.analysis_cell_index])
        assert np.isfinite(group.bias_model_deg).all()
