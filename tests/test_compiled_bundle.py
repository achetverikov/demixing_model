from types import SimpleNamespace

import numpy as np

from model_fit_to_data.compiled_bundle import fitting_targets_from_compiled


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
