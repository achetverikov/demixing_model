"""Strict shadow reader for precompiled DM-WNM empirical inputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompiledWNMGroup:
    fit_group_id: str
    fit_group_values: dict
    analysis_cell_ids: tuple[str, ...]
    analysis_cell_values: tuple[dict, ...]
    analysis_cell_index: np.ndarray
    prediction_coordinates_deg: np.ndarray
    prediction_condition_index: np.ndarray
    feature_operator: np.ndarray
    coordinate_count: int
    prediction_capacity: int
    row_id: np.ndarray
    coordinate_model_deg: np.ndarray
    bias_model_deg: np.ndarray
    trial_condition_index: np.ndarray


def fitting_targets_from_compiled(group, shared):
    """Adapt upstream numerical products to the WNM scorer's array container."""
    try:
        from fitting_targets import FittingTargets
    except ModuleNotFoundError:
        from model_fit_to_data.fitting_targets import FittingTargets

    cells = np.asarray(group.analysis_cell_index, dtype=np.int32)
    density = np.asarray(shared["density_asymmetry"])[cells]
    return FittingTargets(
        condition_names=group.analysis_cell_ids,
        feat_indices=np.arange(len(shared["feature_grid_model_deg"]), dtype=np.int32),
        target_bias=np.asarray(shared["smoothed_bias_deg"])[cells],
        bias_weights=np.asarray(shared["effective_support"])[cells],
        target_density=density,
        target_bias_curve=np.asarray(shared["smoothed_bias_deg"])[cells],
        target_d=np.asarray(shared["bwcrps_target_distance"])[cells],
        fd_weights=np.asarray(shared["bwcrps_support_weight"])[cells],
        bias_fd_weights=np.asarray(shared["bwcrps_bias_weight"])[cells],
        density_target_var=np.var(density, axis=1),
        density_degenerate=np.var(density, axis=1) < 1e-10,
        density_bandwidth=tuple(
            np.asarray(shared["density_bandwidth_model_deg"])[cells].tolist()),
        feature_operator=group.feature_operator,
        matched_density_target=density,
        matched_density_degenerate=np.var(density, axis=1) < 1e-10,
        near_constant_warnings=(),
        prediction_coordinates=group.prediction_coordinates_deg,
        prediction_condition_index=group.prediction_condition_index,
        smoothed_support=np.asarray(shared["effective_support"])[cells],
        prediction_coordinate_count=group.coordinate_count,
        prediction_capacity=group.prediction_capacity,
        feature_coordinate_mode="exact",
    )


def load_compiled_wnm_bundle(path):
    """Load arrays only; empirical geometry and target construction stay upstream."""
    from contextual_biases_database import read_compiled_bundle

    products, assignments, manifest = read_compiled_bundle(path)
    if manifest["population"] != "signed_bias":
        raise ValueError("DM-WNM production objectives require the signed_bias population")
    if manifest["coordinate_policy"] != "exact_bounded_extrapolation":
        raise ValueError("DM-WNM requires exact_bounded_extrapolation coordinates")
    arrays = products["dm_wnm_inputs"]
    groups = []
    for index, fit_group_id in enumerate(arrays["fit_group_id"]):
        prefix = f"group_{index:04d}"
        group_manifest = manifest["fit_groups"][index]
        if str(fit_group_id) != group_manifest["fit_group_id"]:
            raise ValueError("compiled fit-group order differs between arrays and manifest")
        count = int(arrays[f"{prefix}_coordinate_count"])
        capacity = int(arrays[f"{prefix}_prediction_capacity"])
        coordinates = arrays[f"{prefix}_prediction_coordinates_deg"]
        operator = arrays[f"{prefix}_feature_operator"]
        if count > capacity or len(coordinates) != capacity or operator.shape[-1] != capacity:
            raise ValueError(f"invalid packed coordinate shape in {fit_group_id}")
        row_id = arrays[f"{prefix}_row_id"]
        if len(np.unique(row_id)) != len(row_id):
            raise ValueError(f"duplicate signed row IDs in {fit_group_id}")
        trial_condition_index = arrays[f"{prefix}_trial_condition_index"]
        if len(trial_condition_index) != len(row_id):
            raise ValueError(f"compiled trial assignments differ in length for {fit_group_id}")
        analysis_cells = group_manifest["analysis_cells"]
        groups.append(CompiledWNMGroup(
            str(fit_group_id), dict(group_manifest["values"]),
            tuple(cell["analysis_cell_id"] for cell in analysis_cells),
            tuple(dict(cell["values"]) for cell in analysis_cells),
            arrays[f"{prefix}_analysis_cell_index"], coordinates,
            arrays[f"{prefix}_prediction_condition_index"], operator, count, capacity,
            row_id, arrays[f"{prefix}_coordinate_model_deg"],
            arrays[f"{prefix}_bias_model_deg"], trial_condition_index))
    return groups, products["shared_empirical_targets"], assignments, manifest
