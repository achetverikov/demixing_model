"""Strict shadow reader for precompiled DM-WNM empirical inputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompiledWNMGroup:
    fit_group_id: str
    analysis_cell_index: np.ndarray
    prediction_coordinates_deg: np.ndarray
    prediction_condition_index: np.ndarray
    feature_operator: np.ndarray
    coordinate_count: int
    prediction_capacity: int
    row_id: np.ndarray
    coordinate_model_deg: np.ndarray
    bias_model_deg: np.ndarray


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
        count = int(arrays[f"{prefix}_coordinate_count"])
        capacity = int(arrays[f"{prefix}_prediction_capacity"])
        coordinates = arrays[f"{prefix}_prediction_coordinates_deg"]
        operator = arrays[f"{prefix}_feature_operator"]
        if count > capacity or len(coordinates) != capacity or operator.shape[-1] != capacity:
            raise ValueError(f"invalid packed coordinate shape in {fit_group_id}")
        row_id = arrays[f"{prefix}_row_id"]
        if len(np.unique(row_id)) != len(row_id):
            raise ValueError(f"duplicate signed row IDs in {fit_group_id}")
        groups.append(CompiledWNMGroup(
            str(fit_group_id), arrays[f"{prefix}_analysis_cell_index"], coordinates,
            arrays[f"{prefix}_prediction_condition_index"], operator, count, capacity,
            row_id, arrays[f"{prefix}_coordinate_model_deg"],
            arrays[f"{prefix}_bias_model_deg"]))
    return groups, products["shared_empirical_targets"], assignments, manifest
