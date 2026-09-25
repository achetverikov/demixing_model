import json

import numpy as np
import pytest

from model_fit_to_data.compare_wnm_surface_fits import (
    _boundary_hits,
    _fit_groups,
    _markdown_table,
    _validate_fingerprints,
    _validate_pair,
)
from model_fit_to_data.run_fingerprint import objective_versions_for

pytestmark = pytest.mark.development


def _fit(data):
    return {
        "data_df": np.asarray(data),
        **{f"{objective}_fitted_params": np.array([10.0, 20.0, 30.0, 0.0])
           for objective in ("likelihood", "bias_weighted_crps", "density",
                             "smoothed_exp")},
    }


def test_validate_pair_requires_identical_prepared_trials_and_order():
    wnm = {"a": _fit([[2.0, 1.0]]), "b": _fit([[4.0, -1.0]])}
    surface = {"a": _fit([[2.0, 1.0]]), "b": _fit([[4.0, -1.0]])}
    assert _validate_pair(wnm, surface) == ("a", "b")

    with pytest.raises(ValueError, match="prepared trials differ"):
        _validate_pair(wnm, {"a": _fit([[2.0, 2.0]]), "b": surface["b"]})
    with pytest.raises(ValueError, match="condition order"):
        _validate_pair(wnm, {"b": surface["b"], "a": surface["a"]})


def test_boundary_report_uses_each_familys_declared_feature_domain():
    params = np.array([[2.5, 200.0, 5.0, 0.0], [5.0, 20.0, 5.0, 0.0]])
    assert _boundary_hits(params, "wnm") == (
        "sd_feat1_c0@low,sd_feat2_c0@high,sd_spat@low")
    assert _boundary_hits(params, "surface_nn") == (
        "sd_feat2_c0@high,sd_feat1_c1@low,sd_spat@low")


def test_fit_groups_preserve_subject_experiment_units_and_order():
    conditions = (
        "S1#color_1#low___low", "S1#color_1#high___high",
        "S1#color_2#low___low", "S2#color_1#low___low",
    )
    assert _fit_groups(conditions) == {
        "S1#color_1": conditions[:2],
        "S1#color_2": conditions[2:3],
        "S2#color_1": conditions[3:],
    }
    with pytest.raises(ValueError, match="separator"):
        _fit_groups(("malformed",))


def test_markdown_table_has_no_optional_tabulate_dependency():
    import pandas as pd

    rendered = _markdown_table(pd.DataFrame({"family": ["wnm"], "loss": [1.25]}))
    assert "| family | loss |" in rendered
    assert "| wnm    | 1.25 |" in rendered


def test_comparison_refuses_legacy_objective_fingerprints(tmp_path):
    paths = {family: tmp_path / family for family in ("wnm", "surface_nn")}
    for family, path in paths.items():
        path.mkdir()
        payload = {
            "data_sha256": "same-data",
            "objective_versions": objective_versions_for(
                family, ("density", "smoothed_exp")),
        }
        (path / "extended_run_fingerprint.json").write_text(json.dumps({
            "digest": "unused-by-reader", "payload": payload,
        }))

    _validate_fingerprints(paths["wnm"], paths["surface_nn"])

    sidecar = paths["surface_nn"] / "extended_run_fingerprint.json"
    stored = json.loads(sidecar.read_text())
    stored["payload"]["objective_versions"]["density"] = "legacy"
    sidecar.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="requires"):
        _validate_fingerprints(paths["wnm"], paths["surface_nn"])
