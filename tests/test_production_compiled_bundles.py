from pathlib import Path

import yaml

from model_fit_to_data.compiled_bundle import load_compiled_wnm_bundle


DATABASE = Path(__file__).resolve().parents[2] / "contextual_biases_database"


def test_dm_loads_every_cataloged_production_bundle():
    registry = yaml.safe_load((DATABASE / "production_bundles.yaml").read_text())
    loaded = []
    for item in registry["bundles"]:
        if "dm" not in item["consumers"]:
            continue
        groups, shared, assignments, manifest = load_compiled_wnm_bundle(
            DATABASE / item["path"])
        assert groups
        assert len(assignments) == sum(len(group.row_id) for group in groups)
        assert manifest["dataset_id"] == item["dataset"]
        assert len(shared["analysis_cell_id"]) >= len(groups)
        loaded.append((item["dataset"], item["variant"]))
    assert len(loaded) == 10
