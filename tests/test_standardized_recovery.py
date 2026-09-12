import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODEL_FIT = Path(__file__).resolve().parents[1] / "model_fit_to_data"
if str(MODEL_FIT) not in sys.path:
    sys.path.insert(0, str(MODEL_FIT))

import standardized_recovery as SR  # noqa: E402
import wnm_scoring as S  # noqa: E402


def protocol(**updates):
    value = {
        "schema": SR.SCHEMA,
        "surrogate": {"family": "wnm", "n_samples": 20},
        "cases": [{
            "name": "crossed",
            "condition_feature_sds": [[15.0, 45.0], [60.0, 20.0]],
            "sd_spat": 25.0,
        }],
    }
    value.update(updates)
    return value


def test_protocol_defaults_cover_every_supported_objective():
    normalized = SR.normalize_protocol(protocol())

    assert normalized["objectives"] == list(S.SUPPORTED_METHODS)
    assert normalized["surrogate"]["family"] == "wnm"
    assert normalized["dataset_source"]["kind"] == "wnm_closed_loop"
    assert normalized["n_starts"] == 64


def test_protocol_refuses_surface_backend_and_unidentified_motor_fit():
    with pytest.raises(ValueError, match="only the new WNM"):
        SR.normalize_protocol(protocol(surrogate={"family": "surface_nn", "n_samples": 20}))
    with pytest.raises(ValueError, match="cannot identify a fitted motor SD"):
        SR.normalize_protocol(protocol(objectives=["expectation"], fit_motor=True))


def test_checkpoint_rejects_code_or_dataset_drift(tmp_path):
    path = tmp_path / "fit.json"
    path.write_text(json.dumps({"code_digest": "old", "dataset_digest": "data"}))

    with pytest.raises(ValueError, match="code_digest"):
        SR._checkpoint(path, {"code_digest": "new", "dataset_digest": "data"})
    assert SR._checkpoint(
        path, {"code_digest": "old", "dataset_digest": "data"})["code_digest"] == "old"


def test_npz_simulator_source_uses_the_same_condition_dataset_contract(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    values = np.column_stack([np.arange(4), np.arange(4) + 10]).astype(np.float32)
    np.savez(source / "crossed__n4__r0.npz", c0=values, c1=values + 1)
    normalized = SR.normalize_protocol(protocol(
        trial_counts=[4], dataset_source={"kind": "npz", "directory": str(source)}))
    case = SR.recovery_cases(normalized, 4)[0]

    datasets = SR._source_dataset(normalized, None, case, 0, 4, 0)
    SR._validate_datasets(datasets, case)

    assert tuple(datasets) == ("c0", "c1")
    np.testing.assert_array_equal(datasets["c0"], values)


def test_aggregate_keeps_trial_count_and_dissimilarity_strata(tmp_path):
    (tmp_path / "cells").mkdir()
    row = {
        "case": "crossed", "trial_count": 400, "replicate": 0,
        "objective": "density", "dataset_digest": "abc",
    }
    cell = {
        "fit": {"row": row, "names": ["sd_spat"], "truth": [25.0],
                "recovered": [30.0]},
        "curves": [row | {
            "condition": "c0", "metric": "bias", "source": "recovered",
            "dissimilarity": 40.0, "value": 2.0, "bin_n_trials": None,
        }],
    }
    (tmp_path / "cells" / "one.json").write_text(json.dumps(cell))
    manifest = {"protocol_digest": "p", "code_digest": "c", "expected_cells": 1}

    SR._aggregate(tmp_path, manifest)

    parameters = pd.read_csv(tmp_path / "recovery_parameters.csv")
    curves = pd.read_csv(tmp_path / "recovery_curves.csv")
    assert parameters.loc[0, "trial_count"] == 400
    assert parameters.loc[0, "log_ratio"] == pytest.approx(np.log(30.0 / 25.0))
    assert curves.loc[0, "dissimilarity"] == 40.0
    assert curves.loc[0, "trial_count"] == 400
