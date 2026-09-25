from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shared import surrogate
from surface_simulator_for_predictions.surface_simulator import simulate_surfaces_from_file


WNM = surrogate.WNM_DEFAULTS[20]
pytestmark = pytest.mark.skipif(not WNM.exists(), reason="no packaged WNM artifact installed")


def test_public_prediction_api_uses_wnm_by_default(tmp_path):
    input_path = tmp_path / "parameters.csv"
    output_path = tmp_path / "predictions.csv"
    pd.DataFrame([{
        "sd_feat1": 10.0,
        "sd_feat2": 30.0,
        "sd_spat": 20.0,
    }]).to_csv(input_path, index=False)

    simulate_surfaces_from_file(
        str(input_path), 20, str(output_path), skip_motor_noise=True)

    frame = pd.read_csv(output_path)
    assert frame.loc[0, "surrogate_family"] == "wnm"
    assert frame.loc[0, "surrogate_artifact"] == WNM.name
    for column in ("mu1_density_curve", "mu1_expectation_curve", "sd_curve"):
        assert column in frame
        assert "nan" not in str(frame.loc[0, column]).lower()


def test_historical_surface_prediction_requires_explicit_source(tmp_path):
    if not surrogate.SURFACE_DEFAULTS[20].exists():
        pytest.skip("historical surface checkpoint not installed")
    input_path = tmp_path / "parameters.csv"
    output_path = tmp_path / "predictions.csv"
    pd.DataFrame([{
        "sd_feat1": 10.0,
        "sd_feat2": 30.0,
        "sd_spat": 20.0,
    }]).to_csv(input_path, index=False)

    simulate_surfaces_from_file(
        str(input_path), 20, str(output_path), skip_motor_noise=True,
        surface_source="nn")

    frame = pd.read_csv(output_path)
    assert frame.loc[0, "surrogate_family"] == "surface_nn"
