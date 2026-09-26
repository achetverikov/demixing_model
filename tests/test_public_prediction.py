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


@pytest.mark.parametrize("bad_motor", [-1.0, float("nan"), float("inf")])
def test_public_prediction_rejects_invalid_motor_sd_before_model_execution(tmp_path, bad_motor):
    input_path = tmp_path / "parameters.csv"
    output_path = tmp_path / "predictions.csv"
    pd.DataFrame([{
        "sd_feat1": 10.0,
        "sd_feat2": 30.0,
        "sd_spat": 20.0,
        "sd_motor": bad_motor,
    }]).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="sd_motor values must be finite and non-negative"):
        simulate_surfaces_from_file(
            str(input_path), 20, str(output_path), skip_motor_noise=False)


def test_raw_prediction_uses_stored_surface_and_retains_mu2(tmp_path):
    import pickle
    from shared.config import config
    from shared.utils import AveragedSurface

    input_path = tmp_path / "parameters.csv"
    output_path = tmp_path / "predictions.csv"
    surfaces_dir = tmp_path / "averaged_surfaces_20samples"
    surfaces_dir.mkdir()
    feat = config.create_grid("feat_diff")
    mu1 = config.create_grid("mu1_bias")
    mu2 = config.create_grid("mu2_bias")
    surface = AveragedSurface(
        feat_diff_grid=feat, mu1_bias_grid=mu1, mu2_bias_grid=mu2,
        mu1_comp1_surface=np.full((len(mu1), len(feat)), -np.log(360.0)),
        mu1_comp2_surface=np.full((len(mu1), len(feat)), -np.log(360.0)),
        mu2_comp1_surface=np.full((len(mu2), len(feat)), -np.log(len(mu2))),
        mu2_comp2_surface=np.full((len(mu2), len(feat)), -np.log(len(mu2))),
    )
    (surfaces_dir / "averaged_sf1_10.0_sf2_30.0_sp_20.0.pkl").write_bytes(
        pickle.dumps({"surface": surface}))
    pd.DataFrame([{
        "sd_feat1": 10.0,
        "sd_feat2": 30.0,
        "sd_spat": 20.0,
    }]).to_csv(input_path, index=False)

    simulate_surfaces_from_file(
        str(input_path), 20, str(output_path), skip_motor_noise=True,
        surface_source="raw", averaged_surfaces_dir=str(surfaces_dir))

    frame = pd.read_csv(output_path)
    assert frame.loc[0, "surrogate_family"] == "averaged_surfaces"
    assert bool(frame.loc[0, "has_mu2_data"])
    assert "nan" not in str(frame.loc[0, "mu2_expectation_curve"]).lower()

    with pytest.raises(ValueError, match="unknown surface_source"):
        simulate_surfaces_from_file(str(input_path), 20, str(output_path),
                                    skip_motor_noise=True, surface_source="nn")


def test_direct_prediction_script_without_pythonpath(tmp_path):
    """Public CLI contract: a checkout command must not rely on CI's PYTHONPATH."""
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(root / "surface_simulator_for_predictions" /
                             "surface_simulator.py"), "--help"], cwd=tmp_path,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--input-path" in result.stdout
