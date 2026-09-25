import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fit_model_to_data import plan_continuous_prediction_capacities  # noqa: E402


def test_capacity_plan_uses_supported_signed_bias_coordinates_without_clamping():
    features = np.array([0.0, 0.25, 0.5, 1.0, 180.0] * 6)
    frame = pd.DataFrame({"dissimilarity": features, "bias": np.ones(len(features))})
    groups = {"participant": {"condition": frame}}
    engine = SimpleNamespace(
        predictor=SimpleNamespace(domain={"feat_diff": (0.5, 180.0)}),
        skip_motor_noise=True)
    assignment = plan_continuous_prediction_capacities(
        groups, engine, "dissimilarity", "bias", angle_scale_to_model=1.0)
    assert assignment["participant"].coordinate_count == 3
    assert assignment["participant"].capacity == 3


def test_sub_support_coordinates_are_included_as_exact_extrapolations():
    frame = pd.DataFrame({
        "dissimilarity": np.array([0.25, 0.5, 1.0] * 5),
        "bias": np.ones(15),
    })
    engine = SimpleNamespace(
        predictor=SimpleNamespace(domain={"feat_diff": (0.5, 180.0)}),
        skip_motor_noise=True)
    assignment = plan_continuous_prediction_capacities(
        {"participant": {"condition": frame}}, engine,
        "dissimilarity", "bias", angle_scale_to_model=1.0)
    assert assignment["participant"].coordinate_count == 3
