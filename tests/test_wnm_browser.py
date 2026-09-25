import numpy as np
from pathlib import Path

from shared import surrogate
from surface_browser.core.wnm_view import evaluate_predictor
from shared.prediction import predictor_from_surrogate


def test_browser_uses_direct_wnm_predictions_for_both_components():
    predictor = predictor_from_surrogate(surrogate.load_surrogate(
        family="wnm", n_samples=20))
    view = evaluate_predictor(
        predictor, (10.0, 40.0, 25.0),
        feat_diff=np.array([2.0, 40.0, 100.0], dtype=np.float32),
        bias_grid=np.arange(-180.0, 180.0, 20.0, dtype=np.float32))

    assert view.log_density.shape == (2, 3, 18)
    assert view.mean_bias.shape == view.circular_sd.shape == (2, 3)
    assert view.density_asymmetry.shape == (2, 3)
    assert view.identity["surrogate_family"] == "wnm"
    assert np.all(np.isfinite(view.log_density))

    swapped = evaluate_predictor(
        predictor, (40.0, 10.0, 25.0), feat_diff=view.feat_diff,
        bias_grid=view.bias_grid)
    np.testing.assert_allclose(view.log_density[1], swapped.log_density[0])
    np.testing.assert_allclose(view.mean_bias[1], swapped.mean_bias[0])


def test_browser_opens_direct_wnm_view_without_surface_files(monkeypatch):
    from streamlit.testing.v1 import AppTest

    browser = Path(__file__).resolve().parents[1] / "surface_browser"
    monkeypatch.chdir(browser)
    app = AppTest.from_file("main_app.py", default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "🌊 Demixing-model Prediction Browser"
    assert len(app.get("plotly_chart")) == 2
