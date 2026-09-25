from pathlib import Path
from types import SimpleNamespace

import pytest

from model_fit_to_data.create_unified_subject_plots import (
    _resolve_plot_circ_space,
    _result_plot_identity,
    organize_results_by_subject,
)
from model_fit_to_data import plot_pdf_slices


def test_bundle_native_plot_identity_uses_analysis_cell_metadata():
    key = "fischer_signed_v1:cell:opaque"
    result = {
        "analysis_cell_values": {
            "experiment_id": "fischer_whitney_2014_1b",
            "subject_id": "4",
            "condition_id": "orientation",
        }
    }

    assert _result_plot_identity(key, result) == (
        "4", "fischer_whitney_2014_1b", "orientation"
    )
    grouped = organize_results_by_subject({key: result})
    entry = grouped["4"]["fischer_whitney_2014_1b"][0]
    assert entry["condition_name"] == key
    assert entry["noise_condition"] == "orientation"


def test_bundle_native_report_order_maps_to_existing_pairing_labels():
    first = {
        "analysis_cell_values": {
            "experiment_id": "color_2",
            "subject_id": "S9",
            "report_order": 1,
            "condition_id": "low - high",
        }
    }
    second = {
        "analysis_cell_values": {
            "experiment_id": "color_2",
            "subject_id": "S9",
            "report_order": 2,
            "condition_id": "low - high",
        }
    }

    assert _result_plot_identity("opaque:first", first) == (
        "S9", "color_2_first", "low - high"
    )
    assert _result_plot_identity("opaque:second", second) == (
        "S9", "color_2_second", "low - high"
    )


def test_legacy_plot_identity_fallback_is_unchanged():
    assert _result_plot_identity("S12#color_1#high - low", {}) == (
        "S12", "color_1", "high - low"
    )
    assert _result_plot_identity("S1.color.1_low - high", {}) == (
        "S1", "color.1", "low - high"
    )


def test_plot_circular_period_is_inferred_and_conflicts_raise():
    results = {
        "opaque-a": {"circ_space": 180},
        "opaque-b": {"circ_space": 180.0},
    }
    assert _resolve_plot_circ_space(results) == 180
    assert _resolve_plot_circ_space(results, 180) == 180
    with pytest.raises(ValueError, match="disagrees"):
        _resolve_plot_circ_space(results, 360)


def test_plot_circular_period_rejects_mixed_result_sets():
    with pytest.raises(ValueError, match="one circular period"):
        _resolve_plot_circ_space({
            "a": {"circ_space": 180},
            "b": {"circ_space": 360},
        })


def test_standalone_pdf_plot_uses_wnm_backend_from_run_identity(monkeypatch, tmp_path):
    results = {
        "opaque": {
            "circ_space": 180,
            "density_fitted_params": [10.0, 20.0, 30.0, 0.0],
        }
    }
    checkpoint = Path("pretrained/wnm_k12_20samples.pkl")
    loaded = SimpleNamespace(family=plot_pdf_slices.surrogate.FAMILY_WNM)
    predictor = object()
    calls = {}

    monkeypatch.setattr(plot_pdf_slices, "load_extended_results", lambda _: results)
    monkeypatch.setattr(
        plot_pdf_slices.surrogate, "checkpoint_for_run",
        lambda results_path, explicit, n_samples: checkpoint,
    )
    monkeypatch.setattr(
        plot_pdf_slices.surrogate, "load_surrogate",
        lambda checkpoint_path: loaded,
    )
    monkeypatch.setattr(plot_pdf_slices, "predictor_from_surrogate", lambda obj: predictor)

    def fake_create_pdf(loaded_results, backend, output_dir, **kwargs):
        calls["results"] = loaded_results
        calls["backend"] = backend
        calls["output_dir"] = output_dir
        calls["kwargs"] = kwargs

    monkeypatch.setattr(plot_pdf_slices, "create_pdf_slice_plots", fake_create_pdf)
    monkeypatch.setattr(
        plot_pdf_slices.sys, "argv",
        [
            "plot_pdf_slices.py",
            "--results-path", str(tmp_path / "extended_fit_results.pkl"),
            "--output-dir", str(tmp_path),
            "--optimizer", "bias_weighted_crps",
        ],
    )

    plot_pdf_slices.main()

    assert calls["results"] is results
    assert calls["backend"] is predictor
    assert calls["output_dir"] == str(tmp_path)
    assert calls["kwargs"]["optimizer_names"] == ["bias_weighted_crps"]
    assert calls["kwargs"]["circ_space"] == 180
