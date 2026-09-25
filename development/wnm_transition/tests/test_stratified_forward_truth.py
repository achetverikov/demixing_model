import json
import pickle
from argparse import Namespace

import numpy as np
import pandas as pd

from development.wnm_transition import stratified_forward_truth as truth


def test_case_table_uses_surface_native_and_same_parameter_wnm_losses(tmp_path):
    conditions = [f"S{i}#exp#condition" for i in range(1, 7)]
    pd.DataFrame({
        "condition": conditions, "family": "surface_nn",
        "fitted_objective": "smoothed_exp", "sd_feat1": 10.0,
        "sd_feat2": 20.0, "sd_spat": 30.0, "sd_motor": 0.0,
    }).to_csv(tmp_path / "paired_parameters.csv", index=False)
    pd.DataFrame({
        "condition": conditions, "family": "surface_nn",
        "fitted_objective": "smoothed_exp", "scoring_objective": "smoothed_exp",
        "common_wnm_loss": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    }).to_csv(tmp_path / "common_wnm_cross_scores.csv", index=False)
    surface = tmp_path / "surface.pkl"
    with surface.open("wb") as handle:
        pickle.dump({condition: {"smoothed_exp_loss": 1.0}
                     for condition in conditions}, handle)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": "paired_wnm_surface_comparison",
        "surface_results": str(surface),
    }))

    cases = truth.build_case_table(tmp_path, "smoothed_exp")

    assert cases.forward_loss_gap.tolist() == [0.0, 1.0, 3.0, 7.0, 15.0, 31.0]
    selected = truth.select_cases(cases, 1).set_index("gap_stratum")
    assert selected.loc["low", "condition"] == conditions[0]
    assert selected.loc["medium", "condition"] == conditions[2]
    assert selected.loc["high", "condition"] == conditions[-1]


def test_fidelity_summary_stays_stratified_by_dissimilarity_band():
    frame = pd.DataFrame({
        "gap_stratum": ["high"] * 4,
        "condition": ["case"] * 4,
        "component": [1] * 4,
        "dissimilarity_band": [band[2] for band in truth.BANDS],
        "pred_mean_bias_error": [1.0] * 4,
        "production_mean_bias_error": [2.0] * 4,
        "ref_circ_sd": [10.0] * 4,
        "pred_circ_sd": [11.0] * 4,
        "production_circ_sd": [12.0] * 4,
        "ref_density_asym": [0.0] * 4,
        "pred_density_asym": [0.1] * 4,
        "production_density_asym": [0.2] * 4,
        "nll_density_model": [3.0] * 4,
        "nll_production_interp": [4.0] * 4,
        "l1_density_model": [0.05] * 4,
        "l1_production": [0.1] * 4,
    })

    summary = truth._metric_summary(
        frame, ["gap_stratum", "component", "dissimilarity_band"])

    assert len(summary) == 4
    assert np.allclose(summary.mean_bias_rmse_deg_surface_minus_wnm, 1.0)
    assert np.allclose(summary.mean_nll_surface_minus_wnm, 1.0)
    assert np.allclose(summary.mean_density_l1_surface_minus_wnm, 0.05)


def test_dissimilarity_band_boundaries_are_explicit():
    values = [0, 17.999, 18, 59.999, 60, 119.999, 120, 180]
    assert truth.dissimilarity_band(values).tolist() == [
        "[0,18)", "[0,18)", "[18,60)", "[18,60)",
        "[60,120)", "[60,120)", "[120,180]", "[120,180]",
    ]


def test_summarize_writes_only_dissimilarity_stratified_outputs(tmp_path):
    comparison = tmp_path / "comparison.csv"
    rows = []
    for stratum, condition in (("high", "case-high"), ("low", "case-low")):
        for feat_diff in (2.0, 20.0, 80.0, 140.0):
            rows.append({
                "stratum": f"gap_{stratum}::{condition}", "component": 1,
                "feat_diff": feat_diff, "pred_mean_bias_error": 1.0,
                "production_mean_bias_error": 2.0, "ref_circ_sd": 10.0,
                "pred_circ_sd": 11.0, "production_circ_sd": 12.0,
                "ref_density_asym": 0.0, "pred_density_asym": 0.1,
                "production_density_asym": 0.2, "nll_density_model": 3.0,
                "nll_production_interp": 4.0, "l1_density_model": 0.05,
                "l1_production": 0.1,
            })
    pd.DataFrame(rows).to_csv(comparison, index=False)

    truth.summarize_panel(Namespace(comparison=comparison, out_dir=tmp_path))

    by_case = pd.read_csv(tmp_path / "fidelity_by_case_and_dissimilarity_band.csv")
    by_stratum = pd.read_csv(
        tmp_path / "fidelity_by_gap_stratum_and_dissimilarity_band.csv")
    evidence = json.loads((tmp_path / "fidelity_summary.json").read_text())
    assert len(by_case) == 8
    assert len(by_stratum) == 8
    assert set(by_stratum.dissimilarity_band) == {band[2] for band in truth.BANDS}
    assert evidence["decision_status"] == "diagnostic_only"
    assert "not a corpus-average sample" in evidence["sampling_scope"]
