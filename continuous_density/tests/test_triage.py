import pandas as pd
import pytest

from continuous_density import triage_trajectories as triage


def test_error_summary_keeps_trajectories_and_components_separate():
    rows = []
    for component in (1, 2):
        for diff in (2, 4, 6):
            for method, offset in (("raw 100k", 0), ("model", component * 0.1)):
                rows.append({"sd_feat1": 10, "sd_feat2": 20, "sd_ident": 30,
                             "which_comp": component, "dist_feat": diff,
                             "method": method, "mean_bias": diff + offset,
                             "response_sd": 20 + offset,
                             "density_asymmetry": 0.2 + offset})
    summary = triage.error_summary(pd.DataFrame(rows), "model")
    assert len(summary) == 2
    assert set(summary.component) == {1, 2}
    assert summary.set_index("component").loc[2,
           "density_asymmetry_mean_signed_error"] == pytest.approx(0.2)


def test_mode_summary_counts_multimodality_and_switches():
    frame = pd.DataFrame({"sd_feat1": [10] * 4, "sd_feat2": [20] * 4,
                          "sd_ident": [40] * 4, "component": [2] * 4,
                          "feat_diff": [2, 4, 6, 8], "mode_count": [1, 2, 2, 1],
                          "secondary_mode_mass": [0, .2, .3, 0]})
    summary = triage.mode_summary(frame).iloc[0]
    assert summary.multimodal_points == 2
    assert summary.mode_count_switches == 2
    assert summary.max_secondary_mode_mass == .3
