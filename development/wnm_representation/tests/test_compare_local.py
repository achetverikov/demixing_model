import numpy as np
import pandas as pd

from development.wnm_representation import compare_local_representations as compare


def test_comparison_is_trajectory_and_metric_specific():
    rows = []
    for family, offset in (("a", 0.1), ("b", 0.2)):
        for diff in (2, 4, 6, 8):
            rows.append({"family": family, "size": 4, "feat_diff": diff,
                         "raw_mean_bias": diff, "pred_mean_bias": diff + offset,
                         "raw_response_sd": 10 + diff, "pred_response_sd": 10 + diff,
                         "raw_density_asymmetry": diff / 10,
                         "pred_density_asymmetry": diff / 10 - offset,
                         "heldout_nll": 2 + offset,
                         "circular_wasserstein_deg": offset})
    table = compare.comparison_table(pd.DataFrame(rows))
    assert len(table) == 2 * 5
    assert set(table.metric) == {"mean_bias", "response_sd", "density_asymmetry",
                                 "excess_nll", "circular_wasserstein_deg"}
    a_nll = table[(table.family == "a") & (table.metric == "excess_nll")].iloc[0]
    assert a_nll.mean_signed_error == 0
    assert np.isfinite(table["amplitude_slope"][:-4]).any()

