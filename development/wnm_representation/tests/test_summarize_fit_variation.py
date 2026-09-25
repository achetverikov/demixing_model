import pandas as pd

from development.wnm_representation import summarize_fit_variation as variation


def test_compare_reports_wrapped_mean_difference():
    first = pd.DataFrame({"family": ["x"], "size": [1], "feat_diff": [0],
        "pred_density_asymmetry": [0], "pred_mean_bias": [179],
        "pred_resultant": [1], "pred_response_sd": [1], "heldout_nll": [1],
        "circular_wasserstein_deg": [0]})
    second = first.copy()
    second["pred_mean_bias"] = -179
    rows = variation.compare(first, second)
    mean = next(row for row in rows if row["metric"] == "pred_mean_bias")
    assert mean["fitting_half_max_abs_difference"] == 2
