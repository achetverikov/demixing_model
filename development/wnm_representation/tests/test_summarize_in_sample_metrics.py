import pandas as pd

from development.wnm_representation import summarize_in_sample_metrics as summary


def test_in_sample_summary_keeps_fitting_and_heldout_separate():
    frame = pd.DataFrame({"family": ["x", "x"], "size": [1, 1],
        "feat_diff": [0, 2], "pred_mean_bias": [0, 1],
        "pred_resultant": [0.9, 0.8], "pred_response_sd": [10, 11],
        "pred_density_asymmetry": [0, 0.1],
        "raw_mean_bias": [0, 2], "raw_resultant": [0.9, 0.8],
        "raw_response_sd": [10, 12], "raw_density_asymmetry": [0, 0.2],
        "train_raw_mean_bias": [0, 1], "train_raw_resultant": [0.9, 0.8],
        "train_raw_response_sd": [10, 11], "train_raw_density_asymmetry": [0, 0.1],
        "train_nll": [1, 1], "heldout_nll": [1.1, 1.1]})
    result = summary.summarize(frame)
    fitting = result[(result.split == "fitting") & (result.metric == "response_sd")]
    assert fitting.iloc[0].max_abs_error == 0
