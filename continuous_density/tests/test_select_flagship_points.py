import numpy as np
import pandas as pd

from continuous_density import select_flagship_points as flagships


def test_candidate_point_score_uses_worst_normalized_metric():
    frame = pd.DataFrame({
        "raw_resultant": [0.9], "pred_mean_bias": [0.0], "raw_mean_bias": [0.0],
        "pred_density_asymmetry": [0.006], "raw_density_asymmetry": [0.0],
        "pred_response_sd": [10.0], "raw_response_sd": [10.0],
        "circular_wasserstein_deg": [0.1],
    })
    np.testing.assert_allclose(flagships.candidate_point_scores(frame), [1.2])
