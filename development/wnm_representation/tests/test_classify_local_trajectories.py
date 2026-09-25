import pandas as pd

from development.wnm_representation import classify_local_trajectories as classify


def test_smaller_wrapped_pass_with_k12_failure_is_optimizer_unresolved():
    frame = pd.DataFrame({"family": ["wrapped_mixture", "wrapped_mixture"],
        "size": [8, 12], "all_coherent_pass": [True, False],
        "binding_coherent_metrics": ["", "density_asymmetry"],
        "wasserstein_upper_deg": [0.2, 0.3],
        "excess_nll_coherent_upper": [0.0, 0.0]})
    result = classify.classify(frame)
    assert result["classification"].startswith("optimizer-unresolved")
