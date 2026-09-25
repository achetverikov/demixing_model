import numpy as np

from development.wnm_representation import fit_local_wrapped_mixture as fit


def test_wrapped_initialization_is_valid():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(3, 1000)) * 10
    dist = fit.initialize_from_samples(samples, 4)
    assert dist["mu"].shape == (3, 4)
    assert np.allclose(np.exp(dist["log_pi"]).sum(axis=1), 1)
    assert np.all(dist["sigma"] > 0)


def test_wrapped_metrics_report_core_and_distribution_metrics():
    rng = np.random.default_rng(1)
    samples = np.stack([rng.normal(20, 5, 5000), rng.normal(-30, 8, 5000)])
    dist = {"log_pi": np.zeros((2, 1)),
            "mu": np.array([[20.0], [-30.0]]),
            "sigma": np.array([[5.0], [8.0]])}
    metrics = fit.fitted_metrics(dist, samples)
    assert set(("raw_mean_bias", "pred_mean_bias", "raw_resultant", "pred_resultant",
                "raw_response_sd", "pred_response_sd", "raw_density_asymmetry",
                "pred_density_asymmetry", "heldout_nll",
                "circular_wasserstein_deg")) <= set(metrics)
    assert max(metrics["circular_wasserstein_deg"]) < 0.5


def test_component_diagnostics_detect_duplicate_and_vanishing_components():
    dist = {"log_pi": np.array([[0.0, 0.0, -20.0]]),
            "mu": np.array([[10.0, 10.2, 100.0]]),
            "sigma": np.array([[2.0, 2.1, 0.26]])}
    result = fit.component_diagnostics(dist)
    assert result["duplicate_component_pairs"][0] == 1
    assert result["vanishing_components"][0] == 1
    assert result["near_floor_components"][0] == 1
