import numpy as np
import pytest

from development.wnm_representation import reference_noise_floor as nf


def test_split_half_reports_full_n_standard_error():
    rng = np.random.default_rng(0)
    samples = rng.choice([-1.0, 1.0], size=(2000, 2000))
    differences = nf.split_half_differences(samples)
    summary = nf.summarize_split_half(differences)
    assert summary["density_asymmetry"]["per_full_n_se"] == pytest.approx(
        1 / np.sqrt(2000), rel=0.08)


def test_trajectory_mean_se_detects_shared_covariance():
    rng = np.random.default_rng(1)
    common = rng.normal(size=(4000, 1))
    draws = common + rng.normal(scale=0.1, size=(4000, 20))
    summary = nf.trajectory_mean_se(draws)
    assert summary["inflation_factor"] > 4
    assert summary["covariance_aware_se"] > summary["independence_approximation_se"]

