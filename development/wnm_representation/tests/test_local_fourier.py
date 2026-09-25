import numpy as np

from development.wnm_representation import fit_local_fourier
from development.wnm_representation import local_fit_common as common


def test_shared_split_preserves_rowwise_pairing():
    samples = np.stack([np.arange(20), np.arange(20) + 100])
    train, test = common.split_outcomes(samples, seed=3)
    assert np.all(train[1] - train[0] == 100)
    assert np.all(test[1] - test[0] == 100)
    assert set(train[0]).isdisjoint(test[0])


def test_local_fourier_fit_reports_all_core_metrics():
    rng = np.random.default_rng(4)
    train = np.stack([np.degrees(rng.vonmises(mu, 3, 3000)) for mu in (0.1, 0.3)])
    test = np.stack([np.degrees(rng.vonmises(mu, 3, 2000)) for mu in (0.1, 0.3)])
    frame, parameters = fit_local_fourier.fit_trajectory(
        train, test, np.array([20.0, 40.0]), [2], 1024, 300)
    assert set(("raw_mean_bias", "pred_mean_bias", "raw_resultant", "pred_resultant",
                "raw_response_sd", "pred_response_sd", "raw_density_asymmetry",
                "pred_density_asymmetry", "heldout_nll",
                "circular_wasserstein_deg")) <= set(frame)
    assert frame.fit_success.all()
    assert parameters["k2_theta"].shape == (2, 4)

