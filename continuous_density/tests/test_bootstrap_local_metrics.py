import numpy as np

from continuous_density import bootstrap_local_metrics as bootstrap


def test_complex_equivalence_uses_radial_error():
    rng = np.random.default_rng(2)
    passing = bootstrap._complex_result(
        rng.normal(scale=1e-4, size=(500, 8))
        + 1j * rng.normal(scale=1e-4, size=(500, 8)))
    assert passing["pointwise_pass"]
    failing = bootstrap._complex_result(np.full((500, 8), 0.01 + 0.01j))
    assert not failing["pointwise_pass"]
    assert not failing["coherent_pass"]


def test_split_reference_summary_reports_known_asymmetry_difference():
    train = np.tile([10.0, 20.0, 30.0, -10.0], (3, 100))
    test = np.tile([10.0, -20.0, -30.0, -10.0], (3, 100))
    result = bootstrap.split_reference_summary(train, test)
    assert result["density_asymmetry"]["mean_signed_error"] == 1.0
