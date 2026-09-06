import numpy as np

from continuous_density import local_distribution_gate as gate


def test_candidate_parser_preserves_path():
    candidate = gate.parse_candidate("w8:wrapped_mixture:8:/tmp/parameters.npz")
    assert candidate.label == "w8"
    assert candidate.size == 8
    assert str(candidate.parameters) == "/tmp/parameters.npz"


def test_paired_nll_intervals_detect_identical_and_worse_candidate():
    identical = gate.one_sided_mean_intervals(np.zeros((5, 100)))
    assert identical["coherent_upper"] == 0
    worse = gate.one_sided_mean_intervals(np.full((5, 100), 0.01))
    assert worse["coherent_upper"] > 0.009


def test_wasserstein_split_half_is_zero_for_identical_histograms():
    samples = np.tile(np.array([-10.0, 10.0, -10.0, 10.0]), (3, 1))
    result = gate.split_half_wasserstein(samples, samples)
    assert result["mean_deg"] == 0


def test_candidate_row_extracts_one_saved_trajectory_point(tmp_path):
    path = tmp_path / "fourier.npz"
    np.savez(path, k1_theta=np.zeros((3, 2)))
    candidate = gate.Candidate("f1", "maxent_fourier", 1, path)
    samples = np.array([[0.0, 10.0]])
    assert gate.candidate_logpdf(candidate, samples, row=2).shape == (1, 2)
    assert gate.candidate_mass(candidate, row=2).shape == (1, 720)
