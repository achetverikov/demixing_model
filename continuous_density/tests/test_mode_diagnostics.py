import numpy as np

from continuous_density.mode_diagnostics import circular_modes


def test_mode_diagnostic_finds_two_separated_modes():
    rng = np.random.default_rng(4)
    samples = np.r_[rng.normal(-60, 4, 7000), rng.normal(70, 5, 3000)]
    positions, masses = circular_modes(samples)
    assert len(positions) == 2
    assert np.any(np.abs(positions + 60) < 3)
    assert np.any(np.abs(positions - 70) < 3)
    np.testing.assert_allclose(sorted(masses), [0.3, 0.7], atol=0.03)


def test_mode_diagnostic_respects_circular_seam():
    rng = np.random.default_rng(7)
    samples = (rng.normal(179, 3, 10000) + 180) % 360 - 180
    positions, masses = circular_modes(samples)
    assert len(positions) == 1
    assert abs(abs(positions[0]) - 180) < 4
    np.testing.assert_allclose(masses, [1.0], atol=0.01)
