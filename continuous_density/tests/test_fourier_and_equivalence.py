"""Tests for Task 2.0 Fourier, curve, and uncertainty infrastructure."""

import numpy as np
import pytest

from continuous_density import equivalence
from continuous_density import fourier_moments as fm
from continuous_density import maxent_fourier as mf


def test_fourier_asymmetry_converges_to_direct_sign_mass():
    rng = np.random.default_rng(2)
    samples = np.concatenate([rng.normal(12, 4, 160_000),
                              rng.normal(-35, 10, 40_000)])
    coefficients = fm.empirical_coefficients(samples, 151)
    direct = fm.direct_asymmetry(samples)
    error_15 = abs(fm.asymmetry_from_coefficients(coefficients, 15)[0] - direct[0])
    error_151 = abs(fm.asymmetry_from_coefficients(coefficients, 151)[0] - direct[0])
    assert error_151 < 0.01
    assert error_151 < error_15


def test_required_bandwidth_requires_staying_inside_margin():
    coefficients = np.zeros((1, 6), dtype=complex)
    coefficients[0, 1] = 1j * np.pi / 4
    coefficients[0, 3] = 1j * 0.3
    reference = fm.asymmetry_from_coefficients(coefficients, 5)
    # K=1 is close to the final value only under a loose crossing criterion;
    # the omitted k=3 term then moves the partial sum away.
    required = fm.required_asymmetry_bandwidth(coefficients, reference, 0.01)
    assert required[0] >= 3


def test_full_spectrum_detects_symmetric_narrow_structure():
    rng = np.random.default_rng(3)
    samples = np.concatenate([rng.normal(-60, 2, 50_000),
                              rng.normal(60, 2, 50_000)])
    coefficients = fm.empirical_coefficients(samples, 20)[0]
    assert abs(fm.direct_asymmetry(samples)[0]) < 0.01
    assert abs(coefficients[6].real) > 0.9
    assert abs(coefficients[6].imag) < 0.01


def test_linear_fourier_reconstructs_known_cosine_density():
    grid = np.linspace(-180, 180, 720, endpoint=False)
    coefficients = np.zeros(4, dtype=complex)
    coefficients[2] = 0.2
    density = fm.linear_fourier_density(coefficients, grid)[0]
    expected = (1 + 0.4 * np.cos(2 * np.radians(grid))) / 360
    assert np.allclose(density, expected)


def test_fejer_reconstruction_is_nonnegative_and_normalized():
    rng = np.random.default_rng(31)
    coefficients = fm.empirical_coefficients(rng.normal(20, 3, 20_000), 30)
    grid = np.linspace(-180, 180, 720, endpoint=False)
    density = fm.fejer_density(coefficients, grid, 30)[0]
    assert density.min() >= 0
    assert density.sum() * 0.5 == pytest.approx(1.0, abs=2e-12)


def test_circular_wasserstein_respects_the_seam():
    grid = np.linspace(-180, 180, 360, endpoint=False)
    p = np.zeros(360)
    q = np.zeros(360)
    p[np.argmin(abs(grid + 179))] = 1
    q[np.argmin(abs(grid - 179))] = 1
    assert fm.circular_wasserstein_grid(p, q, 1.0)[0] == pytest.approx(2.0)


def test_curve_summary_finds_coherent_run_and_slope():
    reference = np.arange(8.0)
    predicted = 0.8 * reference + np.array([1, 1, 1, -1, -1, -1, -1, 0])
    summary = equivalence.curve_summary(predicted, reference)
    assert summary["longest_sign_run"] == 5
    assert summary["amplitude_slope"] < 1
    assert summary["max_index"] == 6


def test_equivalence_margin_does_not_change_with_precision():
    rng = np.random.default_rng(4)
    imprecise = rng.normal(0.001, 0.003, (2000, 10))
    precise = rng.normal(0.001, 0.0001, (2000, 10))
    a = equivalence.equivalence_from_bootstrap(imprecise, 0.005, 0.002)
    b = equivalence.equivalence_from_bootstrap(precise, 0.005, 0.002)
    assert not a.coherent_pass
    assert b.coherent_pass
    assert b.coherent_estimate == pytest.approx(0.001, abs=2e-5)


def test_shared_index_bootstrap_preserves_cross_point_covariance():
    rng = np.random.default_rng(5)
    common = rng.normal(size=4000)
    samples = np.stack([common + offset for offset in (0.0, 1.0, 2.0)])
    predicted = samples.mean(axis=1)

    def mean(x):
        return x.mean(axis=1)

    shared = equivalence.bootstrap_trajectory(
        samples, mean, predicted, 500, seed=6, shared_indices=True)
    independent = equivalence.bootstrap_trajectory(
        samples, mean, predicted, 500, seed=6, shared_indices=False)
    assert np.corrcoef(shared.T)[0, 1] > 0.99
    assert abs(np.corrcoef(independent.T)[0, 1]) < 0.15


def test_fast_shared_circular_bootstrap_preserves_covariance_and_statistics():
    rng = np.random.default_rng(51)
    common = rng.normal(10, 4, 2000)
    samples = np.stack([common, common + 20])
    draws = equivalence.bootstrap_circular_statistics_shared(
        samples, 300, seed=52, bootstrap_chunk=20)
    assert np.corrcoef(draws["mean_bias"].T)[0, 1] > 0.99
    assert draws["mean_bias"].mean(axis=0) == pytest.approx(
        [samples[0].mean(), samples[1].mean()], abs=0.1)
    assert draws["density_asymmetry"].shape == (300, 2)


def test_independent_circular_bootstrap_does_not_create_row_covariance():
    rng = np.random.default_rng(53)
    samples = np.stack([rng.normal(0, 5, 2000), rng.normal(0, 5, 2000)])
    draws = equivalence.bootstrap_circular_statistics_independent(
        samples, 300, seed=54, bootstrap_chunk=20)
    assert abs(np.corrcoef(draws["mean_bias"].T)[0, 1]) < 0.15


def test_maxent_fourier_is_normalized_periodic_and_recovers_moments():
    rng = np.random.default_rng(7)
    samples = np.degrees(rng.vonmises(mu=np.radians(35), kappa=3, size=30_000))
    result = mf.fit_maxent_fourier(samples[:20_000], 4, quadrature_size=2048)
    assert result.success
    grid = mf.periodic_grid(4096)
    density = result.density.density_grid(grid)
    assert density.sum() * (360 / len(grid)) == pytest.approx(1.0, abs=2e-6)
    assert result.density.logpdf(np.array([-180.0])) == pytest.approx(
        result.density.logpdf(np.array([180.0])), abs=1e-10)
    empirical = fm.empirical_coefficients(samples[:20_000], 4)[0, 1:]
    fitted = result.density.moments(4)[1:]
    assert np.max(np.abs(empirical - fitted)) < 2e-4
    heldout = mf.heldout_metrics(result.density, samples[20_000:])
    assert abs(equivalence.wrap_deg(
        heldout["pred_mean_bias"] - heldout["raw_mean_bias"])) < 1.0


def test_maxent_quadrature_convergence():
    density = mf.MaxentFourierDensity(np.array([1.2, -0.4, 0.3, 0.1]), 2048)
    check = mf.quadrature_convergence(density, (512, 1024, 2048))
    assert max(abs(np.asarray(check["mass"]) - 1)) < 1e-10
    assert check["max_logz_step"] < 1e-10
