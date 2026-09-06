import numpy as np

from continuous_density import periodic_spline as spline


def test_periodic_spline_basis_partitions_unity_and_wraps():
    x = np.linspace(-180, 180, 721)
    values = spline.basis(x, 24)
    np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(values[0], values[-1], atol=1e-12)


def test_periodic_spline_density_normalizes_and_is_continuous():
    density = spline.PeriodicSplineDensity(np.linspace(-1, 1, 23))
    grid = np.linspace(-180, 180, 2880, endpoint=False)
    assert abs(density.density_grid(grid).sum() * 0.125 - 1) < 1e-10
    np.testing.assert_allclose(density.logpdf([-180]), density.logpdf([180]), atol=1e-12)


def test_periodic_spline_fit_improves_over_uniform():
    rng = np.random.default_rng(2)
    samples = rng.normal(40, 8, 5000)
    fit = spline.fit_periodic_spline(samples, 24)
    assert fit.density.logpdf(samples).mean() > -np.log(360) + 1
