"""Correctness tests for the conditional wrapped-normal mixture.

Pure density mathematics — no simulator, no GPU work.

Lives in the maintained production test suite
because ``wrapped_mixture_model`` is the production surrogate's density
implementation: these are the contracts every fit, export and plot depends on,
so they must run in the suite the project actually runs.

Run from the repo root::

    JAX_PLATFORMS=cpu python -m pytest tests/test_wrapped_mixture.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shared import wnm as wm

DENSE = jnp.arange(-180.0, 180.0, 0.05)
DX = 0.05
# Spans both branches of the adaptive wrapped normal and reaches well past the
# sigma where a fixed spatial wrap window would stop integrating to 1.
SIGMAS = [0.5, 2.0, 10.0, 45.0, 120.0, 300.0, 600.0, 1000.0]


def _fourier_wn_pdf(b, mu, sigma):
    """Independent wrapped-normal reference via its Fourier series, in float64.

    ``WN(b) = 1/360 * (1 + 2 sum_n exp(-n^2 s^2 / 2) cos(n (b - mu)))`` with all
    angles in radians.  It converges quickly for *large* sigma, exactly where the
    spatial sum used by the model converges slowly, so agreement between the two
    is a real cross-check rather than a restatement.  The term count is chosen so
    the truncated tail is below double precision.
    """
    d = np.radians(np.asarray(b, dtype=np.float64) - float(mu))
    s = np.radians(float(sigma))
    n = np.arange(1, int(9.0 / s) + 2)
    terms = np.exp(-0.5 * (n * s) ** 2) * np.cos(n * d[..., None])
    return (1.0 + 2.0 * terms.sum(-1)) / wm.PERIOD


@pytest.mark.parametrize('sigma', SIGMAS)
def test_wrapped_normal_normalisation(sigma):
    logp = wm.wrapped_normal_logpdf(DENSE, jnp.float32(17.0), jnp.float32(sigma))
    assert float(jnp.sum(jnp.exp(logp)) * DX) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize('sigma', SIGMAS)
def test_wrapped_normal_matches_fourier_series(sigma):
    mu = jnp.float32(-63.0)
    ours = np.asarray(jnp.exp(wm.wrapped_normal_logpdf(DENSE, mu, jnp.float32(sigma))),
                      dtype=np.float64)
    theirs = _fourier_wn_pdf(DENSE, mu, sigma)
    # float32 evaluation against a float64 reference: tolerance tracks the peak.
    assert np.max(np.abs(ours - theirs)) < 1e-6 * max(1.0, theirs.max() / 1e-3)


def test_branch_switch_is_continuous():
    """Spatial and Fourier branches agree in a band around ``SIGMA_SWITCH``."""
    mu = jnp.float32(23.0)
    for sigma in [55.0, 58.0, 60.0, 62.0, 65.0]:
        below = wm.wrapped_normal_logpdf(DENSE, mu, jnp.float32(sigma - 1e-3))
        above = wm.wrapped_normal_logpdf(DENSE, mu, jnp.float32(sigma + 1e-3))
        assert float(jnp.max(jnp.abs(below - above))) < 1e-3


def test_large_sigma_gradients_are_finite():
    """Very wide predicted sigma stays differentiable (both branches evaluated)."""
    def total_logp(sigma):
        return jnp.sum(wm.wrapped_normal_logpdf(
            jnp.asarray([-120.0, 0.0, 90.0]), jnp.float32(5.0), sigma))
    for sigma in (5.0, 60.0, 500.0):
        g = jax.grad(total_logp)(jnp.float32(sigma))
        assert bool(jnp.isfinite(g))


def test_wrap_continuity_at_pm180():
    """The density is periodic: +180 and -180 are the same angle."""
    mu, sigma = jnp.float32(150.0), jnp.float32(25.0)
    lo = wm.wrapped_normal_logpdf(jnp.float32(-180.0), mu, sigma)
    hi = wm.wrapped_normal_logpdf(jnp.float32(180.0), mu, sigma)
    assert float(lo) == pytest.approx(float(hi), abs=1e-6)
    # and no discontinuity crossing the seam
    seam = jnp.linspace(179.9, 180.1, 21)
    vals = wm.wrapped_normal_logpdf(seam, mu, sigma)
    assert float(jnp.max(jnp.abs(jnp.diff(vals)))) < 1e-2


def _dummy_dist(n=3, k=4, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, k))
    return {'log_pi': jnp.asarray(logits - jax.scipy.special.logsumexp(logits, -1, keepdims=True)),
            'mu': jnp.asarray(rng.uniform(-180, 180, (n, k))),
            'sigma': jnp.asarray(rng.uniform(1.0, 90.0, (n, k)))}


def test_mixture_normalisation():
    dist = _dummy_dist()
    dens = jnp.exp(wm.mixture_logpdf_grid(DENSE, dist))
    mass = jnp.sum(dens, axis=1) * DX
    assert np.allclose(np.asarray(mass), 1.0, atol=1e-5)


def test_grid_and_pointwise_agree():
    """`mixture_logpdf` and `mixture_logpdf_grid` are the same function."""
    dist = _dummy_dist()
    b = jnp.asarray([-179.0, 0.0, 91.5])
    point = wm.mixture_logpdf(b, dist)
    grid = jnp.diagonal(wm.mixture_logpdf_grid(b, dist))
    assert np.allclose(np.asarray(point), np.asarray(grid), atol=1e-6)


def test_analytic_moments_match_quadrature():
    dist = _dummy_dist()
    dens = jnp.exp(wm.mixture_logpdf_grid(DENSE, dist))
    z = jnp.exp(1j * jnp.radians(DENSE))
    numeric = jnp.sum(dens * z[None, :], axis=1) * DX
    analytic = wm.circular_moment(dist, 1)
    assert np.allclose(np.asarray(numeric), np.asarray(analytic), atol=1e-5)


def test_motor_noise_adds_variance():
    """Convolution with a wrapped normal is exact for this family."""
    dist = _dummy_dist(n=1, k=3, seed=7)
    sd_motor = 20.0
    widened = wm.add_motor_noise(dist, jnp.asarray([sd_motor]))
    assert np.allclose(np.asarray(widened['sigma']) ** 2,
                       np.asarray(dist['sigma']) ** 2 + sd_motor ** 2, atol=1e-4)


def test_motor_noise_matches_numeric_convolution():
    """Analytic widening equals a dense circular convolution of the densities."""
    dist = _dummy_dist(n=1, k=3, seed=3)
    sd_motor = 30.0
    grid = jnp.arange(-180.0, 180.0, 0.5)
    dx = 0.5
    dens = jnp.exp(wm.mixture_logpdf_grid(grid, dist))[0]
    kernel = jnp.exp(wm.wrapped_normal_logpdf(grid, jnp.float32(0.0),
                                              jnp.float32(sd_motor)))
    # circular convolution on the periodic grid
    conv = jnp.real(jnp.fft.ifft(jnp.fft.fft(dens) * jnp.fft.fft(jnp.fft.ifftshift(kernel)))) * dx
    widened = jnp.exp(wm.mixture_logpdf_grid(grid, wm.add_motor_noise(
        dist, jnp.asarray([sd_motor]))))[0]
    assert float(jnp.max(jnp.abs(conv - widened))) < 1e-5


def test_mirror_params_is_an_involution():
    x = jnp.asarray([[10.0, 80.0, 30.0, 45.0], [5.0, 5.0, 200.0, 2.0]])
    assert np.allclose(np.asarray(wm.mirror_params(wm.mirror_params(x))), np.asarray(x))
    assert np.allclose(np.asarray(wm.mirror_params(x))[:, [0, 1]],
                       np.asarray(x)[:, [1, 0]])


def test_featurise_range_and_symmetry():
    x = jnp.asarray([[5.0, 200.0, 5.0, 2.0], [200.0, 5.0, 200.0, 180.0]])
    f = wm.featurise(x)
    assert f.shape == (2, 6)
    assert float(jnp.max(jnp.abs(f[:, :4]))) <= 1.0 + 1e-5
    # swapping sd_feat1/sd_feat2 swaps exactly the first two features
    fm = wm.featurise(wm.mirror_params(x))
    assert np.allclose(np.asarray(fm)[:, [0, 1]], np.asarray(f)[:, [1, 0]], atol=1e-6)
    assert np.allclose(np.asarray(fm)[:, 2:], np.asarray(f)[:, 2:], atol=1e-6)


def _init_model(k=6, seed=0):
    model = wm.ConditionalWrappedMixture(n_components=k)
    x = jnp.asarray([[10.0, 40.0, 25.0, 30.0], [150.0, 7.0, 90.0, 175.0]])
    variables = model.init(jax.random.PRNGKey(seed), x)
    return model, variables, x


def test_model_outputs_are_valid_and_normalised():
    model, variables, x = _init_model()
    dist = model.apply(variables, x)
    assert np.allclose(np.asarray(jnp.exp(dist['log_pi']).sum(-1)), 1.0, atol=1e-6)
    assert float(jnp.min(dist['sigma'])) >= model.min_scale
    assert float(jnp.max(jnp.abs(dist['mu']))) <= 180.0 + 1e-4
    mass = jnp.sum(jnp.exp(wm.mixture_logpdf_grid(DENSE, dist)), axis=1) * DX
    assert np.allclose(np.asarray(mass), 1.0, atol=1e-5)


def test_batch_and_single_predictions_agree():
    model, variables, x = _init_model()
    batch = model.apply(variables, x)
    single = model.apply(variables, x[1:2])
    for key in ('log_pi', 'mu', 'sigma'):
        assert np.allclose(np.asarray(batch[key][1]), np.asarray(single[key][0]),
                           atol=1e-5)


def test_gradients_are_finite():
    model, variables, x = _init_model()
    b = jnp.asarray([-179.5, 12.0])
    w = jnp.ones(2)

    def loss(v):
        dist = model.apply(v, x)
        return -jnp.sum(w * wm.mixture_logpdf(b, dist)) / jnp.sum(w)

    grads = jax.grad(loss)(variables)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(grads))


def test_extreme_bias_values_are_stable():
    """Unwrapped inputs (e.g. 540 degrees) give the same density as their angle."""
    model, variables, x = _init_model()
    dist = model.apply(variables, x)
    a = wm.mixture_logpdf(jnp.asarray([170.0, -3.0]), dist)
    b = wm.mixture_logpdf(jnp.asarray([170.0 + 720.0, -3.0 - 360.0]), dist)
    assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)
    assert bool(jnp.all(jnp.isfinite(a)))


def test_analytic_asymmetry_matches_monte_carlo_for_narrow_zero_peak():
    rng = np.random.default_rng(20)
    dist = {"log_pi": jnp.log(jnp.asarray([[0.8, 0.2]])),
            "mu": jnp.asarray([[0.8, -20.0]]),
            "sigma": jnp.asarray([[1.5, 8.0]])}
    component = rng.choice(2, size=500_000, p=[0.8, 0.2])
    raw = rng.normal(np.array([0.8, -20.0])[component],
                     np.array([1.5, 8.0])[component])
    raw = (raw + 180) % 360 - 180
    empirical = ((raw > 0).sum() - (raw < 0).sum()) / len(raw)
    analytic = float(wm.density_asymmetry(dist)[0])
    assert analytic == pytest.approx(empirical, abs=0.003)


def test_analytic_asymmetry_avoids_two_degree_grid_bias():
    dist = {"log_pi": jnp.zeros((1, 1)), "mu": jnp.asarray([[0.8]]),
            "sigma": jnp.asarray([[1.5]])}
    analytic = float(wm.density_asymmetry(dist)[0])
    grid = jnp.arange(-180.0, 180.0, 2.0)
    density = np.exp(np.asarray(wm.mixture_logpdf_grid(grid, dist))[0])
    coarse = 2.0 * (density[np.asarray(grid) > 0].sum()
                    - density[np.asarray(grid) < 0].sum())
    assert abs(analytic - coarse) > 0.05


@pytest.mark.parametrize("sigma", [0.25, 12.0, 59.9, 60.0, 60.1, 200.0, 500.0])
def test_asymmetry_hybrid_matches_high_wrap_spatial_reference(sigma):
    dist = {"log_pi": jnp.log(jnp.asarray([[0.35, 0.65]])),
            "mu": jnp.asarray([[-179.9, 37.0]]),
            "sigma": jnp.asarray([[sigma, sigma * 0.7]])}
    positive = wm.wrapped_normal_interval_probability(
        dist["mu"], dist["sigma"], 0.0, 180.0, 16)
    negative = wm.wrapped_normal_interval_probability(
        dist["mu"], dist["sigma"], -180.0, 0.0, 16)
    reference = jnp.sum(jnp.exp(dist["log_pi"]) * (positive - negative), axis=-1)
    np.testing.assert_allclose(np.asarray(wm.density_asymmetry(dist)),
                               np.asarray(reference), atol=3e-7, rtol=3e-7)


def test_asymmetry_value_and_gradient_are_continuous_at_representation_switch():
    dist = {"log_pi": jnp.zeros((1, 1)), "mu": jnp.asarray([[37.0]])}

    def asymmetry(sigma):
        return wm.density_asymmetry({**dist, "sigma": sigma[None, None]})[0]

    below = jnp.float32(wm.SIGMA_SWITCH - 1e-3)
    above = jnp.float32(wm.SIGMA_SWITCH + 1e-3)
    assert float(jnp.abs(asymmetry(below) - asymmetry(above))) < 2e-5
    assert float(jnp.abs(jax.grad(asymmetry)(below) -
                         jax.grad(asymmetry)(above))) < 2e-5
