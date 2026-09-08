"""Contracts for the shared prediction layer.

The failure modes this layer exists to prevent are all silent ones: motor noise
applied to a fitted curve but not to the likelihood that scores it, component 2
obtained by flipping a sign instead of swapping the two feature SDs, a narrow
density's mass read off a 2-degree grid, a surrogate extrapolating outside its
training domain because nothing checked. Every one of those produces plausible
numbers, so each gets a test here.

The surface-parity test is the load-bearing one: ``SurfacePredictor`` restates
the historical discrete asymmetry, and if that restatement is not exactly
``compute_single_density_asymmetry`` then the two families are being compared
through two different estimators.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_density import wrapped_mixture_model as wm  # noqa: E402
from shared import surrogate  # noqa: E402
from shared.mu1_axis import mu1_grid, mu1_cell_width  # noqa: E402
from shared.prediction import (  # noqa: E402
    PARAM_ORDER, SPATIAL_SEPARATION, SurfacePredictor, WrappedMixturePredictor,
    LEGACY_DOMAIN, domain_from_meta, dprime_from_sd_spat, gaussian_curve_smoother,
    legal_warmup_params, mirror_params, predictor_from_surrogate, sd_spat_from_dprime,
    validate_params)
from shared.utils import compute_single_density_asymmetry  # noqa: E402

ARTIFACT = surrogate.WNM_DEFAULTS[20]
needs_artifact = pytest.mark.skipif(not ARTIFACT.exists(),
                                    reason="no packaged WNM artifact installed")

# Narrow, broad, asymmetric in both orders, low and high d-prime, both ends of
# the feature axis.
# Spans the declared domain: sd_feat down to 2.5 (well below the production
# fitting floor of 5, and the region that motivates this surrogate), sd_spat only
# to 5, and both feature-noise orders.
PARAMS = jnp.asarray([
    [10.0, 10.0, 10.0, 2.0],
    [10.0, 120.0, 10.0, 30.0],
    [120.0, 10.0, 10.0, 30.0],
    [60.0, 60.0, 30.0, 90.0],
    [200.0, 200.0, 200.0, 179.0],
], dtype=jnp.float32)


@pytest.fixture(scope="module")
def predictor():
    return predictor_from_surrogate(surrogate.load_surrogate(checkpoint_path=ARTIFACT))


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

def test_dprime_and_sd_spat_are_one_number_in_two_views():
    for sd_spat in (5.0, 21.0, 42.0, 200.0):
        dprime = float(dprime_from_sd_spat(sd_spat))
        assert dprime == pytest.approx(SPATIAL_SEPARATION / sd_spat)
        assert float(sd_spat_from_dprime(dprime)) == pytest.approx(sd_spat)

def test_mirror_swaps_only_the_two_feature_sds():
    out = np.asarray(mirror_params(PARAMS))
    np.testing.assert_array_equal(out[:, 0], np.asarray(PARAMS)[:, 1])
    np.testing.assert_array_equal(out[:, 1], np.asarray(PARAMS)[:, 0])
    np.testing.assert_array_equal(out[:, 2:], np.asarray(PARAMS)[:, 2:])


@needs_artifact
def test_component_two_is_a_swap_not_a_sign_flip(predictor):
    """Adding a sign flip on top of the swap would invert every component-2 curve.

    Attraction and repulsion are the scientific reading of that sign, so this is
    the difference between a result and its opposite.
    """
    swapped = predictor.component_distribution(PARAMS, 2)
    reference = predictor.distribution(mirror_params(PARAMS))
    for key in ("log_pi", "mu", "sigma"):
        np.testing.assert_array_equal(np.asarray(swapped[key]), np.asarray(reference[key]))

    # And it is genuinely a different prediction from component 1, not a no-op:
    # the asymmetric rows must disagree.
    comp1 = np.asarray(predictor.signed_arc_asymmetry(PARAMS))
    comp2 = np.asarray(wm.density_asymmetry(swapped, predictor.arc_wraps))
    asymmetric = np.asarray(PARAMS)[:, 0] != np.asarray(PARAMS)[:, 1]
    assert np.any(np.abs(comp1[asymmetric] - comp2[asymmetric]) > 1e-4)
    # Nor is it the negation of component 1.
    assert not np.allclose(comp2, -comp1, atol=1e-3)


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------

def test_out_of_domain_parameters_raise_and_name_the_column():
    with pytest.raises(ValueError, match="sd_feat2"):
        validate_params(np.array([[10.0, 400.0, 10.0, 30.0]]), domain=LEGACY_DOMAIN)
    with pytest.raises(ValueError, match="feat_diff"):
        validate_params(np.array([[10.0, 10.0, 10.0, 300.0]]), domain=LEGACY_DOMAIN)
    with pytest.raises(ValueError, match="non-finite"):
        validate_params(np.array([[10.0, np.nan, 10.0, 30.0]]), domain=LEGACY_DOMAIN)
    with pytest.raises(ValueError, match=r"shape"):
        validate_params(np.array([[10.0, 10.0, 10.0]]), domain=LEGACY_DOMAIN)
    with pytest.raises(ValueError, match="missing bounds"):
        validate_params(np.array([[10.0, 10.0, 10.0, 30.0]]),
                        domain={"sd_feat1": (5.0, 200.0)})


def test_warmup_rows_are_inside_the_domain():
    """Compilation warm-up used ``jnp.ones((n, 3))``: an SD of 1, out of domain.

    Shapes are all compilation needs, so the illegal values cost nothing there --
    but the same rows are used to pad batches, where they would either trip
    validation or be predicted at and quietly mixed into a result.
    """
    rows = legal_warmup_params(7, (5.0, 200.0), (0.0, 180.0))
    assert rows.shape == (7, 4)
    validate_params(rows, domain=LEGACY_DOMAIN)


@needs_artifact
def test_predictor_refuses_out_of_domain_inputs(predictor):
    with pytest.raises(ValueError, match="corpus hull"):
        predictor.distribution(jnp.asarray([[1.0, 1.0, 1.0, 30.0]], jnp.float32))


# ---------------------------------------------------------------------------
# Motor noise lives in this layer
# ---------------------------------------------------------------------------

@needs_artifact
def test_motor_noise_widens_variance_analytically(predictor):
    """Wrapped normals are closed under circular convolution: variances add."""
    noisy = predictor.with_motor_noise(15.0)
    base = predictor.distribution(PARAMS)
    widened = noisy.distribution(PARAMS)

    np.testing.assert_allclose(np.asarray(widened["sigma"]),
                               np.sqrt(np.asarray(base["sigma"]) ** 2 + 15.0 ** 2),
                               rtol=1e-6)
    # Weights and means are untouched by convolution with a zero-mean kernel.
    np.testing.assert_array_equal(np.asarray(widened["mu"]), np.asarray(base["mu"]))
    np.testing.assert_array_equal(np.asarray(widened["log_pi"]), np.asarray(base["log_pi"]))


@needs_artifact
def test_motor_noise_reaches_every_quantity_not_just_the_curve(predictor):
    """The failure this prevents: a fit smoothed by motor noise, scored without it."""
    noisy = predictor.with_motor_noise(20.0)
    assert not np.allclose(np.asarray(predictor.circular_sd(PARAMS)),
                           np.asarray(noisy.circular_sd(PARAMS)))
    assert not np.allclose(np.asarray(predictor.log_density(PARAMS, jnp.zeros(len(PARAMS)))),
                           np.asarray(noisy.log_density(PARAMS, jnp.zeros(len(PARAMS)))))
    assert not np.allclose(np.asarray(predictor.cell_probabilities(PARAMS)),
                           np.asarray(noisy.cell_probabilities(PARAMS)))
    # Motor noise cannot sharpen a distribution.
    assert np.all(np.asarray(noisy.circular_sd(PARAMS))
                  >= np.asarray(predictor.circular_sd(PARAMS)) - 1e-4)


@needs_artifact
def test_motor_noise_is_recorded_in_the_identity(predictor):
    assert predictor.identity().as_dict()["surrogate_sd_motor"] == 0.0
    assert predictor.with_motor_noise(12.5).identity().as_dict()["surrogate_sd_motor"] == 12.5


# ---------------------------------------------------------------------------
# Grid densities are for display; probabilities are for scoring
# ---------------------------------------------------------------------------

def test_a_narrow_component_is_mis_massed_by_a_grid_but_not_by_integration():
    """Why distributional scores must not go through the 180-row surface.

    ``min_scale`` is a quarter degree, well under the 2-degree reporting cell, so
    a component can sit almost entirely inside one cell. Sampling its peak and
    renormalising then reports a mass that depends on where the peak fell
    relative to the cell centre.
    """
    dist = {"log_pi": jnp.zeros((1, 1)),
            "mu": jnp.asarray([[0.9]]),          # inside a cell, off its centre
            "sigma": jnp.asarray([[0.3]])}       # far narrower than the cell

    exact = float(wm.wrapped_normal_interval_probability(
        dist["mu"], dist["sigma"], 0.0, 180.0, 8)[0, 0])

    grid = mu1_grid()
    density = jnp.exp(wm.mixture_logpdf_grid(grid, dist, 4))[0]
    renormalised = density / jnp.sum(density)
    grid_mass = float(jnp.sum(jnp.where(grid > 0, renormalised, 0.0)))

    assert exact > 0.99
    assert abs(grid_mass - exact) > 0.05, (
        "a grid happened to agree here; the test is meant to exhibit the "
        f"discrepancy (grid {grid_mass:.4f} vs exact {exact:.4f})")


@needs_artifact
@pytest.mark.parametrize("sd_motor", [0.0, 30.0, 150.0])
def test_cell_probabilities_are_a_distribution_across_the_domain(predictor, sd_motor):
    """Wrap truncation check: 8 wraps must still integrate to 1 at broad scales."""
    probs = np.asarray(predictor.with_motor_noise(sd_motor).cell_probabilities(PARAMS))
    assert probs.shape == (len(PARAMS), len(mu1_grid()))
    assert np.all(probs >= -1e-9)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)


@needs_artifact
def test_cell_probabilities_are_jittable_and_differentiable(predictor):
    """Distributional objectives differentiate through integrated cell mass."""
    def objective(sd_feat1):
        rows = PARAMS.at[0, 0].set(sd_feat1)
        probabilities = predictor.cell_probabilities(rows, validate=False)
        return probabilities[0, 90] + 0.5 * probabilities[0, 91]

    value, gradient = jax.jit(jax.value_and_grad(objective))(10.0)
    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient))
    assert not np.isclose(float(gradient), 0.0)


@needs_artifact
def test_grid_density_integrates_to_one(predictor):
    density = np.exp(np.asarray(predictor.grid_log_density(PARAMS)))
    np.testing.assert_allclose(density.sum(axis=-1) * mu1_cell_width(), 1.0, atol=1e-3)


@needs_artifact
def test_log_density_is_continuous_not_a_cell_lookup(predictor):
    """Two biases inside one reporting cell must give different log densities."""
    row = PARAMS[:1]
    a = float(predictor.log_density(row, jnp.asarray([0.1]))[0])
    b = float(predictor.log_density(row, jnp.asarray([0.9]))[0])
    assert a != b


# ---------------------------------------------------------------------------
# Asymmetry: raw versus fitting-smoothed
# ---------------------------------------------------------------------------

@needs_artifact
def test_smoothed_curve_is_the_raw_curve_through_the_shared_smoother(predictor):
    """The estimator is 'analytic curve, then the existing smoother' -- in that order."""
    feat = jnp.arange(2.0, 182.0, 2.0, dtype=jnp.float32)
    curve = jnp.stack([jnp.full_like(feat, 30.0), jnp.full_like(feat, 60.0),
                       jnp.full_like(feat, 20.0), feat], axis=-1)

    raw = predictor.signed_arc_asymmetry(curve)
    smoothed = predictor.smoothed_asymmetry_curve(curve, 10.0)
    np.testing.assert_array_equal(np.asarray(smoothed),
                                  np.asarray(gaussian_curve_smoother(raw, 10.0)))
    assert not np.allclose(np.asarray(raw), np.asarray(smoothed))


@needs_artifact
def test_smoothing_refuses_a_batch_that_is_not_one_curve(predictor):
    mixed = jnp.asarray([[10.0, 10.0, 10.0, 2.0], [60.0, 60.0, 30.0, 4.0]], jnp.float32)
    with pytest.raises(ValueError, match="one curve"):
        predictor.smoothed_asymmetry_curve(mixed, 10.0)


# ---------------------------------------------------------------------------
# The surface backend keeps its historical conventions exactly
# ---------------------------------------------------------------------------

def test_surface_asymmetry_matches_the_historical_implementation():
    """Both families must be compared through one estimator, not two."""
    rng = np.random.default_rng(5)
    grid = mu1_grid()
    surfaces = jnp.asarray(np.log(np.abs(rng.normal(size=(3, len(grid), 90))) + 1e-3))
    indices = jnp.arange(0, 90, 5)

    predictor = SurfacePredictor(surfaces, n_samples=20, artifact="test.pkl")
    got = np.asarray(predictor.signed_arc_asymmetry(indices))
    expected = np.stack([np.asarray(compute_single_density_asymmetry(
        surface, indices, grid, apply_smoothing=False)) for surface in surfaces])
    np.testing.assert_array_equal(got, expected)


def test_surface_smoothed_curve_matches_the_historical_implementation():
    rng = np.random.default_rng(6)
    grid = mu1_grid()
    surfaces = jnp.asarray(np.log(np.abs(rng.normal(size=(2, len(grid), 90))) + 1e-3))
    indices = jnp.arange(0, 90, 3)

    predictor = SurfacePredictor(surfaces, n_samples=20, artifact="test.pkl")
    got = np.asarray(predictor.smoothed_asymmetry_curve(indices, 10.0))
    expected = np.stack([np.asarray(compute_single_density_asymmetry(
        surface, indices, grid, apply_smoothing=True, smoothing_sigma=10.0))
        for surface in surfaces])
    np.testing.assert_array_equal(got, expected)


def test_surface_predictor_will_not_pretend_to_apply_motor_noise():
    surfaces = jnp.zeros((1, len(mu1_grid()), 90))
    predictor = SurfacePredictor(surfaces, n_samples=20, artifact="test.pkl")
    with pytest.raises(NotImplementedError, match="FFT"):
        predictor.with_motor_noise(10.0)

def _dense_grid():
    return jnp.arange(-180.0, 180.0, 0.02)


def test_motor_noise_equals_numerical_circular_convolution():
    """Analytic variance addition against an explicit FFT convolution."""
    dist = {"log_pi": jnp.log(jnp.asarray([[0.3, 0.7]])),
            "mu": jnp.asarray([[-25.0, 40.0]]),
            "sigma": jnp.asarray([[12.0, 55.0]])}
    grid = _dense_grid()
    sd_motor = 23.0

    analytic = np.exp(np.asarray(
        wm.mixture_logpdf_grid(grid, wm.add_motor_noise(dist, sd_motor), 4))[0])
    base = np.exp(np.asarray(wm.mixture_logpdf_grid(grid, dist, 4))[0])

    x = np.asarray(grid)
    kernel = sum(np.exp(-0.5 * ((x + shift * 360.0) / sd_motor) ** 2)
                 for shift in range(-4, 5))
    kernel /= kernel.sum()
    numerical = np.real(np.fft.ifft(
        np.fft.fft(base) * np.fft.fft(np.roll(kernel, -len(kernel) // 2))))

    np.testing.assert_allclose(analytic, numerical, atol=1e-7)


@pytest.mark.parametrize("sigma", [0.5, 12.0, 55.0, 200.0])
def test_analytic_asymmetry_equals_dense_quadrature(sigma):
    """Closed-form sign mass against quadrature, across narrow and broad scales."""
    dist = {"log_pi": jnp.log(jnp.asarray([[0.35, 0.65]])),
            "mu": jnp.asarray([[-25.0, 40.0]]),
            "sigma": jnp.asarray([[sigma, sigma * 1.5]])}
    grid = _dense_grid()
    step = 0.02

    density = np.exp(np.asarray(wm.mixture_logpdf_grid(grid, dist, 4))[0])
    x = np.asarray(grid)
    quadrature = float(density[x > 0].sum() * step - density[x < 0].sum() * step)
    analytic = float(wm.density_asymmetry(dist, 8)[0])

    assert abs(analytic - quadrature) < 1e-4, (
        f"sigma={sigma}: analytic {analytic:.8f} vs quadrature {quadrature:.8f}")


# ---------------------------------------------------------------------------
# Regressions from the 2026-09-06 audit
# ---------------------------------------------------------------------------

@needs_artifact
def test_the_domain_is_derived_from_the_corpus_not_the_featurisation(predictor):
    """The featurisation constants bracket [5, 200] because that is what they
    rescale to [-1, 1] -- an input-scaling choice, not a statement about what the
    network was shown. Reading the domain off them understated the trained
    feature-noise region by a factor of two (16% of the corpus's sd_feat values
    sit below 5) while overstating it at feat_diff = 0, which the corpus never
    visits.
    """
    domain = predictor.domain
    assert set(domain) == set(PARAM_ORDER)
    assert domain["sd_feat1"] == (2.5, 200.0)
    assert domain["sd_feat2"] == (2.5, 200.0)
    # sd_spat is 42/d' with d' capped at 8.4, so it does not reach below 5 --
    # which is why one shared interval for all three SDs could not be right.
    assert domain["sd_spat"] == (5.0, 200.0)
    assert domain["feat_diff"] == (0.5, 180.0)

    predictor.distribution(jnp.asarray([[2.5, 2.5, 5.0, 0.5],
                                        [200.0, 200.0, 200.0, 180.0]], jnp.float32))


@needs_artifact
def test_the_declared_domain_never_understates_the_corpus(predictor):
    """Declaring less than was trained refuses predictions the network can make.

    The declared box rounds the corpus hull outward to the design's round
    numbers, accepting a sliver of extrapolation (largest: 1.89 degrees at
    sd_feat1's top). It must never round inward.
    """
    hull = predictor.meta["corpus_hull"]
    for key, (lo, hi) in predictor.domain.items():
        hull_lo, hull_hi = hull[key]
        assert lo <= hull_lo + 1e-3, f"{key} low bound excludes trained parameters"
        assert hi >= hull_hi - 1e-3, f"{key} high bound excludes trained parameters"

    # And the accepted extrapolation is recorded rather than left implicit.
    overhang = predictor.meta["declared_domain_overhang"]
    assert max(max(v) for v in overhang.values()) < 2.0


@needs_artifact
def test_untrained_regions_are_still_refused(predictor):
    """A declared domain is not a licence to extrapolate arbitrarily."""
    with pytest.raises(ValueError, match="feat_diff"):
        predictor.distribution(jnp.asarray([[10.0, 10.0, 10.0, 0.0]], jnp.float32))
    with pytest.raises(ValueError, match="sd_feat1"):
        predictor.distribution(jnp.asarray([[2.0, 10.0, 10.0, 30.0]], jnp.float32))
    with pytest.raises(ValueError, match="sd_spat"):
        predictor.distribution(jnp.asarray([[10.0, 10.0, 4.0, 30.0]], jnp.float32))


def test_a_wnm1_artifact_keeps_the_domain_it_claims():
    """Old artifacts are honoured as written, not widened on their behalf."""
    legacy = domain_from_meta({"supported_sd_range": [5.0, 200.0],
                               "supported_feat_diff_range": [0.0, 180.0]})
    assert legacy["sd_feat1"] == (5.0, 200.0)
    assert legacy["feat_diff"] == (0.0, 180.0)
    assert domain_from_meta({}) == LEGACY_DOMAIN


@needs_artifact
@pytest.mark.parametrize("bad", [-20.0, -1e-9, float("nan"), float("inf")])
def test_motor_noise_rejects_negative_and_non_finite(predictor, bad):
    """Variance addition squares the SD, so -20 is silently identical to +20
    while the identity records -20; a NaN would spread through every downstream
    number as a plausible-looking absence.
    """
    with pytest.raises(ValueError):
        predictor.with_motor_noise(bad)


@needs_artifact
def test_zero_motor_noise_is_still_allowed(predictor):
    assert predictor.with_motor_noise(0.0).identity().as_dict()["surrogate_sd_motor"] == 0.0


@needs_artifact
def test_smoothing_refuses_a_shuffled_or_unevenly_spaced_feature_axis(predictor):
    """The kernel is symmetric about each row and its width is in grid steps, so
    the rows must be the feature axis, in order, evenly spaced. Either violation
    still yields a smooth-looking curve -- one that averaged the wrong points.
    """
    def rows(feat):
        feat = np.asarray(feat, dtype=np.float64)
        return jnp.asarray(np.stack([np.full(len(feat), 30.0), np.full(len(feat), 60.0),
                                     np.full(len(feat), 20.0), feat], axis=-1), jnp.float32)

    with pytest.raises(ValueError, match="evenly spaced"):
        predictor.smoothed_asymmetry_curve(rows([2.0, 4.0, 10.0, 12.0]), 10.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        predictor.smoothed_asymmetry_curve(rows([2.0, 8.0, 4.0, 10.0]), 10.0)
    with pytest.raises(ValueError, match="at least two"):
        predictor.smoothed_asymmetry_curve(rows([2.0]), 10.0)

    # The well-formed axis still works.
    predictor.smoothed_asymmetry_curve(rows(np.arange(2.0, 182.0, 2.0)), 10.0)


def test_scoring_cell_masses_underflow_only_where_it_cannot_matter():
    """float32 cell probabilities return exact zeros where the true mass is tiny.

    That is a deliberate trade, not an oversight, and this pins the bound that
    makes it defensible: the zeroed cells must hold so little mass that no energy
    score can notice. The same underflow in a per-trial *log* likelihood is not
    tolerable -- log(0) is not a small number -- and that column is computed
    separately in float64.
    """
    from scipy.stats import norm

    from shared.mu1_axis import mu1_cell_width, mu1_grid

    artifact = surrogate.WNM_DEFAULTS[20]
    if not artifact.exists():
        pytest.skip("no packaged WNM artifact")
    predictor = predictor_from_surrogate(surrogate.load_surrogate(checkpoint_path=artifact))

    # The corpus's narrowest corner, where components are tightest.
    rows = jnp.asarray([[2.5, 2.5, 5.0, 2.0]], jnp.float32)
    probabilities = np.asarray(predictor.cell_probabilities(rows, validate=False))[0]

    centres = np.asarray(mu1_grid())
    half = mu1_cell_width() / 2.0
    dist = predictor.distribution(rows, validate=False)
    mu = np.asarray(dist["mu"], np.float64)
    sigma = np.asarray(dist["sigma"], np.float64)
    weights = np.asarray(np.exp(np.asarray(dist["log_pi"], np.float64)), np.float64)

    exact = np.zeros(len(centres))
    for shift in np.arange(-8, 9) * 360.0:
        upper = (centres[:, None] + half + shift - mu) / sigma
        lower = (centres[:, None] - half + shift - mu) / sigma
        exact += np.sum(weights * (norm.cdf(upper) - norm.cdf(lower)), axis=-1)

    lost = exact[probabilities == 0.0].sum()
    assert lost < 1e-10, (
        f"float32 zeroed cells hold {lost:.3e} of mass; at that size the underflow could "
        "move an energy score and the scoring path would need float64 too")
    # And the surviving mass is still a distribution.
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-5)
