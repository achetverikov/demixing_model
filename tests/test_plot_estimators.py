"""The plotting estimators, pinned against their pre-routing reference.

Row 3 of the subject plots compares a model circular SD against an empirical one
taken over ~10-degree feature bins. That comparison only means anything if both
sides measure the same thing, which is why the model side pools: evaluating the
model at one 2-degree column omits the between-column spread that pooling trials
introduces, and would read on the plot as the model underestimating variability.

So the pooling is an estimator definition, not an implementation detail, and
routing it through the shared prediction layer has to leave it numerically
untouched. `tests/data/plot_estimators_golden.npz` was recorded by
`record_plot_estimators_golden.py` from the plotting code before that routing.

The mixture reaches the same estimator by a different route -- weight-averaging
per-column analytic moments rather than integrating a mixed grid -- because
mixing densities is linear in the density. The surface path keeps its integral,
since those are the numbers the deployed plots were produced with.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "plot_estimators_golden.npz"
CHECKPOINT = ROOT / "pretrained" / "model_epoch1425_10ktrain_20samples.pkl"

if not GOLDEN_PATH.exists():
    raise RuntimeError(
        f"{GOLDEN_PATH} is missing. It is the pre-routing reference for the plotting "
        "estimators and is tracked in git; restore it rather than re-recording it, which "
        "against rerouted code would make this check vacuous.")

pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(),
                                reason="no pretrained surface checkpoint")

from shared import surrogate  # noqa: E402
from shared.config import config  # noqa: E402
from shared.prediction import SurfacePredictor, predictor_from_surrogate  # noqa: E402

CASES = ["continuous", "discrete_levels", "sparse", "empty_bins"]


@pytest.fixture(scope="module")
def golden():
    return np.load(GOLDEN_PATH)


@pytest.fixture(scope="module")
def surfaces():
    from grid_based_multi_condition_optimizer_jax_loops import (
        GridBasedMultiConditionOptimizer)

    optimizer = GridBasedMultiConditionOptimizer(
        str(CHECKPOINT), {"dummy": jnp.zeros((4, 2))}, skip_motor_noise=True)
    return optimizer._predict_batch_fixed_size(
        jnp.asarray([[25.0, 40.0, 30.0]], dtype=jnp.float32), verbosity=0)


@pytest.mark.parametrize("case", CASES)
def test_the_surface_pooled_sd_is_unchanged_by_the_routing(golden, surfaces, case):
    """Exact equality, not a tolerance: this is the same computation on the same
    inputs, and a tolerance would hide the drift the check exists for."""
    weights = jnp.asarray(golden[f"{case}/bin_weights"])[None, :, :]
    predictor = SurfacePredictor(surfaces, n_samples=20, artifact=CHECKPOINT.name)

    got = np.asarray(predictor.pooled_circular_sd(weights))[0]
    reference = golden[f"{case}/pooled_model_sd"]

    np.testing.assert_array_equal(np.isnan(got), np.isnan(reference))
    finite = np.isfinite(reference)
    np.testing.assert_array_equal(got[finite], reference[finite])


@pytest.mark.parametrize("case", CASES)
def test_empty_bins_stay_missing_rather_than_becoming_zero(golden, surfaces, case):
    """A zero would plot as a real measurement of no variability."""
    weights = jnp.asarray(golden[f"{case}/bin_weights"])[None, :, :]
    predictor = SurfacePredictor(surfaces, n_samples=20, artifact=CHECKPOINT.name)
    got = np.asarray(predictor.pooled_circular_sd(weights))[0]

    empty = np.asarray(golden[f"{case}/bin_weights"]).sum(axis=1) == 0
    assert np.all(np.isnan(got[empty]))
    assert not np.any(np.isnan(got[~empty]))


def test_pooling_changes_the_answer_on_a_continuous_design(golden):
    """If it did not, the pooling would be untested decoration and a single
    column would do."""
    pooled = golden["continuous/pooled_model_sd"]
    unpooled = golden["continuous/unpooled_model_sd"]
    both = np.isfinite(pooled) & np.isfinite(unpooled)
    assert np.max(np.abs(pooled[both] - unpooled[both])) > 0.1


def test_pooling_collapses_on_a_discrete_design(golden):
    """A bin whose trials all sit at one feature difference mixes exactly one
    column, so the pooled value must equal the unpooled one -- otherwise the
    model is inflated by columns no trial ever visited (the Moors case)."""
    pooled = golden["discrete_levels/pooled_model_sd"]
    unpooled = golden["discrete_levels/unpooled_model_sd"]
    both = np.isfinite(pooled) & np.isfinite(unpooled)
    assert both.any()
    np.testing.assert_allclose(pooled[both], unpooled[both], atol=1e-9)


def test_the_mixture_reaches_the_same_estimator_analytically(golden):
    """Mixing densities is linear, so the pooled first moment is the
    weight-average of the per-column moments -- each closed form for this family.
    The two routes must agree, or the families are being plotted through
    different estimators.
    """
    artifact = surrogate.WNM_DEFAULTS[20]
    if not artifact.exists():
        pytest.skip("no packaged WNM artifact")

    predictor = predictor_from_surrogate(surrogate.load_surrogate(checkpoint_path=artifact))
    feat_grid = np.asarray(config.create_grid('feat_diff'), dtype=np.float32)
    params = jnp.stack([
        jnp.full(len(feat_grid), 25.0), jnp.full(len(feat_grid), 40.0),
        jnp.full(len(feat_grid), 30.0), jnp.asarray(feat_grid)], axis=-1).astype(jnp.float32)

    weights = jnp.asarray(golden["continuous/bin_weights"])
    analytic = np.asarray(predictor.pooled_circular_sd(params, weights, validate=False))

    # The grid route, built from the mixture's own densities.
    from shared.mu1_axis import mu1_grid, periodic_integral

    densities = jnp.exp(predictor.grid_log_density(params, validate=False))
    mixtures = jnp.einsum('bf,fm->bm', weights, densities)
    angles = jnp.radians(mu1_grid())
    mass = periodic_integral(mixtures, axis=1)
    cosine = periodic_integral(mixtures * jnp.cos(angles)[None, :], axis=1)
    sine = periodic_integral(mixtures * jnp.sin(angles)[None, :], axis=1)
    resultant = np.asarray(jnp.sqrt(cosine ** 2 + sine ** 2) / jnp.where(mass > 0, mass, 1.0))
    grid_route = np.where(np.asarray(mass) > 0,
                          np.degrees(np.sqrt(-2.0 * np.log(np.clip(resultant, 1e-12, 1.0)))),
                          np.nan)

    both = np.isfinite(analytic) & np.isfinite(grid_route)
    assert both.any()
    np.testing.assert_allclose(analytic[both], grid_route[both], atol=1e-4)


def test_the_mixture_marks_empty_bins_missing_too():
    artifact = surrogate.WNM_DEFAULTS[20]
    if not artifact.exists():
        pytest.skip("no packaged WNM artifact")

    predictor = predictor_from_surrogate(surrogate.load_surrogate(checkpoint_path=artifact))
    feat_grid = np.asarray(config.create_grid('feat_diff'), dtype=np.float32)
    params = jnp.stack([
        jnp.full(len(feat_grid), 25.0), jnp.full(len(feat_grid), 40.0),
        jnp.full(len(feat_grid), 30.0), jnp.asarray(feat_grid)], axis=-1).astype(jnp.float32)

    weights = np.zeros((3, len(feat_grid)))
    weights[0, 10] = 1.0          # one populated bin
    # rows 1 and 2 stay empty
    got = np.asarray(predictor.pooled_circular_sd(params, jnp.asarray(weights),
                                                  validate=False))
    assert np.isfinite(got[0])
    assert np.all(np.isnan(got[1:]))


# ---------------------------------------------------------------------------
# The call site, not just the helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES)
def test_the_plotting_entry_point_still_produces_the_reference(golden, surfaces, case):
    """A test that pins the helper while the driver calls it differently proves
    nothing. This goes through the function the plots actually call."""
    import create_unified_subject_plots as plots

    weights = jnp.asarray(golden[f"{case}/bin_weights"])[None, :, :]
    got = np.asarray(plots.compute_predicted_sd_curves_batch_pooled(surfaces, weights))[0]
    reference = golden[f"{case}/pooled_model_sd"]

    np.testing.assert_array_equal(np.isnan(got), np.isnan(reference))
    finite = np.isfinite(reference)
    np.testing.assert_array_equal(got[finite], reference[finite])


# ---------------------------------------------------------------------------
# Where a clamp decides the answer
# ---------------------------------------------------------------------------
#
# The golden fixtures never approach the resultant clamp, so they cannot detect
# a change to it. Routing this estimator did change it once -- [1e-10, 1-1e-10]
# rewritten as a tidier [1e-12, 1] -- and on a surface whose first moment cancels
# in float32 that moved the answer by 37 degrees while all 16 golden tests
# stayed green. These cases go where the fixtures do not.


def _pre_routing_pooled_sd(log_surfaces_batch, bin_weights_batch):
    """The deployed implementation before the routing, transcribed verbatim."""
    from shared.mu1_axis import mu1_grid, periodic_integral

    grid = mu1_grid()
    probabilities = jnp.exp(log_surfaces_batch)
    weights = jnp.asarray(bin_weights_batch)
    mixtures = jnp.einsum('smf,sbf->smb', probabilities, weights)
    angles = jnp.radians(grid)
    mass = periodic_integral(mixtures, axis=1)
    mean_cos = periodic_integral(mixtures * jnp.cos(angles)[None, :, None], axis=1)
    mean_sin = periodic_integral(mixtures * jnp.sin(angles)[None, :, None], axis=1)
    r = jnp.sqrt(mean_cos ** 2 + mean_sin ** 2) / jnp.where(mass > 0, mass, jnp.nan)
    r_safe = jnp.minimum(jnp.maximum(r, 1e-10), 1.0 - 1e-10)
    return jnp.degrees(jnp.sqrt(-2 * jnp.log(r_safe)))


def _extreme_cases():
    from shared.mu1_axis import mu1_grid

    grid = np.asarray(mu1_grid())
    n = len(grid)

    cancelling = np.full((1, n, 1), 1e-30)
    for angle in (-180.0, -90.0, 0.0):
        cancelling[0, int(np.argmin(np.abs(grid - angle))), 0] = 1.0 / 3.0

    spike = np.full((1, n, 1), 1e-30)
    spike[0, n // 2, 0] = 1.0

    uniform = np.full((1, n, 1), 1.0 / n)

    rng = np.random.default_rng(0)
    random = np.abs(rng.normal(size=(2, n, 5)))
    random /= random.sum(axis=1, keepdims=True)
    random_weights = rng.random((2, 4, 5))
    random_weights /= random_weights.sum(axis=-1, keepdims=True)

    return {
        "cancelling first moment": (cancelling, np.ones((1, 1, 1))),
        "near-delta, resultant to 1": (spike, np.ones((1, 1, 1))),
        "uniform, resultant to 0": (uniform, np.ones((1, 1, 1))),
        "random": (random, random_weights),
    }


@pytest.mark.parametrize("case", list(_extreme_cases()))
def test_the_clamp_is_the_deployed_one(case):
    """Bit-identity where the clamp, not the data, decides the answer."""
    densities, weights = _extreme_cases()[case]
    log_surfaces = jnp.log(jnp.asarray(densities))
    weights = jnp.asarray(weights)

    predictor = SurfacePredictor(log_surfaces, n_samples=20, artifact="extremes")
    routed = np.asarray(predictor.pooled_circular_sd(weights))
    reference = np.asarray(_pre_routing_pooled_sd(log_surfaces, weights))

    np.testing.assert_array_equal(np.isnan(routed), np.isnan(reference))
    finite = np.isfinite(reference)
    np.testing.assert_array_equal(routed[finite], reference[finite])


def test_the_upper_clamp_is_inert_in_float32_and_that_is_the_deployed_behaviour():
    """A degenerate case the estimator has always had, pinned rather than fixed.

    The clamp reads ``min(max(r, 1e-10), 1 - 1e-10)``, but 1 - 1e-10 is exactly
    1.0 in float32 (eps is 1.19e-7), so the upper bound never binds: a
    distribution concentrated in one cell gives r == 1, log(1) == 0, and an SD of
    -0.0 degrees. That is what the deployed code returns, and this test pins it
    to the deployed value rather than to what the clamp appears to intend.

    Changing it would be a defensible fix -- an SD of zero for a distribution
    with one cell of support is arguably right, and arguably a floor is wanted --
    but it is a change to a plotted number, so it belongs in a decision, not in a
    routing commit. Recorded in OPEN_DECISIONS.md.
    """
    from shared.mu1_axis import mu1_grid

    n = len(np.asarray(mu1_grid()))
    spike = np.full((1, n, 1), 0.0)
    spike[0, n // 2, 0] = 1.0
    log_surfaces = jnp.log(jnp.asarray(spike) + 1e-300)
    weights = jnp.ones((1, 1, 1))

    routed = float(np.asarray(
        SurfacePredictor(log_surfaces, n_samples=20, artifact="delta")
        .pooled_circular_sd(weights))[0, 0])
    reference = float(np.asarray(_pre_routing_pooled_sd(log_surfaces, weights))[0, 0])

    assert routed == reference
    assert abs(routed) == 0.0, (
        "the single-cell case no longer returns zero; the clamp's behaviour changed")
