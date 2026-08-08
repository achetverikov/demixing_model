"""A cached curve must be the curve the surrogate would have produced.

This is the assumption the exhaustive backend rests on and the one nothing else
can check: the cache is read *instead of* calling the model, so if the builder
writes the wrong curve for a lattice point -- a transposed parameter order, an
off-by-one in the slab index, the wrong density-target settings -- every fit is
wrong and every checksum still passes, because the bytes are exactly what was
written.

So the oracle here is the live model. The cache is built through the production
builder and then compared, point by point, against `predict_nn` +
`generate_nn_density_asymmetry_batch` called directly on the same triples. The
comparison is deliberately made at named, distinct parameter values rather than
in bulk, so a mapping error shows up as a mismatch rather than as a permutation
that happens to have the same summary statistics.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

CHECKPOINT = ROOT / "pretrained" / "model_epoch1425_10ktrain_20samples.pkl"
pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason="no pretrained checkpoint")

import jax.numpy as jnp

import curve_cache as cc
import density_objective as do
from grid_based_multi_condition_optimizer_jax_loops import (
    GridBasedMultiConditionOptimizer,
    _compute_curve_losses,
    generate_nn_density_asymmetry_batch,
    predict_nn,
)

LOW, HIGH, STEP = 20.0, 40.0, 10.0


@pytest.fixture(scope="module")
def built_cache(tmp_path_factory):
    """Build through the production CLI, not a test-local shortcut: the builder
    is what has to be right."""
    out_root = tmp_path_factory.mktemp("caches")
    subprocess.run(
        [sys.executable, str(ROOT / "model_fit_to_data" / "build_curve_cache.py"),
         "--checkpoint-path", str(CHECKPOINT), "--out-root", str(out_root),
         "--step", str(STEP), "--low", str(LOW), "--high", str(HIGH), "--verify"],
        cwd=ROOT, check=True, capture_output=True,
        env={**dict(__import__("os").environ), "JAX_PLATFORMS": "cpu"},
    )
    cache_dir = next(Path(out_root).glob("curve_cache_*"))
    return cc.CachedCurveSource(cache_dir)


@pytest.fixture(scope="module")
def live_optimizer():
    return GridBasedMultiConditionOptimizer(str(CHECKPOINT), None, skip_motor_noise=True)


def live_curves(optimizer, triples):
    """(sd_feat1, sd_feat2, sd_spat) -> density-asymmetry curves, straight from the model."""
    params = jnp.asarray(np.asarray(triples, dtype=float))
    surfaces = predict_nn(optimizer, params)
    return np.asarray(generate_nn_density_asymmetry_batch(
        surfaces, weights_sd=optimizer.emp_density_weights_sd,
        smoothing_sigma=optimizer.density_smoothing_sigma))


def test_every_cached_curve_matches_the_live_model(built_cache, live_optimizer):
    """Exhaustive over this small lattice: every (sd_spat, sd_feat1, sd_feat2)."""
    for spat_index, sd_spat in enumerate(built_cache.sd_spat_values):
        centered, means, _ = built_cache.slab(spat_index)
        triples = [(f1, f2, sd_spat) for f1, f2 in built_cache.feat_pairs]
        expected = live_curves(live_optimizer, triples)
        # The cache stores centered curves plus their means; the sum is the curve.
        reconstructed = np.asarray(centered) + np.asarray(means)[:, None]
        np.testing.assert_allclose(reconstructed, expected, rtol=1e-4, atol=1e-6)


def test_the_parameter_mapping_is_not_transposed(built_cache, live_optimizer):
    """sd_feat1 and sd_feat2 are not interchangeable, and a transposed build would
    still produce a valid-looking cache. Checked at an asymmetric pair, where the
    two orderings give genuinely different curves."""
    pairs = built_cache.feat_pairs
    asymmetric = [i for i, (f1, f2) in enumerate(pairs) if f1 != f2]
    assert asymmetric, "fixture needs at least one sd_feat1 != sd_feat2 pair"
    index = asymmetric[0]
    f1, f2 = pairs[index]
    sd_spat = float(built_cache.sd_spat_values[0])

    centered, means, _ = built_cache.slab(0)
    cached = np.asarray(centered[index]) + float(means[index])
    correct, swapped = live_curves(live_optimizer, [(f1, f2, sd_spat), (f2, f1, sd_spat)])

    assert not np.allclose(correct, swapped, rtol=1e-3), (
        "fixture no longer distinguishes the two orderings")
    np.testing.assert_allclose(cached, correct, rtol=1e-4, atol=1e-6)


def test_the_slab_index_selects_the_right_sd_spat(built_cache, live_optimizer):
    """An off-by-one in the slab axis would fit every condition at the wrong
    shared parameter while looking entirely healthy."""
    for spat_index in range(len(built_cache.sd_spat_values)):
        sd_spat = float(built_cache.sd_spat_values[spat_index])
        centered, means, _ = built_cache.slab(spat_index)
        cached = np.asarray(centered[0]) + float(means[0])
        f1, f2 = built_cache.feat_pairs[0]
        expected = live_curves(live_optimizer, [(f1, f2, sd_spat)])[0]
        np.testing.assert_allclose(cached, expected, rtol=1e-4, atol=1e-6)


def test_cached_statistics_describe_the_cached_curve(built_cache):
    """The dot product and the statistics must describe the same curve, or the
    score is a mixture of two different ones."""
    for spat_index in range(len(built_cache.sd_spat_values)):
        centered, means, variances = built_cache.slab(spat_index)
        centered = np.asarray(centered)
        np.testing.assert_allclose(centered.mean(axis=1), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(variances), centered.var(axis=1),
                                   rtol=1e-4, atol=1e-9)


def test_scoring_from_the_cache_matches_scoring_the_live_curve(built_cache, live_optimizer):
    """End to end: the batched cache score equals the hierarchical path's own
    scorer applied to the live curve. This is the Phase 3 acceptance check."""
    rng = np.random.default_rng(0)
    grid = np.linspace(2, 180, built_cache.n_points)
    target = 0.05 * np.sin(np.pi * grid / 180.0) + 0.004 * rng.normal(size=built_cache.n_points)

    centered_target, target_mean, target_var = do.target_statistics(target)
    for spat_index, sd_spat in enumerate(built_cache.sd_spat_values):
        centered, means, variances = built_cache.slab(spat_index)
        from_cache = np.asarray(do.ccc_loss_batched(
            jnp.asarray(centered) @ centered_target, means, variances,
            target_mean, target_var, built_cache.n_points))

        triples = [(f1, f2, sd_spat) for f1, f2 in built_cache.feat_pairs]
        live = live_curves(live_optimizer, triples)
        targets = np.repeat(target[None, :], len(live), axis=0)
        from_model = np.asarray(_compute_curve_losses(
            jnp.asarray(live), jnp.asarray(targets), loss_type="ccc", is_angular=False))

        np.testing.assert_allclose(from_cache, from_model, rtol=1e-4, atol=1e-6)
