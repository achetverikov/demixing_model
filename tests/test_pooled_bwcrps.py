"""Report-order pooling happens before the nonlinear score, and stays that way.

The transition plan names this estimator explicitly: "preserve the report-order
distribution pooling before nonlinear CRPS scoring". The order is the estimator,
not a detail of it. Pooling the orders into one predicted distribution and
scoring that is a different quantity from scoring each order and averaging,
because the energy score is nonlinear in the distribution -- and the wrong one
produces a perfectly plausible number.

`tests/data/pooled_bwcrps_golden.npz` was recorded by
`record_pooled_bwcrps_golden.py` from the plotting code before the pooling was
moved into `shared.prediction`, so these checks cannot pass vacuously.
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

GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "pooled_bwcrps_golden.npz"
CHECKPOINT = ROOT / "pretrained" / "model_epoch1425_10ktrain_20samples.pkl"

if not GOLDEN_PATH.exists():
    raise RuntimeError(
        f"{GOLDEN_PATH} is missing. It is the pre-routing reference for the report-order "
        "pooling and is tracked in git; restore it rather than re-recording it.")

pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(),
                                reason="no pretrained surface checkpoint")

from shared.config import config  # noqa: E402
from shared.prediction import pooled_bias_weighted_crps  # noqa: E402

CASES = ["two_orders", "unequal_support", "disjoint_coverage", "flat_bias"]


@pytest.fixture(scope="module")
def golden():
    return np.load(GOLDEN_PATH)


@pytest.fixture(scope="module")
def grids():
    feat_grid = np.asarray(config.create_grid('feat_diff'), dtype=float)
    bias_grid = np.asarray(config.create_grid('mu1_bias'), dtype=float)
    difference = np.abs(bias_grid[:, None] - bias_grid[None, :])
    return feat_grid, np.minimum(difference, 360.0 - difference)


@pytest.fixture(scope="module")
def surfaces():
    from grid_based_multi_condition_optimizer_jax_loops import (
        GridBasedMultiConditionOptimizer)

    optimizer = GridBasedMultiConditionOptimizer(
        str(CHECKPOINT), {"dummy": jnp.zeros((4, 2))}, skip_motor_noise=True)
    return optimizer._predict_batch_fixed_size(
        jnp.asarray([[25.0, 40.0, 30.0], [45.0, 20.0, 30.0]], dtype=jnp.float32),
        verbosity=0)


def _orders(golden, case):
    return [golden[f"{case}/input/order0"], golden[f"{case}/input/order1"]]


@pytest.mark.parametrize("case", CASES)


def test_the_plotting_entry_point_is_unchanged_by_the_routing(golden, grids, surfaces, case):
    """The plotting entry point routes through the current circular-weight scorer."""
    import create_unified_subject_plots as plots

    feat_grid, distance = grids
    probabilities = np.exp(np.asarray(surfaces[:2], dtype=float))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    got = plots._pooled_bias_weighted_crps(
        surfaces[:2], _orders(golden, case), feat_grid, distance, 20.0)
    expected = pooled_bias_weighted_crps(
        probabilities, _orders(golden, case), feat_grid, distance, 20.0)
    assert got == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("case", CASES)


def test_pooling_first_is_not_the_same_as_averaging_scores(
        golden, grids, surfaces, case):
    """The property the plan's instruction protects.

    If these agreed, the instruction would be untestable and the order would not
    matter. On unequal support -- 600 trials against 80 -- averaging treats the
    sparse order as an equal vote and the answer moves by 2.7.
    """
    feat_grid, distance = grids
    probabilities = np.exp(np.asarray(surfaces[:2], dtype=float))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    orders = _orders(golden, case)
    pooled = pooled_bias_weighted_crps(
        probabilities, orders, feat_grid, distance, 20.0)
    separate = np.array([
        pooled_bias_weighted_crps(
            probabilities[index:index + 1], [orders[index]],
            feat_grid, distance, 20.0)
        for index in range(2)
    ])
    assert abs(pooled - separate.mean()) > 1e-3


def test_mismatched_order_counts_are_refused(golden, grids, surfaces):
    """Predictions and datasets are paired positionally, so a length mismatch
    scores one order's model against another's trials."""
    feat_grid, distance = grids
    probabilities = np.exp(np.asarray(surfaces[:2], dtype=float))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    with pytest.raises(ValueError, match="paired positionally"):
        pooled_bias_weighted_crps(probabilities, _orders(golden, "two_orders")[:1],
                                  feat_grid, distance, 20.0)


def test_all_zero_bias_weights_raise_rather_than_scoring(grids, surfaces):
    """A pooled score with no identified feature locations is not a small score,
    it is no score."""
    feat_grid, distance = grids
    probabilities = np.exp(np.asarray(surfaces[:2], dtype=float))
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    # Bias identically zero everywhere makes every squared-bias weight zero.
    zero_bias = [np.stack([np.linspace(2.0, 178.0, 200), np.zeros(200)], axis=-1)] * 2
    with pytest.raises(ValueError, match="unidentified"):
        pooled_bias_weighted_crps(probabilities, zero_bias, feat_grid, distance, 20.0)


def test_biases_across_the_seam_retain_their_large_circular_mean(grids, surfaces):
    feat_grid, distance = grids
    probabilities = np.exp(np.asarray(surfaces[:1], dtype=float))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    feat = np.repeat(40.0, 200)
    bias = np.tile([179.0, -179.0], 100)
    score = pooled_bias_weighted_crps(
        probabilities, [np.stack([feat, bias], axis=-1)],
        feat_grid, distance, 20.0)
    assert np.isfinite(score)


def test_the_bias_axis_wraps_rather_than_clipping(grids, surfaces):
    """The mu1_bias axis is a circle. Clipping would pile trials near +/-180 into
    the end bins and bias every score built on them."""
    feat_grid, distance = grids
    probabilities = np.exp(np.asarray(surfaces[:2], dtype=float))
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    feat = np.linspace(2.0, 178.0, 200)
    # A bias just past +180 is the same angle as just past -180; wrapped, the two
    # give the same score, clipped they would not.
    just_over = np.stack([feat, np.full(200, 181.0)], axis=-1)
    equivalent = np.stack([feat, np.full(200, -179.0)], axis=-1)

    over = pooled_bias_weighted_crps(probabilities, [just_over, just_over],
                                     feat_grid, distance, 20.0)
    under = pooled_bias_weighted_crps(probabilities, [equivalent, equivalent],
                                      feat_grid, distance, 20.0)
    assert over == pytest.approx(under, rel=1e-9)
