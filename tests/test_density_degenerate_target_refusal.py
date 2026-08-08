"""A constant density target makes the density objective refuse to run.

Runs the real surrogate on a tiny grid (CPU). The decision under test is that a
degenerate condition raises rather than being dropped from the fit: a condition
that silently vanishes from a group leaves nothing downstream can act on, and the
group's shared parameters would then be fitted to a different set of conditions
than the run reports. The refusal is scoped to the density objectives -- a flat
density-asymmetry target says nothing about whether the same condition can be fit
by likelihood or CRPS, and blocking those would be a much worse failure than the
one being prevented.
"""
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

from grid_based_multi_condition_optimizer_jax_loops import (
    DEGENERATE_TARGET_EPS,
    GridBasedMultiConditionOptimizer,
)

# One coarse stage: the question is whether the run is refused, not where the
# optimum lands.
TINY_FIT = dict(shared_grid_size=4, feat_grid_size=4, stop_after_first_stage=True, verbosity=0)


def informative_condition(seed):
    rng = np.random.default_rng(seed)
    fd = rng.uniform(2, 180, 300)
    return jnp.asarray(np.column_stack([fd, 12 * np.sin(np.pi * fd / 180) + rng.normal(0, 15, 300)]))


def constant_target_condition(seed):
    """Every bias exactly 0, so the signed density asymmetry is identically 0."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(np.column_stack([rng.uniform(2, 180, 300), np.zeros(300)]))


@pytest.fixture(scope="module")
def mixed_optimizer():
    return GridBasedMultiConditionOptimizer(
        str(CHECKPOINT),
        {"good": informative_condition(0), "flat": constant_target_condition(1)},
        skip_motor_noise=True,
    )


def test_only_the_constant_target_condition_is_flagged(mixed_optimizer):
    assert list(mixed_optimizer.density_degenerate_conditions) == [False, True]
    assert mixed_optimizer.density_target_var[0] > 1e-6
    assert mixed_optimizer.density_target_var[1] < DEGENERATE_TARGET_EPS


@pytest.mark.parametrize("method", ["density", "density_legacy"])
def test_density_refuses_and_names_the_offending_condition(mixed_optimizer, method):
    with pytest.raises(ValueError) as excinfo:
        mixed_optimizer.fit_hierarchical_grid(fitting_method=method, **TINY_FIT)
    message = str(excinfo.value)
    assert "flat" in message and "constant" in message
    assert "good" not in message.split("conditions:")[-1], "must name only the offenders"
    assert method in message


def test_a_single_degenerate_condition_is_enough_to_refuse(mixed_optimizer):
    """Not "all conditions degenerate" -- one is a hard stop, because the group's
    shared parameters would otherwise be fitted to a silently reduced set."""
    assert int(np.sum(mixed_optimizer.density_degenerate_conditions)) == 1
    with pytest.raises(ValueError):
        mixed_optimizer.fit_hierarchical_grid(fitting_method="density", **TINY_FIT)


def test_non_density_objectives_are_unaffected(mixed_optimizer):
    """The refusal must not leak: a flat density target is no reason to block a
    likelihood fit of the same conditions."""
    result = mixed_optimizer.fit_hierarchical_grid(fitting_method="likelihood", **TINY_FIT)
    for condition in ("good", "flat"):
        entry = result["condition_results"][condition]
        assert np.isfinite(entry["sd_feat1"]) and np.isfinite(entry["loss"])


def test_a_clean_group_still_fits(mixed_optimizer):
    """The guard is inert when no target is degenerate -- which is every real
    dataset today (minimum measured target variance 2.2e-04)."""
    mixed_optimizer.update_dataset({"good": informative_condition(0),
                                    "also_good": informative_condition(5)})
    assert not mixed_optimizer.density_degenerate_conditions.any()
    result = mixed_optimizer.fit_hierarchical_grid(fitting_method="density", **TINY_FIT)
    assert np.isfinite(result["condition_results"]["good"]["loss"])
    assert np.isfinite(float(result["best_loss"]))
