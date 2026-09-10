"""Extracting target construction must not have moved a single target value.

``tests/data/fitting_targets_golden.npz`` was recorded by
``tests/record_fitting_targets_golden.py`` from the surface optimizer *before*
``model_fit_to_data/fitting_targets.py`` existed. Both the extracted builder and
the optimizer that now calls it are checked against that reference, so this
catches an extraction that changed a definition and an optimizer that stopped
routing through the extraction.

Equality is exact, not approximate. These are the same computations on the same
inputs; a tolerance here would hide exactly the kind of drift the check exists
for -- a reordered reduction, a bandwidth resolved from a different pool, a
padded row leaking into an unpadded core.

The fixtures deliberately include conditions of unequal length (padding), sparse
data (where pooled and per-condition bandwidths diverge), tightly concentrated
errors (where the bandwidth floor is reachable), a sign reversal along the
feature axis (so a transposed axis shows as a sign error), and both the 180- and
360-degree data periods.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "fitting_targets_golden.npz"
if not GOLDEN_PATH.exists():
    # Version-controlled, so absence is not "not recorded yet" -- it means the
    # reference was deleted or the checkout is incomplete. Skipping here would
    # turn the one check that protects the target definitions into silence, and
    # a skipped test reads as coverage in a summary line.
    raise RuntimeError(
        f"{GOLDEN_PATH} is missing. It is the pre-extraction reference for every "
        "fitting target and is tracked in git; restore it rather than re-recording "
        "it, which against extracted code would make this check vacuous.")

import jax.numpy as jnp  # noqa: E402

import fit_model_to_data as F  # noqa: E402
from fitting_targets import (build_fitting_targets, resolve_density_bandwidths)  # noqa: E402
from density_objective import degenerate_targets  # noqa: E402
from grid_based_multi_condition_optimizer_jax_loops import (  # noqa: E402
    compute_bwcrps_condition_targets, compute_target_bias_curve_core)
from shared.config import config  # noqa: E402

#: Recorded array -> attribute on the extracted result.
FIELDS = {
    "unified_feat_indices": "feat_indices",
    "unified_target_bias": "target_bias",
    "unified_bias_weights": "bias_weights",
    "unified_target_density": "target_density",
    "unified_target_bias_curve": "target_bias_curve",
    "unified_target_d": "target_d",
    "unified_fd_weights": "fd_weights",
    "unified_bias_fd_weights": "bias_fd_weights",
    "density_target_var": "density_target_var",
}


@pytest.fixture(scope="module")
def golden():
    return np.load(GOLDEN_PATH)


def _cases(golden):
    return sorted({key.split("/")[0] for key in golden.files})


def _inputs(golden, case):
    prefix = f"{case}/input/"
    # np.savez preserves insertion order, which is the condition order the
    # targets' first axis follows; rebuilding in a different order would compare
    # the right arrays against the wrong conditions.
    return {key[len(prefix):]: golden[key]
            for key in golden.files if key.startswith(prefix)}


def _grids():
    feat_diff_grid = config.create_grid('feat_diff')
    mu1_bias_grid = config.create_grid('mu1_bias')
    diff = jnp.abs(mu1_bias_grid[:, None] - mu1_bias_grid[None, :])
    return feat_diff_grid, jnp.minimum(diff, 360.0 - diff), len(mu1_bias_grid)


def _build(datasets):
    feat_diff_grid, d_circ, n_mu1_bias = _grids()
    return build_fitting_targets(
        {name: jnp.asarray(values) for name, values in datasets.items()},
        feat_diff_grid=feat_diff_grid, d_circ_matrix=d_circ, n_mu1_bias=n_mu1_bias,
        emp_density_weights_sd=F.DENSITY_CURVE_SPEC["emp_density_weights_sd"],
        density_bandwidth_rule=F.DENSITY_CURVE_SPEC["density_bandwidth_rule"],
        density_bandwidth_mode=F.DENSITY_CURVE_SPEC["density_bandwidth_mode"],
        degenerate_targets=degenerate_targets,
        bwcrps_condition_targets=compute_bwcrps_condition_targets,
        target_bias_curve_core=compute_target_bias_curve_core)


def test_the_reference_covers_every_objective(golden):
    """A target that is not recorded is not protected by any of this."""
    cases = _cases(golden)
    assert len(cases) == 8, cases
    assert any("180" in case for case in cases) and any("360" in case for case in cases)
    for case in cases:
        for recorded in FIELDS:
            assert f"{case}/{recorded}" in golden.files, f"{case} is missing {recorded}"


@pytest.mark.parametrize("case", [
    "unequal_180", "unequal_360", "sparse_180", "sparse_360",
    "narrow_180", "narrow_360", "reversal_180", "reversal_360"])
def test_extracted_builder_reproduces_the_pre_extraction_targets(golden, case):
    if f"{case}/unified_target_density" not in golden.files:
        pytest.skip(f"{case} not in the recorded reference")

    targets = _build(_inputs(golden, case))
    for recorded, attribute in FIELDS.items():
        np.testing.assert_array_equal(
            np.asarray(getattr(targets, attribute)), golden[f"{case}/{recorded}"],
            err_msg=f"{case}: {attribute} differs from the pre-extraction reference")


def test_condition_order_is_the_order_given(golden):
    """Targets are positional; a reordered result mislabels every condition."""
    datasets = _inputs(golden, "unequal_180")
    targets = _build(datasets)
    assert targets.condition_names == tuple(datasets)

    reversed_datasets = {k: datasets[k] for k in reversed(list(datasets))}
    reversed_targets = _build(reversed_datasets)
    assert reversed_targets.condition_names == tuple(reversed_datasets)
    np.testing.assert_array_equal(
        np.asarray(reversed_targets.target_density)[::-1],
        np.asarray(targets.target_density))


def test_no_surrogate_is_needed_to_build_targets(golden):
    """Targets come from data alone.

    If building one required a checkpoint, a second search engine could only
    reach the targets through the backend it exists to replace.
    """
    before = set(sys.modules)
    _build(_inputs(golden, "sparse_180"))
    leaked = [name for name in set(sys.modules) - before
              if "mirror_aware" in name or "wrapped_mixture" in name]
    assert not leaked, f"target construction pulled in a surrogate: {leaked}"


# ---------------------------------------------------------------------------
# Bandwidth resolution
# ---------------------------------------------------------------------------

def test_pooled_and_average_share_one_bandwidth_across_conditions():
    """The reason production is pooled: conditions differ in error spread by
    construction, so a per-condition bandwidth varies the target's smoothing
    along the very axis the experiment manipulates."""
    rng = np.random.default_rng(1)
    tight = jnp.asarray(rng.normal(0, 2.0, 400))
    broad = jnp.asarray(rng.normal(0, 30.0, 400))

    for mode in ("pooled", "average"):
        bandwidths = resolve_density_bandwidths([tight, broad], "sj", mode)
        assert len(set(float(b) for b in bandwidths)) == 1, mode

    per_condition = resolve_density_bandwidths([tight, broad], "sj", "per_condition")
    assert float(per_condition[0]) != float(per_condition[1])


def test_unknown_bandwidth_settings_raise():
    values = [jnp.asarray(np.random.default_rng(2).normal(0, 5.0, 100))]
    with pytest.raises(ValueError, match="density_bandwidth_rule"):
        resolve_density_bandwidths(values, "scott", "pooled")
    with pytest.raises(ValueError, match="density_bandwidth_mode"):
        resolve_density_bandwidths(values, "sj", "clever")


def test_the_resolved_bandwidth_is_reported(golden):
    """Recorded rather than left implicit: it is part of what a fit means."""
    targets = _build(_inputs(golden, "sparse_180"))
    assert len(targets.density_bandwidth) == len(targets.condition_names)
    assert all(np.isfinite(b) and b > 0 for b in targets.density_bandwidth)


def test_empty_input_raises_rather_than_returning_empty_targets(golden):
    """An empty frame must not be indistinguishable from 'input absent'."""
    with pytest.raises(ValueError, match="no conditions"):
        _build({})


def test_matched_operator_preserves_surface_clamp_for_dummy_feature_rows():
    """WNM-only targets must not reject legacy surface initialization fixtures."""
    targets = _build({"dummy": np.zeros((4, 2), dtype=np.float32)})
    operator = np.asarray(targets.feature_operator[0])
    np.testing.assert_allclose(operator[:, 0], 1.0, atol=1e-7)
    np.testing.assert_allclose(operator[:, 1:], 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# The call site, not just the helper
# ---------------------------------------------------------------------------

CHECKPOINT = ROOT / "pretrained" / "model_epoch1425_10ktrain_20samples.pkl"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="no pretrained surface checkpoint")
@pytest.mark.parametrize("case", ["unequal_180", "sparse_360", "narrow_180", "reversal_360"])
def test_the_optimizer_still_produces_the_pre_extraction_targets(golden, case):
    """A test that pins the helper while the driver calls it differently proves
    nothing. This goes through the optimizer that production actually runs and
    reads the attributes the JIT objectives actually consume.
    """
    from grid_based_multi_condition_optimizer_jax_loops import (
        GridBasedMultiConditionOptimizer)

    optimizer = GridBasedMultiConditionOptimizer(
        checkpoint_path=str(CHECKPOINT),
        condition_datasets={name: jnp.asarray(values)
                            for name, values in _inputs(golden, case).items()},
        emp_density_weights_sd=F.DENSITY_CURVE_SPEC["emp_density_weights_sd"],
        density_smoothing_sigma=F.DENSITY_CURVE_SPEC["density_smoothing_sigma"],
        density_bandwidth_rule=F.DENSITY_CURVE_SPEC["density_bandwidth_rule"],
        density_bandwidth_mode=F.DENSITY_CURVE_SPEC["density_bandwidth_mode"],
    )

    for recorded in FIELDS:
        np.testing.assert_array_equal(
            np.asarray(getattr(optimizer, recorded)), golden[f"{case}/{recorded}"],
            err_msg=f"{case}: optimizer.{recorded} differs from the pre-extraction reference")
