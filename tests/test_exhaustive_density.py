"""The exhaustive backend: exactness of the factorisation, and its contract.

The claim the whole backend rests on is that at a fixed shared parameter, taking
each condition's own minimum and summing is the *exact* joint optimum -- not an
approximation. That is checked here against brute force over the full joint
product space, which is the only oracle that can settle it, rather than by
re-deriving the argument in a comment.

The second thing checked is the return contract. The backend's output is consumed
by `fit_model_to_data.process_subject`, which reads the same keys from whichever
backend ran; a shape difference there would surface as a KeyError deep inside a
refit, or worse, as a silently missing field.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import density_objective as do
from exhaustive_density import (
    InMemoryCurveSource,
    SUPPORTED_OBJECTIVES,
    fit_exhaustive_density,
)

N_POINTS = 30


def make_source(rng, n_spat=5, n_feat=3, n_points=N_POINTS):
    """A small lattice of smooth, distinguishable curves."""
    grid = np.linspace(2, 180, n_points)
    sd_spat_values = np.linspace(10.0, 90.0, n_spat)
    feat_values = np.linspace(20.0, 60.0, n_feat)
    # Enumeration order the tie policy names: sd_feat1 outer, sd_feat2 inner.
    feat_pairs = np.array([(f1, f2) for f1 in feat_values for f2 in feat_values])

    curves = np.empty((n_spat, len(feat_pairs), n_points))
    for i, spat in enumerate(sd_spat_values):
        for j, (f1, f2) in enumerate(feat_pairs):
            amplitude = 0.05 * (1 + 0.01 * spat)
            phase = 0.01 * f1 + 0.005 * f2
            curves[i, j] = (amplitude * np.sin(np.pi * grid / 180.0 + phase)
                            + 0.002 * rng.normal(size=n_points))
    return InMemoryCurveSource(sd_spat_values, feat_pairs, curves), curves


def brute_force_joint_optimum(curves, targets):
    """The joint optimum with no factorisation: every combination of a shared
    sd_spat with an INDEPENDENT feature pair per condition."""
    import itertools
    n_spat, n_pairs, _ = curves.shape
    n_conditions = len(targets)
    best = (np.inf, None, None)
    for spat_index in range(n_spat):
        for assignment in itertools.product(range(n_pairs), repeat=n_conditions):
            total = sum(
                float(do.ccc_loss(curves[spat_index, assignment[c]], targets[c]))
                for c in range(n_conditions)
            )
            if total < best[0] - 1e-12:
                best = (total, spat_index, assignment)
    return best


def test_factorisation_is_exact_not_approximate():
    """Per-condition minima summed == full joint search over the product space."""
    rng = np.random.default_rng(0)
    source, curves = make_source(rng)
    targets = np.stack([curves[2, 4] * 1.1, curves[1, 0] * 0.9])

    result = fit_exhaustive_density(source, targets, ["c1", "c2"], verbosity=0)
    joint_total, joint_spat, joint_assignment = brute_force_joint_optimum(curves, targets)

    assert result["best_loss"] == pytest.approx(joint_total, rel=1e-6, abs=1e-7)
    assert result["shared_params"]["sd_spat"] == pytest.approx(
        float(source.sd_spat_values[joint_spat]))
    for c, name in enumerate(["c1", "c2"]):
        expected = source.feat_pairs[joint_assignment[c]]
        assert result["condition_results"][name]["sd_feat1"] == pytest.approx(expected[0])
        assert result["condition_results"][name]["sd_feat2"] == pytest.approx(expected[1])


def test_factorisation_holds_with_more_conditions():
    """Four conditions: the product space is n_pairs**4, where getting the
    factorisation wrong would be most visible."""
    rng = np.random.default_rng(3)
    source, curves = make_source(rng, n_spat=3, n_feat=2)
    targets = np.stack([curves[1, 0] * 1.2, curves[0, 3] * 0.8,
                        curves[2, 1] * 1.05, curves[1, 2] * 0.95])

    result = fit_exhaustive_density(source, targets, list("abcd"), verbosity=0)
    joint_total, _, _ = brute_force_joint_optimum(curves, targets)
    assert result["best_loss"] == pytest.approx(joint_total, rel=1e-6, abs=1e-7)


def test_a_perfect_match_is_found_exactly():
    """Both targets taken from the SAME slab, so one shared sd_spat can satisfy
    both -- otherwise the shared parameter is a compromise and zero loss is not
    reachable, which would be a property of the fixture rather than the search."""
    rng = np.random.default_rng(5)
    source, curves = make_source(rng)
    targets = np.stack([curves[3, 7], curves[3, 2]])

    result = fit_exhaustive_density(source, targets, ["c1", "c2"], verbosity=0)
    assert result["best_loss"] == pytest.approx(0.0, abs=1e-6)
    assert result["shared_params"]["sd_spat"] == pytest.approx(float(source.sd_spat_values[3]))
    assert result["condition_results"]["c1"]["sd_feat1"] == pytest.approx(source.feat_pairs[7][0])
    assert result["condition_results"]["c2"]["sd_feat2"] == pytest.approx(source.feat_pairs[2][1])


# --- the tie policy ----------------------------------------------------------

def test_ties_resolve_to_the_smallest_sd_spat_then_the_earliest_pair():
    """Exhaustive search over a lattice ties far more often than a zoom does, so
    the rule is pinned rather than left to argmin's incidental behaviour."""
    grid = np.linspace(2, 180, N_POINTS)
    curve = 0.05 * np.sin(np.pi * grid / 180.0)
    n_spat, n_pairs = 4, 6
    # Every candidate identical: every total ties, so only the policy decides.
    curves = np.broadcast_to(curve, (n_spat, n_pairs, N_POINTS)).copy()
    sd_spat_values = np.array([10.0, 20.0, 30.0, 40.0])
    feat_pairs = np.array([(f1, f2) for f1 in (5.0, 15.0) for f2 in (7.0, 17.0, 27.0)])
    source = InMemoryCurveSource(sd_spat_values, feat_pairs, curves)

    result = fit_exhaustive_density(source, curve[None, :], ["only"], verbosity=0)
    assert result["shared_params"]["sd_spat"] == 10.0
    assert result["condition_results"]["only"]["sd_feat1"] == 5.0
    assert result["condition_results"]["only"]["sd_feat2"] == 7.0


def test_a_later_sd_spat_must_strictly_beat_the_incumbent():
    rng = np.random.default_rng(7)
    source, curves = make_source(rng, n_spat=3, n_feat=2)
    # Slabs 0 and 2 identical, and the target exactly matches one of their
    # curves, so both reach loss 0 and the totals tie exactly. (Scaling the
    # target instead would not tie: these curves differ across sd_spat in
    # amplitude, and 1 - CCC is not invariant to scaling one side alone.)
    curves[2] = curves[0]
    source = InMemoryCurveSource(source.sd_spat_values, source.feat_pairs, curves)
    targets = curves[0, 1][None, :]

    result = fit_exhaustive_density(source, targets, ["c"], verbosity=0)
    assert result["best_loss"] == pytest.approx(0.0, abs=1e-6)
    assert result["shared_params"]["sd_spat"] == pytest.approx(float(source.sd_spat_values[0]))


# --- the return contract -----------------------------------------------------

def test_return_shape_matches_what_process_subject_reads():
    """These are exactly the accesses `fit_model_to_data.process_subject` makes on
    a `fit_hierarchical_grid` result."""
    rng = np.random.default_rng(11)
    source, curves = make_source(rng)
    targets = np.stack([curves[1, 1] * 1.1, curves[2, 5] * 0.8])

    result = fit_exhaustive_density(source, targets, ["c1", "c2"], verbosity=0)

    shared = result["shared_params"]
    assert set(shared) >= {"sd_spat", "sd_motor"}
    assert isinstance(result["best_loss"], float)
    assert isinstance(result.get("stage_times", []), list)
    for name in ("c1", "c2"):
        entry = result["condition_results"][name]
        assert set(entry) >= {"condition_name", "sd_feat1", "sd_feat2", "loss"}
        assert all(np.isfinite([entry["sd_feat1"], entry["sd_feat2"], entry["loss"]]))


def test_reported_losses_are_the_scored_losses():
    """A per-condition loss must be the objective evaluated at the parameters
    reported for that condition -- not, say, at another condition's."""
    rng = np.random.default_rng(13)
    source, curves = make_source(rng)
    targets = np.stack([curves[1, 1] * 1.1, curves[2, 5] * 0.8])

    result = fit_exhaustive_density(source, targets, ["c1", "c2"], verbosity=0)
    spat_index = int(np.argmin(np.abs(source.sd_spat_values
                                      - result["shared_params"]["sd_spat"])))
    for c, name in enumerate(["c1", "c2"]):
        entry = result["condition_results"][name]
        pair_index = int(np.argmin(np.abs(source.feat_pairs
                                          - np.array([entry["sd_feat1"], entry["sd_feat2"]])
                                          ).sum(axis=1)))
        recomputed = float(do.ccc_loss(curves[spat_index, pair_index], targets[c]))
        assert entry["loss"] == pytest.approx(recomputed, rel=1e-5, abs=1e-6)
    assert result["best_loss"] == pytest.approx(
        sum(result["condition_results"][n]["loss"] for n in ("c1", "c2")), rel=1e-6)


# --- refusals ----------------------------------------------------------------

def test_density_legacy_is_refused():
    rng = np.random.default_rng(17)
    source, curves = make_source(rng)
    with pytest.raises(ValueError, match="density_legacy|supports"):
        fit_exhaustive_density(source, curves[0, :1], ["c"], objective="density_legacy",
                               verbosity=0)
    assert "density_legacy" not in SUPPORTED_OBJECTIVES


def test_grid_mismatch_is_refused_not_broadcast():
    """A target on a different feat_diff grid must not be silently compared."""
    rng = np.random.default_rng(19)
    source, _ = make_source(rng)
    with pytest.raises(ValueError, match="same feat_diff grid|points"):
        fit_exhaustive_density(source, np.zeros((1, N_POINTS + 2)) + np.linspace(0, 1, N_POINTS + 2),
                               ["c"], verbosity=0)


def test_constant_target_is_refused():
    rng = np.random.default_rng(23)
    source, _ = make_source(rng)
    with pytest.raises(ValueError, match="constant"):
        fit_exhaustive_density(source, np.full((1, N_POINTS), 0.02), ["flat"], verbosity=0)


def test_condition_name_count_must_match_targets():
    rng = np.random.default_rng(29)
    source, curves = make_source(rng)
    with pytest.raises(ValueError, match="condition names"):
        fit_exhaustive_density(source, curves[0, :2], ["only_one"], verbosity=0)


def test_source_requires_an_ordered_lattice():
    """The tie policy resolves to the smallest sd_spat, which means nothing if
    the lattice is unordered."""
    rng = np.random.default_rng(31)
    _, curves = make_source(rng, n_spat=3, n_feat=2)
    feat_pairs = np.array([(f1, f2) for f1 in (20.0, 60.0) for f2 in (20.0, 60.0)])
    with pytest.raises(ValueError, match="ascending"):
        InMemoryCurveSource([30.0, 10.0, 20.0], feat_pairs, curves)
