"""Contracts for the shared surrogate resolver and the packaged WNM artifacts.

Two families now answer the same question, and the ways that can go wrong are
silent rather than loud: a run scored with the 100-observation model but recorded
as the 20-observation one, a research fit loaded as though it were production, a
family inferred from a filename that happens to contain the right substring.  All
of those produce plausible numbers.  These tests pin the refusals.

The packaging test is the one that matters scientifically: packaging copies
weights, so a packaged artifact that does not reproduce its research checkpoint
exactly means the architecture was rebuilt wrongly and every prediction from it
is a different model's.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import surrogate  # noqa: E402

PANEL = np.array([
    [10.0, 10.0, 10.0, 2.0],
    [10.0, 120.0, 10.0, 30.0],
    [200.0, 200.0, 200.0, 90.0],
    [5.0, 200.0, 5.0, 179.0],
], dtype=np.float32)

INSTALLED_WNM = [(n, path) for n, path in surrogate.WNM_DEFAULTS.items() if path.exists()]


def _write(tmp_path, name, blob):
    path = tmp_path / name
    with open(path, "wb") as handle:
        pickle.dump(blob, handle)
    return path


# ---------------------------------------------------------------------------
# Family detection: content decides, never the filename
# ---------------------------------------------------------------------------

def test_family_comes_from_content_not_filename(tmp_path):
    """A WNM artifact named like a surface checkpoint is still a WNM artifact."""
    misleading = _write(tmp_path, "surface_legacy_epoch1425_10ktrain_20samples.pkl",
                        {"variables": {}, "model_config": {"n_components": 8}, "meta": {}})
    assert surrogate.detect_family(misleading) == surrogate.FAMILY_WNM

    other = _write(tmp_path, "current_wnm_k12_20samples.pkl", {"apply_fn": object(), "params": {}})
    assert surrogate.detect_family(other) == surrogate.FAMILY_SURFACE_NN


def test_research_fit_is_refused_with_a_pointer_to_the_packager(tmp_path):
    fit = _write(tmp_path, "wnm_k12_full_n20_alldata-abc-best.pkl",
                 {"variables": {}, "selected_step": 90000, "validation_nll": 3.9})
    with pytest.raises(ValueError, match="surrogate_training/wnm/package_artifact"):
        surrogate.detect_family(fit)


def test_unrecognised_layout_raises(tmp_path):
    junk = _write(tmp_path, "whatever.pkl", {"weights": 1})
    with pytest.raises(ValueError, match="unrecognised checkpoint layout"):
        surrogate.detect_family(junk)


# ---------------------------------------------------------------------------
# Resolution: no guessed identities, no silent default between observer models
# ---------------------------------------------------------------------------

def test_resolving_a_default_requires_an_explicit_sample_count():
    with pytest.raises(ValueError, match="n_samples is required"):
        surrogate.resolve_checkpoint(surrogate.FAMILY_SURFACE_NN)


def test_unknown_family_and_sample_count_are_rejected():
    with pytest.raises(ValueError, match="unknown surrogate family"):
        surrogate.resolve_checkpoint("mixture_of_hopes", 20)
    with pytest.raises(ValueError, match="not one of"):
        surrogate.resolve_checkpoint(surrogate.FAMILY_SURFACE_NN, 50)
    with pytest.raises(ValueError, match="no default surface_nn artifact"):
        surrogate.resolve_checkpoint(surrogate.FAMILY_SURFACE_NN, 100)


def test_explicit_path_wins_over_the_defaults(tmp_path):
    explicit = tmp_path / "somewhere" / "custom.pkl"
    assert surrogate.resolve_checkpoint(None, 20, explicit) == explicit


@pytest.mark.legacy_surface
def test_unregistered_surface_checkpoint_has_no_inferable_identity(tmp_path):
    """`..._20samples.pkl` in the name is not evidence of the observer model."""
    stranger = tmp_path / "model_epoch900_10ktrain_20samples.pkl"
    with pytest.raises(ValueError, match="not in SURFACE_CHECKPOINT_REGISTRY"):
        surrogate._surface_sample_count(stranger, None)
    assert surrogate._surface_sample_count(stranger, 20) == 20


@pytest.mark.legacy_surface
def test_registered_surface_checkpoint_rejects_a_contradicting_request():
    known = Path("surface_legacy_epoch1425_10ktrain_20samples.pkl")
    assert surrogate._surface_sample_count(known, None) == 20
    assert surrogate._surface_sample_count(known, 20) == 20
    with pytest.raises(ValueError, match="different observer models"):
        surrogate._surface_sample_count(known, 100)


# ---------------------------------------------------------------------------
# Loading the installed WNM artifacts
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not INSTALLED_WNM, reason="no packaged WNM artifact installed")
@pytest.mark.parametrize("n_samples,path", INSTALLED_WNM)
def test_installed_artifact_declares_its_own_identity(n_samples, path):
    loaded = surrogate.load_surrogate(checkpoint_path=path)
    assert loaded.family == surrogate.FAMILY_WNM
    assert loaded.n_samples == n_samples

    meta = loaded.meta
    assert meta["artifact_schema"] == "wnm/2"
    assert meta["parameter_order"] == ["sd_feat1", "sd_feat2", "sd_spat", "feat_diff"]
    assert meta["period_degrees"] == 360.0
    assert meta["spatial_separation_degrees"] == 42.0
    # The alldata artifacts select their step in sample; a consumer that reports
    # validation_nll as held-out would be wrong, so the flag must be present.
    assert meta["validation_nll_is_in_sample"] is True
    assert meta["checkpoint_selection"] == "in_sample_slice"

    identity = loaded.identity()
    assert identity["dm_version"] == f"wnm_k12_{n_samples}samples"
    assert identity["surrogate_family"] == "wnm"
    assert identity["surrogate_n_samples"] == n_samples
    assert identity["surrogate_artifact"] == Path(path).name


@pytest.mark.parametrize(
    ("family", "artifact", "expected"),
    [
        ("wnm", "current_wnm_k12_20samples.pkl", "wnm_k12_20samples"),
        (
            "surface_nn",
            "surface_legacy_epoch1425_10ktrain_20samples.pkl",
            "surface_nn_epoch1425_10ktrain_20samples",
        ),
    ],
)
def test_dm_version_identifies_the_implementation(family, artifact, expected):
    assert surrogate.dm_version(family, artifact) == expected


@pytest.mark.skipif(not INSTALLED_WNM, reason="no packaged WNM artifact installed")
@pytest.mark.parametrize("n_samples,path", INSTALLED_WNM)
def test_requesting_the_wrong_sample_count_raises(n_samples, path):
    other = 100 if n_samples == 20 else 20
    with pytest.raises(ValueError, match="different observer models"):
        surrogate.load_surrogate(n_samples=other, checkpoint_path=path)


@pytest.mark.skipif(not INSTALLED_WNM, reason="no packaged WNM artifact installed")
@pytest.mark.parametrize("n_samples,path", INSTALLED_WNM)
def test_requesting_the_wrong_family_raises(n_samples, path):
    with pytest.raises(ValueError, match="family="):
        surrogate.load_surrogate(family=surrogate.FAMILY_SURFACE_NN, checkpoint_path=path)


@pytest.mark.skipif(not INSTALLED_WNM, reason="no packaged WNM artifact installed")
@pytest.mark.parametrize("n_samples,path", INSTALLED_WNM)
def test_packaged_artifact_reproduces_its_research_checkpoint(n_samples, path):
    """Packaging copies weights. Any difference is a reconstruction bug."""
    loaded = surrogate.load_surrogate(checkpoint_path=path)
    source = Path(loaded.meta["source_fit"])

    candidates = list((REPO_ROOT.parent / "results").glob(f"*/run_cache/*/{source.name}"))
    if not candidates:
        pytest.skip(f"research checkpoint {source.name} is not on this machine")

    with open(candidates[0], "rb") as handle:
        research = pickle.load(handle)

    packaged = loaded.payload["model"].apply(loaded.payload["variables"], PANEL)
    reference = loaded.payload["model"].apply(research["variables"], PANEL)
    for key in ("log_pi", "mu", "sigma"):
        np.testing.assert_array_equal(np.asarray(packaged[key]), np.asarray(reference[key]))


@pytest.mark.skipif(not INSTALLED_WNM, reason="no packaged WNM artifact installed")
def test_production_loading_does_not_need_the_corpus_or_training_scripts():
    """A normal checkout must be able to load and run the artifact on its own.

    The plan requires production inference not to import sibling experiment
    scripts or reach for the corpus; the packaged artifact carries its own
    architecture precisely so it does not have to.
    """
    n_samples, path = INSTALLED_WNM[0]
    before = set(sys.modules)
    loaded = surrogate.load_surrogate(checkpoint_path=path)
    dist = loaded.payload["model"].apply(loaded.payload["variables"], PANEL)
    assert np.all(np.isfinite(np.asarray(dist["mu"])))

    newly_imported = set(sys.modules) - before
    leaked = [name for name in newly_imported
              if "train_wnm" in name or "corpus_io" in name or name == "common"]
    assert not leaked, f"production load imported research modules: {leaked}"


# ---------------------------------------------------------------------------
# Search bounds belong to the surrogate, not to a module constant
# ---------------------------------------------------------------------------

@pytest.mark.legacy_surface
def test_surface_search_bounds_are_its_documented_training_range():
    """Substituting these for the old config constants must be a strict no-op.

    ``config.param_grid_low``/``param_range_high`` were 5.0/200.0, so if these
    differ, the surface backend's search changed when it should not have.
    """
    from shared.config import config

    bounds = surrogate.search_bounds(surrogate.SURFACE_DOMAIN)
    assert bounds["sd_feat"] == (config.param_grid_low, config.param_range_high)
    assert bounds["sd_spat"] == (config.param_grid_low, config.param_range_high)


@pytest.mark.skipif(not INSTALLED_WNM, reason="no packaged WNM artifact installed")
def test_wnm_search_bounds_open_the_narrow_feature_region():
    """The coverage that motivates the replacement has to reach the search.

    The mixture was trained down to sd_feat 2.5 but only to sd_spat 5, because
    sd_spat is 42/d' with d-prime capped at 8.4. Bounds that ignored that
    difference would either refuse trained feature noise or invite spatial
    extrapolation.
    """
    from shared.prediction import domain_from_meta

    _, path = INSTALLED_WNM[0]
    bounds = surrogate.search_bounds(domain_from_meta(surrogate.load_surrogate(
        checkpoint_path=path).meta))
    assert bounds["sd_feat"] == (2.5, 200.0)
    assert bounds["sd_spat"] == (5.0, 200.0)
    assert bounds["sd_feat"][0] < bounds["sd_spat"][0]


def test_the_two_feature_sds_share_one_interval():
    """They are exchangeable; a bound reachable for one but not the other would
    break the symmetry that component 2 is derived from."""
    lopsided = {"sd_feat1": (2.5, 198.0), "sd_feat2": (3.0, 200.0),
                "sd_spat": (5.0, 200.0), "feat_diff": (0.5, 180.0)}
    bounds = surrogate.search_bounds(lopsided)
    assert bounds["sd_feat"] == (3.0, 198.0)  # the intersection, not the union


# ---------------------------------------------------------------------------
# Exactly two production artifacts, selected by observer model
# ---------------------------------------------------------------------------

def test_wnm_is_the_production_family():
    assert surrogate.production_family() == surrogate.FAMILY_WNM


def test_exactly_two_checkpoints_are_production_at_a_time():
    """One per observer model, and one switch that says which family they come
    from. A caller asks by n_samples; nothing downstream should name a file."""
    paths = {n: surrogate.production_checkpoint(n)
             for n in surrogate.SUPPORTED_SAMPLE_COUNTS}
    assert len(paths) == 2
    assert len(set(paths.values())) == 2, "the two observer models share an artifact"
    for n_samples, path in paths.items():
        loaded = surrogate.load_surrogate(checkpoint_path=path)
        assert loaded.family == surrogate.production_family()
        assert loaded.n_samples == n_samples


def test_an_unsupported_sample_count_is_refused():
    """20 and 100 are two observer models, not a resolution knob, so there is
    nothing sensible between or beyond them."""
    with pytest.raises(ValueError, match="two different observer models"):
        surrogate.production_checkpoint(50)

@pytest.mark.legacy_surface
def test_other_checkpoints_stay_reachable_by_name():
    """Production is the default, not a restriction: a script that takes a
    checkpoint parameter can still load any installed artifact."""
    historical = surrogate.SURFACE_DEFAULTS[20]
    if not historical.exists():
        pytest.skip("historical checkpoint not installed")
    assert historical != surrogate.production_checkpoint(20)
    loaded = surrogate.load_surrogate(checkpoint_path=historical)
    assert loaded.n_samples == 20
