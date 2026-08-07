"""Cache identity: the thing that makes reading a cache instead of the model safe.

A cache built from a different checkpoint, grid, or density-target setting yields
curves that are perfectly plausible and simply wrong, and no downstream check
would notice. So the tests here are about *refusal*: does a cache that should not
be read get read anyway?

The one failure mode a checksum cannot catch is a wrong row-to-parameter mapping
-- the bytes are intact, they just mean something else -- which is why the axis
order and the pair enumeration are asserted against the parameters they claim to
describe, not just against the manifest that states them.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import curve_cache as cc
import density_objective as do
from exhaustive_density import fit_exhaustive_density

N_POINTS = 12


KEY_ARGS = dict(
    checkpoint_sha256="a" * 64,
    sd_spat_grid=(5.0, 200.0, 1.0),
    feat_grid=(5.0, 200.0, 1.0),
    feat_diff_range=(2, 180),
    feat_diff_step=2,
    mu1_bias_range=(-180, 180),
    mu1_bias_step=2,
    emp_density_weights_sd=20.0,
    density_smoothing_sigma=None,
    encoding={"kind": "exact", "dtype": "float32"},
)


def build(tmp_path, n_spat=3, n_feat=2, seed=0, key="testkey"):
    rng = np.random.default_rng(seed)
    sd_spat = cc.lattice(10.0, 30.0, 10.0)[:n_spat]
    feat_pairs = cc.feat_pair_lattice(cc.lattice(20.0, 20.0 + 10.0 * (n_feat - 1), 10.0))
    curves = rng.normal(0.0, 0.05, (len(sd_spat), len(feat_pairs), N_POINTS))
    cache_dir = cc.cache_dir_for(tmp_path, key)
    cc.write_cache(cache_dir, cache_key=key, sd_spat_values=sd_spat,
                   feat_pairs=feat_pairs, curves=curves)
    return cache_dir, curves, sd_spat, feat_pairs


# --- the cache key -----------------------------------------------------------

def test_key_is_stable():
    assert cc.compute_cache_key(**KEY_ARGS) == cc.compute_cache_key(**KEY_ARGS)


@pytest.mark.parametrize("override", [
    {"checkpoint_sha256": "b" * 64},
    {"sd_spat_grid": (5.0, 200.0, 2.0)},
    {"feat_grid": (10.0, 200.0, 1.0)},
    {"feat_diff_range": (2, 178)},
    {"feat_diff_step": 4},
    {"mu1_bias_step": 4},
    {"mu1_bias_range": (-180, 178)},
    {"emp_density_weights_sd": 25.0},
    {"density_smoothing_sigma": 5.0},
    {"encoding": {"kind": "pca", "rank": 12, "dtype": "float16"}},
])
def test_everything_that_changes_a_curve_changes_the_key(override):
    assert cc.compute_cache_key(**{**KEY_ARGS, **override}) != cc.compute_cache_key(**KEY_ARGS)


def test_exact_and_pca_builds_cannot_share_a_directory():
    exact = cc.compute_cache_key(**KEY_ARGS)
    pca = cc.compute_cache_key(**{**KEY_ARGS, "encoding": {"kind": "pca", "rank": 12}})
    assert cc.cache_dir_for("/tmp", exact) != cc.cache_dir_for("/tmp", pca)


def test_mu1_grid_convention_is_in_the_key():
    """The circularity fix changed this axis. A cache built before it holds
    different curves and must not be readable under the current convention."""
    pre_fix = cc.compute_cache_key(**{**KEY_ARGS, "mu1_bias_range": (-180, 182)})
    assert pre_fix != cc.compute_cache_key(**KEY_ARGS)


# --- what is stored ----------------------------------------------------------

def test_curves_are_stored_centered_with_their_statistics(tmp_path):
    cache_dir, raw, _, _ = build(tmp_path)
    source = cc.CachedCurveSource(cache_dir)
    centered, means, variances = source.slab(1)

    np.testing.assert_allclose(means, raw[1].mean(axis=1), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(variances, raw[1].var(axis=1), rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(centered.mean(axis=1), 0.0, atol=1e-7)
    np.testing.assert_allclose(centered, raw[1] - raw[1].mean(axis=1, keepdims=True),
                               rtol=1e-4, atol=1e-7)


def test_row_to_parameter_mapping_survives_the_round_trip(tmp_path):
    """The failure no checksum can catch: intact bytes, wrong meaning."""
    cache_dir, raw, sd_spat, feat_pairs = build(tmp_path, n_spat=3, n_feat=3)
    source = cc.CachedCurveSource(cache_dir)

    np.testing.assert_allclose(source.sd_spat_values, sd_spat)
    np.testing.assert_allclose(source.feat_pairs, feat_pairs)
    for spat_index in range(len(sd_spat)):
        centered, _, _ = source.slab(spat_index)
        expected = raw[spat_index] - raw[spat_index].mean(axis=1, keepdims=True)
        np.testing.assert_allclose(centered, expected, rtol=1e-4, atol=1e-7)


def test_pair_enumeration_is_sd_feat1_outer():
    """The order the exhaustive tie policy names."""
    pairs = cc.feat_pair_lattice(np.array([1.0, 2.0, 3.0]))
    assert pairs.tolist() == [[1, 1], [1, 2], [1, 3],
                              [2, 1], [2, 2], [2, 3],
                              [3, 1], [3, 2], [3, 3]]


def test_manifest_records_the_mapping_not_just_the_shape(tmp_path):
    cache_dir, _, _, _ = build(tmp_path)
    manifest = json.loads((cache_dir / cc.MANIFEST_NAME).read_text())
    assert manifest["axis_order"] == ["sd_spat", "feat_pair", "feat_diff"]
    assert "sd_feat1 outer" in manifest["feat_pair_enumeration"]
    assert manifest["curves_are_centered"] is True
    assert set(manifest["checksums"]) == {"curves", "means", "variances",
                                          "sd_spat", "feat_pairs"}


# --- refusals ----------------------------------------------------------------

def test_an_unfinished_cache_is_refused(tmp_path):
    cache_dir, _, _, _ = build(tmp_path)
    (cache_dir / cc.COMPLETION_MARKER).unlink()
    with pytest.raises(cc.CacheIncompleteError, match="never finished"):
        cc.CachedCurveSource(cache_dir)


def test_the_marker_is_written_last(tmp_path):
    """A build killed partway must not read as a short cache. Everything the
    marker vouches for has to exist before it does."""
    cache_dir, _, _, _ = build(tmp_path)
    marker_mtime = (cache_dir / cc.COMPLETION_MARKER).stat().st_mtime_ns
    for name in ("curves", "means", "variances", "sd_spat", "feat_pairs"):
        assert (cache_dir / f"{name}.npy").stat().st_mtime_ns <= marker_mtime
    assert (cache_dir / cc.MANIFEST_NAME).stat().st_mtime_ns <= marker_mtime


def test_a_damaged_array_is_caught_by_verify(tmp_path):
    cache_dir, _, _, _ = build(tmp_path)
    cc.read_manifest(cache_dir, verify=True)          # clean

    curves = np.load(cache_dir / "curves.npy")
    curves[0, 0, 0] += 1.0
    np.save(cache_dir / "curves.npy", curves)

    with pytest.raises(cc.CacheCorruptError, match="checksum"):
        cc.read_manifest(cache_dir, verify=True)


def test_a_stale_cache_format_is_refused(tmp_path):
    cache_dir, _, _, _ = build(tmp_path)
    manifest = json.loads((cache_dir / cc.MANIFEST_NAME).read_text())
    manifest["cache_format"] = cc.CACHE_FORMAT + 1
    (cache_dir / cc.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(cc.CacheCorruptError, match="cache format"):
        cc.CachedCurveSource(cache_dir)


def test_a_raw_curve_cache_is_refused(tmp_path):
    """Reading raw curves as if they were centered would silently reintroduce
    the float32 cancellation the centering exists to avoid."""
    cache_dir, _, _, _ = build(tmp_path)
    manifest = json.loads((cache_dir / cc.MANIFEST_NAME).read_text())
    manifest["curves_are_centered"] = False
    (cache_dir / cc.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(cc.CacheCorruptError, match="centered"):
        cc.CachedCurveSource(cache_dir)


def test_shape_disagreeing_with_the_manifest_is_refused(tmp_path):
    cache_dir, _, _, _ = build(tmp_path)
    manifest = json.loads((cache_dir / cc.MANIFEST_NAME).read_text())
    manifest["n_pairs"] += 1
    (cache_dir / cc.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(cc.CacheCorruptError, match="row-to-parameter"):
        cc.CachedCurveSource(cache_dir)


# --- build on demand ---------------------------------------------------------

def test_open_or_build_builds_once_then_reuses(tmp_path):
    calls = []

    def build_fn(cache_dir):
        calls.append(cache_dir)
        rng = np.random.default_rng(1)
        sd_spat = cc.lattice(10.0, 20.0, 10.0)
        pairs = cc.feat_pair_lattice(np.array([20.0, 30.0]))
        cc.write_cache(cache_dir, cache_key="k", sd_spat_values=sd_spat, feat_pairs=pairs,
                       curves=rng.normal(0, 0.05, (2, 4, N_POINTS)))

    first = cc.open_or_build(tmp_path, "k", build_fn)
    second = cc.open_or_build(tmp_path, "k", build_fn)
    assert len(calls) == 1
    assert first.n_points == second.n_points == N_POINTS


def test_open_or_build_does_not_leave_a_readable_cache_when_the_build_fails(tmp_path):
    def failing_build(cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "curves.npy", np.zeros((1, 1, N_POINTS), dtype=np.float32))
        raise RuntimeError("surrogate died halfway")

    with pytest.raises(RuntimeError, match="halfway"):
        cc.open_or_build(tmp_path, "k", failing_build)
    with pytest.raises(cc.CacheIncompleteError):
        cc.CachedCurveSource(cc.cache_dir_for(tmp_path, "k"))


# --- end to end with the backend --------------------------------------------

def test_a_cached_source_drives_the_exhaustive_backend(tmp_path):
    """The point of the abstraction: the backend cannot tell disk from memory."""
    cache_dir, raw, sd_spat, feat_pairs = build(tmp_path, n_spat=3, n_feat=3, seed=4)
    source = cc.CachedCurveSource(cache_dir)
    targets = np.stack([raw[1, 4] * 1.2, raw[2, 0] * 0.7])

    result = fit_exhaustive_density(source, targets, ["c1", "c2"], verbosity=0)

    # Independent brute force over the same lattice, straight from the raw curves.
    best = (np.inf, None)
    for i in range(len(sd_spat)):
        total = sum(min(float(do.ccc_loss(raw[i, j], targets[c]))
                        for j in range(len(feat_pairs)))
                    for c in range(2))
        if total < best[0]:
            best = (total, i)
    assert result["best_loss"] == pytest.approx(best[0], rel=1e-4, abs=1e-6)
    assert result["shared_params"]["sd_spat"] == pytest.approx(float(sd_spat[best[1]]))
