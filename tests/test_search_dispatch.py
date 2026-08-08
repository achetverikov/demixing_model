"""`--search exhaustive` reaches density and nothing else, end to end.

Dispatch is the part of this work most likely to be wrong in a way no number
reveals: a run that silently stayed hierarchical produces perfectly good fits,
just not the ones the command line claims. So these tests run the real
`run_fitting` on real data and record **which backend was called for which
method**, rather than inferring it from fitted values -- with a coarse cache the
two searches can land on the same parameters by coincidence, and an inference
test would then pass while dispatch was broken.

The hierarchical grid is shrunk for the same reason a cache step of 10 deg is
used: this file is about routing, and a production-grid fit on CPU takes tens of
minutes. Search quality is `test_exhaustive_density`'s job.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

CHECKPOINT = ROOT / "pretrained" / "model_epoch1500_10ktrain_20samples.pkl"
DATA = ROOT.parent / "example_data" / "moors_prepared.csv"
pytestmark = pytest.mark.skipif(
    not (CHECKPOINT.exists() and DATA.exists()), reason="needs checkpoint and example data")

import curve_cache as cc
import fit_model_to_data as F
from grid_based_multi_condition_optimizer_jax_loops import GridBasedMultiConditionOptimizer
from shared.config import config

CACHE_STEP = 10.0
#: Routing, not search quality. One coarse stage keeps a CPU run to seconds.
TINY_GRID = {'shared_grid_size': 4, 'feat_grid_size': 4,
             'min_grid_step': 40.0, 'zoom_factor': 0.5}


@pytest.fixture(scope="module")
def cache_root(tmp_path_factory):
    """A coarse cache spanning the production parameter range."""
    root = tmp_path_factory.mktemp("caches")
    key = cc.default_cache_key(
        checkpoint_path=CHECKPOINT, low=config.param_grid_low,
        high=config.param_range_high, step=CACHE_STEP,
        emp_density_weights_sd=F.DENSITY_CURVE_SPEC['emp_density_weights_sd'],
        density_smoothing_sigma=F.DENSITY_CURVE_SPEC['density_smoothing_sigma'],
    )
    optimizer = GridBasedMultiConditionOptimizer(str(CHECKPOINT), None, skip_motor_noise=True,
                                                 **F.DENSITY_CURVE_SPEC)
    sd_spat = cc.lattice(config.param_grid_low, config.param_range_high, CACHE_STEP)
    pairs = cc.feat_pair_lattice(sd_spat)
    curves = cc.build_curve_lattice(optimizer, sd_spat, pairs, verbosity=0)
    cc.write_cache(cc.cache_dir_for(root, key), cache_key=key, sd_spat_values=sd_spat,
                   feat_pairs=pairs, curves=curves)
    return root, key


@pytest.fixture
def tiny_hierarchical(monkeypatch):
    monkeypatch.setitem(F.HIERARCHICAL_GRID_SPEC, 'shared_grid_size', TINY_GRID['shared_grid_size'])
    monkeypatch.setitem(F.HIERARCHICAL_GRID_SPEC, 'feat_grid_size', TINY_GRID['feat_grid_size'])
    monkeypatch.setitem(F.HIERARCHICAL_GRID_SPEC, 'min_grid_step', TINY_GRID['min_grid_step'])


def run_fit(output_dir, methods, **kwargs):
    F.run_fitting(
        data_path=str(DATA), checkpoint_path=str(CHECKPOINT), output_dir=str(output_dir),
        methods=methods, max_subjects=1, min_trials=30, circ_space=180,
        results_dir=str(ROOT / "results"), **kwargs,
    )
    return F.load_results(str(output_dir))[0]


@pytest.fixture
def record_backends(monkeypatch):
    """Record which backend ran for which method."""
    calls = {'exhaustive': [], 'hierarchical': []}
    real_exhaustive = F.fit_exhaustive_density
    real_hierarchical = GridBasedMultiConditionOptimizer.fit_hierarchical_grid

    def spy_exhaustive(source, targets, names, objective="density", **kwargs):
        calls['exhaustive'].append(objective)
        return real_exhaustive(source, targets, names, objective=objective, **kwargs)

    def spy_hierarchical(self, *args, fitting_method="likelihood", **kwargs):
        calls['hierarchical'].append(fitting_method)
        return real_hierarchical(self, *args, fitting_method=fitting_method, **kwargs)

    monkeypatch.setattr(F, 'fit_exhaustive_density', spy_exhaustive)
    monkeypatch.setattr(GridBasedMultiConditionOptimizer, 'fit_hierarchical_grid',
                        spy_hierarchical)
    return calls


def test_only_density_is_routed_to_the_exhaustive_backend(
        cache_root, tmp_path, record_backends, tiny_hierarchical):
    """A curve cache holds density curves and nothing else, so a global backend
    switch would be wrong; dispatch is per method."""
    root, _ = cache_root
    run_fit(tmp_path / "mixed", ["density", "likelihood"], search="exhaustive",
            curve_cache_root=str(root), curve_cache_step=CACHE_STEP)

    assert record_backends['exhaustive'] == ["density"]
    assert record_backends['hierarchical'] == ["likelihood"]


def test_without_the_flag_density_stays_hierarchical(tmp_path, record_backends, tiny_hierarchical):
    """The default is unchanged, so every existing invocation keeps its behaviour."""
    run_fit(tmp_path / "default", ["density"])
    assert record_backends['exhaustive'] == []
    assert record_backends['hierarchical'] == ["density"]


def test_exhaustive_results_land_on_the_cache_lattice(cache_root, tmp_path):
    """Corroborates the routing from the other side: an exhaustive fit can only
    return lattice points."""
    root, _ = cache_root
    results = run_fit(tmp_path / "exh", ["density"], search="exhaustive",
                      curve_cache_root=str(root), curve_cache_step=CACHE_STEP)
    lattice = cc.lattice(config.param_grid_low, config.param_range_high, CACHE_STEP)

    assert results, "no conditions fitted"
    for entry in results.values():
        for value in np.asarray(entry["density_fitted_params"])[:3]:
            assert np.min(np.abs(lattice - value)) < 1e-6, f"{value} is off the cache lattice"


def test_exhaustive_refuses_motor_noise(tmp_path):
    """sd_motor is a fourth axis the cache does not span; scanning anyway would
    silently fit every group at sd_motor = 0."""
    with pytest.raises(ValueError, match="sd_motor|fourth axis"):
        run_fit(tmp_path / "motor", ["density"], search="exhaustive",
                curve_cache_root=str(tmp_path / "cache"), skip_motor_noise=False)


def test_exhaustive_requires_a_cache_path(tmp_path):
    with pytest.raises(ValueError, match="curve-cache"):
        run_fit(tmp_path / "nocache", ["density"], search="exhaustive")


def test_the_backend_is_recorded_in_the_run_fingerprint(cache_root, tmp_path):
    """Two runs that searched differently must not resume onto each other: their
    results are not interchangeable."""
    import run_fingerprint as rf
    root, key = cache_root

    out = tmp_path / "fp"
    run_fit(out, ["density"], search="exhaustive",
            curve_cache_root=str(root), curve_cache_step=CACHE_STEP)
    payload = rf.read_fingerprint_sidecar(out)["payload"]
    assert payload["search_backend"] == "exhaustive_1deg"
    assert payload["curve_cache_key"] == key

    with pytest.raises(rf.StaleResultsError, match="search_backend"):
        run_fit(out, ["density"])          # hierarchical, same directory
