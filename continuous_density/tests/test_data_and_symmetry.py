"""Tests for the design, the sample store and the mirror relation.

The simulator-level symmetry check (``test_simulator_mirror_relation``) needs a
GPU run of the EM and is skipped unless ``DM_RUN_SIM_TESTS=1``.

Run from the repo root::

    PYTHONPATH=. python -m pytest continuous_density/tests/test_data_and_symmetry.py
"""

import gzip
import os
import pickle

import numpy as np
import pytest

from continuous_density import data as data_mod
from continuous_density import design as design_mod
from continuous_density import existing_samples


def _fake_store(n_rows=40, n_sims=7, seed=0):
    rng = np.random.default_rng(seed)
    design = design_mod.sobol_design(n_rows, seed=seed)
    bias = rng.uniform(-180, 180, (n_rows, n_sims, 2)).astype(np.float32)
    return data_mod.SampleStore(design, bias)


def test_sobol_design_covers_the_domain():
    d = design_mod.sobol_design(512, seed=1)
    assert d.shape == (512, 4)
    assert d[:, :3].min() >= design_mod.SD_BOUNDS[0] - 1e-3
    assert d[:, :3].max() <= design_mod.SD_BOUNDS[1] + 1e-3
    assert d[:, 3].min() >= design_mod.FEAT_DIFF_BOUNDS[0] - 1e-3
    assert d[:, 3].max() <= design_mod.FEAT_DIFF_BOUNDS[1] + 1e-3
    # off-grid by construction: essentially nothing lands on the 5-degree grid
    on_grid = np.mean(np.isclose(d[:, :3] % 5.0, 0.0, atol=1e-3))
    assert on_grid < 0.01


def test_low_dprime_augmentation_design_targets_off_grid_unequal_noise():
    d = design_mod.low_dprime_augmentation_design(512, seed=4)
    assert d.shape == (512, 4)
    assert np.all(d[:, 0] < d[:, 1])
    assert np.all((d[:, 2] >= 50.) & (d[:, 2] <= 140.))
    assert np.all((40. / d[:, 2] >= 40. / 140.) & (40. / d[:, 2] <= .8))
    assert np.max(d[:, 1] / d[:, 0]) > 7.
    assert np.any((d[:, 0] > 50.) & (d[:, 1] / d[:, 0] < 1.25))
    assert np.mean(np.isclose(d[:, :3] % 5., 0., atol=1e-3)) < .01


def test_mixed_batch_uses_requested_augmentation_fraction():
    from continuous_density import train
    primary = _fake_store(n_rows=4, n_sims=3)
    augmentation = _fake_store(n_rows=4, n_sims=3, seed=1)
    primary.bias[:] = 1.
    augmentation.bias[:] = 2.
    _, bias, _ = train.mixed_batch(primary, augmentation, .25,
                                   np.random.default_rng(0), 100)
    assert np.sum(bias == 2.) == 25
    assert np.sum(bias == 1.) == 75


def test_validation_design_is_stratified_and_off_grid():
    d, strata = design_mod.validation_design(per_stratum=8)
    assert d.shape[0] == len(strata) == 8 * len(design_mod._STRATA)
    assert len(set(strata)) == len(design_mod._STRATA)
    assert np.mean(np.isclose(d[:, :3] % 5.0, 0.0, atol=1e-3)) < 0.01


def test_stress_trajectory_design_is_off_grid_and_grouped():
    d, labels = design_mod.stress_trajectory_design(n_curves=5, points_per_curve=9)
    assert d.shape == (45, 4) and len(labels) == 45
    assert np.all((d[:, :3] >= 5) & (d[:, :3] <= 200))
    assert not np.any(np.isclose(d[:, :3] % 5.0, 0.0, atol=1e-3))
    for label in set(labels):
        block = d[np.asarray(labels) == label]
        assert len(block) == 9
        assert np.all(block[:, :3] == block[0, :3])
        assert block[0, 3] == pytest.approx(2.0)
        assert block[-1, 3] == pytest.approx(180.0)


def test_uev_design_is_canonical_and_uses_spatial_dprime_definition():
    d, labels = design_mod.uev_design(feature_step=2.0)
    n_pairs = len(design_mod.UEV_SD_FEAT) * (len(design_mod.UEV_SD_FEAT) + 1) // 2
    assert d.shape == (n_pairs * len(design_mod.UEV_SPAT_DPRIME) * 90, 4)
    assert len(labels) == len(d)
    assert np.all(d[:, 0] <= d[:, 1])
    assert set(d[:, 2]) == {20.0, 40.0, 80.0}
    assert set(design_mod.UEV_SPAT_DIFF / d[:, 2]) == set(design_mod.UEV_SPAT_DPRIME)
    assert set(d[:, 3]) == set(np.arange(2.0, 181.0, 2.0))

    with pytest.raises(ValueError, match='divide'):
        design_mod.uev_design(feature_step=3.0)


def test_resumable_shard_helpers_verify_and_assemble(tmp_path):
    from continuous_density import generate_training_data as gen
    assert gen._shard_ranges(7, 3) == [(0, 3), (3, 6), (6, 7)]
    design = np.arange(20, dtype=np.float32).reshape(5, 4)
    records = []
    for row_start, row_stop in gen._chunk_ranges(5, 2):
        for sim_start, sim_stop in gen._chunk_ranges(5, 3):
            path = tmp_path / f'{row_start}_{sim_start}.npz'
            value = row_start * 10 + sim_start
            bias = np.full((row_stop - row_start, sim_stop - sim_start, 2),
                           value, dtype=np.float32)
            coordinates = np.asarray([row_start, row_stop, sim_start, sim_stop])
            gen._save_npz(path, False, design=design[row_start:row_stop], bias=bias,
                          coordinates=coordinates)
            assert gen._valid_shard(path, design[row_start:row_stop],
                                    sim_stop - sim_start, coordinates)
            assert not gen._valid_shard(path, design[row_start:row_stop],
                                        sim_stop - sim_start, coordinates + 1)
            records.append((path, row_start, row_stop, sim_start, sim_stop))
    assembled = gen._assemble_chunks(records, 5, 5)
    assert assembled.shape == (5, 5, 2)
    assert np.all(assembled[:2, :3] == 0)
    assert np.all(assembled[:2, 3:] == 3)
    assert np.all(assembled[-1, :3] == 40)
    assert not list(tmp_path.glob('.*.tmp'))


def test_simulation_keys_are_shard_invariant_and_crn_aware():
    import jax
    from continuous_density import sim_interface
    key = jax.random.PRNGKey(12)
    whole = np.asarray(sim_interface.simulation_keys(key, 5, row_offset=7,
                                                     simulation_offset=250))
    split = np.concatenate([
        np.asarray(sim_interface.simulation_keys(key, 2, row_offset=7,
                                                 simulation_offset=250)),
        np.asarray(sim_interface.simulation_keys(key, 3, row_offset=9,
                                                 simulation_offset=250)),
    ])
    assert np.array_equal(whole, split)
    assert len(np.unique(whole, axis=0)) == 5
    crn = np.asarray(sim_interface.simulation_keys(
        key, 5, common_random_numbers=True, row_offset=7, simulation_offset=250))
    assert np.all(crn == crn[0])
    later = np.asarray(sim_interface.simulation_keys(
        key, 1, common_random_numbers=True, simulation_offset=500))
    assert not np.array_equal(crn[0], later[0])


def test_label_cases_finds_two_modes():
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(-60, 8, 4000), rng.normal(60, 8, 4000)])
    unimodal = rng.normal(10, 15, 8000)
    labels = design_mod.label_cases(np.stack([bimodal, unimodal]))
    assert labels['n_modes'][0] == 2
    assert labels['n_modes'][1] == 1
    assert labels['mean_bias'][1] == pytest.approx(10.0, abs=1.0)


def test_batch_applies_the_mirror_to_component_two():
    store = _fake_store()
    rng = np.random.default_rng(0)
    x, b, w = store.batch(rng, 4096)
    assert x.shape == (4096, 4) and b.shape == (4096,)
    assert np.all(w == 1.0)
    # every drawn x is either a design row or its mirror
    rows = {tuple(np.round(r, 4)) for r in store.design}
    mirrors = {tuple(np.round(r, 4)) for r in store.mirrored}
    assert all(tuple(np.round(r, 4)) in rows or tuple(np.round(r, 4)) in mirrors
               for r in x[:200])


def test_non_finite_outcomes_get_zero_weight():
    store = _fake_store(n_rows=4, n_sims=3)
    store.bias[:] = np.nan
    store.finite = np.isfinite(store.bias)
    _, b, w = store.batch(np.random.default_rng(0), 64)
    assert np.all(w == 0.0)
    assert np.all(np.isfinite(b))  # nan_to_num keeps the loss finite
    assert store.n_observations == 0


def test_split_is_by_parameter_triple_not_by_simulation():
    design = np.repeat(np.array([[10., 20., 30., 4.], [50., 60., 70., 8.]],
                                dtype=np.float32), 5, axis=0)
    design[:, 3] = np.tile(np.arange(5, dtype=np.float32) * 2 + 2, 2)
    bias = np.zeros((10, 3, 2), dtype=np.float32)
    train, val = data_mod.split_by_params(data_mod.SampleStore(design, bias),
                                          val_fraction=0.5, seed=0)
    train_keys = {tuple(k) for k in train.param_key()}
    val_keys = {tuple(k) for k in val.param_key()}
    assert not (train_keys & val_keys)
    assert train.n_rows == val.n_rows == 5


def test_split_is_mirror_aware():
    """(sd1, sd2, sp) and its mirror (sd2, sd1, sp) never straddle the split.

    The mirror augmentation turns a training row into an observation at the
    swapped parameters, so grouping on the raw triple would leak a held-out
    combination through its mirrored twin.
    """
    base = np.array([[10., 20., 30.], [20., 10., 30.],   # a mirror pair
                     [40., 50., 60.], [50., 40., 60.],   # another mirror pair
                     [70., 70., 80.], [90., 15., 25.]], dtype=np.float32)
    design = np.column_stack([base, np.full(len(base), 5.0, np.float32)])
    bias = np.zeros((len(base), 3, 2), dtype=np.float32)
    store = data_mod.SampleStore(design, bias)

    # the two SD orderings share a canonical key ...
    keys = store.canonical_param_key()
    assert np.array_equal(keys[0], keys[1]) and np.array_equal(keys[2], keys[3])

    # ... so across many seeds a pair is always wholly train or wholly validation
    for seed in range(30):
        train, val = data_mod.split_by_params(store, val_fraction=0.4, seed=seed)
        train_c = {tuple(k) for k in train.canonical_param_key()}
        val_c = {tuple(k) for k in val.canonical_param_key()}
        assert not (train_c & val_c)
        train_raw = {tuple(np.round(r, 3)) for r in train.param_key()}
        val_raw = {tuple(np.round(r, 3)) for r in val.param_key()}
        # the raw mirror of any training triple must not appear in validation
        for r in train_raw:
            assert (r[1], r[0], r[2]) not in val_raw


def test_split_requires_two_mirror_distinct_groups():
    design = np.array([[10., 20., 30., 4.], [20., 10., 30., 8.]], np.float32)
    store = data_mod.SampleStore(design, np.zeros((2, 3, 2), np.float32))
    with pytest.raises(ValueError, match='mirror-distinct'):
        data_mod.split_by_params(store)


def _write_sample_file(directory, sf1, sf2, sp, n_bytes, run=None, stub=False):
    tag = f"_r{run}" if run is not None else ""
    name = f"samples_sf1_{sf1}_sf2_{sf2}_sp_{sp}{tag}_a1b2c3d4.pkl.gz"
    if stub:
        payload = {'parameters': {}, 'mu1_samples': np.zeros(0), 'stub': True}
    else:  # incompressible payload so the gzipped file clears the size floor
        rng = np.random.default_rng(abs(hash((sf1, sf2, sp, run))) % 2**32)
        payload = {'parameters': {},
                   'mu1_samples': rng.standard_normal(n_bytes).astype(np.float16)}
    with gzip.open(directory / name, 'wb') as f:
        pickle.dump(payload, f)
    return directory / name


def test_list_files_keeps_usable_and_drops_stubs(tmp_path):
    """Usable raw files are chosen *before* any n_files subsample, not stubs."""
    # 3 usable files (with a mirror-run variant) and 20 tiny stubs
    _write_sample_file(tmp_path, '10.0', '20.0', '30.0', 80000)
    _write_sample_file(tmp_path, '10.0', '20.0', '30.0', 80000, run=0)
    _write_sample_file(tmp_path, '40.0', '50.0', '60.0', 80000)
    for i in range(20):
        _write_sample_file(tmp_path, f'{i}.0', '5.0', '5.0', 0, stub=True)

    usable = existing_samples.list_files(tmp_path, min_bytes=1000)
    assert len(usable) == 3
    assert all(p.stat().st_size >= 1000 for p, _ in usable)
    # the default threshold also excludes the sub-kilobyte stubs
    assert len(existing_samples.list_files(tmp_path)) == 3


def test_circular_modes_counts_locations_and_masses():
    grid = np.linspace(-180.0, 180.0, 360, endpoint=False)

    def wn(center, sd):
        d = np.mod(grid - center + 180.0, 360.0) - 180.0
        return np.exp(-0.5 * (d / sd) ** 2)

    dens = 0.7 * wn(-60.0, 8.0) + 0.3 * wn(70.0, 8.0)
    modes = design_mod.circular_modes(dens, grid, min_mass=0.05)
    assert modes['n_modes'] == 2
    # sorted by mass: primary near -60 (~0.7), secondary near 70 (~0.3)
    assert modes['locations'][0] == pytest.approx(-60.0, abs=2.0)
    assert modes['locations'][1] == pytest.approx(70.0, abs=2.0)
    assert modes['masses'][0] == pytest.approx(0.7, abs=0.05)
    assert modes['masses'][1] == pytest.approx(0.3, abs=0.05)
    # a tiny bump below min_mass is not counted
    small = wn(0.0, 8.0) + 0.01 * wn(150.0, 4.0)
    assert design_mod.circular_modes(small, grid, min_mass=0.05)['n_modes'] == 1


def test_label_cases_excludes_nonfinite_outcomes():
    rng = np.random.default_rng(0)
    good = rng.normal(20.0, 6.0, 5000)
    with_nans = np.concatenate([good, np.full(5000, np.nan)])
    clean = design_mod.label_cases(good[None, :])
    dirty = design_mod.label_cases(with_nans[None, :])
    # NaNs must not shift the mean toward 0 nor invent a mode
    assert dirty['mean_bias'][0] == pytest.approx(clean['mean_bias'][0], abs=0.5)
    assert dirty['n_modes'][0] == clean['n_modes'][0] == 1


def _train_tiny(store, seed=0, steps=1500):
    import jax
    import jax.numpy as jnp
    import optax
    from continuous_density import wrapped_mixture_model as wm

    model = wm.ConditionalWrappedMixture(n_components=4, hidden_dims=(16, 16))
    x0 = store.design[:1]
    variables = model.init(jax.random.PRNGKey(seed), x0)
    x_all, b_all = store.all_rows()
    x_all = jnp.asarray(x_all)
    b_all = jnp.asarray(b_all)

    @jax.jit
    def loss_fn(v):
        dist = model.apply(v, x_all)
        return -jnp.mean(wm.mixture_logpdf(b_all, dist))

    tx = optax.adam(3e-3)
    opt_state = tx.init(variables)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    for _ in range(steps):
        _, grads = grad_fn(variables)
        updates, opt_state = tx.update(grads, opt_state, variables)
        variables = optax.apply_updates(variables, updates)
    return model, variables


def test_augmentation_teaches_the_mirror_relation():
    """A model trained on mirror-augmented data serves *both* components.

    This synthetic fixture happens to be odd under swapping SDs, but the actual
    theoretical relation only requires component 2 to be predicted at the
    mirrored input.  After a short deterministic fit, both component errors and
    NLLs must be finite and small.
    """
    import jax

    from continuous_density import evaluate as ev
    from continuous_density import wrapped_mixture_model as wm

    rng = np.random.default_rng(0)
    sd1, sd2, sp = np.meshgrid(np.array([20., 45., 90.]), np.array([25., 60., 120.]),
                               np.array([30., 80.]), indexing='ij')
    design = np.column_stack([sd1.ravel(), sd2.ravel(), sp.ravel(),
                              np.full(sd1.size, 40.0)]).astype(np.float32)
    mean = 18.0 * (np.log(design[:, 0]) - np.log(design[:, 1]))  # odd under mirror
    s = 12.0
    n = 400
    comp1 = mean[:, None] + s * rng.standard_normal((len(design), n))
    comp2 = (-mean)[:, None] + s * rng.standard_normal((len(design), n))
    bias = np.mod(np.stack([comp1, comp2], axis=-1) + 180.0, 360.0) - 180.0
    store = data_mod.SampleStore(design, bias.astype(np.float32))

    trained_model, trained_vars = _train_tiny(store)
    trained = ev.mirror_symmetry_diagnostic(trained_model, trained_vars, store)

    assert trained['comp1_mean_abs_error'] < 6.0
    assert trained['comp2_mean_abs_error'] < 6.0
    assert np.isfinite(trained['comp1_nll'])
    assert np.isfinite(trained['comp2_nll'])


def test_all_rows_matches_flatten_with_mirror():
    from continuous_density import sim_interface
    store = _fake_store(n_rows=6, n_sims=4)
    x1, b1 = store.all_rows()
    x2, b2 = sim_interface.flatten_with_mirror(store.design, store.bias)
    assert np.array_equal(x1, x2) and np.array_equal(b1, b2)


@pytest.mark.skipif(os.environ.get('DM_RUN_SIM_TESTS') != '1',
                    reason='runs the EM simulator on the GPU; set DM_RUN_SIM_TESTS=1')
def test_simulator_mirror_relation():
    """Component 2 at (sd1, sd2) is distributed like component 1 at (sd2, sd1).

    This is the assumption the mirror augmentation rests on, checked against the
    simulator itself rather than assumed.
    """
    import jax

    from continuous_density import sim_interface

    design = np.array([[12.0, 70.0, 40.0, 33.0],
                       [70.0, 12.0, 40.0, 33.0]], dtype=np.float32)
    bias = np.asarray(sim_interface.simulate(
        jax.random.PRNGKey(0), design, n_simulations=4000, n_samples=100))
    comp2_first = bias[0, :, 1]
    comp1_mirror = bias[1, :, 0]
    m = [np.mean(np.exp(1j * np.radians(v))) for v in (comp2_first, comp1_mirror)]
    assert abs(m[0] - m[1]) < 0.05
