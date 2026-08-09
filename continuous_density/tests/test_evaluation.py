"""Tests for the batched, finite-aware evaluation and the production reader.

Pure array mathematics on tiny fixtures — no simulator, no GPU, no reference
corpus.

Run from the repo root::

    PYTHONPATH=. python -m pytest continuous_density/tests/test_evaluation.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from continuous_density import compare_existing_model as cmp
from continuous_density import data as data_mod
from continuous_density import design as design_mod
from continuous_density import evaluate as ev
from continuous_density import wrapped_mixture_model as wm
from shared.mu1_axis import mu1_cell_width, mu1_grid_np, mu1_size


def _model(k=6, seed=0):
    model = wm.ConditionalWrappedMixture(n_components=k)
    x = jnp.asarray([[10.0, 40.0, 25.0, 30.0], [150.0, 7.0, 90.0, 175.0]])
    variables = model.init(jax.random.PRNGKey(seed), x)
    return model, variables


# ---------------------------------------------------------------------------
# fix 3: chunked / finite-aware NLL and moments, no cases x grid x samples tensor
# ---------------------------------------------------------------------------

def test_model_case_nll_matches_naive_and_ignores_nonfinite():
    model, variables = _model()
    rng = np.random.default_rng(0)
    params = np.array([[10., 40., 25., 30.], [150., 7., 90., 175.],
                       [60., 60., 50., 90.]], dtype=np.float32)
    bias = rng.uniform(-180, 180, (3, 5000)).astype(np.float32)
    bias[0, :10] = np.nan  # non-finite outcomes must be excluded, not zeroed

    nll, n_ref = ev.model_case_nll(model, variables, params, bias,
                                   case_block=2, chunk=777)

    # naive per-case reference over the finite samples only
    for c in range(3):
        finite = np.isfinite(bias[c])
        dist = model.apply(variables, jnp.asarray(params[c:c + 1]))
        lp = np.asarray(wm.mixture_logpdf(jnp.asarray(bias[c][finite]),
                                          {k: jnp.repeat(v, finite.sum(), 0)
                                           for k, v in dist.items()}))
        assert nll[c] == pytest.approx(-lp.mean(), abs=1e-4)
        assert n_ref[c] == int(finite.sum())


def test_empirical_moments_exclude_nonfinite():
    rng = np.random.default_rng(1)
    good = rng.normal(30.0, 5.0, 4000)
    dirty = np.concatenate([good, np.full(4000, np.nan)])[None, :]
    clean = ev.empirical_moments(good[None, :])
    out = ev.empirical_moments(dirty)
    assert out['mean_bias'][0] == pytest.approx(clean['mean_bias'][0], abs=0.3)
    assert out['circ_sd'][0] == pytest.approx(clean['circ_sd'][0], abs=0.3)
    assert out['moment_real'][0] == pytest.approx(clean['moment_real'][0])
    assert out['moment_imag'][0] == pytest.approx(clean['moment_imag'][0])


def test_reference_density_ignores_nonfinite_and_integrates_to_one():
    rng = np.random.default_rng(2)
    bias = rng.normal(0.0, 20.0, (2, 3000)).astype(np.float32)
    bias[1, :500] = np.nan
    grid = ev.EVAL_GRID
    dx = float(grid[1] - grid[0])
    dens = ev.reference_density(bias, grid)
    mass = dens.sum(axis=1) * dx
    assert np.allclose(mass, 1.0, atol=1e-3)


def test_fft_reference_density_matches_direct_kernel_sum():
    rng = np.random.default_rng(22)
    samples = rng.normal(17.3, 18.0, 600)
    grid = np.arange(-180.0, 180.0, 1.0)
    got = ev.reference_density(samples[None, :], grid, kappa=40.0)[0]
    delta = np.radians(grid[:, None] - samples[None, :])
    direct = np.exp(40.0 * (np.cos(delta) - 1.0)).mean(axis=1)
    direct /= direct.sum()
    assert np.max(np.abs(got - direct)) < 2e-4


# ---------------------------------------------------------------------------
# fix 4: production log-density is renormalised after feat_diff interpolation
# ---------------------------------------------------------------------------

def _normalized_surfaces(M, n_feat, seed=0):
    rng = np.random.default_rng(seed)
    n_mu1 = mu1_size()
    raw = rng.normal(size=(M, n_mu1, n_feat)).astype(np.float64)
    # normalise each (case, feat) column to a per-degree density, like production
    dx = mu1_cell_width()
    logZ = jax.scipy.special.logsumexp(raw, axis=1, keepdims=True)
    return np.asarray(raw - logZ - np.log(dx))


def test_interpolated_profile_is_normalized():
    from shared.config import config
    feat_grid = np.asarray(config.create_grid('feat_diff'), dtype=np.float64)
    surfaces = _normalized_surfaces(4, len(feat_grid))
    dx = mu1_cell_width()

    # a feat_diff strictly between two grid nodes -> genuine interpolation
    feat_diff = np.full(4, float(feat_grid[10] + feat_grid[11]) / 2.0)
    prof = cmp.interpolated_profile(surfaces, feat_diff)
    mass = np.exp(prof).sum(axis=1) * dx
    assert np.allclose(mass, 1.0, atol=1e-9)

    # nearest reading of an already-normalised column stays normalised too
    near = cmp.nearest_profile(surfaces, feat_diff)
    assert np.allclose(np.exp(near).sum(axis=1) * dx, 1.0, atol=1e-9)


def test_normalize_mu1_logdensity_is_idempotent():
    surfaces = _normalized_surfaces(3, 5)
    prof = surfaces[:, :, 2]
    once = cmp.normalize_mu1_logdensity(prof)
    twice = cmp.normalize_mu1_logdensity(once)
    assert np.allclose(once, twice, atol=1e-12)


def test_single_surface_interpolation_handles_many_trials():
    from shared.config import config
    feat_grid = np.asarray(config.create_grid('feat_diff'), dtype=np.float64)
    surface = _normalized_surfaces(1, len(feat_grid))[0]
    feat_diff = np.linspace(feat_grid[0], feat_grid[-1], 17)
    got = cmp.interpolated_profile_single(surface, feat_diff)
    repeated = np.repeat(surface[None, :, :], len(feat_diff), axis=0)
    want = cmp.interpolated_profile(repeated, feat_diff)
    assert np.allclose(got, want)


def test_production_prediction_is_batched():
    calls = []

    def apply_fn(params, x):
        del params
        calls.append(len(x))
        return x[:, :, None]

    design = np.arange(44, dtype=np.float32).reshape(11, 4)
    got = cmp._predict_in_batches(apply_fn, None, design, batch_size=4)
    assert calls == [4, 4, 3]
    assert np.array_equal(got[:, :, 0], design[:, :3])


def test_production_condition_predicts_one_surface():
    from continuous_density import fit_demo
    calls = []
    n_feat = 90
    surface = _normalized_surfaces(1, n_feat)[0]

    def predictor(design):
        calls.append(design.copy())
        return surface[None, :, :]

    feat_diff = np.array([2.0, 20.5, 90.0, 180.0])
    nll = fit_demo.production_predictor_condition_nll(
        predictor, [20., 30., 40.], feat_diff, np.zeros(4))
    assert np.isfinite(nll)
    assert len(calls) == 1 and calls[0].shape == (1, 4)


def test_observer_sample_count_mismatch_is_rejected(tmp_path):
    model_meta = {'source_meta': {'n_samples': 20}}
    with pytest.raises(ValueError, match='n_samples=20'):
        data_mod.require_matching_n_samples(model_meta, {'n_samples': 100})
    assert data_mod.require_matching_n_samples(model_meta, {'n_samples': 20}) == 20
    assert cmp.checkpoint_n_samples(tmp_path / 'model_epoch1425_10ktrain_20samples.pkl') == 20
    assert cmp.checkpoint_n_samples(tmp_path / 'custom.pkl') is None


# ---------------------------------------------------------------------------
# fix 5: predicted secondary-mode diagnostics
# ---------------------------------------------------------------------------

def test_predicted_modes_recover_a_known_bimodal_mixture():
    from continuous_density import design as design_mod
    grid = ev.MODE_GRID
    dist = {'log_pi': jnp.log(jnp.asarray([[0.65, 0.35]])),
            'mu': jnp.asarray([[-50.0, 80.0]]),
            'sigma': jnp.asarray([[9.0, 9.0]])}
    dens = np.asarray(jnp.exp(wm.mixture_logpdf_grid(jnp.asarray(grid), dist)))[0]
    modes = design_mod.circular_modes(dens, grid)
    assert modes['n_modes'] == 2
    assert modes['locations'][0] == pytest.approx(-50.0, abs=2.0)
    assert modes['masses'][0] == pytest.approx(0.65, abs=0.05)


def test_mixture_logpdf_samples_agrees_with_pointwise():
    model, variables = _model()
    params = np.array([[10., 40., 25., 30.], [150., 7., 90., 175.]], dtype=np.float32)
    dist = model.apply(variables, jnp.asarray(params))
    samples = jnp.asarray([[-100.0, 0.0, 100.0], [12.0, -33.0, 179.0]])
    got = np.asarray(wm.mixture_logpdf_samples(samples, dist))
    for i in range(2):
        di = {k: v[i:i + 1] for k, v in dist.items()}
        want = np.asarray(wm.mixture_logpdf(
            samples[i], {k: jnp.repeat(v, 3, 0) for k, v in di.items()}))
        assert np.allclose(got[i], want, atol=1e-5)


def test_trajectory_summary_uses_complex_moment_residuals_at_seam():
    angles_ref = np.array([178., 179., -179., -178.])
    angles_pred = angles_ref + 1.0
    frame = pd.DataFrame({
        'stratum': ['trajectory_seam_0'] * 4,
        'component': [1] * 4,
        'feat_diff': [2., 30., 90., 180.],
        'ref_mean_bias': angles_ref,
        'pred_mean_bias': angles_pred,
        'ref_moment_real': np.cos(np.radians(angles_ref)),
        'ref_moment_imag': np.sin(np.radians(angles_ref)),
        'pred_moment_real': np.cos(np.radians(angles_pred)),
        'pred_moment_imag': np.sin(np.radians(angles_pred)),
        'nll': np.ones(4), 'kl_ref_model': np.ones(4),
        'pred_n_modes': np.ones(4), 'ref_n_modes': np.ones(4),
    })
    row = ev.trajectory_summary(frame).iloc[0]
    assert row.circular_moment_rmse == pytest.approx(2 * np.sin(np.radians(0.5)))
    assert row.circular_moment_residual_d2_rms < 0.002


def test_validation_coverage_reports_missing_behaviours():
    frame = pd.DataFrame({
        'ref_mean_bias': [2., -2., 0.], 'ref_n_modes': [1, 2, 1],
        'ref_circ_sd': [20., 100., 10.], 'ref_seam_mass': [0.01, 0.2, 0.01],
    })
    coverage = ev.validation_coverage(frame).set_index('category')
    assert coverage.loc['attraction (mean bias > 1 deg)', 'n_cases'] == 1
    assert coverage.loc['repulsion (mean bias < -1 deg)', 'n_cases'] == 1
    assert coverage.loc['multimodal reference', 'n_cases'] == 1


def test_training_trajectory_metrics_report_maxima_and_group_names():
    from continuous_density import train

    model, variables = _model(k=4)
    design, labels = design_mod.low_dprime_trajectory_design(2, 3, seed=3)
    dist = model.apply(variables, jnp.asarray(design))
    mean, _ = wm.mean_and_resultant(dist)
    rng = np.random.default_rng(5)
    comp1 = np.asarray(mean)[:, None] + rng.normal(0, 5, (len(design), 40))
    mirrored = np.asarray(wm.mirror_params(design))
    mean2, _ = wm.mean_and_resultant(model.apply(variables, jnp.asarray(mirrored)))
    comp2 = np.asarray(mean2)[:, None] + rng.normal(0, 7, (len(design), 40))
    store = data_mod.SampleStore(design, np.stack([comp1, comp2], axis=-1))
    got = train.trajectory_validation_metrics(model, variables, store, labels)
    assert np.isfinite(got['trajectory_nll'])
    assert got['worst_trajectory_nll'] >= got['trajectory_nll']
    assert got['max_mean_error'] >= 0 and got['max_sd_error'] >= 0
    assert got['max_mean_error_identified'] >= 0
    assert got['max_sd_error_identified'] >= 0
    assert got['moment_min_resultant'] == .2
    assert ':component' in got['max_mean_error_group']
