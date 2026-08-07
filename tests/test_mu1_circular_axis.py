"""Executable guards for the circular mu1_bias axis (CIRCULARITY_FIX_PLAN).

The failure mode this migration has is *not* a crash: it is code that keeps
running and returns a plausible wrong number.  So most tests here are written
as A/B comparisons against an explicit **faulty** implementation, and each one
is checked to actually fail on that faulty version — a test that cannot fail is
worse than no test, because it looks like coverage.

Run with:  PYTHONPATH=. pytest tests/test_mu1_circular_axis.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp

from shared.config import config
from shared import mu1_axis
from shared.mu1_axis import (LEGACY_MU1_GRID_SIZE, assert_mu1_axis, bin_indices,
                             legacy_mu1_axis, mu1_cell_width, mu1_grid,
                             mu1_size, periodic_integral, sign_masks,
                             trim_legacy_grid, trim_legacy_rows)


# ---------------------------------------------------------------------------
# 2 / 8. Grid identity and config consistency
# ---------------------------------------------------------------------------

def test_grid_identity_exact_values():
    grid = np.asarray(mu1_grid())
    assert grid.size == 180
    assert grid[0] == -180
    assert grid[-1] == 178
    assert np.allclose(np.diff(grid), 2.0)


def test_config_consistency_and_no_over_application():
    assert len(config.create_grid('mu1_bias')) == config.mu1_bias_grid_size
    # The fix applies to the ONE circular axis. feat_diff and mu2_bias are
    # bounded intervals and must still be inclusive.
    feat = np.asarray(config.create_grid('feat_diff'))
    mu2 = np.asarray(config.create_grid('mu2_bias'))
    assert feat[0] == config.feat_diff_range[0] and feat[-1] == config.feat_diff_range[1]
    assert mu2[0] == config.mu2_bias_range[0] and mu2[-1] == config.mu2_bias_range[1]
    assert len(feat) == config.feat_diff_grid_size
    assert config.mu1_surface_shape == (180, len(feat))


def test_cell_width_is_period_over_n_not_range_over_n_minus_1():
    assert mu1_cell_width() == pytest.approx(2.0)
    # The two traps: the inclusive convention, and deriving the step from the
    # half-open grid's own endpoints.
    grid = np.asarray(mu1_grid())
    assert (grid[-1] - grid[0]) / grid.size == pytest.approx(358 / 180)  # 1.989 — wrong
    assert 360 / (grid.size - 1) == pytest.approx(2.0112, abs=1e-4)      # also wrong


def test_assert_mu1_axis_rejects_legacy_and_wrong_values():
    assert_mu1_axis(mu1_grid())
    assert_mu1_axis(180)
    with pytest.raises(ValueError, match="legacy"):
        assert_mu1_axis(LEGACY_MU1_GRID_SIZE)
    # Right length, wrong values: an inclusive linspace with a 2.0112 step.
    faulty = np.linspace(-180, 180, 180)
    with pytest.raises(ValueError):
        assert_mu1_axis(faulty)


# ---------------------------------------------------------------------------
# 1. Sign-split exclusion — the decisive one
# ---------------------------------------------------------------------------

def _naive_sign_masks(grid):
    """The most likely implementation mistake: drop the row, keep `< 0`."""
    grid = np.asarray(grid)
    return grid > 0, grid < 0


def _asymmetry(probs, pos_mask, neg_mask, dx):
    pos = np.where(np.asarray(pos_mask)[:, None], probs, 0.0).sum(axis=0) * dx
    neg = np.where(np.asarray(neg_mask)[:, None], probs, 0.0).sum(axis=0) * dx
    return pos - neg


def test_sign_split_all_mass_at_antipode_is_exactly_zero():
    from shared.utils import compute_single_density_asymmetry

    grid = mu1_grid()
    log_surf = np.full((mu1_size(), 4), -1e9)
    log_surf[0, :] = 0.0  # all mass at -180, the antipode
    asym = compute_single_density_asymmetry(
        jnp.asarray(log_surf), jnp.arange(4), grid, apply_smoothing=False)
    assert np.allclose(np.asarray(asym), 0.0)

    # The same fixture under the naive mask returns a large negative value, so
    # this test fails loudly on precisely the error it guards.
    probs = np.exp(log_surf)
    naive = _asymmetry(probs, *_naive_sign_masks(grid), mu1_cell_width())
    assert np.max(np.abs(naive)) > 1.0


def test_sign_split_symmetric_with_antipodal_mass_is_zero():
    from shared.utils import compute_single_density_asymmetry

    grid = np.asarray(mu1_grid())
    # A less degenerate companion: paired ±angle mass plus substantial mass at
    # the antipode, which is where the two implementations diverge.
    probs = np.zeros((grid.size, 3))
    for angle, w in [(20.0, 0.3), (100.0, 0.15)]:
        probs[np.argmin(np.abs(grid - angle))] += w
        probs[np.argmin(np.abs(grid + angle))] += w
    probs[0] += 0.5  # antipode
    probs = probs / probs.sum(axis=0, keepdims=True) / mu1_cell_width()

    asym = compute_single_density_asymmetry(
        jnp.log(jnp.asarray(probs)), jnp.arange(3), jnp.asarray(grid),
        apply_smoothing=False)
    assert np.allclose(np.asarray(asym), 0.0, atol=1e-6)

    naive = _asymmetry(probs, *_naive_sign_masks(grid), mu1_cell_width())
    assert np.max(np.abs(naive)) > 0.1  # the defect signal is large here


def test_sign_masks_exclude_zero_and_antipode_only():
    grid = np.asarray(mu1_grid())
    pos, neg = (np.asarray(m) for m in sign_masks(jnp.asarray(grid)))
    assert not pos[grid == 0] and not neg[grid == 0]
    assert not pos[grid == -180] and not neg[grid == -180]
    # Everything else is on exactly one side.
    interior = (grid != 0) & (grid != -180)
    assert np.all(pos[interior] ^ neg[interior])
    assert pos.sum() == neg.sum() == 89


# ---------------------------------------------------------------------------
# 3. Normalisation — both normalisers, separate code paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["normalize_to_density", "normalize_to_density_flexible"])
def test_normalisers_integrate_to_one(which):
    if which == "normalize_to_density":
        from shared.surface_functions import normalize_to_density as fn
    else:
        from neural_network_optimization.mirror_aware_model import (
            normalize_to_density_flexible as fn)

    rng = np.random.default_rng(0)
    log_probs = jnp.asarray(rng.normal(size=(2, mu1_size(), 5)))
    density = np.exp(np.asarray(fn(log_probs)))
    integral = density.sum(axis=1) * mu1_cell_width()
    assert np.allclose(integral, 1.0, atol=1e-5)


def test_normalisers_reject_a_legacy_shaped_input():
    from shared.surface_functions import normalize_to_density

    log_probs = jnp.zeros((1, LEGACY_MU1_GRID_SIZE, 3))
    with pytest.raises(ValueError):
        normalize_to_density(log_probs)


# ---------------------------------------------------------------------------
# 4. Binning wraps, not clips
# ---------------------------------------------------------------------------

def test_bin_indices_wrap_both_endpoints_to_the_same_bin():
    assert int(bin_indices(jnp.asarray(-180.0))) == 0
    assert int(bin_indices(jnp.asarray(180.0))) == 0
    # The value that actually mis-bins under a clip: (179, 180) rounds to index
    # 180 and would be clipped onto the last row (+178) instead of wrapping.
    assert int(bin_indices(jnp.asarray(179.5))) == 0
    naive = int(np.clip(round((179.5 + 180) / 2), 0, mu1_size() - 1))
    assert naive == 179 and naive != 0


def test_both_optimizer_binning_paths_wrap():
    """Value check on the two in-fitter binning expressions (not just the helper)."""
    bias = np.array([-180.0, 180.0, 179.5, 0.0, 178.0])
    expected = np.array([0, 0, 0, 90, 179])

    # _prepare_all_condition_data path
    assert np.array_equal(np.asarray(bin_indices(jnp.asarray(bias))), expected)
    # _precompute_target_curves path (numpy twin, same convention)
    got = np.mod(np.round((bias - config.mu1_bias_range[0]) / config.mu1_bias_step
                          ).astype(int), mu1_size())
    assert np.array_equal(got, expected)
    # postprocess path
    from model_fit_to_data.postprocess_fitted_likelihoods import model_grid_indices
    got = model_grid_indices(bias, grid_min=config.mu1_bias_range[0],
                             grid_step=config.mu1_bias_step,
                             grid_size=config.mu1_bias_grid_size, circular=True)
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# 5 / 11. Legacy trim, never resize
# ---------------------------------------------------------------------------

def test_trim_preserves_ratios_between_retained_rows():
    rng = np.random.default_rng(1)
    rows = rng.normal(size=(LEGACY_MU1_GRID_SIZE, 4)).astype(np.float32)
    rows[-1] = rows[0]  # the identity the legacy data satisfies

    trimmed = np.asarray(trim_legacy_rows(jnp.asarray(rows)))
    assert trimmed.shape[0] == mu1_size()
    # Bit-exact retention is what distinguishes a trim from an interpolation.
    assert np.array_equal(trimmed, rows[:-1])

    # An interpolating "resize" to the same row count passes any shape check but
    # perturbs the rows relative to one another — hence this stronger assertion.
    import jax
    resized = np.asarray(jax.image.resize(
        jnp.asarray(rows), (mu1_size(), 4), method='linear'))
    assert not np.allclose(resized, rows[:-1])


def test_trim_refuses_when_endpoint_rows_differ():
    rows = np.zeros((LEGACY_MU1_GRID_SIZE, 3))
    rows[-1] = 1.0
    with pytest.raises(ValueError, match="same angle"):
        trim_legacy_rows(jnp.asarray(rows))


def test_trim_is_a_noop_on_already_periodic_rows():
    rows = jnp.zeros((mu1_size(), 3))
    assert np.asarray(trim_legacy_rows(rows)).shape[0] == mu1_size()


def test_trim_legacy_grid_matches_config_grid():
    legacy = np.arange(-180, 180 + 2, 2)
    assert np.array_equal(trim_legacy_grid(legacy), np.asarray(mu1_grid()))


def test_legacy_axis_context_manager_restores_state():
    assert mu1_size() == 180
    with legacy_mu1_axis():
        assert mu1_size() == LEGACY_MU1_GRID_SIZE
        assert len(config.create_grid('mu1_bias')) == LEGACY_MU1_GRID_SIZE
    assert mu1_size() == 180
    assert config.mu1_inclusive_legacy is False


# ---------------------------------------------------------------------------
# 6. Motor convolution wraps
# ---------------------------------------------------------------------------

def test_motor_convolution_is_symmetric_across_the_seam():
    from model_fit_to_data.grid_based_multi_condition_optimizer_jax_loops import (
        apply_motor_noise_with_precomputed_kernel, create_motor_noise_kernel_fft)

    n = mu1_size()
    kernel_fft = create_motor_noise_kernel_fft(6.0, n)
    log_surf = np.full((1, n, 1), -1e9)
    log_surf[0, 0, 0] = 0.0  # impulse at index 0 (= -180, the seam)

    out = np.asarray(apply_motor_noise_with_precomputed_kernel(
        jnp.asarray(log_surf), kernel_fft))[0, :, 0]
    p = np.exp(out - out.max())
    # Response must be symmetric about index 0, i.e. p[k] == p[-k] across the
    # wrap.  Tolerance is absolute against the unit peak: the FFT round-trip is
    # float32, so exact equality is not available.
    for k in range(1, 20):
        assert abs(p[k] - p[-k]) < 1e-6


def test_motor_kernel_fft_period_matches_the_angular_period():
    from model_fit_to_data.grid_based_multi_condition_optimizer_jax_loops import (
        create_motor_noise_kernel_fft)
    n = mu1_size()
    assert n * config.mu1_bias_step == 360  # the FFT period IS the angular period
    assert create_motor_noise_kernel_fft(3.0, n).shape == (n,)


# ---------------------------------------------------------------------------
# 9. The independent generator
# ---------------------------------------------------------------------------

def test_surface_computation_generator_emits_the_periodic_axis():
    """surface_computation builds its own arange, bypassing config entirely."""
    src = (Path(__file__).resolve().parents[1]
           / "surface_computation" / "jax_fit_main.py").read_text()
    assert "jnp.arange(mu1_bias_range[0], mu1_bias_range[1], mu1_bias_step)" in src
    assert "mu1_bias_range[1] + mu1_bias_step" not in src

    mu1_vals = np.arange(-180, 180, 2)  # the expression above, evaluated
    assert np.array_equal(mu1_vals, np.asarray(mu1_grid()))


# ---------------------------------------------------------------------------
# 10. The empirical TARGET side
# ---------------------------------------------------------------------------

def _mirrored_seam_fixture(n=4000, seed=3):
    """Mirrored (feat_diff, ±bias) pairs with mass at the seam.

    Both properties are required and neither is optional:
      * mirrored PAIRS (not merely aggregate symmetry) — feature-dependent
        weights are applied before the KDE is summed, so marginal symmetry can
        still produce a nonzero asymmetry;
      * mass AT the seam — a seam-free fixture puts no mass on the antipode row,
        so including or excluding it changes nothing and the defect signal is
        exactly 0.0, i.e. the test cannot fail.
    Bandwidth needs a positive IQR, not merely positive variance, so the bias
    values are continuous rather than a few repeated levels.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    feat = rng.uniform(4, 176, size=half)
    bias = ((rng.normal(180.0, 15.0, size=half) + 180) % 360) - 180
    return (np.concatenate([feat, feat]).astype(np.float64),
            np.concatenate([bias, -bias]).astype(np.float64))


def _faulty_empirical_asymmetry(real_feat_diff, real_bias, feat_diff_grid,
                                weights_sd=20, circ_space=360):
    """The naive `< 0` endpoint mask on the empirical target side."""
    import jax
    max_diss = circ_space / 2
    bias_std = jnp.std(real_bias)
    n = len(real_bias)
    kernel_bw = 0.9 * jnp.minimum(
        bias_std,
        (jnp.quantile(real_bias, 0.75) - jnp.quantile(real_bias, 0.25)) / 1.34
    ) * (n ** (-1 / 5))
    n_bias_points = 180
    dx = (2 * max_diss) / n_bias_points
    bias_range = -max_diss + dx * jnp.arange(n_bias_points)

    dist_matrix = feat_diff_grid[:, None] - real_feat_diff[None, :]
    log_w = -0.5 * (dist_matrix / weights_sd) ** 2
    w = jnp.exp(log_w - jax.scipy.special.logsumexp(log_w, axis=1, keepdims=True))
    k = jnp.exp(-0.5 * ((bias_range[:, None] - real_bias[None, :]) / kernel_bw) ** 2)
    k = k / (kernel_bw * jnp.sqrt(2 * jnp.pi))
    density = w @ k.T

    pos = jnp.where((bias_range > 0)[None, :], density, 0.0).sum(axis=1) * dx
    neg = jnp.where((bias_range < 0)[None, :], density, 0.0).sum(axis=1) * dx
    return pos - neg


def test_empirical_target_grid_is_half_open():
    """The target axis carried the identical dual-endpoint defect."""
    from shared.utils import _compute_empirical_density_asymmetry_core
    import inspect
    src = inspect.getsource(_compute_empirical_density_asymmetry_core)
    assert "linspace(-max_diss, max_diss, 181)" not in src


def test_empirical_target_symmetric_fixture_is_zero():
    from shared.utils import _compute_empirical_density_asymmetry_core

    feat, bias = _mirrored_seam_fixture()
    grid = jnp.arange(4, 181, 4).astype(float)
    _, asym = _compute_empirical_density_asymmetry_core(
        jnp.asarray(feat), jnp.asarray(bias), grid)

    # Tolerance measured, both bounds: the worst residual on a correct run is
    # ~3.8e-6 (float32), the defect signal on this fixture is ~1.2e-2. 1e-4 sits
    # ~26x above the noise floor and ~124x below the signal. 1e-6 would be BELOW
    # the noise floor and would flake on correct code.
    assert np.max(np.abs(np.asarray(asym))) < 1e-4

    # And the fixture is not vacuous: the naive endpoint mask fails it.
    faulty = _faulty_empirical_asymmetry(
        jnp.asarray(feat), jnp.asarray(bias), grid)
    assert np.max(np.abs(np.asarray(faulty))) > 1e-3


def test_empirical_and_model_sign_splits_agree_on_the_endpoint_rule():
    """Exports disagree with the fit if only one side excludes the antipode."""
    import inspect
    from shared.utils import (_compute_empirical_density_asymmetry_core,
                              compute_single_density_asymmetry)
    emp = inspect.getsource(_compute_empirical_density_asymmetry_core)
    assert "bias_range > -max_diss" in emp
    assert "sign_masks" in inspect.getsource(compute_single_density_asymmetry)


# ---------------------------------------------------------------------------
# 12 / 14. expectation_loss coordinates and periodic quadrature values
# ---------------------------------------------------------------------------

def test_expectation_loss_reads_the_config_grid():
    import inspect
    from neural_network_optimization import loss_functions
    src = inspect.getsource(loss_functions.expectation_loss)
    assert "mu1_grid()" in src
    assert "jnp.linspace(" not in src  # never reconstruct the coordinates


def test_periodic_quadrature_integrates_the_uniform_density_to_one():
    uniform = np.full((mu1_size(), 1), 1.0 / 360.0)
    assert float(periodic_integral(jnp.asarray(uniform), axis=0)[0]) == pytest.approx(1.0)
    # Trapezoid over x=grid spans only 358 deg and half-weights the ends.
    naive = float(jnp.trapezoid(jnp.asarray(uniform[:, 0]), x=mu1_grid()))
    assert naive == pytest.approx(358 / 360, abs=1e-6)
    assert abs(naive - 1.0) > 1e-3


def test_periodic_smoothness_penalises_the_wrap_step():
    from neural_network_optimization.loss_functions import smoothness_regularization

    x = np.zeros((1, mu1_size(), 3))
    x[0, 0, :] = 1.0  # a step that exists ONLY across the wrap and at row 0/1
    loss = float(smoothness_regularization(jnp.asarray(x)))
    # n differences (not n-1): the wrap-around gradient carries real weight, so
    # each of the 3 columns contributes two unit steps (into and out of row 0).
    expected = 2.0 / mu1_size()
    assert loss == pytest.approx(expected, rel=1e-6)

    naive = float(jnp.mean(jnp.diff(jnp.asarray(x), axis=1) ** 2))
    assert naive < expected  # jnp.diff never compares the last row to the first


# ---------------------------------------------------------------------------
# The seam neighbourhood in shared/averaging.py — bin-aligned rotation
# equivariance.  Not covered by any of the other invariants.
# ---------------------------------------------------------------------------

def _removed(samples, min_neighbors):
    from shared.averaging import prefilter_isolated_samples
    _, n = prefilter_isolated_samples(jnp.asarray(samples),
                                      min_neighbors=min_neighbors)
    return int(n)


def _rot(samples, k, bias_bin_size=8):
    return ((samples + k * bias_bin_size + 180) % 360) - 180


@pytest.mark.parametrize("min_neighbors", [17, 20, 24])
def test_prefilter_neighbourhood_wraps_at_the_seam(min_neighbors):
    """Bin-aligned rotation equivariance.

    Rotations must be integer multiples of bias_bin_size: the histogram bins are
    fixed, so an arbitrary rotation moves samples between bins and equivariance
    cannot hold even for correct code.

    24 feat rows give 3 feat bins, so the middle one has genuine *unclipped*
    feat neighbours — the real code path rather than a degenerate corner.  Only
    rows 8-15 sit near the seam: hist[1,44]=8 and hist[1,0]=16.  For a sample at
    178 the clipped bias neighbourhood {43,44,44} sums to 16 while the wrapped
    one {43,44,0} sums to 24, so any threshold in (16, 24] separates them.  The
    measured discriminating window is exactly [17, 24].
    """
    A = np.zeros((24, 3))
    A[8:16] = np.array([178., -178., -178.])

    # Correct (wrapping) behaviour: nothing is isolated, at the seam or anywhere.
    assert _removed(A, min_neighbors) == 0
    for k in range(1, 45):
        assert _removed(_rot(A, k), min_neighbors) == 0


def test_prefilter_seam_fixture_discriminates_against_clipping():
    """The fixture must fail on the faulty implementation, or it proves nothing."""
    A = np.zeros((24, 3))
    A[8:16] = np.array([178., -178., -178.])

    n_bias_bins, bias_bin_size, feat_bin_size = 45, 8, 8
    hist = np.zeros((3, n_bias_bins))
    for f in range(24):
        for v in A[f]:
            hist[f // feat_bin_size, int((v + 180) // bias_bin_size)] += 1

    def clipped_sum(f_bin, b_bin):
        return sum(hist[np.clip(f_bin + df, 0, 2), np.clip(b_bin + db, 0, n_bias_bins - 1)]
                   for df in (-1, 0, 1) for db in (-1, 0, 1))

    # A sample at 178 (bias bin 44, feat bin 1) sees only 16 neighbours when the
    # neighbourhood is clipped rather than wrapped -> removed at threshold 20.
    assert clipped_sum(1, 44) == 16
    assert clipped_sum(1, 44) < 20 <= clipped_sum(1, 0)


# ---------------------------------------------------------------------------
# 15. Legacy surfaces are never silently dropped
# ---------------------------------------------------------------------------

def _legacy_surface_payload():
    from shared.utils import Surface
    feat = np.asarray(config.create_grid('feat_diff'))
    mu2 = np.asarray(config.create_grid('mu2_bias'))
    mu1_legacy = np.arange(-180, 180 + 2, 2)
    rng = np.random.default_rng(7)
    mu1_surf = rng.normal(size=(mu1_legacy.size, feat.size))
    mu1_surf[-1] = mu1_surf[0]  # the identity legacy data satisfies
    return {"surface": Surface(
        feat_diff_grid=jnp.asarray(feat),
        mu1_bias_grid=jnp.asarray(mu1_legacy),
        mu2_bias_grid=jnp.asarray(mu2),
        mu1_comp1_surface=jnp.asarray(mu1_surf),
        mu1_comp2_surface=jnp.asarray(mu1_surf * 0.5),
        mu2_comp1_surface=jnp.zeros((mu2.size, feat.size)),
        mu2_comp2_surface=jnp.zeros((mu2.size, feat.size)),
    ), "parameters": {"sd_feat1": 30.0, "sd_feat2": 40.0, "sd_spat": 50.0}}


def _write_legacy_surface(directory: Path) -> Path:
    import pickle
    path = directory / "surface_sf1_30.0_sf2_40.0_sp_50.0_x.pkl"
    path.write_bytes(pickle.dumps(_legacy_surface_payload()))
    return path


def test_legacy_surfaces_raise_rather_than_being_silently_dropped(tmp_path):
    """The one failure that looks like success: a shorter list, not an error."""
    from shared.surface_folder_parsing import load_filtered_surfaces
    _write_legacy_surface(tmp_path)
    with pytest.raises(ValueError, match="migrat"):
        load_filtered_surfaces(str(tmp_path), low=10, high=100)


def test_load_filtered_surfaces_accepts_migrated_surfaces(tmp_path):
    import pickle
    from cloud.migrate_surfaces_to_periodic_mu1 import migrate_loose_file
    from shared.surface_folder_parsing import load_filtered_surfaces

    path = _write_legacy_surface(tmp_path)
    assert migrate_loose_file(path) is True
    loaded = load_filtered_surfaces(str(tmp_path), low=10, high=100)
    assert len(loaded) == 1
    assert loaded[0]["surface"].mu1_comp1_surface.shape[0] == mu1_size()


# ---------------------------------------------------------------------------
# 11. The migration script
# ---------------------------------------------------------------------------

def test_migration_trims_arrays_and_embedded_grid_and_is_idempotent(tmp_path):
    import pickle
    from cloud.migrate_surfaces_to_periodic_mu1 import migrate_loose_file

    path = _write_legacy_surface(tmp_path)
    original = pickle.loads(path.read_bytes())["surface"]

    assert migrate_loose_file(path) is True
    migrated = pickle.loads(path.read_bytes())["surface"]

    assert len(migrated.mu1_bias_grid) == mu1_size()
    assert migrated.mu1_comp1_surface.shape[0] == mu1_size()
    assert np.array_equal(np.asarray(migrated.mu1_bias_grid), np.asarray(mu1_grid()))
    # Trim, not interpolate: retained rows are bit-identical.
    assert np.array_equal(np.asarray(migrated.mu1_comp1_surface),
                          np.asarray(original.mu1_comp1_surface)[:-1])
    # mu2 is a linear axis and must be untouched.
    assert migrated.mu2_comp1_surface.shape == original.mu2_comp1_surface.shape

    assert migrate_loose_file(path) is False  # idempotent


def test_migration_refuses_when_endpoint_rows_differ(tmp_path):
    import pickle
    from cloud.migrate_surfaces_to_periodic_mu1 import (MigrationError,
                                                          migrate_loose_file)
    payload = _legacy_surface_payload()
    surf = payload["surface"]
    rows = np.asarray(surf.mu1_comp1_surface).copy()
    rows[-1] += 1.0
    object.__setattr__(surf, "mu1_comp1_surface", jnp.asarray(rows))
    path = tmp_path / "surface_sf1_30.0_sf2_40.0_sp_50.0_x.pkl"
    path.write_bytes(pickle.dumps(payload))

    with pytest.raises(MigrationError, match="same angle"):
        migrate_loose_file(path)


def test_migration_handles_bundles_without_touching_manifests(tmp_path):
    import gzip
    import pickle
    from cloud.migrate_surfaces_to_periodic_mu1 import migrate_bundle

    bundle = tmp_path / "surface_bundle_chunk_00000_00001.pkl.gz"
    member = "averaged_sf1_30.0_sf2_40.0_sp_50.0.pkl"
    with gzip.open(bundle, "wb") as f:
        pickle.dump({"surfaces": {member: pickle.dumps(_legacy_surface_payload())}}, f)
    manifest = tmp_path / "surface_bundle_chunk_00000_00001.manifest.json"
    manifest.write_text('{"surfaces": ["%s"]}' % member)
    before = manifest.read_text()

    assert migrate_bundle(bundle) == 1
    with gzip.open(bundle, "rb") as f:
        surfaces = pickle.load(f)["surfaces"]
    surface = pickle.loads(surfaces[member])["surface"]
    assert surface.mu1_comp1_surface.shape[0] == mu1_size()
    assert manifest.read_text() == before
    assert migrate_bundle(bundle) == 0  # idempotent


# ---------------------------------------------------------------------------
# 11. Checkpoint metadata: mismatch refusal, both directions
# ---------------------------------------------------------------------------

class _FakeApplyFn:
    """Module-level (hence picklable) stand-in for a checkpoint's apply_fn.

    ``rows="config"`` mimics the real model, whose output row count is read from
    config at call time — which is exactly why a legacy checkpoint would
    silently interpolate rather than emit 181 rows to trim.
    """

    def __init__(self, rows):
        self.rows = rows

    def __call__(self, params, x):
        n = config.mu1_bias_grid_size if self.rows == "config" else self.rows
        return jnp.zeros((jnp.asarray(x).shape[0], n, 5))


def _write_checkpoint(path, rows, *, with_metadata):
    import pickle
    from shared.mu1_axis import GRID_CONVENTION

    data = {"params": {}, "opt_state": {}, "step": 0, "epoch": 1, "loss": 0.0,
            "timestamp": "now", "apply_fn": _FakeApplyFn(rows)}
    if with_metadata:
        data["grid_convention"] = GRID_CONVENTION
        data["mu1_bias_grid_size"] = mu1_size()
    path.write_bytes(pickle.dumps(data))
    return path


def test_legacy_checkpoint_output_is_trimmed_not_resized(tmp_path):
    """No metadata => legacy: run at 181 rows, then trim + renormalise."""
    from shared.utils import load_checkpoint

    path = _write_checkpoint(tmp_path / "legacy.pkl", "config", with_metadata=False)
    state, _ = load_checkpoint(path)
    out = np.asarray(state.apply_fn({}, jnp.ones((2, 3))))
    assert out.shape[1] == mu1_size()
    # Renormalised on the periodic axis.
    assert np.allclose(np.exp(out).sum(axis=1) * mu1_cell_width(), 1.0, atol=1e-5)


def test_checkpoint_metadata_mismatch_is_refused_both_directions(tmp_path):
    from shared.utils import load_checkpoint

    # Declares periodic, emits legacy row count.
    path = _write_checkpoint(tmp_path / "a.pkl", LEGACY_MU1_GRID_SIZE, with_metadata=True)
    state, _ = load_checkpoint(path)
    with pytest.raises(ValueError, match="grid_convention"):
        state.apply_fn({}, jnp.ones((1, 3)))

    # No metadata (so: legacy) but emits the periodic row count.
    path = _write_checkpoint(tmp_path / "b.pkl", mu1_size(), with_metadata=False)
    state, _ = load_checkpoint(path)
    with pytest.raises(ValueError, match="legacy"):
        state.apply_fn({}, jnp.ones((1, 3)))


def test_unknown_grid_convention_is_refused(tmp_path):
    import pickle
    from shared.utils import load_checkpoint

    path = _write_checkpoint(tmp_path / "c.pkl", "config", with_metadata=True)
    data = pickle.loads(path.read_bytes())
    data["grid_convention"] = "something_else"
    path.write_bytes(pickle.dumps(data))
    with pytest.raises(ValueError, match="unknown grid_convention"):
        load_checkpoint(path)
