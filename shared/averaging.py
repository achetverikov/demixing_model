"""Core KDE averaging functions shared between the grid pipeline and the
standalone create_averaged_surfaces_from_samples script.

Public API
----------
make_averaging_functions(config, bias_bandwidth, feat_bandwidth)
    Compile and return (vmap_circular, vmap_spatial, feat_diff_steps,
    feat_diff_grid, mu1_bias_grid, mu2_bias_grid, ref_sum).

average_sample_pair(mu1_a, mu2_a, mu1_b, mu2_b, sf1, sf2,
                    vmap_circular, vmap_spatial, feat_diff_steps)
    Combine two in-memory sample arrays and return an AveragedSurface.
    For off-diagonal pairs (sf1 != sf2) pass canonical as _a, mirror as _b.
    For diagonal pairs (sf1 == sf2) pass r0 as _a, r1 as _b.
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde

from shared.utils import AveragedSurface


# ---------------------------------------------------------------------------
# Outlier pre-filter
# ---------------------------------------------------------------------------

@jax.jit
def prefilter_isolated_samples(samples, feat_bin_size=8, bias_bin_size=8,
                                min_neighbors=3):
    """Flag isolated samples as NaN using a 2-D histogram density check.

    Bins the (feat_diff_index × bias_value) space and marks any sample whose
    3×3 bin neighbourhood has fewer than min_neighbors total samples as NaN.
    NaN-flagged samples get zero weight in the KDE.

    Note the bin sizes are in different units: feat_bin_size counts grid steps
    (8 steps = 16° on the 2° grid) while bias_bin_size is in degrees. At grid
    edges/corners the 3×3 neighbour indices are clipped independently, so the
    same boundary bin is counted more than once (up to 4× at a corner) — the
    effective threshold there is laxer than "min_neighbors across 9 distinct
    bins" (MODEL_PIPELINE_FOR_AGENTS.md S3.2b).
    Returns (filtered_samples, n_removed).
    """
    feat_diff_n, sample_n = samples.shape

    feat_coords = jnp.arange(feat_diff_n)[:, None].repeat(sample_n, axis=1).flatten()
    bias_coords = samples.flatten()

    feat_bins = jnp.arange(0, feat_diff_n + feat_bin_size, feat_bin_size)
    bias_bins = jnp.arange(-180, 180 + bias_bin_size, bias_bin_size)

    hist, _, _ = jnp.histogram2d(feat_coords, bias_coords,
                                  bins=[feat_bins, bias_bins])

    def check_sample_neighborhood(feat_idx, sample_idx):
        bias_val = samples[feat_idx, sample_idx]
        feat_bin_idx = feat_idx // feat_bin_size
        bias_bin_idx = jnp.clip(
            (bias_val + 180) // bias_bin_size, 0, hist.shape[1] - 1
        ).astype(jnp.int32)
        neighbor_sum = 0
        for df in [-1, 0, 1]:
            for db in [-1, 0, 1]:
                f_idx = jnp.clip(feat_bin_idx + df, 0, hist.shape[0] - 1)
                b_idx = jnp.clip(bias_bin_idx + db, 0, hist.shape[1] - 1)
                neighbor_sum += hist[f_idx, b_idx]
        return jnp.where(neighbor_sum >= min_neighbors, bias_val, jnp.nan)

    filtered = jax.vmap(
        jax.vmap(check_sample_neighborhood, in_axes=(None, 0)),
        in_axes=(0, None),
    )(jnp.arange(feat_diff_n), jnp.arange(sample_n))

    return filtered, jnp.sum(jnp.isnan(filtered))


# ---------------------------------------------------------------------------
# Weighted KDE
# ---------------------------------------------------------------------------

def _sum_wrapped_pdf(kde, bias_grid):
    kde_sum = (kde.evaluate(bias_grid)
               + kde.evaluate(bias_grid + 360)
               + kde.evaluate(bias_grid - 360))
    epsilon = 1e-10
    return jnp.log(jax.nn.softplus(kde_sum / epsilon) * epsilon)


def weighted_kde_single_step_bounded(i, samples, bias_grid, feat_diff_steps,
                                      bias_bandwidth, feat_bandwidth, ref_sum,
                                      circular=True):
    """Compute weighted KDE log-density for one feat_diff index."""
    current_step = feat_diff_steps[i]
    distances = jnp.abs(feat_diff_steps - current_step)
    weights = jnp.exp(-0.5 * (distances / feat_bandwidth) ** 2)

    max_window_size = 15
    half_window = max_window_size // 2
    start_idx = jnp.maximum(0, i - half_window)

    window_weights = jax.lax.dynamic_slice(weights, (start_idx,), (max_window_size,))
    window_samples = jax.lax.dynamic_slice(
        samples, (start_idx, 0), (max_window_size, samples.shape[1])
    )

    valid_weights = window_weights
    current_sum = jnp.sum(valid_weights)
    scale_factor = jnp.where(current_sum > 0, ref_sum / current_sum, 1.0)
    min_weight = jnp.min(jnp.where(valid_weights > 0, valid_weights, jnp.inf))
    min_weight = jnp.where(jnp.isfinite(min_weight), min_weight, 0.0)
    compensated = min_weight + (valid_weights - min_weight) * scale_factor
    scaled_weights = compensated / jnp.sum(compensated)

    weighted_samples = window_samples.flatten()
    sample_weights = jnp.repeat(scaled_weights, window_samples.shape[1])
    nan_mask = jnp.isnan(weighted_samples)
    sample_weights = jnp.where(nan_mask, 0.0, sample_weights)
    weighted_samples = jnp.where(nan_mask, 0.0, weighted_samples)

    kde = gaussian_kde(weighted_samples, bw_method=bias_bandwidth,
                       weights=sample_weights)
    if circular:
        return _sum_wrapped_pdf(kde, bias_grid)
    return kde.logpdf(bias_grid)


# ---------------------------------------------------------------------------
# Factory: compile vmap functions once per run
# ---------------------------------------------------------------------------

def make_averaging_functions(config, bias_bandwidth: float = 0.075,
                              feat_bandwidth: float = 3.0):
    """Compile and return everything needed to run KDE averaging.

    ``feat_bandwidth`` is the Gaussian SD in feature-difference grid steps,
    not degrees.  With the standard 2-degree grid, the default value of 3.0
    therefore corresponds to a 6-degree SD.

    Returns a dict with keys:
        vmap_circular, vmap_spatial,
        feat_diff_steps, feat_diff_grid, mu1_bias_grid, mu2_bias_grid,
        ref_sum
    """
    feat_diff_grid = config.create_grid('feat_diff')
    mu1_bias_grid = config.create_grid('mu1_bias')
    mu2_bias_grid = config.create_grid('mu2_bias')
    feat_diff_steps = jnp.arange(len(feat_diff_grid))

    # Reference weight sum for boundary compensation
    center = len(feat_diff_steps) // 2
    center_dists = jnp.abs(feat_diff_steps - center)
    center_w = jnp.exp(-0.5 * (center_dists / feat_bandwidth) ** 2)
    ref_sum = jnp.sum(jnp.where(center_w >= 0.01, center_w, 0.0))

    @jax.jit
    def kde_circular(i, samples):
        return weighted_kde_single_step_bounded(
            i, samples, mu1_bias_grid, feat_diff_steps,
            bias_bandwidth, feat_bandwidth, ref_sum, True,
        )

    @jax.jit
    def kde_spatial(i, samples):
        return weighted_kde_single_step_bounded(
            i, samples, mu2_bias_grid, feat_diff_steps,
            bias_bandwidth, feat_bandwidth, ref_sum, False,
        )

    vmap_circular = jax.vmap(kde_circular, in_axes=(0, None))
    vmap_spatial = jax.vmap(kde_spatial, in_axes=(0, None))

    return {
        'vmap_circular': vmap_circular,
        'vmap_spatial': vmap_spatial,
        'feat_diff_steps': feat_diff_steps,
        'feat_diff_grid': feat_diff_grid,
        'mu1_bias_grid': mu1_bias_grid,
        'mu2_bias_grid': mu2_bias_grid,
        'ref_sum': ref_sum,
        'bias_bandwidth': bias_bandwidth,
        'feat_bandwidth': feat_bandwidth,
    }


# ---------------------------------------------------------------------------
# In-memory averaging
# ---------------------------------------------------------------------------

def average_sample_pair(mu1_a, mu2_a, mu1_b, mu2_b,
                        sf1: float, sf2: float,
                        avg_fns: dict) -> AveragedSurface:
    """Combine two in-memory sample arrays into one AveragedSurface.

    For off-diagonal (sf1 < sf2): pass canonical as _a, mirror (sf2,sf1) as _b.
    For diagonal (sf1 == sf2): pass r0 as _a, r1 as _b.

    mu1_* shape: (feat_diff_n, n_simulations, 2)  — two feature components
    mu2_* shape: (feat_diff_n, n_simulations, 2)  — two spatial components
    """
    vmap_circular = avg_fns['vmap_circular']
    vmap_spatial = avg_fns['vmap_spatial']
    feat_diff_steps = avg_fns['feat_diff_steps']
    feat_diff_grid = avg_fns['feat_diff_grid']
    mu1_bias_grid = avg_fns['mu1_bias_grid']
    mu2_bias_grid = avg_fns['mu2_bias_grid']
    bias_bandwidth = avg_fns['bias_bandwidth']
    feat_bandwidth = avg_fns['feat_bandwidth']

    n_expected = len(feat_diff_steps)

    # Drop leading feat_diff=0 row if present (old format compatibility)
    for arr in [mu1_a, mu2_a, mu1_b, mu2_b]:
        if arr is not None and arr.shape[0] > n_expected:
            pass  # handled below

    def _trim(arr):
        return arr[1:] if arr is not None and arr.shape[0] > n_expected else arr

    mu1_a, mu2_a = _trim(mu1_a), _trim(mu2_a)
    mu1_b, mu2_b = _trim(mu1_b), _trim(mu2_b)

    # Combine samples
    if sf1 != sf2:
        # Mirror flips the component order (sf2,sf1) → (sf1,sf2) symmetry
        mu1_combined = jnp.concatenate(
            [mu1_a, jnp.flip(mu1_b, axis=2)], axis=1
        )
    else:
        mu1_combined = jnp.concatenate([mu1_a, mu1_b], axis=1)

    mu2_combined = jnp.concatenate([mu2_a, mu2_b], axis=1)

    # Pre-filter isolated samples
    if sf1 == sf2:
        mu1_all = jnp.concatenate([mu1_combined[:, :, 0],
                                    mu1_combined[:, :, 1]], axis=1)
        mu1_filtered, _ = prefilter_isolated_samples(mu1_all)
        mu1_comp1_surface = vmap_circular(
            jnp.arange(len(feat_diff_steps)), mu1_filtered
        ).T
        mu1_comp2_surface = mu1_comp1_surface
    else:
        comp1_f, _ = prefilter_isolated_samples(mu1_combined[:, :, 0])
        comp2_f, _ = prefilter_isolated_samples(mu1_combined[:, :, 1])
        mu1_comp1_surface = vmap_circular(
            jnp.arange(len(feat_diff_steps)), comp1_f
        ).T
        mu1_comp2_surface = vmap_circular(
            jnp.arange(len(feat_diff_steps)), comp2_f
        ).T

    mu2_flat = mu2_combined.reshape(mu2_combined.shape[0], -1)
    mu2_flat_f, _ = prefilter_isolated_samples(mu2_flat)
    mu2_surface = vmap_spatial(
        jnp.arange(len(feat_diff_steps)), mu2_flat_f
    ).T

    return AveragedSurface(
        feat_diff_grid=feat_diff_grid,
        mu1_bias_grid=mu1_bias_grid,
        mu2_bias_grid=mu2_bias_grid,
        mu1_comp1_surface=mu1_comp1_surface,
        mu1_comp2_surface=mu1_comp2_surface,
        mu2_comp1_surface=mu2_surface,
        mu2_comp2_surface=mu2_surface,
        n_sample_files_used=2,
        kde_parameters={
            'bias_bandwidth': bias_bandwidth,
            'feat_bandwidth': feat_bandwidth,
        },
    )
