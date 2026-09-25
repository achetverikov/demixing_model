"""Empirical curve estimators and bandwidth rules used by maintained fitting.

This module contains data-derived estimators only. It deliberately has no
checkpoint, training, surface-I/O, plotting, or result-path dependencies.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np

KDE_WRAPS = 1
KDE_BW_FLOOR = 1e-6
_SJ_DELMAX = 1000.0  # R's cutoff: skip pair distances beyond sqrt(DELMAX) bandwidths
BIAS_SCALE_SAFETY = 0.5

def silverman_bandwidth(real_bias, floor=KDE_BW_FLOOR):
    """Silverman's rule-of-thumb bandwidth, the default bias-KDE bandwidth.

    Exposed separately so a caller can compute ONE bandwidth over several
    conditions and pass it back in via `kernel_bw`, rather than letting each
    condition pick its own (see `_compute_empirical_density_asymmetry_core`).

    NOT identical to R's `bw.nrd0`: it uses the population SD (jnp.std, ddof=0)
    where R uses the sample SD, so this runs narrower by exactly sqrt(n/(n-1)) --
    2.5% at n=20, 0.02% at n=3000 (measured; a ddof=1 version reproduces bw.nrd0
    to 2e-14). The 1.34 constant does match. The discrepancy is kept deliberately:
    this path exists to reproduce fits made before 2026-08 bit-for-bit, and
    "correcting" it would defeat that. Use `sheather_jones_bandwidth` for a rule
    that is faithful to R.
    """
    real_bias = jnp.asarray(real_bias)
    bias_std = jnp.std(real_bias)
    iqr = jnp.quantile(real_bias, 0.75) - jnp.quantile(real_bias, 0.25)
    bw = 0.9 * jnp.minimum(bias_std, iqr / 1.34) * (len(real_bias) ** (-1 / 5))
    return jnp.maximum(bw, floor)


_SJ_DELMAX = 1000.0  # R's cutoff: skip pair distances beyond sqrt(DELMAX) bandwidths

def _sj_pair_histogram(x, nb=1000):
    """Pair-distance counts on R's binning, so each functional evaluation is O(nb).

    R bins for the same reason -- it is what makes an O(n^2) selector tractable and
    is the whole speed story behind `stats::bw.SJ`. The convention has to be copied
    exactly or the bandwidths disagree by several percent: R assigns each VALUE a bin
    via truncation toward zero (`trunc(abs(x)/d) * sign(x)`, matching the C cast in
    band_den_bin) and then uses the difference of bin indices as the distance --
    which is NOT the same as binning the distances, because the two truncations do
    not compose. Distance counts then come from the autocorrelation of the bin
    counts, as C_bw_den_binned does.

    Returns (bin_width, counts) with counts[k] the number of unordered pairs whose
    bin indices differ by k.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.ptp(x) * 1.01
    if not np.isfinite(rng) or rng <= 0:
        return None, None
    d = rng / nb

    idx = (np.trunc(np.abs(x) / d) * np.sign(x)).astype(np.int64)
    idx -= idx.min()
    bin_counts = np.bincount(idx, minlength=nb + 1).astype(np.float64)

    ac = np.correlate(bin_counts, bin_counts, mode="full")
    counts = ac[len(bin_counts) - 1:][:nb].copy()
    counts[0] = (counts[0] - n) / 2.0   # drop the i == j diagonal, then unorder
    return d, counts

def _sj_functional(counts, d, n, h, order):
    """R's bw_phi4 / bw_phi6: the integrated squared density-derivative functionals.

    phi4(u) = exp(-u^2/2) (u^4 - 6u^2 + 3), phi6(u) = exp(-u^2/2) (u^6 - 15u^4 +
    45u^2 - 15), summed over ordered pairs (off-diagonal twice, plus the diagonal
    term phi4(0) = 3 / phi6(0) = -15).
    """
    delta = (np.arange(len(counts)) * d / h) ** 2
    keep = delta < _SJ_DELMAX
    delta = delta[keep]
    weight = counts[keep]
    if order == 4:
        term = np.exp(-delta / 2) * (delta ** 2 - 6 * delta + 3)
        total = 2 * np.sum(term * weight) + 3 * n
        power = 5.0
    else:
        term = np.exp(-delta / 2) * (delta ** 3 - 15 * delta ** 2 + 45 * delta - 15)
        total = 2 * np.sum(term * weight) - 15 * n
        power = 7.0
    return total / (n * (n - 1) * h ** power * np.sqrt(2 * np.pi))

def sheather_jones_bandwidth(real_bias, nb=1000, fallback=True):
    """Sheather-Jones solve-the-equation bandwidth, a port of `stats::bw.SJ`.

    R's own documentation recommends this over the Silverman default it ships:
    "The default, \"nrd0\", has remained the default for historical and
    compatibility reasons, rather than as a general recommendation, where e.g.
    \"SJ\" would rather fit" (?density). Silverman is the more fragile rule -- it
    takes min(sd, IQR/1.34), which misjudges heavy-tailed or multimodal error
    distributions; on one moors cell it came out 5.1x the SJ value.

    scipy has no SJ, hence the port. Constants, the DELMAX cutoff, the pair
    binning and the bracket expansion all follow R's implementation so the two
    agree; verified against `stats::bw.SJ` on real data.

    Args:
        real_bias: samples (a single condition's, or pooled across conditions).
        nb: pair-distance bins, as in R.
        fallback: if True, fall back to Silverman when the sample is too sparse for
            SJ to have a solution. R raises instead; a fitting pipeline should not
            die on one degenerate cell.
    """
    x = np.asarray(real_bias, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 data points")

    scale = min(np.std(x, ddof=1), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349)
    d, counts = _sj_pair_histogram(x, nb=nb)

    def sd_h(h):
        return _sj_functional(counts, d, n, h, order=4)

    def td_h(h):
        return _sj_functional(counts, d, n, h, order=6)

    def _fallback(reason):
        if not fallback:
            raise RuntimeError(f"bw.SJ failed: {reason}")
        warnings.warn(f"Sheather-Jones bandwidth failed ({reason}); "
                      "falling back to Silverman", RuntimeWarning)
        return float(silverman_bandwidth(x))

    if d is None or scale <= 0:
        return _fallback("degenerate sample")

    td = -td_h(1.23 * scale * n ** (-1 / 9))
    if not np.isfinite(td) or td <= 0:
        return _fallback("sample is too sparse to find TD")
    alph2 = 1.357 * (sd_h(1.24 * scale * n ** (-1 / 7)) / td) ** (1 / 7)
    if not np.isfinite(alph2):
        return _fallback("sample is too sparse to find alph2")

    c1 = 1 / (2 * np.sqrt(np.pi) * n)

    def f_sd(h):
        return (c1 / sd_h(alph2 * h ** (5 / 7))) ** (1 / 5) - h

    hmax = 1.144 * scale * n ** (-1 / 5)
    lower, upper = 0.1 * hmax, hmax
    for itry in range(1, 100):
        try:
            if f_sd(lower) * f_sd(upper) <= 0:
                break
        except (FloatingPointError, ValueError):
            return _fallback("functional evaluation failed")
        if itry % 2:
            upper *= 1.2
        else:
            lower /= 1.2
    else:
        return _fallback("no solution in the specified range of bandwidths")

    from scipy.optimize import brentq
    return float(brentq(f_sd, lower, upper, xtol=0.1 * 0.1 * hmax))


BIAS_SCALE_SAFETY = 0.5
"""Scaled bias values must stay within this fraction of the half-period.

The rescaling below turns the circular axis into an effectively linear one, which
is only legitimate while no mass is near the wrap. 0.5 keeps the largest scaled
error at most halfway to the antipode.
"""

def rescale_bias_for_grid(real_bias, kernel_bw, dx, max_diss,
                          safety=BIAS_SCALE_SAFETY):
    """Widen a too-narrow bias distribution instead of over-smoothing it.

    The density asymmetry is a signed MASS difference, and scaling every error by
    the same positive factor preserves each error's sign -- so the estimand is
    unchanged while the bandwidth scales with the data, lifting the kernel above
    the grid it has to be sampled on. Scaling the data by k against a fixed grid
    is exactly equivalent to evaluating on a grid k times finer; this is the
    cheaper way to get that resolution, since the grid is fixed by the model
    surfaces.

    Measured on real moors cells against the bandwidth the data actually support:
    scaling lands within 0.91% of it (median, max 1.94%), where flooring the
    bandwidth instead leaves 4.53% median and 38.8% max.

    The catch is that scaling is a LINEAR operation on a CIRCULAR axis: valid only
    while the scaled data stay clear of the wrap. Narrow distributions -- the ones
    that need this -- are far from it, but motion-direction data carry a
    180-degree-off reversal mode sitting exactly there, so the headroom is checked
    rather than assumed and the factor is capped to preserve it. Where the cap
    binds, the caller's bandwidth floor still applies.

    Returns:
        (scaled_bias, scaled_bandwidth, scale)
    """
    real_bias = jnp.asarray(real_bias)
    max_abs = jnp.max(jnp.abs(real_bias))
    wanted = dx / jnp.maximum(kernel_bw, 1e-12)             # lifts the kernel to one cell
    allowed = (safety * max_diss) / jnp.maximum(max_abs, 1e-12)
    scale = jnp.maximum(jnp.minimum(wanted, allowed), 1.0)
    return real_bias * scale, kernel_bw * scale, scale

def _compute_empirical_density_asymmetry_core(real_feat_diff, real_bias, feat_diff_grid, weights_sd=20, circ_space=360,
                                              kernel_bw=None):
    """
    Core computation for empirical density asymmetry with all matrix operations.

    Args:
        real_feat_diff: Array of feature difference values, shape (n_trials,)
        real_bias: Array of bias values, shape (n_trials,)
        feat_diff_grid: Grid of feature difference values to compute asymmetry at
        weights_sd: Standard deviation for Gaussian weights in feat_diff dimension
        circ_space: Circular space size (360 for full circle)
        kernel_bw: Bias-KDE bandwidth. None (default) estimates it from THIS call's
            trials via Sheather-Jones, so each condition gets its own smoothing --
            and conditions differ in error spread by construction, which is the axis
            the experiment manipulates. Pass an explicit value to share one bandwidth
            across conditions (`silverman_bandwidth` over the pooled trials, or the
            mean of the per-condition estimates).

    Returns:
        distances: feat_diff_grid values
        asymmetry_values: Asymmetry values at each grid point

    Notes:
        The bias KDE is WRAPPED (see KDE_WRAPS): the axis is a circle, so a kernel
        centred near +max_diss reappears just past -max_diss rather than being
        truncated. On orientation/colour data this is a no-op to ~1e-13 because
        errors sit near 0 and the bandwidth is a few degrees, but motion-direction
        data routinely carry a second mode of 180-degree-off reversals sitting
        exactly on the boundary, where truncation both loses mass and mis-signs it.

        Each trial's kernel carries unit mass, so the returned signed difference is
        a proportion of a unit-mass distribution. circhelp divides instead by
        (P+ + P-), excluding the sign-ambiguous 0-cell -- a different denominator
        convention, applied consistently on our model and target sides alike.
    """
    max_diss = circ_space / 2

    # Bandwidth: Sheather-Jones over this call's trials unless the caller supplies
    # one. SJ rather than Silverman so this library default matches the BBZ twin
    # (_empirical_density_asymmetry_np) -- two defaults is how the next divergence
    # starts. The fitter never relies on it; it always passes an explicit value.
    if kernel_bw is None:
        kernel_bw = sheather_jones_bandwidth(np.asarray(real_bias))
    kernel_bw = jnp.maximum(jnp.asarray(kernel_bw), KDE_BW_FLOOR)


    # Use provided feat_diff_grid or create default distances
    if feat_diff_grid is not None:
        distances = feat_diff_grid
    else:
        distances = jnp.arange(4, int(max_diss * 2) + 1, 4)  # [4, 8, ..., circ_space]
    # 180 points spanning [-max_diss, +max_diss): step 2° for circ_space=360 (the
    # model-space default used by the fitter), step 1° for circ_space=180.  The
    # axis is circular, so it is half-open — +max_diss is the same angle as
    # -max_diss and must not occupy a second cell (it used to, via a 181-point
    # inclusive linspace, mirroring the model-side defect).
    n_bias_points = 180
    dx_empirical = (2 * max_diss) / n_bias_points
    bias_range = -max_diss + dx_empirical * jnp.arange(n_bias_points)

    # RESOLUTION. A kernel narrower than a cell falls BETWEEN grid points and the
    # rectangle rule stops representing the density: at bw = dx/20 a trial centred
    # on a cell contributes 7.98x its mass and one half a cell away contributes
    # 0.0000, so the target would follow where trials sit within a cell rather than
    # the data. Sheather-Jones reaches bw = 0.09 model degrees on real moors cells
    # (bw/dx = 0.046), where the target was wrong by 278% of curve range.
    #
    # Rather than floor the bandwidth -- which fixes the arithmetic by deliberately
    # over-smoothing, biasing the estimate on exactly the cells that triggered it --
    # widen the distribution to fit the grid. Sign is preserved, so the signed mass
    # difference is unchanged. See rescale_bias_for_grid. The floor stays as the
    # backstop for when the wrap guard caps the scaling.
    real_bias, kernel_bw, _bias_scale = rescale_bias_for_grid(
        real_bias, kernel_bw, dx_empirical, max_diss)
    kernel_bw = jnp.maximum(kernel_bw, dx_empirical / 2.0)
    
    # Vectorized weight matrix computation with logsumexp for numerical stability
    dist_matrix = distances[:, None] - real_feat_diff[None, :]  # Broadcasting
    log_weights = -0.5 * (dist_matrix / weights_sd) ** 2
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights, axis=1, keepdims=True)
    weight_matrix = jnp.exp(log_weights_normalized)
    
    # Vectorized kernel matrix computation: (n_bias_points, n_datapoints), summed
    # over KDE_WRAPS periodic images so the kernel closes around the circle.
    bias_matrix = bias_range[:, None] - real_bias[None, :]  # Broadcasting
    offsets = circ_space * jnp.arange(-KDE_WRAPS, KDE_WRAPS + 1, dtype=bias_matrix.dtype)
    kernel_matrix = jnp.sum(
        jnp.exp(-0.5 * ((bias_matrix[None, :, :] + offsets[:, None, None]) / kernel_bw) ** 2),
        axis=0)
    kernel_matrix = kernel_matrix / (kernel_bw * jnp.sqrt(2 * jnp.pi))  # Normalize kernel
    
    # Matrix multiplication: (n_distances, n_bias_points)
    # weight_matrix @ kernel_matrix.T gives density for each (distance, bias_point) combination
    density_matrix = weight_matrix @ kernel_matrix.T
    
    # Split into positive and negative regions for asymmetry computation.  The
    # antipode (-max_diss) is as sign-ambiguous as 0 and is excluded from both
    # masks, matching the model side (compute_single_density_asymmetry).
    pos_mask = bias_range > 0
    neg_mask = (bias_range < 0) & (bias_range > -max_diss)

    # Sum densities in positive and negative regions for each distance.
    # dx_empirical (the periodic cell width) is set with the grid above.

    # Use jnp.where instead of boolean indexing to avoid concreteness issues
    pos_densities_matrix = jnp.where(pos_mask[None, :], density_matrix, 0.0)
    neg_densities_matrix = jnp.where(neg_mask[None, :], density_matrix, 0.0)
    
    pos_densities = jnp.sum(pos_densities_matrix, axis=1) * dx_empirical  # Shape: (n_distances,)
    neg_densities = jnp.sum(neg_densities_matrix, axis=1) * dx_empirical  # Shape: (n_distances,)
    
    asymmetry_values = pos_densities - neg_densities
    return distances, asymmetry_values

def compute_target_bias_rolling_curve_core(real_feat_diff, real_bias, feat_diff_grid, weights_sd=20):
    """
    Gaussian-weighted (rolling) circular-mean bias curve over feat_diff_grid.

    Same feat_diff-space Gaussian weighting as _compute_empirical_density_asymmetry_core,
    but normalized to sum to 1 over trials (a true kernel-weighted average) and applied to
    cos/sin of bias instead of splitting into positive/negative asymmetry.

    Args:
        real_feat_diff: Array of feature difference values, shape (n_trials,)
        real_bias: Array of bias values in degrees, shape (n_trials,)
        feat_diff_grid: Grid of feature difference values to compute the curve at
        weights_sd: Standard deviation for Gaussian weights in feat_diff dimension

    Returns:
        target_bias_curve: Circular mean bias in degrees at each feat_diff_grid point
    """
    dist_matrix = feat_diff_grid[:, None] - real_feat_diff[None, :]  # (n_grid, n_trials)
    log_weights = -0.5 * (dist_matrix / weights_sd) ** 2
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights, axis=1, keepdims=True)
    weight_matrix = jnp.exp(log_weights_normalized)  # each row sums to 1

    bias_rad = jnp.radians(real_bias)
    cos_curve = weight_matrix @ jnp.cos(bias_rad)
    sin_curve = weight_matrix @ jnp.sin(bias_rad)

    return jnp.degrees(jnp.arctan2(sin_curve, cos_curve))

