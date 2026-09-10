"""Empirical fitting targets, built once from data and shared by every engine.

Every objective compares a model curve against something derived from the
subject's actual trials, and that something is what this module builds: the
binned circular means for ``expectation``, the rolling-mean curve for
``smoothed_exp``, the wrapped-KDE signed-mass curve for ``density``, and the
empirical bias distributions the CRPS variants score against.

It used to live inside the surface optimizer's ``_precompute_target_curves``,
which meant a second search engine could only reach it by constructing that
optimizer -- and constructing it means loading a surface checkpoint. Targets do
not depend on any surrogate, so requiring one to build them would have tied a
future engine to the backend it is meant to replace. Nothing here loads a model.

The definitions are deliberately *not* revised in the move. They are the
contracts the deployed results were produced under, recorded in the transition
plan's target inventory, and a change to any of them changes what a fit means,
not just how fast it is reached. ``tests/test_fitting_targets.py`` holds them to
a reference recorded from the optimizer before this module existed.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from shared.config import config
from shared.utils import (_compute_empirical_density_asymmetry_core,
                          compute_target_bias_rolling_curve_core,
                          sheather_jones_bandwidth, silverman_bandwidth)


@dataclass(frozen=True)
class FittingTargets:
    """Everything the objectives need from the data, and nothing from a model.

    Attributes:
        condition_names: condition order every array's first axis follows.
        feat_indices: feature-grid positions the binned bias target occupies.
        target_bias: ``(n_conditions, n_bins)`` circular mean per bin.
        bias_weights: ``(n_conditions, n_bins)`` trial counts per bin.
        target_density: ``(n_conditions, n_feat)`` empirical signed-mass curve.
        target_bias_curve: ``(n_conditions, n_feat)`` rolling circular mean.
        target_d: ``(n_conditions, n_feat, n_bias)`` expected circular distance
            from each bias bin to the empirical distribution.
        fd_weights: ``(n_conditions, n_feat)`` binary support mask, balanced CRPS.
        bias_fd_weights: ``(n_conditions, n_feat)`` squared smoothed-bias weights,
            bias-weighted CRPS.
        density_target_var: variance of each density target curve.
        density_degenerate: which conditions have a density target too flat to fit.
        density_bandwidth: the bias-KDE bandwidth actually used per condition.
        near_constant_warnings: conditions whose density target is defined but
            numerically unstable under CCC, reported rather than refused.
    """

    condition_names: tuple
    feat_indices: jnp.ndarray
    target_bias: jnp.ndarray
    bias_weights: jnp.ndarray
    target_density: jnp.ndarray
    target_bias_curve: jnp.ndarray
    target_d: jnp.ndarray
    fd_weights: jnp.ndarray
    bias_fd_weights: jnp.ndarray
    density_target_var: np.ndarray
    density_degenerate: np.ndarray
    density_bandwidth: tuple
    feature_operator: jnp.ndarray
    matched_density_target: jnp.ndarray
    matched_density_degenerate: np.ndarray
    near_constant_warnings: tuple


def resolve_density_bandwidths(bias_by_condition, rule: str, mode: str):
    """Resolve the bias-KDE bandwidth for each condition.

    The bandwidth is always resolved here and passed explicitly, never left to
    the KDE core's default. Relying on the default meant ``'silverman'`` silently
    tracked whatever that default happened to be, so flipping it -- as the SJ
    adoption did -- would have changed the one mode whose entire purpose is to
    reproduce pre-2026-08 fits.

    ``'per_condition'`` lets each condition estimate its own bandwidth, but
    conditions differ in error spread by construction, so the target's smoothing
    then varies along the very axis the experiment manipulates (measured up to
    3.29x within one csh2026 subject). ``'pooled'`` and ``'average'`` share one
    bandwidth across the subject's conditions, matching what circhelp does within
    a single ``density_asymmetry`` call. Production is pooled SJ.
    """
    if rule == 'silverman':
        estimate_bw = silverman_bandwidth
    elif rule == 'sj':
        estimate_bw = sheather_jones_bandwidth
    else:
        raise ValueError(
            f"unknown density_bandwidth_rule {rule!r}; expected 'silverman' or 'sj'")

    if mode == 'pooled':
        shared = estimate_bw(jnp.concatenate(list(bias_by_condition)))
        return [shared] * len(bias_by_condition)
    if mode == 'average':
        shared = jnp.mean(jnp.stack([jnp.asarray(estimate_bw(b)) for b in bias_by_condition]))
        return [shared] * len(bias_by_condition)
    if mode == 'per_condition':
        return [estimate_bw(b) for b in bias_by_condition]
    raise ValueError(
        f"unknown density_bandwidth_mode {mode!r}; "
        "expected 'per_condition', 'pooled' or 'average'")


def build_fitting_targets(condition_datasets, feat_diff_grid, d_circ_matrix,
                          n_mu1_bias: int, emp_density_weights_sd: float,
                          density_bandwidth_rule: str, density_bandwidth_mode: str,
                          degenerate_targets, bwcrps_condition_targets,
                          target_bias_curve_core) -> FittingTargets:
    """Build every empirical target from one pass over the conditions' trials.

    Args:
        condition_datasets: ordered mapping ``{condition_name: (n_trials, 2)}``
            of ``[feat_diff, bias]`` in model degrees. Unpadded and real.
        feat_diff_grid: the feature-difference grid targets are defined on.
        d_circ_matrix: circular distance matrix over the bias grid, for CRPS.
        n_mu1_bias: number of bias bins.
        emp_density_weights_sd: Gaussian feature-weight SD, in model degrees.
        density_bandwidth_rule: ``'sj'`` or ``'silverman'``.
        density_bandwidth_mode: ``'pooled'``, ``'average'`` or ``'per_condition'``.
        degenerate_targets: predicate marking density targets too flat to fit.
        bwcrps_condition_targets: the CRPS-variant target builder.
        target_bias_curve_core: the binned circular-mean core for ``expectation``.

    Those three are injected rather than imported. Two of them live in the
    optimizer module, which imports this one, so importing them back would be
    circular; and moving them would break the cross-repository BWCRPS and
    density-target parity tests that import them at their current paths. The
    definitions stay with the objectives that own them.

    Returns:
        :class:`FittingTargets`.
    """
    condition_names = tuple(condition_datasets)
    dataframes = [jnp.asarray(condition_datasets[name]) for name in condition_names]
    if not dataframes:
        raise ValueError("no conditions given; targets cannot be built from nothing")

    all_feat_diff = [df[:, 0] for df in dataframes]
    all_bias_values = [df[:, 1] for df in dataframes]

    # The binned bias core is vmapped, so conditions are padded to a common
    # width with the repeated last observation and the real count is passed
    # alongside; the core masks on it.
    max_trials = max(len(vals) for vals in all_feat_diff)
    padded_feat_diff = jnp.stack([
        jnp.pad(vals, (0, max_trials - len(vals)),
                constant_values=vals[-1] if len(vals) > 0 else 0)
        for vals in all_feat_diff])
    padded_bias_values = jnp.stack([
        jnp.pad(vals, (0, max_trials - len(vals)),
                constant_values=vals[-1] if len(vals) > 0 else 0)
        for vals in all_bias_values])
    trial_counts = jnp.array([len(vals) for vals in all_feat_diff])

    vectorized_bias_compute = jax.vmap(target_bias_curve_core, in_axes=(0, 0, 0))
    feat_indices, target_bias, bias_weights = vectorized_bias_compute(
        padded_feat_diff, padded_bias_values, trial_counts)

    # Everything below runs over the REAL, unpadded arrays. The density core has
    # no trial-count mask, so vmapping it over the rectangular padded arrays
    # would feed the repeated last-observation rows into the empirical density --
    # both as a mass clump at the last trial's coordinates and through the
    # std/quantile/n bandwidth terms. This is a one-time precompute, not part of
    # the JIT objective, so looping costs nothing. See codex_audit.md #3.
    bandwidths = resolve_density_bandwidths(all_bias_values, density_bandwidth_rule,
                                            density_bandwidth_mode)

    def feature_operator(feat_diff_vals):
        feat = np.asarray(feat_diff_vals)
        grid = np.asarray(feat_diff_grid)
        weights = np.exp(-0.5 * ((grid[:, None] - feat[None, :])
                                 / emp_density_weights_sd) ** 2)
        weights /= weights.sum(axis=1, keepdims=True)
        indices = np.rint((feat - grid[0]) / config.feat_diff_step).astype(int)
        # Match the existing surface trial-index contract. Production fitting
        # filters into the model domain before target construction, but small
        # simulation/plotting fixtures historically pass 0-degree dummy rows.
        # Those rows clamp to the nearest surface column; rejecting them here
        # would make a new WNM-only target field break unchanged surface paths.
        indices = np.clip(indices, 0, len(grid) - 1)
        assignment = np.zeros((len(feat), len(grid)))
        assignment[np.arange(len(feat)), indices] = 1.0
        return weights, weights @ assignment

    operators = []
    matched_density = []
    for values, bandwidth in zip(dataframes, bandwidths):
        trial_weights, operator = feature_operator(values[:, 0])
        bias = np.asarray(values[:, 1])
        shifts = np.arange(-8, 9) * 360.0
        def arc_probability(low, high):
            return np.sum(
                ndtr((high + shifts[:, None] - bias[None, :]) / bandwidth)
                - ndtr((low + shifts[:, None] - bias[None, :]) / bandwidth), axis=0)
        soft_sign = arc_probability(0.0, 180.0) - arc_probability(-180.0, 0.0)
        operators.append(operator)
        matched_density.append(trial_weights @ soft_sign)

    def density_curve(feat_diff_vals, bias_vals, kernel_bw):
        _, asymmetry_values = _compute_empirical_density_asymmetry_core(
            feat_diff_vals, bias_vals, feat_diff_grid,
            weights_sd=emp_density_weights_sd, kernel_bw=kernel_bw)
        return asymmetry_values

    target_density = jnp.stack([
        density_curve(dataframes[i][:, 0], dataframes[i][:, 1], bandwidths[i])
        for i in range(len(condition_names))])

    # Same curve-vs-curve idea as the density asymmetry curve, but for the
    # circular-mean bias itself instead of hard 8-degree bins.
    target_bias_curve = jnp.stack([
        compute_target_bias_rolling_curve_core(dataframes[i][:, 0], dataframes[i][:, 1],
                                               feat_diff_grid,
                                               weights_sd=emp_density_weights_sd)
        for i in range(len(condition_names))])

    # Degenerate density targets are decided once, from the target alone, and
    # recorded rather than acted on: the refusal is scoped to the density
    # objectives, because a flat density target says nothing about whether the
    # same condition can be fit by likelihood or CRPS.
    density_target_var = np.var(np.asarray(target_density), axis=1)
    density_degenerate = degenerate_targets(target_density)

    warnings = []
    for i, name in enumerate(condition_names):
        # Near-constant targets are defined but numerically unstable under CCC.
        # Warn; the refusal is at eps and nowhere else. Scale-relative, so this
        # does not fire on a genuinely small but well-resolved curve.
        scale = float(np.max(np.abs(np.asarray(target_density)[i])))
        if (not density_degenerate[i] and scale > 0
                and np.sqrt(density_target_var[i]) < 1e-3 * scale):
            warnings.append(
                f"  WARNING: {name} density target is near-constant "
                f"(sd={np.sqrt(density_target_var[i]):.2e} vs max|curve|={scale:.2e}); "
                "CCC is unstable here — treat its density fit with suspicion.")

    # Empirical bias distributions shared by balanced_crps and bias_weighted_crps.
    # Q[c, j, k] is the Gaussian-weighted empirical probability of bias bin k at
    # feature grid point j; target_d[c, j, k] = sum_l D_circ[k, l] Q[c, j, l], the
    # expected circular distance from bin k to the empirical distribution.
    # Precomputed so the JIT branch is cheap.
    d_np = np.asarray(d_circ_matrix)
    fd_grid_np = np.asarray(feat_diff_grid)
    target_d_list, fd_weights_list, fd_bias_weights_list = [], [], []
    for name in condition_names:
        dataset_np = np.asarray(condition_datasets[name])
        target_d, support_mask, weights = bwcrps_condition_targets(
            dataset_np[:, 0], dataset_np[:, 1], fd_grid_np, d_np,
            emp_density_weights_sd, config.mu1_bias_range[0], config.mu1_bias_step,
            n_mu1_bias)
        target_d_list.append(target_d)
        fd_weights_list.append(support_mask)
        fd_bias_weights_list.append(weights)

    return FittingTargets(
        condition_names=condition_names,
        feat_indices=feat_indices[0],  # identical across conditions by construction
        target_bias=target_bias,
        bias_weights=bias_weights,
        target_density=target_density,
        target_bias_curve=target_bias_curve,
        target_d=jnp.array(np.stack(target_d_list)),
        fd_weights=jnp.array(np.stack(fd_weights_list)),
        bias_fd_weights=jnp.array(np.stack(fd_bias_weights_list)),
        density_target_var=density_target_var,
        density_degenerate=density_degenerate,
        density_bandwidth=tuple(float(b) for b in bandwidths),
        feature_operator=jnp.asarray(np.stack(operators), dtype=jnp.float32),
        matched_density_target=jnp.asarray(np.stack(matched_density), dtype=jnp.float32),
        matched_density_degenerate=np.var(np.stack(matched_density), axis=1) < 1e-10,
        near_constant_warnings=tuple(warnings),
    )
