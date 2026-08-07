"""The density objective, defined once for every backend that computes it.

Two search backends score the same objective by different routes: the
hierarchical path generates 3-D log surfaces and collapses each to a curve inside
the JIT, while the exhaustive path reads curves that were collapsed when the
cache was built and scores a whole slab with one matrix product.  Those are
different computations of the same number, so without a single definition they
drift -- and a drift here is invisible: both sides keep producing plausible
losses, and only the ranking changes.

So the definition lives here, in two forms that are held equal by
``tests/test_density_objective_parity.py``:

  ``ccc_loss``            elementwise, one curve at a time; the reference.
  ``ccc_loss_batched``    the same value from precomputed per-curve statistics
                          and a single ``predicted @ centered_target`` product;
                          what a cache-backed scan uses.

Both are ``1 - CCC``.  Nothing here knows about surfaces, caches, or grids.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np

#: Below this variance a density-asymmetry target curve is constant, and the
#: density objective refuses to run against it (user decision, 2026-08-07).
#:
#: CCC is undefined against a constant target -- its denominator collapses with
#: its numerator -- and the value the formula returns there is actively wrong:
#: two identical constant curves score 1, the WORST possible loss, for a perfect
#: match.  The alternative to raising, silently dropping the offending condition,
#: was rejected: a condition vanishing from a group's total leaves nothing
#: downstream can act on, the group's shared parameters would then be fitted to a
#: different set of conditions than the run reports, and a constant target is far
#: more likely to mean something is wrong with the data or the target
#: construction than to be a legitimate input.
#:
#: The test is on the target alone -- not the prediction, not the pair -- so it
#: cannot depend on the parameters being tried and cannot fire partway through a
#: search.  A constant *prediction* against a varying target is a different case
#: and stays scoreable: it earns loss 1 legitimately.
#:
#: Defensive in practice: measured 2026-08-07 over the 681 empirical target curves
#: in ``results/observer_models/*/density_asymmetry_curves.parquet``, the smallest
#: variance in any dataset is 2.2e-04 -- six orders of magnitude above this.
DEGENERATE_TARGET_EPS = 1e-10


def degenerate_targets(targets) -> np.ndarray:
    """Boolean mask of target curves too flat for the density objective.

    Args:
        targets: ``(n_conditions, n_points)`` empirical density-asymmetry curves.

    Returns:
        ``(n_conditions,)`` bool array, True where the curve is constant.
    """
    return np.var(np.asarray(targets), axis=1) < DEGENERATE_TARGET_EPS


def check_targets_fittable(targets, condition_names: Sequence[str], objective: str) -> None:
    """Raise if any target curve is constant. See `DEGENERATE_TARGET_EPS`.

    Call before scoring, from every backend, on the target alone.

    Raises:
        ValueError: naming every offending condition and its measured variance.
    """
    variances = np.var(np.asarray(targets), axis=1)
    degenerate = variances < DEGENERATE_TARGET_EPS
    if not degenerate.any():
        return
    offenders = ", ".join(
        f"{condition_names[i]} (var={variances[i]:.2e})" for i in np.flatnonzero(degenerate)
    )
    raise ValueError(
        f"'{objective}' cannot be fitted: the empirical density-asymmetry target is constant "
        f"(var < {DEGENERATE_TARGET_EPS:.0e}) for {int(degenerate.sum())} of "
        f"{len(variances)} conditions: {offenders}. A constant target carries no information "
        "to fit and CCC is undefined against it, returning its worst value for a perfect "
        "match. Check the trial data and the density-target construction for these "
        "conditions; fit them with a non-density objective, or drop them from the input "
        "explicitly rather than letting a fit silently omit them. (Dropping a condition also "
        "changes the pooled KDE bandwidth, and so the target curves of every OTHER condition "
        "for that subject -- see density_bandwidth_mode.)"
    )


def _safe_denominator(D):
    """Guard the CCC denominator without altering any value that can reach it.

    Unreachable for a non-degenerate target: every scored curve has
    ``D >= var_target >= DEGENERATE_TARGET_EPS``, because `check_targets_fittable`
    ran first.  Kept anyway, and deliberately at the *same* epsilon: a guard set
    looser than the refusal would leave a live band of curves that pass the
    refusal and then silently get their denominator altered, which is exactly the
    distortion the refusal exists to remove.

    ``where`` rather than ``maximum`` so both branches are finite -- dividing by a
    raw ``D`` that can be 0 produces inf/NaN in the dead branch and poisons any
    gradient taken through it.
    """
    return jnp.where(D < DEGENERATE_TARGET_EPS, 1.0, D)


def ccc_loss(predicted, target):
    """``1 - CCC`` between each predicted curve and one target. The reference form.

    ``1 - CCC = MSE / D`` with ``D = var_p + var_t + (mean_p - mean_t)^2``, and
    ``CCC = r * C_b``, so the loss scores shape (precision ``r``) and amplitude
    and offset (accuracy ``C_b``) together.

    Args:
        predicted: ``(n_curves, n_points)`` or ``(n_points,)``.
        target: ``(n_points,)``.

    Returns:
        ``(n_curves,)`` losses, or a scalar for a 1-D input.
    """
    predicted = jnp.asarray(predicted)
    target = jnp.asarray(target)
    scalar = predicted.ndim == 1
    predicted = jnp.atleast_2d(predicted)

    pred_mean = jnp.mean(predicted, axis=1)
    target_mean = jnp.mean(target)
    pred_centered = predicted - pred_mean[:, None]
    target_centered = target - target_mean

    covariance = jnp.mean(pred_centered * target_centered[None, :], axis=1)
    D = (jnp.mean(pred_centered ** 2, axis=1) + jnp.mean(target_centered ** 2)
         + (pred_mean - target_mean) ** 2)
    losses = 1.0 - 2.0 * covariance / _safe_denominator(D)
    return losses[0] if scalar else losses


def target_statistics(target):
    """Precompute the per-target quantities a batched scan reuses.

    Returns:
        ``(centered_target, mean, var)``.
    """
    target = jnp.asarray(target)
    mean = jnp.mean(target)
    centered = target - mean
    return centered, mean, jnp.mean(centered ** 2)


def centered_curves(predicted):
    """Split curves into ``(centered, mean)`` for the batched scorer.

    A cache should store the centered curves and their means, not the raw curves:
    then a scan gets `ccc_loss_batched`'s required centering for free, at no
    cost per candidate.
    """
    predicted = jnp.asarray(predicted)
    mean = jnp.mean(predicted, axis=-1)
    return predicted - mean[..., None], mean


def ccc_loss_batched(centered_dot, pred_mean, pred_var,
                     target_mean, target_var, n_points: int):
    """``1 - CCC`` from precomputed statistics. Same value as `ccc_loss`.

    Written for a cache-backed scan: the caller computes ``centered_dot =
    centered_curves @ centered_target`` for a whole slab with one GEMM, and reads
    ``pred_mean``/``pred_var`` from the cache instead of recomputing them.

    **Both sides of the dot product must be centered.** Centering only the target
    is algebraically identical -- ``(P @ tc)/n == (P @ t)/n - mean_p*mean_t`` --
    but not numerically: the summands are then of order ``mean_p * |tc|`` while
    the covariance they must produce is of order ``sd_p * sd_t``, so in float32
    the result is a difference of much larger numbers. Measured on a curve with
    mean 50 and sd 0.05, that form returned -0.0268 for a loss whose true value
    is 0.0 -- a 2.7e-2 absolute error on a loss in [0, 2], and negative, which
    ``1 - CCC`` cannot legitimately be for identical curves. Centering both sides
    removes the cancellation at its source.

    Args:
        centered_dot: ``(n_curves,)`` dot products of the **centered** curves with
            the **centered** target (see `centered_curves`, `target_statistics`).
        pred_mean, pred_var: ``(n_curves,)`` per-curve mean and variance. For a
            compressed cache these must describe the **reconstructed** curve, or
            the dot product and the statistics describe different curves.
        target_mean, target_var: scalars from `target_statistics`.
        n_points: curve length; must come from the cache manifest rather than a
            literal, since it follows from the feat_diff grid.

    Returns:
        ``(n_curves,)`` losses.
    """
    covariance = centered_dot / n_points
    D = pred_var + target_var + (pred_mean - target_mean) ** 2
    return 1.0 - 2.0 * covariance / _safe_denominator(D)


def ccc_components(predicted, target) -> dict:
    """Decompose ``CCC = r * C_b`` for reporting. NaN where a factor is undefined.

    ``r`` is precision (does the curve have the right shape?), ``C_b`` accuracy
    (the right amplitude and offset?).  Splitting them is what makes an amplitude
    failure visible: a curve 10x too small can still have ``r`` near 1.

    Returns NaN for ``r`` and ``C_b`` when either variance is 0 -- they are
    genuinely undefined there, and 0 would be a plausible-looking lie (a real
    measurement of "no correlation", or of maximal scale mismatch).
    """
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    nan_result = {'ccc': np.nan, 'r': np.nan, 'C_b': np.nan}
    if predicted.size != target.size or predicted.size < 2:
        return dict(nan_result)

    pred_mean, target_mean = predicted.mean(), target.mean()
    pred_var, target_var = predicted.var(), target.var()
    covariance = ((predicted - pred_mean) * (target - target_mean)).mean()
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2

    ccc = np.nan if denominator <= 0 else 2 * covariance / denominator
    if pred_var <= 0 or target_var <= 0:
        return {'ccc': float(ccc), 'r': np.nan, 'C_b': np.nan}

    r = covariance / np.sqrt(pred_var * target_var)
    # C_b written out rather than as CCC/r, so it stays finite as r approaches 0.
    scale_ratio = np.sqrt(pred_var / target_var)
    location_shift = (pred_mean - target_mean) / np.sqrt(np.sqrt(pred_var * target_var))
    C_b = 2.0 / (scale_ratio + 1.0 / scale_ratio + location_shift ** 2)
    return {'ccc': float(ccc), 'r': float(r), 'C_b': float(C_b)}
