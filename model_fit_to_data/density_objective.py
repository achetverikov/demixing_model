"""Refuse degenerate empirical density targets before WNM fitting."""

import numpy as np

# CCC is undefined when the empirical target is constant.
DEGENERATE_TARGET_EPS = 1e-10


def degenerate_targets(targets) -> np.ndarray:
    """Mask empirical density-asymmetry curves that are too flat to fit.

    Args:
        targets: ``(n_conditions, n_points)`` empirical density-asymmetry curves.

    Returns:
        ``(n_conditions,)`` bool array, True where the curve is constant.
    """
    return np.var(np.asarray(targets), axis=1) < DEGENERATE_TARGET_EPS


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
