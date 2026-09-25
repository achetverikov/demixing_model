"""Training-time evaluation helpers for wrapped-normal-mixture surrogates."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from shared import wnm as wm

CASE_BLOCK = 32
SAMPLE_CHUNK = 5000


def empirical_moments(bias: np.ndarray) -> dict:
    """Finite-aware circular first moment and derived summary statistics."""
    b = np.asarray(bias, dtype=np.float64)
    finite = np.isfinite(b)
    z = np.where(finite, np.exp(1j * np.radians(np.where(finite, b, 0.0))), 0.0)
    count = finite.sum(axis=-1)
    m1 = z.sum(axis=-1) / np.maximum(count, 1)
    m1 = np.where(count > 0, m1, np.nan + 1j * np.nan)
    r = np.clip(np.abs(m1), 1e-12, 1.0)
    return {
        "mean_bias": np.degrees(np.angle(m1)),
        "resultant": np.abs(m1),
        "moment_real": np.real(m1),
        "moment_imag": np.imag(m1),
        "circ_sd": np.degrees(np.sqrt(-2.0 * np.log(r))),
    }


def model_case_nll(
    model,
    variables,
    params,
    bias,
    n_wraps: int = 4,
    case_block: int = CASE_BLOCK,
    chunk: int = SAMPLE_CHUNK,
):
    """Per-case mean NLL of raw samples, excluding non-finite outcomes."""
    params = np.asarray(params)
    bias = np.asarray(bias)
    n_cases, n_samples = bias.shape
    nll = np.empty(n_cases)
    n_finite = np.empty(n_cases, dtype=np.int64)
    for a in range(0, n_cases, case_block):
        p = params[a:a + case_block]
        b = bias[a:a + case_block]
        dist = model.apply(variables, jnp.asarray(p))
        total = np.zeros(len(p))
        cnt = np.zeros(len(p))
        for s in range(0, n_samples, chunk):
            bs = b[:, s:s + chunk]
            finite = np.isfinite(bs)
            lp = np.asarray(wm.mixture_logpdf_samples(
                jnp.asarray(np.where(finite, bs, 0.0).astype(np.float32)),
                dist,
                n_wraps,
            ))
            total += np.where(finite, lp, 0.0).sum(axis=1)
            cnt += finite.sum(axis=1)
        block_nll = -total / np.maximum(cnt, 1)
        block_nll[cnt == 0] = np.nan
        nll[a:a + len(p)] = block_nll
        n_finite[a:a + len(p)] = cnt.astype(np.int64)
    return nll, n_finite
