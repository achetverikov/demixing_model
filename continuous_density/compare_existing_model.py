#!/usr/bin/env python3
"""Head-to-head: production surface NN vs. the conditional density model.

Both are scored against the *same* raw high-simulation reference samples. The
production KDE surface is a competitor, never the target.

The production predictor consumes ``(sd_feat1, sd_feat2, sd_ident)`` and emits a
log density on the fixed ``(mu1_bias x feat_diff)`` grid, so evaluating it at a
continuous ``feat_diff`` requires interpolating along that axis.  Two readings
are reported:

``nearest``   the production convention — bin ``feat_diff`` and ``bias`` to the
              nearest grid cell (``mu1_axis.bin_indices``);
``interp``    linear in ``feat_diff`` (in log space) and circularly linear in
              bias, the fairest possible reading of the same checkpoint.

Interpolating two normalised log-density columns and exponentiating gives a
sub-normalised profile, so the interpolated profile is **re-normalised over the
mu1 grid** before it feeds any NLL, moment, asymmetry, mode or density metric.
Every sample-heavy quantity is computed in case blocks / sample chunks and
excludes non-finite EM outcomes, matching ``evaluate.py``.

Usage (from the repo root)::

    PYTHONPATH=. python continuous_density/compare_existing_model.py \
        --model $DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k8.pkl \
        --checkpoint pretrained/model_epoch1500_10ktrain_100samples.pkl \
        --reference $DEMIXING_ARTIFACT_ROOT/continuous_density/validation.npz \
        --out $DEMIXING_ARTIFACT_ROOT/continuous_density/comparison.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from continuous_density import data as data_mod
from continuous_density import design as design_mod
from continuous_density import evaluate as ev
from continuous_density import wrapped_mixture_model as wm
from shared.config import config
from shared.mu1_axis import (bin_indices_np, mu1_cell_width, mu1_grid_np,
                             mu1_size, mu1_step)
from shared.utils import load_checkpoint

_NN_DIR = Path(__file__).resolve().parents[1] / "neural_network_optimization"
if str(_NN_DIR) not in sys.path:
    sys.path.insert(0, str(_NN_DIR))


def _predict_in_batches(apply_fn, params, design: np.ndarray,
                        batch_size: int) -> np.ndarray:
    """Bound production-network device memory independently of case count."""
    parts = []
    for start in range(0, len(design), batch_size):
        part = apply_fn(params, jnp.asarray(design[start:start + batch_size, :3]))
        parts.append(np.asarray(part))
    return np.concatenate(parts, axis=0)


def production_predictor(checkpoint: Path, batch_size: int = 16):
    """Return a memory-bounded production surface predictor."""
    state, _ = load_checkpoint(str(checkpoint))
    apply_fn = jax.jit(state.apply_fn)

    def predict(design: np.ndarray) -> np.ndarray:
        return _predict_in_batches(apply_fn, state.params, np.asarray(design), batch_size)

    return predict


def checkpoint_n_samples(checkpoint: Path):
    """Infer the observer sample count encoded in standard checkpoint names."""
    match = re.search(r'_(20|100)samples(?:\.|_)', checkpoint.name)
    return int(match.group(1)) if match else None


def _feat_diff_weights(feat_diff: np.ndarray):
    """Indices and weights for linear interpolation along the feat_diff grid."""
    grid = np.asarray(config.create_grid('feat_diff'), dtype=np.float64)
    pos = np.clip(np.interp(feat_diff, grid, np.arange(len(grid))), 0, len(grid) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(grid) - 1)
    return lo, hi, (pos - lo)


def normalize_mu1_logdensity(log_density: np.ndarray) -> np.ndarray:
    """Re-normalise a per-degree log density over the mu1 grid (axis 1) to mass 1.

    ``sum_j exp(out) * dx == 1`` for every row.  This is the production
    ``normalize_to_density_flexible`` written for NumPy post-processing; applying
    it after the feat_diff interpolation restores the density that the linear
    combination of two normalised columns loses.
    """
    log_density = np.asarray(log_density, dtype=np.float64)
    dx = mu1_cell_width()
    from scipy.special import logsumexp
    log_norm = logsumexp(log_density, axis=1, keepdims=True)
    return log_density - log_norm - np.log(dx)


def interpolated_profile(surfaces: np.ndarray, feat_diff: np.ndarray) -> np.ndarray:
    """Normalised ``(M, n_mu1)`` log density at each case's continuous feat_diff."""
    lo, hi, frac = _feat_diff_weights(feat_diff)
    m = np.arange(surfaces.shape[0])
    prof = ((1 - frac)[:, None] * surfaces[m, :, lo]
            + frac[:, None] * surfaces[m, :, hi])
    return normalize_mu1_logdensity(prof)


def interpolated_profile_single(surface: np.ndarray,
                                feat_diff: np.ndarray) -> np.ndarray:
    """Read one production surface at many trial-level dissimilarities."""
    lo, hi, frac = _feat_diff_weights(np.asarray(feat_diff))
    prof = ((1 - frac)[:, None] * np.asarray(surface)[:, lo].T
            + frac[:, None] * np.asarray(surface)[:, hi].T)
    return normalize_mu1_logdensity(prof)


def nearest_profile(surfaces: np.ndarray, feat_diff: np.ndarray) -> np.ndarray:
    """Nearest-column ``(M, n_mu1)`` log density (already normalised per column)."""
    lo, hi, frac = _feat_diff_weights(feat_diff)
    m = np.arange(surfaces.shape[0])
    col = np.where(frac < 0.5, lo, hi)
    return surfaces[m, :, col]


def _case_nll_from_scorer(score_fn, bias, chunk: int = 20000) -> np.ndarray:
    """Per-case mean NLL from a ``(M, chunk) -> (M, chunk)`` log-density scorer.

    Chunks over samples and excludes non-finite outcomes, so peak memory is
    ``M x chunk`` rather than ``M x S``.
    """
    bias = np.asarray(bias)
    M, S = bias.shape
    total = np.zeros(M)
    cnt = np.zeros(M)
    for s in range(0, S, chunk):
        b = bias[:, s:s + chunk]
        finite = np.isfinite(b)
        lp = score_fn(np.where(finite, b, 0.0))
        total += np.where(finite, lp, 0.0).sum(axis=1)
        cnt += finite.sum(axis=1)
    nll = -total / np.maximum(cnt, 1)
    nll[cnt == 0] = np.nan
    return nll


def _score_nearest(prof):
    def scorer(bias_chunk):
        rows = bin_indices_np(bias_chunk)
        return np.take_along_axis(prof, rows, axis=1)
    return scorer


def _score_interp(prof):
    dens = np.exp(prof)
    grid0 = mu1_grid_np()[0]
    step = mu1_step()
    n = mu1_size()

    def scorer(bias_chunk):
        pos = (np.asarray(bias_chunk) - grid0) / step
        r0 = np.mod(np.floor(pos).astype(int), n)
        r1 = np.mod(r0 + 1, n)
        t = pos - np.floor(pos)
        d = ((1 - t) * np.take_along_axis(dens, r0, axis=1)
             + t * np.take_along_axis(dens, r1, axis=1))
        return ev.safe_log(d)
    return scorer


def compare(model, variables, store: data_mod.SampleStore, checkpoint: Path,
            n_wraps: int = 4) -> pd.DataFrame:
    """One row per (validation combination, component) with both models' metrics."""
    grid = mu1_grid_np().astype(np.float64)
    dx = mu1_cell_width()
    z = np.exp(1j * np.radians(grid))
    predict_production = production_predictor(checkpoint)
    frames = []
    for c in (0, 1):
        params = store.design if c == 0 else np.asarray(wm.mirror_params(store.design))
        bias = store.bias[:, :, c]

        surfaces = predict_production(params)
        prof_near = nearest_profile(surfaces, params[:, 3])
        prof_interp = interpolated_profile(surfaces, params[:, 3])
        prod = {
            'nll_production_nearest': _case_nll_from_scorer(_score_nearest(prof_near), bias),
            'nll_production_interp': _case_nll_from_scorer(_score_interp(prof_interp), bias),
        }
        prod_dens = np.exp(prof_interp)
        prod_m1 = (prod_dens * z[None, :]).sum(axis=1) * dx
        prod_r = np.clip(np.abs(prod_m1), 1e-12, 1.0)

        dist = model.apply(variables, jnp.asarray(params))
        new_nll, _ = ev.model_case_nll(model, variables, params, bias, n_wraps)
        new_log = ev.model_logdensity_grid(model, variables, params, grid, n_wraps)
        new_dens = np.exp(new_log)

        ref_dens = ev.reference_density(bias, grid)
        emp = ev.empirical_moments(bias)

        ref_modes = [design_mod.circular_modes(d, grid) for d in ref_dens]
        pred_modes = [design_mod.circular_modes(d, grid) for d in new_dens]
        prod_modes = [design_mod.circular_modes(d, grid) for d in prod_dens]

        frame = pd.DataFrame({
            'component': c + 1,
            'sd_feat1': params[:, 0], 'sd_feat2': params[:, 1],
            'sd_ident': params[:, 2], 'feat_diff': params[:, 3],
            'nll_density_model': new_nll,
            **prod,
            'ref_mean_bias': emp['mean_bias'],
            'pred_mean_bias': np.asarray(wm.mean_and_resultant(dist)[0]),
            'production_mean_bias': np.degrees(np.angle(prod_m1)),
            'ref_circ_sd': emp['circ_sd'],
            'pred_circ_sd': np.asarray(wm.circular_sd(dist)),
            'production_circ_sd': np.degrees(np.sqrt(-2.0 * np.log(prod_r))),
            'ref_density_asym': ev.density_asymmetry(ev.safe_log(ref_dens).T),
            'pred_density_asym': ev.density_asymmetry(new_log.T),
            'production_density_asym': ev.density_asymmetry(ev.safe_log(prod_dens).T),
            'l1_density_model': np.sum(np.abs(ref_dens - new_dens), axis=1) * dx,
            'l1_production': np.sum(np.abs(ref_dens - prod_dens), axis=1) * dx,
        })
        frame = pd.concat([frame, ev._mode_frame('ref', ref_modes),
                           ev._mode_frame('pred', pred_modes),
                           ev._mode_frame('production', prod_modes)], axis=1)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    for col, ref in [('pred_mean_bias', 'ref_mean_bias'),
                     ('production_mean_bias', 'ref_mean_bias')]:
        df[col.replace('_mean_bias', '_mean_bias_error')] = ev.wrap_error(df[col] - df[ref])
    df['pred_n_modes_error'] = df['pred_n_modes'] - df['ref_n_modes']
    df['production_n_modes_error'] = df['production_n_modes'] - df['ref_n_modes']
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path,
                   help='pretrained production surface network')
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()

    model, variables, meta = wm.load_model(args.model)
    store, ref_meta = data_mod.load_npz(args.reference)
    target_n = data_mod.require_matching_n_samples(meta, ref_meta)
    checkpoint_n = checkpoint_n_samples(args.checkpoint)
    if target_n is not None and checkpoint_n is not None and target_n != checkpoint_n:
        raise ValueError(f'reference/model target n_samples={target_n}, but production '
                         f'checkpoint name indicates n_samples={checkpoint_n}')
    df = compare(model, variables, store, args.checkpoint, meta.get('n_wraps', 4))
    if 'strata' in ref_meta:
        df['stratum'] = list(ref_meta['strata']) * 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    cols = ['nll_density_model', 'nll_production_nearest', 'nll_production_interp',
            'l1_density_model', 'l1_production']
    print(df[cols].describe().T)
    wins = (df['nll_density_model'] < df['nll_production_interp']).mean()
    print(f"\nDensity model beats the production NN on held-out NLL in "
          f"{wins:.1%} of validation combinations")
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
