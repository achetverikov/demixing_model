#!/usr/bin/env python3
"""Evaluate a conditional density model against raw reference simulations.

The reference is always the raw high-simulation EM output, never a KDE surface.
Dense grids appear here only for reporting quantities (density asymmetry,
integrated density error, plots); the model itself never uses one.

The reference corpus is large (order 100 combinations x 100k EM outcomes x 2
components), so every sample-heavy quantity is computed in **case blocks** and
**sample chunks**: nothing here allocates a ``cases x grid x samples`` tensor or
repeats the parameters across all samples.  Non-finite EM outcomes are excluded
consistently from the moments, the reference/model densities, the mode
diagnostics and the NLL.

Usage (from the repo root)::

    PYTHONPATH=. python continuous_density/evaluate.py \
        --model $DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k8.pkl \
        --reference $DEMIXING_ARTIFACT_ROOT/continuous_density/validation.npz \
        --out $DEMIXING_ARTIFACT_ROOT/continuous_density/eval_k8.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from continuous_density import data as data_mod
from continuous_density import design as design_mod
from continuous_density import wrapped_mixture_model as wm
from shared.mu1_axis import mu1_grid_np
from shared.utils import compute_single_density_asymmetry

#: Grid used only for reporting integrated density error and KL divergence.
EVAL_GRID = np.arange(-180.0, 180.0, 0.5, dtype=np.float32)
#: Coarser grid shared by the reference and predicted mode diagnostics.
MODE_GRID = np.linspace(-180.0, 180.0, 360, endpoint=False).astype(np.float64)

#: Block/chunk sizes that bound the transient tensors of the sample-heavy paths.
CASE_BLOCK = 32
SAMPLE_CHUNK = 5000


def empirical_moments(bias: np.ndarray) -> dict:
    """Finite-aware circular first moment, mean direction, resultant, circular SD.

    ``bias`` is ``(M, S)``; non-finite outcomes are excluded rather than mapped
    to a spurious 0-degree sample.
    """
    b = np.asarray(bias, dtype=np.float64)
    finite = np.isfinite(b)
    z = np.where(finite, np.exp(1j * np.radians(np.where(finite, b, 0.0))), 0.0)
    count = finite.sum(axis=-1)
    m1 = z.sum(axis=-1) / np.maximum(count, 1)
    m1 = np.where(count > 0, m1, np.nan + 1j * np.nan)
    r = np.clip(np.abs(m1), 1e-12, 1.0)
    return {'mean_bias': np.degrees(np.angle(m1)), 'resultant': np.abs(m1),
            'moment_real': np.real(m1), 'moment_imag': np.imag(m1),
            'circ_sd': np.degrees(np.sqrt(-2.0 * np.log(r)))}


def empirical_density_asymmetry(bias: np.ndarray) -> np.ndarray:
    """Raw-sample ``P(bias > 0) - P(bias < 0)`` without KDE broadening."""
    b = np.asarray(bias, dtype=np.float64)
    finite = np.isfinite(b)
    ambiguous = np.isclose(np.abs(b), 180., atol=1e-6)
    positive = finite & (b > 0.) & ~ambiguous
    negative = finite & (b < 0.) & ~ambiguous
    count = finite.sum(axis=-1)
    out = (positive.sum(axis=-1) - negative.sum(axis=-1)) / np.maximum(count, 1)
    return np.where(count > 0, out, np.nan)


def model_logdensity_grid(model, variables, params, grid, n_wraps: int = 4,
                          case_block: int = CASE_BLOCK) -> np.ndarray:
    """Model log density (per degree) on ``grid``: ``(M, 4)`` -> ``(M, G)``.

    Cases are processed in blocks so the transient ``(block, G, K, wraps)`` tensor
    is bounded independently of ``M``.
    """
    params = np.asarray(params)
    grid_j = jnp.asarray(grid)
    parts = []
    for a in range(0, len(params), case_block):
        dist = model.apply(variables, jnp.asarray(params[a:a + case_block]))
        parts.append(np.asarray(wm.mixture_logpdf_grid(grid_j, dist, n_wraps)))
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, len(grid)))


def model_case_nll(model, variables, params, bias, n_wraps: int = 4,
                   case_block: int = CASE_BLOCK, chunk: int = SAMPLE_CHUNK):
    """Per-case mean NLL of the raw samples, excluding non-finite outcomes.

    The mixture parameters are evaluated once per case (never repeated across
    samples) and the sample log densities accumulate in chunks, so peak memory is
    ``block x chunk x K x wraps`` rather than ``M x S``.
    """
    params = np.asarray(params)
    bias = np.asarray(bias)
    M, S = bias.shape
    nll = np.empty(M)
    n_finite = np.empty(M, dtype=np.int64)
    for a in range(0, M, case_block):
        p = params[a:a + case_block]
        b = bias[a:a + case_block]
        dist = model.apply(variables, jnp.asarray(p))
        total = np.zeros(len(p))
        cnt = np.zeros(len(p))
        for s in range(0, S, chunk):
            bs = b[:, s:s + chunk]
            finite = np.isfinite(bs)
            lp = np.asarray(wm.mixture_logpdf_samples(
                jnp.asarray(np.where(finite, bs, 0.0).astype(np.float32)), dist, n_wraps))
            total += np.where(finite, lp, 0.0).sum(axis=1)
            cnt += finite.sum(axis=1)
        block_nll = -total / np.maximum(cnt, 1)
        block_nll[cnt == 0] = np.nan
        nll[a:a + len(p)] = block_nll
        n_finite[a:a + len(p)] = cnt.astype(np.int64)
    return nll, n_finite


def reference_density(bias: np.ndarray, grid=EVAL_GRID, kappa: float = 60.0,
                      chunk: int = 5000) -> np.ndarray:
    """Circular-kernel density of the reference samples, for *reporting only*.

    Excludes non-finite outcomes and chunks over samples (see
    :func:`design.circular_kde`).  It is never a training target and never enters
    model selection, which uses held-out likelihood of the raw samples.
    """
    return design_mod.circular_kde(bias, grid, kappa=kappa, chunk=chunk)


def _mode_frame(prefix: str, mode_dicts) -> pd.DataFrame:
    s = design_mod.secondary_mode_summary(mode_dicts)
    return pd.DataFrame({f'{prefix}_n_modes': s['n_modes'],
                         f'{prefix}_secondary_loc': s['secondary_loc'],
                         f'{prefix}_secondary_mass': s['secondary_mass']})


def per_case_metrics(model, variables, store: data_mod.SampleStore,
                     n_wraps: int = 4, grid=EVAL_GRID) -> pd.DataFrame:
    """Metrics A-G of the task, one row per (validation combination, component).

    ``store.bias[:, :, 0]`` is the component-1 bias; the mirrored component is
    evaluated through :func:`wrapped_mixture_model.mirror_params` rather than
    with a second model.
    """
    grid = np.asarray(grid)
    dx = float(grid[1] - grid[0])
    rows = []

    for c in (0, 1):
        params = store.design if c == 0 else np.asarray(wm.mirror_params(store.design))
        bias = store.bias[:, :, c]

        dist = model.apply(variables, jnp.asarray(params))
        pm, pr = (np.asarray(a) for a in wm.mean_and_resultant(dist))
        psd = np.asarray(wm.circular_sd(dist))

        nll, n_ref = model_case_nll(model, variables, params, bias, n_wraps)
        emp = empirical_moments(bias)

        log_q_grid = model_logdensity_grid(model, variables, params, grid, n_wraps)
        q_dens = np.exp(log_q_grid)
        ref_dens = reference_density(bias, grid)
        seam = np.abs(grid) >= 150.0
        with np.errstate(divide='ignore', invalid='ignore'):
            kl = np.sum(np.where(ref_dens > 0,
                                 ref_dens * (np.log(ref_dens) - log_q_grid), 0.0),
                        axis=1) * dx
        l1 = np.sum(np.abs(ref_dens - q_dens), axis=1) * dx

        ref_kde = reference_density(bias, MODE_GRID, kappa=40.0)
        ref_modes = [design_mod.circular_modes(d, MODE_GRID) for d in ref_kde]
        pred_dens = np.exp(model_logdensity_grid(model, variables, params,
                                                 MODE_GRID, n_wraps))
        pred_modes = [design_mod.circular_modes(d, MODE_GRID) for d in pred_dens]

        block = pd.DataFrame({
            'component': c + 1,
            'sd_feat1': params[:, 0], 'sd_feat2': params[:, 1],
            'sd_ident': params[:, 2], 'feat_diff': params[:, 3],
            'nll': nll,
            'ref_mean_bias': emp['mean_bias'], 'pred_mean_bias': pm,
            'ref_moment_real': emp['moment_real'],
            'ref_moment_imag': emp['moment_imag'],
            'pred_moment_real': np.real(np.asarray(wm.circular_moment(dist, 1))),
            'pred_moment_imag': np.imag(np.asarray(wm.circular_moment(dist, 1))),
            'ref_resultant': emp['resultant'], 'pred_resultant': pr,
            'ref_circ_sd': emp['circ_sd'], 'pred_circ_sd': psd,
            'ref_seam_mass': np.sum(ref_dens[:, seam], axis=1) * dx,
            'pred_seam_mass': np.sum(q_dens[:, seam], axis=1) * dx,
            'kl_ref_model': kl, 'l1_density_error': l1,
            'n_reference_samples': n_ref,
        })
        block = pd.concat([block, _mode_frame('ref', ref_modes),
                           _mode_frame('pred', pred_modes)], axis=1)
        rows.append(block)

    out = pd.concat(rows, ignore_index=True)
    out['mean_bias_error'] = wrap_error(out['pred_mean_bias'] - out['ref_mean_bias'])
    out['circ_sd_error'] = out['pred_circ_sd'] - out['ref_circ_sd']
    out['n_modes_error'] = out['pred_n_modes'] - out['ref_n_modes']
    out['secondary_loc_error'] = wrap_error(out['pred_secondary_loc']
                                            - out['ref_secondary_loc'])
    return out


def wrap_error(x):
    return np.mod(np.asarray(x, dtype=np.float64) + 180.0, 360.0) - 180.0


def predicted_mean_bias(model, variables, params) -> np.ndarray:
    """Predicted circular mean bias (degrees) at each parameter row."""
    dist = model.apply(variables, jnp.asarray(np.asarray(params)))
    return np.asarray(wm.mean_and_resultant(dist)[0])


def mirror_symmetry_diagnostic(model, variables, store: data_mod.SampleStore
                               ) -> dict:
    """How well one model, via the mirror map, serves *both* simulator components.

    The mirror augmentation is not an architectural constraint, so nothing forces
    it to hold; this measures whether training actually instilled it.  Returns the
    mean absolute mean-bias error (degrees) and raw-sample NLL for component 1
    evaluated at ``x`` and component 2 evaluated at ``mirror(x)``.  The known
    symmetry does *not* say that the two biases are negatives of each other; it
    says that component 2 must be read from the model at the swapped input.
    """
    design = store.design
    mirror = np.asarray(wm.mirror_params(design))
    pred1 = predicted_mean_bias(model, variables, design)
    pred2 = predicted_mean_bias(model, variables, mirror)
    emp1 = empirical_moments(store.bias[:, :, 0])['mean_bias']
    emp2 = empirical_moments(store.bias[:, :, 1])['mean_bias']
    nll1, _ = model_case_nll(model, variables, design, store.bias[:, :, 0])
    nll2, _ = model_case_nll(model, variables, mirror, store.bias[:, :, 1])
    return {
        'comp1_mean_abs_error': float(np.mean(np.abs(wrap_error(pred1 - emp1)))),
        'comp2_mean_abs_error': float(np.mean(np.abs(wrap_error(pred2 - emp2)))),
        'comp1_nll': float(np.mean(nll1)),
        'comp2_nll': float(np.mean(nll2)),
    }


def safe_log(density: np.ndarray) -> np.ndarray:
    """Log of a density in float64, with a floor so empty tails stay finite."""
    return np.log(np.maximum(np.asarray(density, dtype=np.float64), 1e-300))


def density_asymmetry(log_density_mu1_grid: np.ndarray) -> np.ndarray:
    """Project density asymmetry, reusing the production implementation.

    Expects log density on the production ``mu1_bias`` grid with shape
    ``(n_mu1_bias, n_cases)``; smoothing is off because the cases here are
    independent parameter combinations, not a feat_diff curve.
    """
    n_cases = log_density_mu1_grid.shape[1]
    return np.asarray(compute_single_density_asymmetry(
        jnp.asarray(log_density_mu1_grid), jnp.arange(n_cases),
        jnp.asarray(mu1_grid_np()), apply_smoothing=False))


def add_density_asymmetry(df: pd.DataFrame, model, variables,
                          store: data_mod.SampleStore, n_wraps: int = 4
                          ) -> pd.DataFrame:
    """Attach analytic mixture and direct raw-sample density asymmetry."""
    parts = []
    for c in (0, 1):
        params = store.design if c == 0 else np.asarray(wm.mirror_params(store.design))
        dist = model.apply(variables, jnp.asarray(params))
        parts.append(pd.DataFrame({
            'component': c + 1,
            'pred_density_asym': np.asarray(wm.density_asymmetry(dist)),
            'ref_density_asym': empirical_density_asymmetry(store.bias[:, :, c]),
        }))
    asym = pd.concat(parts, ignore_index=True)
    asym['density_asym_error'] = asym['pred_density_asym'] - asym['ref_density_asym']
    del asym['component']
    return pd.concat([df.reset_index(drop=True), asym], axis=1)


def trajectory_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Dissimilarity-aware summaries for fixed-SD validation trajectories."""
    if ('stratum' not in df
            or not df.stratum.astype(str).str.startswith('trajectory_').any()):
        return pd.DataFrame()
    rows = []
    trajectories = df[df.stratum.astype(str).str.startswith('trajectory_')]
    for (label, component), group in trajectories.groupby(['stratum', 'component']):
        group = group.sort_values('feat_diff')
        residual = wrap_error(group.pred_mean_bias - group.ref_mean_bias)
        moment_residual = ((group.pred_moment_real.to_numpy()
                            + 1j * group.pred_moment_imag.to_numpy())
                           - (group.ref_moment_real.to_numpy()
                              + 1j * group.ref_moment_imag.to_numpy()))
        rows.append({
            'stratum': label,
            'component': component,
            'n_feat_diff': len(group),
            'mean_bias_curve_rmse': float(np.sqrt(np.mean(residual ** 2))),
            'mean_bias_residual_d2_rms': (float(np.sqrt(np.mean(np.diff(residual, 2) ** 2)))
                                          if len(group) >= 3 else np.nan),
            'circular_moment_rmse': float(np.sqrt(np.mean(np.abs(moment_residual) ** 2))),
            'circular_moment_residual_d2_rms': (
                float(np.sqrt(np.mean(np.abs(np.diff(moment_residual, 2)) ** 2)))
                if len(group) >= 3 else np.nan),
            'mean_nll': float(group.nll.mean()),
            'mean_kl_ref_model': float(group.kl_ref_model.mean()),
            'mode_count_accuracy': float(np.mean(group.pred_n_modes == group.ref_n_modes)),
        })
    return pd.DataFrame(rows)


def validation_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Count interpretable behaviours in a raw-reference validation result.

    These thresholds are diagnostics, not pass/fail criteria.  They make an
    accidentally bland validation design visible before aggregate metrics are
    interpreted.
    """
    categories = {
        'attraction (mean bias > 1 deg)': df.ref_mean_bias > 1.0,
        'repulsion (mean bias < -1 deg)': df.ref_mean_bias < -1.0,
        'near-zero bias (abs <= 1 deg)': np.abs(df.ref_mean_bias) <= 1.0,
        'multimodal reference': df.ref_n_modes > 1,
        'high dispersion (circular SD >= 90 deg)': df.ref_circ_sd >= 90.0,
        'substantial seam mass (>= 0.15)': df.ref_seam_mass >= 0.15,
    }
    return pd.DataFrame({'category': list(categories),
                         'n_cases': [int(np.sum(v)) for v in categories.values()],
                         'fraction': [float(np.mean(v)) for v in categories.values()]})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--reference', required=True, type=Path,
                   help='.npz of high-simulation reference samples')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--repeat-reference', type=Path, default=None,
                   help='independent raw simulation repeat at the identical design')
    p.add_argument('--trajectory-summary', type=Path, default=None)
    args = p.parse_args()

    model, variables, meta = wm.load_model(args.model)
    store, ref_meta = data_mod.load_npz(args.reference)
    data_mod.require_matching_n_samples(meta, ref_meta)
    print(f"K={model.n_components}; {store.n_rows} reference combinations x "
          f"{store.bias.shape[1]} simulations")

    n_wraps = meta.get('n_wraps', 4)
    df = per_case_metrics(model, variables, store, n_wraps=n_wraps)
    df = add_density_asymmetry(df, model, variables, store, n_wraps)
    if 'strata' in ref_meta:
        df['stratum'] = list(ref_meta['strata']) * 2
    df['model'] = args.model.stem

    if args.repeat_reference is not None:
        repeat_store, repeat_meta = data_mod.load_npz(args.repeat_reference)
        data_mod.require_matching_n_samples(meta, repeat_meta)
        if repeat_store.design.shape != store.design.shape or not np.allclose(
                repeat_store.design, store.design, rtol=0, atol=1e-6):
            raise ValueError('repeat reference must use exactly the same parameter design')
        repeat = per_case_metrics(model, variables, repeat_store, n_wraps=n_wraps)
        df['repeat_nll'] = repeat.nll
        df['model_nll_repeat_delta'] = repeat.nll - df.nll
        df['reference_repeat_mean_bias_delta'] = wrap_error(
            repeat.ref_mean_bias - df.ref_mean_bias)
        df['reference_repeat_circ_sd_delta'] = repeat.ref_circ_sd - df.ref_circ_sd

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = trajectory_summary(df)
    if not summary.empty:
        summary_path = args.trajectory_summary or args.out.with_name(
            f'{args.out.stem}_trajectories.csv')
        summary.to_csv(summary_path, index=False)
        print(f'Saved {summary_path}')
    print(df[['nll', 'mean_bias_error', 'circ_sd_error', 'kl_ref_model',
              'l1_density_error', 'density_asym_error', 'n_modes_error']].describe().T)
    coverage = validation_coverage(df)
    print('\nValidation-behaviour coverage:')
    print(coverage.to_string(index=False))
    missing = coverage.loc[coverage.n_cases == 0, 'category'].tolist()
    if missing:
        print('WARNING: validation reference contains no cases for: ' + ', '.join(missing))
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
