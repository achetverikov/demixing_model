#!/usr/bin/env python3
"""Generate the five planned figures from real evaluation artifacts.

This command never simulates or fits.  It reads a trained model, raw reference
NPZ, and CSVs from ``evaluate.py`` / ``compare_existing_model.py``.  Missing
inputs cause an explicit skip; results are never fabricated.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


def _k_from_name(name):
    match = re.search(r'k(\d+)', str(name))
    return int(match.group(1)) if match else None


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure5_k_comparison(eval_paths, out):
    """Plan Figure 5: paired held-out NLL change with mixture components K."""
    frames = []
    for path in eval_paths:
        df = pd.read_csv(path)
        label = df['model'].iloc[0] if 'model' in df and len(df) else path.stem
        k = _k_from_name(label) or _k_from_name(path.stem)
        if k is not None and 'nll' in df:
            frames.append((k, df.reset_index(drop=True)))
    if not frames:
        return None
    frames.sort(key=lambda item: item[0])
    lengths = {len(df) for _, df in frames}
    if len(lengths) != 1:
        raise ValueError('component comparisons require matching evaluation designs')
    n_rows = lengths.pop()
    if n_rows % 2:
        raise ValueError('component comparisons require paired component rows')
    baseline_k, baseline = frames[-1]
    rows = []
    for k, df in frames:
        # Evaluations contain component 1 followed by the mirrored component 2.
        # Average those before estimating uncertainty so each simulation design
        # row, rather than each correlated component, is the sampling unit.
        delta = df.nll.to_numpy() - baseline.nll.to_numpy()
        paired = .5 * (delta[:n_rows // 2] + delta[n_rows // 2:])
        se = paired.std(ddof=1) / np.sqrt(len(paired)) if len(paired) > 1 else 0.
        rows.append((k, paired.mean(), 1.96 * se))
    k, mean, ci = map(np.asarray, zip(*rows))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(k, mean, yerr=ci, fmt='o-', color='#3b6ea5', capsize=3)
    ax.axhline(0, color='#888888', ls='--', lw=1)
    ax.set(xlabel='mixture components K',
           ylabel=f'mean paired NLL − K={baseline_k} (95% CI)',
           title='Accuracy versus mixture size')
    ax.set_xticks(k)
    return _save(fig, out)


def figure3_calibration(eval_path, out):
    """Plan Figure 3: mean bias, circular SD, and density asymmetry calibration."""
    df = pd.read_csv(eval_path)
    pairs = [('ref_mean_bias', 'pred_mean_bias', 'mean bias (deg)'),
             ('ref_circ_sd', 'pred_circ_sd', 'circular SD (deg)'),
             ('ref_density_asym', 'pred_density_asym', 'density asymmetry')]
    if not all({a, b}.issubset(df) for a, b, _ in pairs):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (ref, pred, title) in zip(axes, pairs):
        good = np.isfinite(df[ref]) & np.isfinite(df[pred])
        if ref == 'ref_mean_bias' and 'ref_resultant' in df:
            # Mean direction is unstable and scientifically uninformative when
            # the reference circular resultant is close to zero.
            good &= df.ref_resultant >= .25
            title += ' (reference R ≥ 0.25)'
        x, y = df.loc[good, ref], df.loc[good, pred]
        ax.scatter(x, y, s=10, alpha=.6, color='#3b6ea5')
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], '--', color='#888888', lw=1)
        ax.set(xlabel='reference', ylabel='conditional mixture', title=title)
    return _save(fig, out)


def figure4_nll(comparison_path, out):
    """Plan Figure 4: production NN versus conditional-mixture held-out NLL."""
    df = pd.read_csv(comparison_path)
    need = {'nll_density_model', 'nll_production_interp'}
    if not need.issubset(df):
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    x, y = df.nll_production_interp, df.nll_density_model
    ax.scatter(x, y, s=12, alpha=.6, color='#3b6ea5')
    lo, hi = np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])
    ax.plot([lo, hi], [lo, hi], '--', color='#888888', lw=1)
    wins = np.mean(y < x)
    ax.set(xlabel='production NN NLL', ylabel='conditional-mixture NLL',
           title=f'Held-out NLL (mixture wins {wins:.0%})')
    return _save(fig, out)


def _difficult_rows(df):
    """One row each for attraction, repulsion, near-zero, multimodal, and noisy."""
    candidates = [df.ref_mean_bias.idxmax(), df.ref_mean_bias.idxmin(),
                  df.ref_mean_bias.abs().idxmin(), df.ref_circ_sd.idxmax()]
    if 'ref_secondary_mass' in df:
        candidates.insert(3, df.ref_secondary_mass.fillna(0).idxmax())
    elif 'ref_n_modes' in df:
        candidates.insert(3, df.ref_n_modes.idxmax())
    return list(dict.fromkeys(map(int, candidates)))


def _random_rows(df, n=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(len(df), min(n, len(df)), replace=False).tolist()


def _density_overlays(model_path, reference_path, checkpoint, comparison_path,
                      row_indices, out, title):
    from development.wnm_transition import compare_existing_model as cmp
    from surrogate_training.wnm import data as data_mod
    from development.wnm_transition import evaluate as ev
    from shared import wnm as wm

    model, variables, meta = wm.load_model(model_path)
    store, _ = data_mod.load_npz(reference_path)
    comparison = pd.read_csv(comparison_path).reset_index(drop=True)
    chosen = comparison.iloc[row_indices]
    params = chosen[['sd_feat1', 'sd_feat2', 'sd_ident', 'feat_diff']].to_numpy(np.float32)
    grid = ev.EVAL_GRID
    model_dens = np.exp(ev.model_logdensity_grid(
        model, variables, params, grid, meta.get('n_wraps', 4)))
    ref_dens = []
    for row in row_indices:
        component = int(comparison.loc[row, 'component']) - 1
        source_row = row if component == 0 else row - store.n_rows
        ref_dens.append(ev.reference_density(
            store.bias[source_row:source_row + 1, :, component], grid)[0])
    ref_dens = np.asarray(ref_dens)

    surfaces = cmp.production_predictor(checkpoint)(params)
    profiles = cmp.interpolated_profile(surfaces, params[:, 3])
    prod_dens = np.exp(cmp._score_interp(profiles)(
        np.broadcast_to(grid, (len(params), len(grid)))))

    n = len(row_indices)
    ncol = min(3, n)
    fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol,
                             figsize=(4 * ncol, 3 * int(np.ceil(n / ncol))),
                             squeeze=False)
    for j, ax in enumerate(axes.ravel()):
        if j >= n:
            ax.axis('off')
            continue
        ax.plot(grid, ref_dens[j], color='#555555', lw=1.5, label='raw-reference KDE')
        ax.plot(grid, prod_dens[j], color='#3b6ea5', lw=1.2, label='production NN')
        ax.plot(grid, model_dens[j], color='#c1523f', lw=1.2, label='conditional mixture')
        row = chosen.iloc[j]
        ax.set_title(f"c{int(row.component)} sd=({row.sd_feat1:.1f},"
                     f"{row.sd_feat2:.1f},{row.sd_ident:.1f}) d={row.feat_diff:.1f}",
                     fontsize=8)
        ax.set_xlim(-180, 180)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle(title)
    return _save(fig, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', nargs='*', type=Path, default=[])
    parser.add_argument('--comparison', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    overlay_inputs = (args.model, args.reference, args.checkpoint, args.comparison)
    if all(overlay_inputs):
        comparison = pd.read_csv(args.comparison)
        made.append(_density_overlays(*overlay_inputs,
                    _random_rows(comparison), args.out_dir / 'figure1_random_distributions.png',
                    'Random off-grid validation distributions'))
        made.append(_density_overlays(*overlay_inputs,
                    _difficult_rows(comparison), args.out_dir / 'figure2_difficult_cases.png',
                    'Difficult off-grid validation distributions'))
    else:
        print('figures 1-2 skipped: supply model, reference, checkpoint, and comparison')
    if args.eval:
        made.append(figure3_calibration(args.eval[0],
                                        args.out_dir / 'figure3_calibration.png'))
        made.append(figure5_k_comparison(args.eval,
                                         args.out_dir / 'figure5_components.png'))
    else:
        print('figures 3 and 5 skipped: supply evaluation CSVs')
    if args.comparison:
        made.append(figure4_nll(args.comparison, args.out_dir / 'figure4_nll.png'))
    else:
        print('figure 4 skipped: supply comparison CSV')
    for path in filter(None, made):
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
