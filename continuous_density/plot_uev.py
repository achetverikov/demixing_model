#!/usr/bin/env python3
"""Plot raw 100k UEV mean-bias curves against both surrogate models.

Produces two families matching the earlier CSH2026 UEV views, with each spatial
d-prime level in a separate figure.  Raw curves are circular means computed
directly from the saved EM outcomes. The conditional-density curve is analytic;
the optional production curve is read from its predicted KDE surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from continuous_density import data as data_mod
from continuous_density import design as design_mod
from continuous_density import evaluate
from continuous_density import wrapped_mixture_model as wm

COMP_COLORS = {1: '#1d4ed8', 2: '#c2410c'}
METHOD_LS = {'raw 100k': '-', 'conditional density': '--'}
METHOD_LS['production NN'] = ':'


def _dprime(sd_ident):
    return design_mod.UEV_SPAT_DIFF / np.asarray(sd_ident)


def build_summary(model_path: Path, reference_path: Path,
                  checkpoint: Path | None = None) -> pd.DataFrame:
    """Return raw/model circular means for every canonical component curve."""
    model, variables, meta = wm.load_model(model_path)
    store, ref_meta = data_mod.load_npz(reference_path)
    data_mod.require_matching_n_samples(meta, ref_meta)
    if checkpoint is not None:
        from continuous_density import compare_existing_model as comparison
        checkpoint_n = comparison.checkpoint_n_samples(checkpoint)
        model_n = data_mod.model_n_samples(meta)
        if model_n is not None and checkpoint_n is not None and model_n != checkpoint_n:
            raise ValueError(f'density model targets n_samples={model_n}, but production '
                             f'checkpoint indicates n_samples={checkpoint_n}')
        production_predict = comparison.production_predictor(checkpoint)
    expected, _ = design_mod.uev_design(ref_meta.get('uev_feature_step', 2.0))
    if not np.array_equal(store.design, expected):
        raise ValueError('reference is not the expected canonical UEV design')

    frames = []
    for component in (0, 1):
        params = store.design if component == 0 else np.asarray(wm.mirror_params(store.design))
        raw_parts = []
        for start in range(0, store.n_rows, 32):
            raw_parts.append(evaluate.empirical_moments(
                store.bias[start:start + 32, :, component])['mean_bias'])
        raw = np.concatenate(raw_parts)
        pred = evaluate.predicted_mean_bias(model, variables, params)
        values = [('raw 100k', raw), ('conditional density', pred)]
        if checkpoint is not None:
            triples, inverse = np.unique(params[:, :3], axis=0, return_inverse=True)
            surfaces = production_predict(np.column_stack([triples, np.zeros(len(triples))]))
            profiles = comparison.interpolated_profile(surfaces[inverse], params[:, 3])
            density = np.exp(profiles)
            grid = comparison.mu1_grid_np()
            moment = ((density * np.exp(1j * np.radians(grid))[None, :]).sum(axis=1)
                      * comparison.mu1_cell_width())
            values.append(('production NN', np.degrees(np.angle(moment))))
        common = pd.DataFrame({
            'sd_feat1': store.design[:, 0], 'sd_feat2': store.design[:, 1],
            'sd_ident': store.design[:, 2], 'spat_dprime': _dprime(store.design[:, 2]),
            'dist_feat': store.design[:, 3], 'which_comp': component + 1})
        for method, mean_bias in values:
            frame = common.copy()
            frame['method'] = method
            frame['mean_bias'] = mean_bias
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _slug(value):
    return f'{value:g}'.replace('.', 'p')


def _comparison_title(df):
    return ('Raw 100k vs conditional density and production NN'
            if 'production NN' in set(df.method) else 'Raw 100k vs conditional density')


def plot_matrix(df: pd.DataFrame, dprime: float, out: Path):
    """Triangular feature-SD grid, raw solid versus density dashed."""
    vals = design_mod.UEV_SD_FEAT
    fig, axes = plt.subplots(len(vals), len(vals), figsize=(11.2, 9.6),
                             sharex=True, sharey=True)
    for row, sd1 in enumerate(vals):
        for col, sd2 in enumerate(vals):
            ax = axes[row, col]
            if sd2 < sd1:
                ax.axis('off')
                continue
            ax.axhline(0, color='#9ca3af', linewidth=.6)
            sub = df[(df.sd_feat1 == sd1) & (df.sd_feat2 == sd2)
                     & np.isclose(df.spat_dprime, dprime)]
            for method in df.method.unique():
                ls = METHOD_LS[method]
                for comp in (1, 2):
                    cur = sub[(sub.method == method) & (sub.which_comp == comp)].sort_values('dist_feat')
                    ax.plot(cur.dist_feat, cur.mean_bias, color=COMP_COLORS[comp],
                            linestyle=ls, linewidth=1.35 if method == 'raw 100k' else 1.15)
            ax.grid(True, color='#e5e7eb', linewidth=.5)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(f'σ₂={sd2:g}°', fontsize=9)
            if col == row:
                ax.set_ylabel(f'σ₁={sd1:g}°\nmean bias, °', fontsize=8)
            if row == len(vals) - 1 or (col > row and sd1 == vals[-2]):
                ax.set_xlabel('dissimilarity, °', fontsize=8)
    handles = [mlines.Line2D([], [], color=COMP_COLORS[c], lw=2,
                            label=f'component {c}') for c in (1, 2)]
    handles += [mlines.Line2D([], [], color='#555555', ls=METHOD_LS[method], lw=2,
                             label=method) for method in df.method.unique()]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=9)
    sd_ident = design_mod.UEV_SPAT_DIFF / dprime
    fig.suptitle(f'{_comparison_title(df)}: spatial d′={dprime:g} '
                 f'(sd_ident={sd_ident:g}°)\nrows=σ_feat1, columns=σ_feat2', fontsize=11)
    fig.subplots_adjust(left=.08, right=.98, top=.91, bottom=.10,
                        hspace=.28, wspace=.14)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_averaged_uev(df: pd.DataFrame, dprime: float, out: Path):
    """Unequal-variance curves averaged over available higher-noise levels."""
    sub = df[(df.sd_feat2 > df.sd_feat1) & np.isclose(df.spat_dprime, dprime)]
    agg = sub.groupby(['sd_feat1', 'dist_feat', 'which_comp', 'method'], as_index=False).agg(
        mean_bias=('mean_bias', 'mean'))
    sd1_vals = sorted(agg.sd_feat1.unique())
    blues = plt.cm.Blues(np.linspace(.45, .9, len(sd1_vals)))
    oranges = plt.cm.Oranges(np.linspace(.45, .9, len(sd1_vals)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color='#9ca3af', linewidth=.8)
    for i, sd1 in enumerate(sd1_vals):
        for comp, colors in ((1, blues), (2, oranges)):
            for method in df.method.unique():
                ls = METHOD_LS[method]
                cur = agg[(agg.sd_feat1 == sd1) & (agg.which_comp == comp)
                          & (agg.method == method)].sort_values('dist_feat')
                ax.plot(cur.dist_feat, cur.mean_bias, color=colors[i], linestyle=ls,
                        linewidth=1.8 if method == 'raw 100k' else 1.4)
    ax.set(xlabel='Feature dissimilarity, °', ylabel='Mean bias, °',
           title=f'Unequal encoding variability: spatial d′={dprime:g} '
                 f'(sd_ident={design_mod.UEV_SPAT_DIFF / dprime:g}°)')
    ax.grid(True, color='#e5e7eb', linewidth=.6)
    color_handles = [mlines.Line2D([], [], color=blues[i], lw=2,
                                  label=f'lower-noise σ={v:g}°')
                     for i, v in enumerate(sd1_vals)]
    color_handles += [mlines.Line2D([], [], color=oranges[i], lw=2,
                                   label=f'higher-noise item (σ_low={v:g}° group)')
                      for i, v in enumerate(sd1_vals)]
    method_handles = [mlines.Line2D([], [], color='#555555', ls=METHOD_LS[m], lw=2,
                                   label=m) for m in df.method.unique()]
    leg = ax.legend(handles=color_handles, loc='upper right', fontsize=7, ncol=2)
    ax.add_artist(leg)
    ax.legend(handles=method_handles, loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path,
                        help='optional production surface NN for the same raw comparison')
    parser.add_argument('--out-dir', required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args.model, args.reference, args.checkpoint)
    name = ('uev_raw100k_vs_models.csv' if args.checkpoint
            else 'uev_raw100k_vs_density.csv')
    summary.to_csv(args.out_dir / name, index=False)
    for dprime in design_mod.UEV_SPAT_DPRIME:
        suffix = _slug(dprime)
        plot_matrix(summary, dprime, args.out_dir / f'uev_grid_dprime_{suffix}.png')
        plot_averaged_uev(summary, dprime,
                          args.out_dir / f'uev_averaged_dprime_{suffix}.png')
        print(f'wrote UEV figures for spatial d′={dprime:g}')


if __name__ == '__main__':
    main()
