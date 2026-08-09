#!/usr/bin/env python3
"""Plot raw 100k UEV mean-bias and response-SD curves against both surrogates.

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
from shared.mu1_axis import mu1_grid_np

COMP_COLORS = {1: '#1d4ed8', 2: '#c2410c'}
METHOD_LS = {'raw 100k': '-', 'conditional density': '--',
             'trajectory NLL': '--', 'moment w=2': '-.', 'moment w=10': ':'}
METHOD_LS['production NN'] = ':'


def _dprime(sd_ident):
    return design_mod.UEV_SPAT_DIFF / np.asarray(sd_ident)


def build_summary(model_path: Path, reference_path: Path,
                  checkpoint: Path | None = None, model_label='conditional density',
                  additional_models=(), reference_extension: Path | None = None
                  ) -> pd.DataFrame:
    """Return raw/model circular means and SDs for every component curve."""
    model, variables, meta = wm.load_model(model_path)
    store, ref_meta = data_mod.load_npz(reference_path)
    data_mod.require_matching_n_samples(meta, ref_meta)
    density_models = [(model_label, model, variables)]
    for label, path in additional_models:
        extra_model, extra_variables, extra_meta = wm.load_model(path)
        data_mod.require_matching_n_samples(extra_meta, ref_meta)
        density_models.append((label, extra_model, extra_variables))
    fallback_styles = ['--', '-.', ':', (0, (3, 1, 1, 1))]
    for i, (label, _, _) in enumerate(density_models):
        METHOD_LS.setdefault(label, fallback_styles[i % len(fallback_styles)])
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
    if reference_extension is not None:
        extension, extension_meta = data_mod.load_npz(reference_extension)
        data_mod.require_matching_n_samples(meta, extension_meta)
        expected_extension, _ = design_mod.uev_extension_design(
            extension_meta['uev_added_sd_feat'], extension_meta['uev_spat_dprime'],
            extension_meta.get('uev_feature_step', 2.0))
        if not np.array_equal(extension.design, expected_extension):
            raise ValueError('extension is not the expected UEV extension design')
        store = data_mod.SampleStore(
            np.concatenate([store.design, extension.design]),
            np.concatenate([store.bias, extension.bias]))

    frames = []
    asym_grid = mu1_grid_np().astype(np.float32)
    for component in (0, 1):
        params = store.design if component == 0 else np.asarray(wm.mirror_params(store.design))
        raw_mean_parts, raw_sd_parts, raw_asym_parts = [], [], []
        for start in range(0, store.n_rows, 32):
            bias = store.bias[start:start + 32, :, component]
            moments = evaluate.empirical_moments(bias)
            raw_mean_parts.append(moments['mean_bias'])
            raw_sd_parts.append(moments['circ_sd'])
            ref_density = evaluate.reference_density(bias, asym_grid)
            raw_asym_parts.append(evaluate.density_asymmetry(
                evaluate.safe_log(ref_density).T))
        raw_mean = np.concatenate(raw_mean_parts)
        raw_sd = np.concatenate(raw_sd_parts)
        raw_asym = np.concatenate(raw_asym_parts)
        values = [('raw 100k', raw_mean, raw_sd, raw_asym)]
        for label, density_model, density_variables in density_models:
            dist = density_model.apply(density_variables, jnp.asarray(params))
            pred_mean = np.asarray(wm.mean_and_resultant(dist)[0])
            pred_sd = np.asarray(wm.circular_sd(dist))
            log_density = evaluate.model_logdensity_grid(
                density_model, density_variables, params, asym_grid)
            pred_asym = evaluate.density_asymmetry(log_density.T)
            values.append((label, pred_mean, pred_sd, pred_asym))
        if checkpoint is not None:
            triples, inverse = np.unique(params[:, :3], axis=0, return_inverse=True)
            surfaces = production_predict(np.column_stack([triples, np.zeros(len(triples))]))
            profiles = comparison.interpolated_profile(surfaces[inverse], params[:, 3])
            density = np.exp(profiles)
            grid = comparison.mu1_grid_np()
            moment = ((density * np.exp(1j * np.radians(grid))[None, :]).sum(axis=1)
                      * comparison.mu1_cell_width())
            prod_r = np.clip(np.abs(moment), 1e-12, 1.0)
            values.append(('production NN', np.degrees(np.angle(moment)),
                           np.degrees(np.sqrt(-2.0 * np.log(prod_r))),
                           evaluate.density_asymmetry(profiles.T)))
        common = pd.DataFrame({
            'sd_feat1': store.design[:, 0], 'sd_feat2': store.design[:, 1],
            'sd_ident': store.design[:, 2], 'spat_dprime': _dprime(store.design[:, 2]),
            'dist_feat': store.design[:, 3], 'which_comp': component + 1})
        for method, mean_bias, response_sd, density_asymmetry in values:
            frame = common.copy()
            frame['method'] = method
            frame['mean_bias'] = mean_bias
            frame['response_sd'] = response_sd
            frame['density_asymmetry'] = density_asymmetry
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _slug(value):
    return f'{value:g}'.replace('.', 'p')


def _comparison_title(df):
    methods = set(df.method)
    if 'production NN' in methods:
        return 'Raw 100k vs conditional density and production NN'
    if len(methods) > 2:
        return 'Raw 100k vs conditional-density objectives'
    return 'Raw 100k vs conditional density'


def plot_matrix(df: pd.DataFrame, dprime: float, out: Path,
                metric: str = 'mean_bias'):
    """Triangular feature-SD grid for one response metric."""
    labels = {'mean_bias': ('mean bias, °', 'Mean bias'),
              'response_sd': ('circular response SD, °', 'Response variability')}
    ylabel, metric_title = labels[metric]
    vals = sorted(df.sd_feat1.unique())
    fig, axes = plt.subplots(len(vals), len(vals), figsize=(11.2, 9.6),
                             sharex=True, sharey=True)
    for row, sd1 in enumerate(vals):
        for col, sd2 in enumerate(vals):
            ax = axes[row, col]
            if sd2 < sd1:
                ax.axis('off')
                continue
            if metric == 'mean_bias':
                ax.axhline(0, color='#9ca3af', linewidth=.6)
            sub = df[(df.sd_feat1 == sd1) & (df.sd_feat2 == sd2)
                     & np.isclose(df.spat_dprime, dprime)]
            for method in df.method.unique():
                ls = METHOD_LS[method]
                for comp in (1, 2):
                    cur = sub[(sub.method == method) & (sub.which_comp == comp)].sort_values('dist_feat')
                    ax.plot(cur.dist_feat, cur[metric], color=COMP_COLORS[comp],
                            linestyle=ls, linewidth=1.35 if method == 'raw 100k' else 1.15)
            ax.grid(True, color='#e5e7eb', linewidth=.5)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(f'σ₂={sd2:g}°', fontsize=9)
            if col == row:
                ax.set_ylabel(f'σ₁={sd1:g}°\n{ylabel}', fontsize=8)
            if row == len(vals) - 1 or (col > row and sd1 == vals[-2]):
                ax.set_xlabel('dissimilarity, °', fontsize=8)
    handles = [mlines.Line2D([], [], color=COMP_COLORS[c], lw=2,
                            label=f'component {c}') for c in (1, 2)]
    handles += [mlines.Line2D([], [], color='#555555', ls=METHOD_LS[method], lw=2,
                             label=method) for method in df.method.unique()]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=9)
    sd_ident = design_mod.UEV_SPAT_DIFF / dprime
    fig.suptitle(f'{metric_title}: {_comparison_title(df).lower()}, spatial d′={dprime:g} '
                 f'(sd_ident={sd_ident:g}°)\nrows=σ_feat1, columns=σ_feat2', fontsize=11)
    fig.subplots_adjust(left=.08, right=.98, top=.91, bottom=.10,
                        hspace=.28, wspace=.14)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_averaged_uev(df: pd.DataFrame, dprime: float, out: Path,
                      metric: str = 'mean_bias'):
    """UEV curves averaged over available higher-noise levels."""
    labels = {'mean_bias': ('Mean bias, °', 'Mean bias'),
              'response_sd': ('Circular response SD, °', 'Response variability'),
              'density_asymmetry': ('P(bias > 0) − P(bias < 0)',
                                    'Density asymmetry')}
    ylabel, metric_title = labels[metric]
    sub = df[(df.sd_feat2 > df.sd_feat1) & np.isclose(df.spat_dprime, dprime)]
    agg = sub.groupby(['sd_feat1', 'dist_feat', 'which_comp', 'method'], as_index=False).agg(
        value=(metric, 'mean'))
    sd1_vals = sorted(agg.sd_feat1.unique())
    blues = plt.cm.Blues(np.linspace(.45, .9, len(sd1_vals)))
    oranges = plt.cm.Oranges(np.linspace(.45, .9, len(sd1_vals)))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=.09, right=.68, bottom=.13, top=.88)
    if metric == 'mean_bias':
        ax.axhline(0, color='#9ca3af', linewidth=.8)
    for i, sd1 in enumerate(sd1_vals):
        for comp, colors in ((1, blues), (2, oranges)):
            for method in df.method.unique():
                ls = METHOD_LS[method]
                cur = agg[(agg.sd_feat1 == sd1) & (agg.which_comp == comp)
                          & (agg.method == method)].sort_values('dist_feat')
                ax.plot(cur.dist_feat, cur.value, color=colors[i], linestyle=ls,
                        linewidth=1.8 if method == 'raw 100k' else 1.4)
    ax.set(xlabel='Feature dissimilarity, °', ylabel=ylabel,
           title=f'{metric_title}, unequal encoding variability: spatial d′={dprime:g} '
                 f'(sd_ident={design_mod.UEV_SPAT_DIFF / dprime:g}°)')
    ax.grid(True, color='#e5e7eb', linewidth=.6)
    color_handles = [mlines.Line2D([], [], color=blues[i], lw=2,
                                  label=f'lower-noise item, σ_low={v:g}°')
                     for i, v in enumerate(sd1_vals)]
    color_handles += [mlines.Line2D([], [], color=oranges[i], lw=2,
                                   label=f'higher-noise item, σ_low={v:g}°')
                      for i, v in enumerate(sd1_vals)]
    method_handles = [mlines.Line2D([], [], color='#555555', ls=METHOD_LS[m], lw=2,
                                   label=m) for m in df.method.unique()]
    leg = ax.legend(handles=color_handles, loc='upper left',
                    bbox_to_anchor=(1.01, 1.0), fontsize=7, ncol=1)
    ax.add_artist(leg)
    ax.legend(handles=method_handles, loc='lower left',
              bbox_to_anchor=(1.01, 0.0), fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_averaged_residual(df: pd.DataFrame, dprime: float, out: Path,
                           metric: str = 'mean_bias'):
    """Plot signed model-minus-raw error, averaged over higher-noise levels."""
    labels = {
        'mean_bias': ('bias', '°'),
        'density_asymmetry': ('density asymmetry', ''),
    }
    metric_label, unit = labels[metric]
    unit_suffix = f', {unit}' if unit else ''
    sub = df[(df.sd_feat2 > df.sd_feat1) & np.isclose(df.spat_dprime, dprime)]
    keys = ['sd_feat1', 'sd_feat2', 'dist_feat', 'which_comp']
    raw = sub[sub.method == 'raw 100k'][keys + [metric]].rename(
        columns={metric: 'raw_value'})
    residual = sub[sub.method != 'raw 100k'].merge(raw, on=keys)
    residual['error'] = residual[metric] - residual.raw_value
    agg = residual.groupby(
        ['sd_feat1', 'dist_feat', 'which_comp', 'method'], as_index=False
    ).agg(error=('error', 'mean'))

    sd1_vals = sorted(agg.sd_feat1.unique())
    colors = plt.cm.viridis(np.linspace(.15, .9, len(sd1_vals)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.subplots_adjust(left=.055, right=.82, bottom=.14, top=.82, wspace=.28)
    for ax, comp, title in zip(
            axes, (1, 2), ('Lower-noise item', 'Higher-noise item')):
        ax.axhline(0, color='#6b7280', linewidth=.8)
        for i, sd1 in enumerate(sd1_vals):
            for method in residual.method.unique():
                cur = agg[(agg.sd_feat1 == sd1) & (agg.which_comp == comp)
                          & (agg.method == method)].sort_values('dist_feat')
                ax.plot(cur.dist_feat, cur.error, color=colors[i],
                        linestyle=METHOD_LS[method], linewidth=1.5)
        ax.set(xlabel='Feature dissimilarity, °', title=title)
        ax.grid(True, color='#e5e7eb', linewidth=.6)
    axes[0].set_ylabel(f'Approximated {metric_label} − raw 100k{unit_suffix}')
    high_noise = agg[agg.which_comp == 2]
    worst = high_noise.groupby(['sd_feat1', 'method'], as_index=False).agg(
        error=('error', 'min'))
    axes[2].axhline(0, color='#6b7280', linewidth=.8)
    for method in residual.method.unique():
        cur = worst[worst.method == method].sort_values('sd_feat1')
        axes[2].plot(cur.sd_feat1, cur.error, color='#374151',
                     linestyle=METHOD_LS[method], marker='o', linewidth=1.5,
                     markersize=3.5)
    axes[2].set(xlabel='Lower feature-noise SD, °',
                ylabel=f'Largest signed error{unit_suffix}',
                title='Higher-noise item\npeak undershoot')
    axes[2].grid(True, color='#e5e7eb', linewidth=.6)
    fig.suptitle(f'Signed {metric_label} error: spatial d′={dprime:g} '
                 f'(sd_ident={design_mod.UEV_SPAT_DIFF / dprime:g}°)\n'
                 'negative values indicate approximation undershoot')
    noise_handles = [mlines.Line2D([], [], color=colors[i], lw=2,
                                   label=f'σ_low={v:g}°')
                     for i, v in enumerate(sd1_vals)]
    method_handles = [mlines.Line2D([], [], color='#555555', ls=METHOD_LS[m], lw=2,
                                   label=m) for m in residual.method.unique()]
    leg = axes[2].legend(handles=noise_handles, loc='upper left',
                         bbox_to_anchor=(1.03, 1.0), fontsize=8)
    axes[2].add_artist(leg)
    axes[2].legend(handles=method_handles, loc='lower left',
                   bbox_to_anchor=(1.03, 0.0), fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--model-label', default='conditional density')
    parser.add_argument('--additional-model', nargs=2, action='append', default=[],
                        metavar=('LABEL', 'PATH'),
                        help='additional density checkpoint and legend label')
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--reference-extension', type=Path,
                        help='raw UEV rows containing additional feature-SD levels')
    parser.add_argument('--checkpoint', type=Path,
                        help='optional production surface NN for the same raw comparison')
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--averaged-only', action='store_true')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    additional = [(label, Path(path)) for label, path in args.additional_model]
    summary = build_summary(args.model, args.reference, args.checkpoint,
                            args.model_label, additional, args.reference_extension)
    name = ('uev_raw100k_vs_models.csv' if args.checkpoint
            else 'uev_raw100k_vs_density.csv')
    summary.to_csv(args.out_dir / name, index=False)
    for dprime in design_mod.UEV_SPAT_DPRIME:
        if not np.any(np.isclose(summary.spat_dprime, dprime)):
            continue
        suffix = _slug(dprime)
        if not args.averaged_only:
            plot_matrix(summary, dprime, args.out_dir / f'uev_grid_dprime_{suffix}.png')
        plot_averaged_uev(summary, dprime,
                          args.out_dir / f'uev_averaged_dprime_{suffix}.png')
        if not args.averaged_only:
            plot_matrix(summary, dprime,
                        args.out_dir / f'uev_response_sd_grid_dprime_{suffix}.png',
                        metric='response_sd')
        plot_averaged_uev(
            summary, dprime,
            args.out_dir / f'uev_response_sd_averaged_dprime_{suffix}.png',
            metric='response_sd')
        plot_averaged_uev(
            summary, dprime,
            args.out_dir / f'uev_density_asymmetry_averaged_dprime_{suffix}.png',
            metric='density_asymmetry')
        plot_averaged_residual(
            summary, dprime,
            args.out_dir / f'uev_bias_residual_averaged_dprime_{suffix}.png')
        plot_averaged_residual(
            summary, dprime,
            args.out_dir / f'uev_density_asymmetry_residual_averaged_dprime_{suffix}.png',
            metric='density_asymmetry')
        print(f'wrote UEV figures for spatial d′={dprime:g}')


if __name__ == '__main__':
    main()
