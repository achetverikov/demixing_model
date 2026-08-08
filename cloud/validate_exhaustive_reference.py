#!/usr/bin/env python3
"""Check the exhaustive backend against the 51-group reference optima.

``reference_optima_temp.csv`` records the 1-degree-lattice optimum for all 51
csh2026 subject x experiment groups, produced during the investigation that
motivated the exhaustive search. Reproducing it exactly is the acceptance
criterion for the cache and the scorer together: the cache must hold the curves
the surrogate would produce, and the scan must find the same argmin.

This is a *lattice-stage* check. It says nothing about sub-degree refinement,
which carries no global guarantee (see the plan's Phase 4 resolution).

Usage (from repo root)::

    python cloud/validate_exhaustive_reference.py \\
        --data-path ../example_data/csh2026_prepared.csv \\
        --checkpoint-path pretrained/model_epoch1500_10ktrain_20samples.pkl \\
        --curve-cache results/curve_caches \\
        --reference reference_optima_temp.csv

Runs on CPU: the scan is linear algebra over the cache, and the surrogate is only
needed to construct the optimizer (the empirical targets come from the trials).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax.numpy as jnp  # noqa: E402

import curve_cache as cc  # noqa: E402
from exhaustive_density import fit_exhaustive_density  # noqa: E402
from fit_model_to_data import (  # noqa: E402
    DENSITY_CURVE_SPEC, group_conditions, load_data,
)
from grid_based_multi_condition_optimizer_jax_loops import (  # noqa: E402
    GridBasedMultiConditionOptimizer,
)
from shared.config import config  # noqa: E402
from shared.utils import filter_data_for_fitting, resolve_input_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--curve-cache', required=True, help='Cache root directory.')
    parser.add_argument('--curve-cache-step', type=float, default=1.0)
    parser.add_argument('--reference', required=True, help='reference_optima_temp.csv')
    parser.add_argument('--circ-space', type=int, default=360, choices=[180, 360])
    parser.add_argument('--min-trials', type=int, default=30)
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='Absolute tolerance on parameters, in degrees.')
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    checkpoint = resolve_input_path(args.checkpoint_path, args.results_dir)

    cache_key = cc.default_cache_key(
        checkpoint_path=checkpoint, low=config.param_grid_low,
        high=config.param_range_high, step=args.curve_cache_step,
        emp_density_weights_sd=DENSITY_CURVE_SPEC['emp_density_weights_sd'],
        density_smoothing_sigma=DENSITY_CURVE_SPEC['density_smoothing_sigma'],
    )
    source = cc.CachedCurveSource(cc.cache_dir_for(args.curve_cache, cache_key))
    print(f"Cache {cache_key}: {len(source.sd_spat_values)} x {len(source.feat_pairs)} lattice")

    model_circ_space = 2 * config.feat_diff_range[1]
    angle_scale = model_circ_space / args.circ_space
    min_diss = config.feat_diff_range[0] / angle_scale
    max_diss = config.feat_diff_range[1] / angle_scale

    df = load_data(args.data_path, 'is_outlier', False)
    groups = group_conditions(df, 'expName', 'subject', 'condition', args.min_trials)
    optimizer = GridBasedMultiConditionOptimizer(
        str(checkpoint), {'dummy': jnp.zeros((100, 2))}, skip_motor_noise=True,
        **DENSITY_CURVE_SPEC)

    rows = []
    started = time.time()
    for group_key, conditions in groups.items():
        datasets = {}
        for cond_key, cond_df in conditions.items():
            clean = filter_data_for_fitting(cond_df, feat_diff_col='abs_td_dist',
                                            bias_col='bias_to_distr_corr', verbose=False,
                                            min_diss=min_diss, max_diss=max_diss)
            if len(clean) < 10:
                continue
            data = clean[['abs_td_dist', 'bias_to_distr_corr']].values.copy() * angle_scale
            datasets[cond_key] = jnp.asarray(data)
        if not datasets:
            continue

        optimizer.update_dataset(datasets)
        result = fit_exhaustive_density(source, optimizer.unified_target_density,
                                        optimizer.condition_names, verbosity=0)

        for cond_key, entry in result['condition_results'].items():
            # Result keys are sanitized ("high___high"); the reference stores the
            # original labels ("high - high"). Recover them from the trials rather
            # than trying to invert the sanitization, which is not injective.
            source_rows = conditions[cond_key]
            rows.append({
                'exp': str(source_rows['expName'].iloc[0]),
                'subject': str(source_rows['subject'].iloc[0]),
                'condition': str(source_rows['condition'].iloc[0]),
                'sd_spat': result['shared_params']['sd_spat'],
                'sd_feat1': entry['sd_feat1'], 'sd_feat2': entry['sd_feat2'],
                'fit_loss': entry['loss'], 'total_loss': result['best_loss'],
            })
        print(f"  {group_key}: sd_spat={result['shared_params']['sd_spat']:.1f} "
              f"total={result['best_loss']:.4f}", flush=True)

    got = pd.DataFrame(rows)
    print(f"\nScanned {got.groupby(['exp', 'subject']).ngroups} groups "
          f"in {time.time() - started:.1f}s")

    # The reference stores sanitized condition labels; match on them the same way.
    merged = reference.merge(got, on=['exp', 'subject', 'condition'],
                             suffixes=('_ref', '_got'), how='outer', indicator=True)
    missing = merged[merged['_merge'] != 'both']
    if len(missing):
        print(f"WARNING: {len(missing)} rows did not match between reference and result")
        print(missing[['exp', 'subject', 'condition', '_merge']].head(10).to_string())

    both = merged[merged['_merge'] == 'both']
    agree = {}
    for column in ('sd_spat', 'sd_feat1', 'sd_feat2'):
        delta = (both[f'{column}_ref'] - both[f'{column}_got']).abs()
        agree[column] = int((delta <= args.tolerance).sum())
        if agree[column] < len(both):
            worst = both.loc[delta.idxmax()]
            print(f"  {column}: {agree[column]}/{len(both)} match; worst "
                  f"{worst['exp']}/{worst['subject']}/{worst['condition']} "
                  f"ref={worst[f'{column}_ref']} got={worst[f'{column}_got']}")

    groups_matching = (
        both.groupby(['exp', 'subject'])
        .apply(lambda g: bool(((g['sd_spat_ref'] - g['sd_spat_got']).abs() <= args.tolerance).all()
                              and ((g['sd_feat1_ref'] - g['sd_feat1_got']).abs()
                                   <= args.tolerance).all()
                              and ((g['sd_feat2_ref'] - g['sd_feat2_got']).abs()
                                   <= args.tolerance).all()),
               include_groups=False)
    )
    n_groups = len(groups_matching)
    n_ok = int(groups_matching.sum())

    loss_delta = (both['fit_loss_ref'] - both['fit_loss_got']).abs()
    print(f"\nParameters: " + ", ".join(f"{k} {v}/{len(both)}" for k, v in agree.items()))
    print(f"Groups fully matching: {n_ok}/{n_groups}")
    print(f"fit_loss: max |delta| = {loss_delta.max():.3e}, median {loss_delta.median():.3e}")

    if n_ok != n_groups:
        raise SystemExit(f"FAILED: {n_groups - n_ok} of {n_groups} groups differ from the "
                         "reference lattice optimum.")
    print("\nPASSED: reproduces the reference lattice optima.")


if __name__ == '__main__':
    main()
