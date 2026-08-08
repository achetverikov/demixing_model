#!/usr/bin/env python3
"""Head-to-head: exhaustive lattice scan vs hierarchical zoom, same code, same data.

This replaces ``validate_exhaustive_reference.py`` as the meaningful acceptance
check. That script compares against ``reference_optima_temp.csv``, which was
produced before the density target changed (the half-open mu1 axis, the closed
bias-KDE kernel, the pooled Sheather-Jones bandwidth, the grid-widening fix), so
its optima are optima of a different landscape. Reproducing it would now mean
those changes had *failed* to take effect.

What can be checked is that the two backends, run through the same current code
on the same data, agree about the same objective. The expected signature is not
"identical": the exhaustive scan is exact on the 1-degree lattice, while the
hierarchical zoom searches between lattice points. So

  * hierarchical wins by a **small** margin where it found a sub-degree point;
  * exhaustive wins by a **larger** margin where the zoom fell into a local basin;
  * neither should ever win by much on the lattice itself.

A large exhaustive loss would mean the cache or the scan is wrong. A large
hierarchical loss means the zoom is missing optima -- which is the finding that
motivated this work in the first place.

Usage (from repo root)::

    python cloud/compare_search_backends.py \\
        --data-path ../example_data/csh2026_prepared.csv \\
        --checkpoint-path pretrained/model_epoch1425_10ktrain_20samples.pkl \\
        --curve-cache results/curve_caches --out backend_comparison.csv
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
    DENSITY_CURVE_SPEC, HIERARCHICAL_GRID_SPEC, group_conditions, load_data,
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
    parser.add_argument('--curve-cache', required=True)
    parser.add_argument('--curve-cache-step', type=float, default=1.0)
    parser.add_argument('--circ-space', type=int, default=360, choices=[180, 360])
    parser.add_argument('--min-trials', type=int, default=30)
    parser.add_argument('--max-groups', type=int, default=None)
    parser.add_argument('--out', default=None, help='Write the per-group table here.')
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()

    checkpoint = resolve_input_path(args.checkpoint_path, args.results_dir)
    key = cc.default_cache_key(
        checkpoint_path=checkpoint, low=config.param_grid_low,
        high=config.param_range_high, step=args.curve_cache_step,
        emp_density_weights_sd=DENSITY_CURVE_SPEC['emp_density_weights_sd'],
        density_smoothing_sigma=DENSITY_CURVE_SPEC['density_smoothing_sigma'],
    )
    source = cc.CachedCurveSource(cc.cache_dir_for(args.curve_cache, key))

    model_circ_space = 2 * config.feat_diff_range[1]
    angle_scale = model_circ_space / args.circ_space
    df = load_data(args.data_path, 'is_outlier', False)
    groups = group_conditions(df, 'expName', 'subject', 'condition', args.min_trials)
    optimizer = GridBasedMultiConditionOptimizer(
        str(checkpoint), {'dummy': jnp.zeros((100, 2))}, skip_motor_noise=True,
        **DENSITY_CURVE_SPEC)

    items = list(groups.items())[:args.max_groups]
    rows = []
    for index, (group_key, conditions) in enumerate(items, start=1):
        datasets = {}
        for cond_key, cond_df in conditions.items():
            clean = filter_data_for_fitting(
                cond_df, feat_diff_col='abs_td_dist', bias_col='bias_to_distr_corr',
                verbose=False, min_diss=config.feat_diff_range[0] / angle_scale,
                max_diss=config.feat_diff_range[1] / angle_scale)
            if len(clean) < 10:
                continue
            data = clean[['abs_td_dist', 'bias_to_distr_corr']].values.copy() * angle_scale
            datasets[cond_key] = jnp.asarray(data)
        if not datasets:
            continue
        optimizer.update_dataset(datasets)

        started = time.time()
        exhaustive = fit_exhaustive_density(source, optimizer.unified_target_density,
                                            optimizer.condition_names, verbosity=0)
        t_exhaustive = time.time() - started

        started = time.time()
        hierarchical = optimizer.fit_hierarchical_grid(
            fitting_method='density', verbosity=0, **HIERARCHICAL_GRID_SPEC)
        t_hierarchical = time.time() - started

        rows.append({
            'group': group_key,
            'exhaustive_loss': float(exhaustive['best_loss']),
            'hierarchical_loss': float(hierarchical['best_loss']),
            'delta': float(exhaustive['best_loss']) - float(hierarchical['best_loss']),
            'exhaustive_sd_spat': exhaustive['shared_params']['sd_spat'],
            'hierarchical_sd_spat': float(hierarchical['shared_params']['sd_spat']),
            't_exhaustive': t_exhaustive,
            't_hierarchical': t_hierarchical,
        })
        print(f"[{index}/{len(items)}] {group_key:22s} exh={rows[-1]['exhaustive_loss']:.5f} "
              f"({t_exhaustive:.1f}s)  hier={rows[-1]['hierarchical_loss']:.5f} "
              f"({t_hierarchical:.1f}s)  delta={rows[-1]['delta']:+.5f}", flush=True)

    table = pd.DataFrame(rows)
    if args.out:
        table.to_csv(args.out, index=False)

    delta = table['delta']
    exhaustive_better = int((delta < -1e-6).sum())
    hierarchical_better = int((delta > 1e-6).sum())
    print(f"\nGroups: {len(table)}")
    print(f"  exhaustive better:            {exhaustive_better:3d}  "
          f"(largest margin {-delta.min():.5f})")
    print(f"  hierarchical better:          {hierarchical_better:3d}  "
          f"(largest margin {delta.max():.5f})")
    print(f"  tie within 1e-6:              {len(table) - exhaustive_better - hierarchical_better:3d}")
    print(f"  |delta| median {delta.abs().median():.6f}, "
          f"as a fraction of the loss {(delta.abs() / table['hierarchical_loss']).median():.2%}")
    print(f"  time per group: exhaustive {table['t_exhaustive'].median():.1f}s, "
          f"hierarchical {table['t_hierarchical'].median():.1f}s "
          f"({table['t_hierarchical'].median() / table['t_exhaustive'].median():.1f}x)")
    spat_gap = (table['exhaustive_sd_spat'] - table['hierarchical_sd_spat']).abs()
    print(f"  sd_spat differs by >1 deg in {int((spat_gap > 1.0).sum())} of {len(table)} groups")


if __name__ == '__main__':
    main()
