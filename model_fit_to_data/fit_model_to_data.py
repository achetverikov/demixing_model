#!/usr/bin/env python3
"""
Fit demixing model to data.

General-purpose fitting script: works with any CSV that has columns for
subject, experiment, condition, feature difference, and bias. Column names
are configurable via CLI arguments.

Usage (from repo root):
    python model_fit_to_data/fit_model_to_data.py \
        --data-path your_data.csv \
        --output-dir results/my_fit

Column name defaults match the example data in example_data/.
"""
import argparse
import json
import pickle
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from grid_based_multi_condition_optimizer_jax_loops import GridBasedMultiConditionOptimizer
from shared.utils import filter_data_for_fitting, resolve_input_path, resolve_results_path


def _sanitize(value: str) -> str:
    """Replace characters that are problematic in filenames/keys."""
    return re.sub(r'[^\w]', '_', str(value)).strip('_')


def load_data(data_path: str, outlier_col: Optional[str], include_outliers: bool) -> pd.DataFrame:
    df = pd.read_csv(data_path, low_memory=False)
    if include_outliers or outlier_col is None or outlier_col not in df.columns:
        if not include_outliers and outlier_col is not None and outlier_col not in df.columns:
            print(f"Warning: outlier column '{outlier_col}' not found — keeping all rows.")
        print(f"Loaded {len(df)} trials (no outlier filtering).")
        return df
    before = len(df)
    df = df[df[outlier_col] != 1].copy()
    print(f"Loaded {before} trials, removed {before - len(df)} outliers, {len(df)} remaining.")
    return df


def load_results(output_dir: str):
    results_file = Path(output_dir) / 'extended_fit_results.pkl'
    if not results_file.exists():
        return {}, set()
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    completed = {key.rsplit('#', 1)[0] for key in results}
    return results, completed


def save_results(results: Dict, output_dir: str, label: str = None):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Convert JAX arrays to numpy before pickling
    serializable = {
        cond: {k: np.array(v) if hasattr(v, '__array__') and hasattr(v, 'device') else v
               for k, v in entry.items() if not k.endswith('_optimization_result')}
        for cond, entry in results.items()
    }
    with open(output_path / 'extended_fit_results.pkl', 'wb') as f:
        pickle.dump(serializable, f)

    progress = {
        'total_completed': len(results),
        'last_completed': label,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'completed_conditions': list(results.keys()),
    }
    with open(output_path / 'extended_progress.json', 'w') as f:
        json.dump(progress, f, indent=2)

    if label:
        print(f"  Saved {len(results)} conditions.")


def group_conditions(
    df: pd.DataFrame,
    exp_col: str,
    subject_col: str,
    condition_col: str,
    min_trials: int,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Group data into {subject#exp: {subject#exp#condition: DataFrame}} structure."""
    groups = {}
    for exp in df[exp_col].unique():
        exp_data = df[df[exp_col] == exp]
        for subject in exp_data[subject_col].unique():
            subject_data = exp_data[exp_data[subject_col] == subject]
            exp_key = f"{_sanitize(subject)}#{_sanitize(exp)}"

            conditions = {}
            for condition in subject_data[condition_col].unique():
                cond_data = subject_data[subject_data[condition_col] == condition]
                if len(cond_data) >= min_trials:
                    cond_key = f"{exp_key}#{_sanitize(condition)}"
                    conditions[cond_key] = cond_data

            if len(conditions) >= 1:
                groups[exp_key] = conditions
            else:
                print(f"  Skipping {exp_key}: no condition has >= {min_trials} trials.")

    return groups


def process_subject(
    subject_id: str,
    subject_conditions: Dict[str, pd.DataFrame],
    optimizer: GridBasedMultiConditionOptimizer,
    methods: List[str],
    x_col: str,
    y_col: str,
    existing_results: Optional[Dict] = None,
    missing_methods: Optional[List[str]] = None,
) -> Dict:
    methods_to_run = missing_methods if missing_methods is not None else methods
    print(f"\nProcessing {subject_id}: {len(subject_conditions)} condition(s)" +
          (f" (adding: {', '.join(methods_to_run)})" if missing_methods else ""))

    # Filter and convert each condition to a JAX array
    condition_datasets = {}
    for cond_key, cond_df in subject_conditions.items():
        clean = filter_data_for_fitting(cond_df, feat_diff_col=x_col, bias_col=y_col, verbose=False)
        if len(clean) < 10:
            print(f"  Skipping {cond_key}: only {len(clean)} valid trials after filtering.")
            continue
        condition_datasets[cond_key] = jnp.asarray(clean[[x_col, y_col]].values)
        print(f"  {cond_key}: {len(clean)} trials")

    if not condition_datasets:
        print(f"  Skipping {subject_id}: no valid conditions remain.")
        return {}

    optimizer.update_dataset(condition_datasets)

    empirical_curves = {
        cond: {
            'target_bias': optimizer.unified_target_bias[i],
            'bias_weights': optimizer.unified_bias_weights[i],
            'target_density': optimizer.unified_target_density[i],
            'bias_feat_indices': optimizer.unified_feat_indices,
            'density_feat_grid': optimizer.feat_diff_grid,
        }
        for i, cond in enumerate(condition_datasets)
    }

    method_results = {}
    for method in methods_to_run:
        print(f"  Running {method}...")
        t0 = time.time()
        result = optimizer.fit_hierarchical_grid(fitting_method=method, verbosity=1,
                                                 shared_grid_size=40, feat_grid_size=20)
        duration = time.time() - t0
        method_results[method] = {'result': result, 'duration': duration}
        shared = result['shared_params']
        print(f"  {method} done in {duration:.1f}s — "
              f"sd_spat={shared['sd_spat']:.1f}, sd_motor={shared['sd_motor']:.1f}")

    condition_results = {}
    for cond_key in condition_datasets:
        entry = existing_results.get(cond_key, {}).copy() if existing_results else {}
        entry.update({
            'condition': cond_key,
            'data_df': condition_datasets[cond_key],
            'n_trials': len(subject_conditions[cond_key]),
            'empirical_curves': empirical_curves.get(cond_key, {}),
        })
        for method, mdata in method_results.items():
            opt = mdata['result']
            cond_res = opt['condition_results'][cond_key]
            shared = opt['shared_params']
            entry.update({
                f'{method}_fitted_params': jnp.array([
                    cond_res['sd_feat1'], cond_res['sd_feat2'],
                    shared['sd_spat'], shared['sd_motor'],
                ]),
                f'{method}_optimization_time': mdata['duration'],
                f'{method}_loss': cond_res['loss'],
            })
        condition_results[cond_key] = entry

    return condition_results


def run_fitting(
    data_path: str,
    checkpoint_path: str,
    output_dir: str,
    exp_col: str = 'expName',
    subject_col: str = 'subject',
    condition_col: str = 'condition',
    x_col: str = 'abs_td_dist',
    y_col: str = 'bias_to_distr_corr',
    outlier_col: Optional[str] = 'is_outlier',
    include_outliers: bool = False,
    methods: List[str] = None,
    min_trials: int = 30,
    resume: bool = True,
    max_subjects: Optional[int] = None,
    corr_weight: float = 0.25,
    skip_motor_noise: bool = True,
    results_dir: str = 'results',
):
    methods = methods or ['density']

    resolved_output = resolve_results_path(output_dir, results_dir)
    resolved_checkpoint = resolve_input_path(checkpoint_path, results_dir)

    print("=== Demixing Model Fitting ===")
    print(f"Data:       {data_path}")
    print(f"Checkpoint: {resolved_checkpoint}")
    print(f"Output:     {resolved_output}")
    print(f"Methods:    {', '.join(methods)}")
    print(f"Columns:    exp={exp_col}, subject={subject_col}, condition={condition_col}, "
          f"x={x_col}, y={y_col}")
    print()

    Path(resolved_output).mkdir(exist_ok=True, parents=True)

    df = load_data(data_path, outlier_col if not include_outliers else None, include_outliers)

    # Validate required columns
    required = [exp_col, subject_col, condition_col, x_col, y_col]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in CSV: {missing_cols}")

    existing_results, completed = load_results(resolved_output) if resume else ({}, set())

    print("Initializing optimizer...")
    dummy = jnp.asarray(np.random.uniform(-180, 180, (100, 2)))
    optimizer = GridBasedMultiConditionOptimizer(
        str(resolved_checkpoint), {'dummy': dummy},
        skip_motor_noise=skip_motor_noise, corr_weight=corr_weight,
    )
    print("Optimizer ready.\n")

    subject_groups = group_conditions(df, exp_col, subject_col, condition_col, min_trials)
    n_total = min(len(subject_groups), max_subjects) if max_subjects else len(subject_groups)
    print(f"Subject×experiment groups: {n_total} | Completed: {len(completed)} | "
          f"Remaining: {n_total - len(completed)}\n")

    n_done = 0
    t_start = time.time()

    for subject_id, subject_conditions in subject_groups.items():
        if max_subjects and n_done >= max_subjects:
            break

        if subject_id in completed:
            sample = next((v for k, v in existing_results.items()
                           if k.startswith(f"{subject_id}#")), None)
            missing = [m for m in methods
                       if sample is None or f'{m}_fitted_params' not in sample]
            if not missing:
                print(f"Skipping {subject_id}: already complete.")
                continue
            subject_existing = {k: v for k, v in existing_results.items()
                                 if k.startswith(f"{subject_id}#")}
        else:
            missing, subject_existing = None, None

        try:
            results = process_subject(
                subject_id, subject_conditions, optimizer, methods,
                x_col, y_col, subject_existing, missing,
            )
            existing_results.update(results)
            n_done += 1
            save_results(existing_results, resolved_output, subject_id)

            elapsed = time.time() - t_start
            eta = (n_total - n_done) * elapsed / n_done
            print(f"  Progress: {n_done}/{n_total} | "
                  f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

        except Exception as e:
            print(f"  Error processing {subject_id}: {e}")
            raise

    save_results(existing_results, resolved_output, 'FINAL')
    total = time.time() - t_start
    print(f"\nDone. {len(existing_results)} conditions fitted in {total/60:.1f}m.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit the demixing model to behavioural data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-path', required=True,
                        help='Path to input CSV.')
    parser.add_argument('--checkpoint-path', default='pretrained/model_epoch_1500.pkl',
                        help='Path to trained NN checkpoint.')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for results.')

    # Column names
    parser.add_argument('--exp-col',       default='expName',           help='Experiment column.')
    parser.add_argument('--subject-col',   default='subject',           help='Subject column.')
    parser.add_argument('--condition-col', default='condition',         help='Condition column.')
    parser.add_argument('--x-col',         default='abs_td_dist',       help='Feature difference column.')
    parser.add_argument('--y-col',         default='bias_to_distr_corr',help='Bias column.')
    parser.add_argument('--outlier-col',   default='is_outlier',
                        help='Outlier flag column (1 = exclude). Ignored if column absent or '
                             '--include-outliers is set.')

    # Fitting options
    parser.add_argument('--include-outliers', action='store_true',
                        help='Skip outlier filtering.')
    parser.add_argument('--include-methods', nargs='+', default=['density'],
                        choices=['density', 'expectation', 'likelihood', 'crps'],
                        help='Optimisation method(s) to run.')
    parser.add_argument('--min-trials', type=int, default=30,
                        help='Minimum trials per condition to include.')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True,
                        help='Resume from existing results.')
    parser.add_argument('--max-subjects', type=int, default=None,
                        help='Stop after this many subject×experiment groups.')
    parser.add_argument('--corr-weight', type=float, default=0.25)
    parser.add_argument('--no-skip-motor-noise', dest='skip_motor_noise',
                        action='store_false', default=True,
                        help='Include motor noise parameter (slower, rarely needed).')
    parser.add_argument('--results-dir', default='results',
                        help='Base directory for relative output paths.')

    args = parser.parse_args()

    run_fitting(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        exp_col=args.exp_col,
        subject_col=args.subject_col,
        condition_col=args.condition_col,
        x_col=args.x_col,
        y_col=args.y_col,
        outlier_col=args.outlier_col,
        include_outliers=args.include_outliers,
        methods=args.include_methods,
        min_trials=args.min_trials,
        resume=args.resume,
        max_subjects=args.max_subjects,
        corr_weight=args.corr_weight,
        skip_motor_noise=args.skip_motor_noise,
        results_dir=args.results_dir,
    )
