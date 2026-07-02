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
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from grid_based_multi_condition_optimizer_jax_loops import (
    GridBasedMultiConditionOptimizer,
    apply_motor_noise_with_precomputed_kernel,
    create_motor_noise_kernel_fft,
)
from shared.utils import filter_data_for_fitting, resolve_input_path, resolve_results_path


LOSS_EVALUATION_METHODS = [
    'density', 'expectation', 'likelihood', 'crps', 'balanced_crps', 'bias_weighted_crps',
]

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
except ImportError:  # pragma: no cover - fallback for minimal environments
    Console = None


USE_RICH = (
    Console is not None
    and os.environ.get("NO_RICH") != "1"
    and sys.stdout.isatty()
)
console = Console(color_system="auto") if USE_RICH else None


def log(message: str = "", style: Optional[str] = None) -> None:
    """Print a message to the console, using Rich styling when available.

    Args:
        message: Text to print.
        style: Optional Rich style string (e.g. ``"green"``, ``"yellow bold"``).
               Ignored in plain-text mode.
    """
    if USE_RICH:
        console.print(message, style=style)
    else:
        print(message)


def make_progress() -> Optional[Progress]:
    """Create a Rich Progress bar configured for model fitting, or None in plain-text mode.

    Returns:
        Configured :class:`rich.progress.Progress` instance, or ``None`` when
        Rich is unavailable (e.g. non-TTY output).
    """
    if not USE_RICH:
        return None
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _sanitize(value: str) -> str:
    """Replace characters that are problematic in filenames/keys."""
    return re.sub(r'[^\w]', '_', str(value)).strip('_')


def _canonical_condition_key(key: str) -> str:
    """Canonicalize legacy result keys to the current sanitized form."""
    parts = str(key).split('#')
    if len(parts) < 3:
        return str(key)
    subject, experiment = parts[0], parts[1]
    condition = "#".join(parts[2:])
    return f"{_sanitize(subject)}#{_sanitize(experiment)}#{_sanitize(condition)}"


def _merge_result_entries(old: Dict, new: Dict) -> Dict:
    """Merge duplicate legacy/current condition entries without dropping methods."""
    merged = dict(old)
    merged.update(new)
    return merged


def canonicalize_result_keys(results: Dict) -> Dict:
    """Collapse legacy unsanitized condition keys onto sanitized keys."""
    canonical = {}
    for key, entry in results.items():
        ckey = _canonical_condition_key(key)
        if ckey in canonical and isinstance(canonical[ckey], dict) and isinstance(entry, dict):
            canonical[ckey] = _merge_result_entries(canonical[ckey], entry)
        else:
            canonical[ckey] = entry
    return canonical


def load_data(data_path: str, outlier_col: Optional[str], include_outliers: bool) -> pd.DataFrame:
    """Load trial data from a CSV file and optionally filter outlier rows.

    Args:
        data_path: Path to the CSV file.
        outlier_col: Column whose value of ``1`` marks a trial as an outlier.
            If ``None`` or the column is absent, no filtering is applied.
        include_outliers: If ``True``, skip outlier filtering regardless of
            ``outlier_col``.

    Returns:
        DataFrame of (filtered) trial rows.
    """
    df = pd.read_csv(data_path, low_memory=False)
    if include_outliers or outlier_col is None or outlier_col not in df.columns:
        if not include_outliers and outlier_col is not None and outlier_col not in df.columns:
            log(f"Warning: outlier column '{outlier_col}' not found - keeping all rows.", "yellow")
        log(f"Loaded {len(df)} trials (no outlier filtering).", "cyan")
        return df
    before = len(df)
    df = df[df[outlier_col] != 1].copy()
    log(f"Loaded {before} trials, removed {before - len(df)} outliers, {len(df)} remaining.", "cyan")
    return df


def load_results(output_dir: str):
    """Load previously saved fit results from disk.

    Args:
        output_dir: Directory that may contain ``extended_fit_results.pkl``.

    Returns:
        Tuple of (results dict, completed set) where results maps condition
        keys to fit result dicts and completed is the set of already-processed
        condition key prefixes (without the trailing ``#<hash>`` suffix).
        Both are empty when no results file exists.
    """
    results_file = Path(output_dir) / 'extended_fit_results.pkl'
    if not results_file.exists():
        return {}, set()
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError) as exc:
        corrupt_path = results_file.with_suffix(
            results_file.suffix + f".corrupt-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        try:
            results_file.replace(corrupt_path)
            log(
                f"Warning: could not load {results_file} ({exc}); moved it to {corrupt_path} "
                "and will refit from scratch.",
                "yellow",
            )
        except OSError:
            log(
                f"Warning: could not load {results_file} ({exc}); will refit from scratch.",
                "yellow",
            )
        return {}, set()
    results = canonicalize_result_keys(results)
    completed = {key.rsplit('#', 1)[0] for key in results}
    return results, completed


def save_results(results: Dict, output_dir: str, label: str = None):
    """Persist fit results to disk and write a JSON progress summary.

    JAX arrays are converted to NumPy before pickling.  Internal
    ``*_optimization_result`` keys are excluded from the saved file.

    Args:
        results: Mapping from condition key to fit result dict.
        output_dir: Output directory (created if absent).
        label: Human-readable label of the last completed condition written
            into ``extended_progress.json``; ``None`` suppresses the log line.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Convert JAX arrays to numpy before pickling
    serializable = {
        cond: {k: np.array(v) if hasattr(v, '__array__') and hasattr(v, 'device') else v
               for k, v in entry.items() if not k.endswith('_optimization_result')}
        for cond, entry in results.items()
    }
    results_file = output_path / 'extended_fit_results.pkl'
    tmp_results_file = output_path / f'.{results_file.name}.tmp'
    with open(tmp_results_file, 'wb') as f:
        pickle.dump(serializable, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_results_file, results_file)

    progress = {
        'total_completed': len(results),
        'last_completed': label,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'completed_conditions': list(results.keys()),
    }
    with open(output_path / 'extended_progress.json', 'w') as f:
        json.dump(progress, f, indent=2)

    if label:
        log(f"  Saved {len(results)} conditions.", "green")


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
                log(f"  Skipping {exp_key}: no condition has >= {min_trials} trials.", "yellow")

    return groups


def evaluate_parameter_losses(
    optimizer: GridBasedMultiConditionOptimizer,
    params_by_condition: jnp.ndarray,
    fitting_methods: List[str],
) -> Dict[str, jnp.ndarray]:
    """Evaluate one fixed parameter set per current condition under each objective."""
    params_by_condition = jnp.asarray(params_by_condition)
    if params_by_condition.ndim != 2 or params_by_condition.shape[0] != optimizer.n_conditions:
        raise ValueError(
            "params_by_condition must have shape (n_conditions, 3 or 4); "
            f"got {params_by_condition.shape} for {optimizer.n_conditions} conditions"
        )

    log_surfaces = optimizer._predict_batch_fixed_size(params_by_condition[:, :3], verbosity=0)

    if params_by_condition.shape[1] >= 4:
        sd_motors = np.array(params_by_condition[:, 3])
        if np.any(sd_motors > 0):
            noisy_surfaces = []
            for surface, sd_motor in zip(log_surfaces, sd_motors):
                if sd_motor > 0:
                    kernel_fft = create_motor_noise_kernel_fft(float(sd_motor), optimizer.n_mu1_bias)
                    surface = apply_motor_noise_with_precomputed_kernel(surface[None], kernel_fft)[0]
                noisy_surfaces.append(surface)
            log_surfaces = jnp.stack(noisy_surfaces)

    cond_params_with_idx = jnp.column_stack([
        jnp.arange(optimizer.n_conditions),
        params_by_condition[:, 0],
        params_by_condition[:, 1],
    ])
    surface_indices = jnp.arange(optimizer.n_conditions)

    losses = {}
    for method in fitting_methods:
        evaluated = optimizer._optimize_conditions_jit(
            log_surfaces,
            cond_params_with_idx,
            surface_indices,
            fitting_method=method,
        )
        losses[method] = evaluated[:, 2]
    return losses


def process_subject(
    subject_id: str,
    subject_conditions: Dict[str, pd.DataFrame],
    optimizer: GridBasedMultiConditionOptimizer,
    methods: List[str],
    x_col: str,
    y_col: str,
    existing_results: Optional[Dict] = None,
    missing_methods: Optional[List[str]] = None,
    angle_scale_to_model: float = 1.0,
    circ_space: int = 360,
    progress: Optional[Progress] = None,
) -> Dict:
    methods_to_run = missing_methods if missing_methods is not None else methods
    log(f"\nProcessing {subject_id}: {len(subject_conditions)} condition(s)" +
        (f" (adding: {', '.join(methods_to_run)})" if missing_methods else ""), "bold cyan")

    # Clamp dissimilarity in the input space to the MODEL-space grid range so the
    # effective floor/ceiling is identical across datasets (the legacy raw-space
    # 4/180 made the floor period-dependent: 8° model for 180° data, 4° for 360°).
    # clip commutes with the positive angle scaling applied just below, so raw
    # bounds = model bounds / scale == clamping model-space to feat_diff_range.
    # See codex_audit.md report-level #3.
    from shared.config import config as _cfg
    min_diss = _cfg.feat_diff_range[0] / angle_scale_to_model
    max_diss = _cfg.feat_diff_range[1] / angle_scale_to_model

    # Filter and convert each condition to a JAX array
    condition_datasets = {}
    for cond_key, cond_df in subject_conditions.items():
        clean = filter_data_for_fitting(cond_df, feat_diff_col=x_col, bias_col=y_col, verbose=False,
                                        min_diss=min_diss, max_diss=max_diss)
        if len(clean) < 10:
            log(f"  Skipping {cond_key}: only {len(clean)} valid trials after filtering.", "yellow")
            continue
        data = clean[[x_col, y_col]].values.copy()
        if angle_scale_to_model != 1.0:
            data[:, 0] = data[:, 0] * angle_scale_to_model
            data[:, 1] = data[:, 1] * angle_scale_to_model
        condition_datasets[cond_key] = jnp.asarray(data)
        log(f"  {cond_key}: {len(clean)} trials", "cyan")

    if not condition_datasets:
        log(f"  Skipping {subject_id}: no valid conditions remain.", "yellow")
        return {}

    # Data-informed cap on motor noise. Motor noise is one additive component of
    # the total response error, so sd_motor can never exceed the empirical error
    # SD. Because it is a shared (across-condition) parameter it must be <= the
    # tightest condition, i.e. min over conditions. Values here are already in the
    # model's 360°-wide space (scaled by angle_scale_to_model), matching the grid.
    # This never binds for likelihood/CRPS (pure speed-up) and clips motor-noise
    # over-attribution for density.
    def _circ_sd_model_deg(vals):
        th = np.deg2rad(np.asarray(vals, dtype=float))  # model space period = 360°
        R = np.abs(np.mean(np.exp(1j * th)))
        return float('inf') if R <= 0 else float(np.degrees(np.sqrt(-2.0 * np.log(R))))

    emp_motor_cap = 50.0
    if not optimizer.skip_motor_noise:
        cond_error_sds = [_circ_sd_model_deg(np.asarray(ds)[:, 1]) for ds in condition_datasets.values()]
        emp_motor_cap = min(cond_error_sds) * 1.1  # small margin for finite-sample noise
        emp_motor_cap = float(max(0.1, min(emp_motor_cap, 50.0)))
        log(f"  Empirical motor-noise cap (min-condition error SD ×1.1): {emp_motor_cap:.1f}° (model space)", "cyan")

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
    method_task = None
    if progress is not None:
        method_task = progress.add_task(f"[magenta]Methods for {subject_id}", total=len(methods_to_run))
    for method in methods_to_run:
        # The 'expectation' objective fits only the circular-mean bias curve, which
        # a symmetric motor-noise convolution leaves exactly invariant — sd_motor is
        # mathematically unidentifiable there. In a motor-noise run it is therefore
        # skipped entirely (fitting it would just reproduce the no-motor result).
        if method == "expectation" and not optimizer.skip_motor_noise:
            log("  Skipping 'expectation': sd_motor is unidentifiable under this "
                "objective (symmetric response convolution preserves the circular mean).", "yellow")
            if progress is not None and method_task is not None:
                progress.advance(method_task)
            continue
        log(f"  Running {method}...", "magenta")
        t0 = time.time()
        status = None
        if USE_RICH:
            status = console.status(f"[bold magenta]Fitting {subject_id} / {method}[/]", spinner="dots")
            status.start()
        try:
            result = optimizer.fit_hierarchical_grid(fitting_method=method, verbosity=1,
                                                     shared_grid_size=40, feat_grid_size=20,
                                                     sd_motor_max=emp_motor_cap)
        finally:
            if status is not None:
                status.stop()
        duration = time.time() - t0
        method_results[method] = {'result': result, 'duration': duration}
        shared = result['shared_params']
        log(f"  {method} done in {duration:.1f}s - "
            f"sd_spat={shared['sd_spat']:.1f}, sd_motor={shared['sd_motor']:.1f}", "green")
        if progress is not None and method_task is not None:
            progress.advance(method_task)
    if progress is not None and method_task is not None:
        progress.remove_task(method_task)

    condition_results = {}
    for cond_key in condition_datasets:
        entry = existing_results.get(cond_key, {}).copy() if existing_results else {}
        entry.update({
            'condition': cond_key,
            'data_df': condition_datasets[cond_key],
            'n_trials': len(subject_conditions[cond_key]),
            'empirical_curves': empirical_curves.get(cond_key, {}),
            'circ_space': circ_space,
            'angle_scale_to_model': angle_scale_to_model,
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
                f'{method}_stage_times': opt.get('stage_times', []),
                f'{method}_loss': cond_res['loss'],
            })
        condition_results[cond_key] = entry

    fitted_methods = sorted({
        key[:-len('_fitted_params')]
        for entry in condition_results.values()
        for key in entry
        if key.endswith('_fitted_params')
    })
    if fitted_methods:
        eval_task = None
        if progress is not None:
            eval_task = progress.add_task(f"[blue]Evaluating losses for {subject_id}", total=len(fitted_methods))
        for fitted_method in fitted_methods:
            param_key = f'{fitted_method}_fitted_params'
            if not all(param_key in condition_results[cond_key] for cond_key in condition_datasets):
                continue
            params_by_condition = jnp.stack([
                jnp.asarray(condition_results[cond_key][param_key])
                for cond_key in condition_datasets
            ])
            losses_by_objective = evaluate_parameter_losses(
                optimizer,
                params_by_condition,
                fitting_methods=LOSS_EVALUATION_METHODS,
            )
            for cond_idx, cond_key in enumerate(condition_datasets):
                eval_losses = {
                    objective: float(losses[cond_idx])
                    for objective, losses in losses_by_objective.items()
                }
                condition_results[cond_key][f'{fitted_method}_evaluation_losses'] = eval_losses
                for objective, loss in eval_losses.items():
                    condition_results[cond_key][f'{fitted_method}_eval_{objective}_loss'] = loss
            if progress is not None and eval_task is not None:
                progress.advance(eval_task)
        if progress is not None and eval_task is not None:
            progress.remove_task(eval_task)

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
    circ_space: int = 360,
):
    methods = methods or ['density']

    from shared.config import config as _cfg
    model_circ_space = 2 * _cfg.feat_diff_range[1]
    angle_scale_to_model = model_circ_space / circ_space

    resolved_output = resolve_results_path(output_dir, results_dir)
    resolved_checkpoint = resolve_input_path(checkpoint_path, results_dir)

    if USE_RICH:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Data", str(data_path))
        table.add_row("Checkpoint", str(resolved_checkpoint))
        table.add_row("Output", str(resolved_output))
        table.add_row("Methods", ", ".join(methods))
        table.add_row("Columns", f"exp={exp_col}, subject={subject_col}, condition={condition_col}, x={x_col}, y={y_col}")
        circ_text = f"{circ_space}°"
        if angle_scale_to_model != 1.0:
            circ_text += f" (angles scaled x{angle_scale_to_model:.3f} into model space)"
        table.add_row("Circular space", circ_text)
        console.print(Panel(table, title="[bold cyan]Demixing Model Fitting[/]", border_style="cyan"))
    else:
        print("=== Demixing Model Fitting ===")
        print(f"Data:       {data_path}")
        print(f"Checkpoint: {resolved_checkpoint}")
        print(f"Output:     {resolved_output}")
        print(f"Methods:    {', '.join(methods)}")
        print(f"Columns:    exp={exp_col}, subject={subject_col}, condition={condition_col}, "
              f"x={x_col}, y={y_col}")
        print(f"Circular space: {circ_space}°"
              + (f" (angles scaled x{angle_scale_to_model:.3f} into model space)"
                 if angle_scale_to_model != 1.0 else ""))
        print()

    Path(resolved_output).mkdir(exist_ok=True, parents=True)

    df = load_data(data_path, outlier_col if not include_outliers else None, include_outliers)

    # Validate required columns
    required = [exp_col, subject_col, condition_col, x_col, y_col]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in CSV: {missing_cols}")

    existing_results, completed = load_results(resolved_output) if resume else ({}, set())

    log("Initializing optimizer...", "bold cyan")
    dummy = jnp.asarray(np.random.uniform(-180, 180, (100, 2)))
    status = None
    if USE_RICH:
        status = console.status("[bold cyan]Loading model and compiling optimizer[/]", spinner="dots")
        status.start()
    try:
        optimizer = GridBasedMultiConditionOptimizer(
            str(resolved_checkpoint), {'dummy': dummy},
            skip_motor_noise=skip_motor_noise, corr_weight=corr_weight,
        )
    finally:
        if status is not None:
            status.stop()
    log("Optimizer ready.\n", "green")

    subject_groups = group_conditions(df, exp_col, subject_col, condition_col, min_trials)
    n_total = min(len(subject_groups), max_subjects) if max_subjects else len(subject_groups)
    log(f"Subject x experiment groups: {n_total} | Completed: {len(completed)} | "
        f"Remaining: {n_total - len(completed)}\n", "bold")

    n_done = 0
    t_start = time.time()

    progress = make_progress()
    progress_cm = progress if progress is not None else None
    if progress_cm is None:
        class _NullProgress:
            def __enter__(self): return None
            def __exit__(self, *args): return False
        progress_cm = _NullProgress()

    with progress_cm as active_progress:
        subject_task = None
        if active_progress is not None:
            subject_task = active_progress.add_task("[cyan]Subjects", total=n_total)

        for subject_id, subject_conditions in subject_groups.items():
            if max_subjects and n_done >= max_subjects:
                break

            if subject_id in completed:
                sample = next((v for k, v in existing_results.items()
                               if k.startswith(f"{subject_id}#")), None)
                metadata_mismatch = (
                    sample is None
                    or sample.get('circ_space', 360) != circ_space
                    or not np.isclose(sample.get('angle_scale_to_model', 1.0), angle_scale_to_model)
                )
                if metadata_mismatch:
                    log(f"Re-fitting {subject_id}: existing results use a different circular-space transform.", "yellow")
                    missing, subject_existing = None, None
                else:
                    subject_existing = {k: v for k, v in existing_results.items()
                                         if k.startswith(f"{subject_id}#")}
                    expected_condition_keys = set(subject_conditions)
                    existing_condition_keys = set(subject_existing)
                    conditions_missing = bool(expected_condition_keys - existing_condition_keys)
                    # A method is missing if it is absent from ANY of the subject's
                    # expected condition entries — not just one sampled condition. Include
                    # conditions that have no saved entry at all, as can happen when a
                    # condition is added after the original fit or an incomplete save omits
                    # it. See codex_audit.md recovery #2.
                    missing = [m for m in methods
                               if not subject_existing
                               or conditions_missing
                               or any(f'{m}_fitted_params' not in entry
                                      for entry in subject_existing.values())]
                    missing_evaluations = [
                        m for m in methods
                        if sample is not None
                        and f'{m}_fitted_params' in sample
                        and any(f'{m}_evaluation_losses' not in entry
                                for entry in subject_existing.values()
                                if f'{m}_fitted_params' in entry)
                    ]
                    if not missing and not missing_evaluations:
                        log(f"Skipping {subject_id}: already complete.", "yellow")
                        if active_progress is not None and subject_task is not None:
                            active_progress.advance(subject_task)
                        continue
                    if missing_evaluations and not missing:
                        log(f"Evaluating saved losses for {subject_id}.", "cyan")
            else:
                missing, subject_existing = None, None

            try:
                results = process_subject(
                    subject_id, subject_conditions, optimizer, methods,
                    x_col, y_col, subject_existing, missing,
                    angle_scale_to_model=angle_scale_to_model,
                    circ_space=circ_space,
                    progress=active_progress,
                )
                existing_results.update(results)
                n_done += 1
                save_results(existing_results, resolved_output, subject_id)

                elapsed = time.time() - t_start
                eta = (n_total - n_done) * elapsed / n_done
                log(f"  Progress: {n_done}/{n_total} | "
                    f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m", "cyan")
                if active_progress is not None and subject_task is not None:
                    active_progress.advance(subject_task)

            except Exception as e:
                log(f"  Error processing {subject_id}: {e}", "bold red")
                raise

    save_results(existing_results, resolved_output, 'FINAL')
    total = time.time() - t_start
    log(f"\nDone. {len(existing_results)} conditions fitted in {total/60:.1f}m.", "bold green")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit the demixing model to behavioural data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-path', required=True,
                        help='Path to input CSV.')
    parser.add_argument('--checkpoint-path', default='pretrained/model_epoch1500_10ktrain_20samples.pkl',
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
                        choices=['density', 'expectation', 'likelihood', 'crps', 'balanced_crps', 'bias_weighted_crps'],
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
    parser.add_argument('--circ-space', type=int, default=360, choices=[180, 360],
                        help='Circular space of the input data. Use 180 for axial orientation '
                             'data in which feature differences span 0–90° and errors span '
                             '±90°; values are doubled into the model 360° space. Use 360 '
                             'when data already match model space.')

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
        circ_space=args.circ_space,
    )
