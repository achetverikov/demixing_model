#!/usr/bin/env python3
"""
Batch Fit Analysis V2 - Grid-Based Multi-Condition Optimizer
"""
import pandas as pd
import numpy as np
import jax.numpy as jnp
import pickle
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional
import warnings
import argparse

warnings.filterwarnings('ignore')

from grid_based_multi_condition_optimizer_jax_loops import GridBasedMultiConditionOptimizer
from shared.config import config
from shared.utils import filter_data_for_fitting, resolve_input_path, resolve_results_path
def load_data(data_path, include_outliers=False, outlier_col = 'is_outlier' ):
    """Load and filter the main dataset."""
    print("Loading data...")
    df = pd.read_csv(data_path, low_memory=False)

    if include_outliers:
        df_clean = df.copy()
        print(f"Including all trials: {len(df_clean)} trials")
    else:
        # Filter out outliers
        df_clean = df[df[outlier_col] != 1].copy()
        print(f"Filtered out {len(df) - len(df_clean)} outliers, {len(df_clean)} trials remaining")
    
    return df_clean

def check_missing_methods(result: Dict, include_methods: List[str], overwrite_methods: List[str] = None) -> List[str]:
    """Check which optimization method results are missing or need recomputation."""
    overwrite_methods = overwrite_methods or []
    return [method for method in include_methods 
            if f'{method}_fitted_params' not in result or method in overwrite_methods]

def load_extended_results(output_dir: str) -> tuple[Dict, set]:
    """Load existing extended results to resume analysis."""
    extended_file, progress_file = Path(output_dir) / 'extended_fit_results.pkl', Path(output_dir) / 'extended_progress.json'
    
    if not extended_file.exists():
        return {}, set()
    
    with open(extended_file, 'rb') as f:
        extended_results = pickle.load(f)
    
    # Convert old naming scheme to new one and discard old color.2 8-condition results
    converted_results = {}
    completed_subjects = set()
    
    for old_cond, result in extended_results.items():
        if '#' in old_cond:
            # Already new naming scheme
            converted_results[old_cond] = result
            parts = old_cond.split('#')
            if len(parts) >= 2:
                completed_subjects.add(f"{parts[0]}#{parts[1]}")
        else:
            # Old naming scheme - check if it's color.2 with report labels (8 conditions per subject)
            if 'color.2_' in old_cond and ('_First_report' in old_cond or '_Second_report' in old_cond):
                # Discard old color.2 results with 8 conditions (noise × report combinations)
                print(f"Discarding old color.2 result: {old_cond}")
                continue
            else:
                # Convert regular experiments to new naming
                # Pattern: S12.color.1_high - low -> S12#color_1#high - low
                match = re.match(r'^(S\d+)\.([^.]+)\.(\d+)_(.+)$', old_cond)
                if match:
                    subj, exp_name, exp_num, noise = match.groups()
                    new_cond = f"{subj}#{exp_name}_{exp_num}#{noise}"
                    converted_results[new_cond] = result
                    completed_subjects.add(f"{subj}#{exp_name}_{exp_num}")
                    print(f"Converted: {old_cond} -> {new_cond}")
                else:
                    # Keep other formats as-is for now
                    converted_results[old_cond] = result
                    match = re.match(r'^(S\d+\.[^.]+\.\d+)', old_cond)
                    if match:
                        completed_subjects.add(match.group(1))
    
    print(f"Found {len(converted_results)} completed conditions from {len(completed_subjects)} subjects after conversion")
    return converted_results, completed_subjects

def optimize_result_for_storage(result: Dict) -> Dict:
    """Convert JAX arrays to numpy and remove heavy objects."""
    if result is None:
        return None
    
    return {k: np.array(v) if hasattr(v, '__array__') and hasattr(v, 'device') else v
            for k, v in result.items() if not k.endswith('_optimization_result')}

def save_extended_results(extended_results: Dict, output_dir: str, subject_id: str = None):
    """Save extended results incrementally with optimization."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Optimize and save results
    optimized_results = {cond: optimize_result_for_storage(result) 
                        for cond, result in extended_results.items()}
    
    with open(output_path / 'extended_fit_results.pkl', 'wb') as f:
        pickle.dump(optimized_results, f)
    
    # Save progress
    progress = {
        'total_completed': len(extended_results),
        'last_completed': subject_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'completed_conditions': list(extended_results.keys())
    }
    with open(output_path / 'extended_progress.json', 'w') as f:
        json.dump(progress, f, indent=2)
    
    if subject_id:
        print(f"  Extended progress saved: {len(extended_results)} conditions completed")

def group_conditions_by_subject_experiment(df_clean: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Group data by subject and experiment for multi-condition optimization."""
    subject_groups = {}
    
    # Process regular experiments
    for exp_name in [exp for exp in df_clean['expName'].unique() if "_fd" not in str(exp) and exp != 'color_2']:
        exp_data = df_clean[df_clean['expName'] == exp_name]
        
        for subject_exp in exp_data['subject_exp'].unique():
            subject_data = exp_data[exp_data['subject_exp'] == subject_exp]
            
            # Convert subject_exp format: S12.color.1 -> S12, color_1
            subj_match = re.match(r'^(S\d+)\.([^.]+)\.(\d+)$', subject_exp)
            if subj_match:
                subj, exp_type, exp_num = subj_match.groups()
                exp_id = f"{exp_type}_{exp_num}"
                
                # Group by noise condition using new naming scheme
                subject_conditions = {f"{subj}#{exp_id}#{noise}": condition_data
                                    for noise in subject_data['noise'].unique()
                                    if len(condition_data := subject_data[subject_data['noise'] == noise]) >= 30}
                
                if len(subject_conditions) >= 2:
                    subject_groups[f"{subj}#{exp_id}"] = subject_conditions
    
    # Handle color_2 experiment - three separate analyses
    if 'color_2' in df_clean['expName'].values:
        exp_data = df_clean[df_clean['expName'] == 'color_2']
        
        for subject_exp in exp_data['subject_exp'].unique():
            subject_data = exp_data[exp_data['subject_exp'] == subject_exp]
            
            # Convert subject_exp format: S12.color.2 -> S12
            subj_match = re.match(r'^(S\d+)\.color\.2$', subject_exp)
            if subj_match:
                subj = subj_match.group(1)
                
                # Analysis 1: Combined (ignoring report order) - group only by noise
                combined_conditions = {}
                for noise in subject_data['noise'].unique():
                    noise_data = subject_data[subject_data['noise'] == noise]
                    if len(noise_data) >= 30:
                        combined_conditions[f"{subj}#color_2#{noise}"] = noise_data
                
                if len(combined_conditions) >= 2:
                    subject_groups[f"{subj}#color_2"] = combined_conditions
                
                # Analysis 2: First report only
                first_report_data = subject_data[subject_data['cue_n_label'] == 'First report']
                first_conditions = {}
                for noise in first_report_data['noise'].unique():
                    noise_data = first_report_data[first_report_data['noise'] == noise]
                    if len(noise_data) >= 30:
                        first_conditions[f"{subj}#color_2_first#{noise}"] = noise_data
                
                if len(first_conditions) >= 2:
                    subject_groups[f"{subj}#color_2_first"] = first_conditions
                
                # Analysis 3: Second report only
                second_report_data = subject_data[subject_data['cue_n_label'] == 'Second report']
                second_conditions = {}
                for noise in second_report_data['noise'].unique():
                    noise_data = second_report_data[second_report_data['noise'] == noise]
                    if len(noise_data) >= 30:
                        second_conditions[f"{subj}#color_2_second#{noise}"] = noise_data
                
                if len(second_conditions) >= 2:
                    subject_groups[f"{subj}#color_2_second"] = second_conditions
    
    return subject_groups

def process_single_subject_multi_condition(
    subject_id: str,
    subject_conditions: Dict[str, pd.DataFrame],
    global_optimizer: GridBasedMultiConditionOptimizer,
    include_methods: List[str],
    existing_results: Dict = None,
    missing_methods: List[str] = None,
    feat_diff_col: str = 'abs_td_dist',
    bias_col: str = 'bias_to_distr_corr'
) -> Dict[str, Dict]:
    """Process a single subject with multiple conditions using the grid-based optimizer."""
    
    is_partial_update = existing_results is not None and missing_methods is not None
    methods_to_run = missing_methods if is_partial_update else include_methods
    
    print(f"Processing {subject_id}: {len(subject_conditions)} conditions" + 
          (f" (partial: {', '.join(missing_methods)})" if is_partial_update else ""))
    
    # Prepare condition datasets (no binning needed - optimizer handles it)
    condition_datasets = {}
    for cond, data in subject_conditions.items():
        print(f"  {cond}: Processing {len(data)} trials")
        
        # Use the utility function to filter data
        clean_data = filter_data_for_fitting(data, feat_diff_col=feat_diff_col, bias_col=bias_col, verbose=True)
        
        # Check if we still have enough data
        if len(clean_data) < 30:
            print(f"  WARNING: {cond} has only {len(clean_data)} valid trials after filtering")
            continue
        
        condition_datasets[cond] = jnp.asarray(clean_data[[feat_diff_col,bias_col]].values)
    
    if len(condition_datasets) < 2:
        print(f"  Skipping {subject_id}: insufficient valid conditions")
        return {}
    
    # Print condition info
    [print(f"  {cond}: {data.shape[0]} trials") for cond, data in condition_datasets.items()]
    
    # Update optimizer with this subject's data
    global_optimizer.update_dataset(condition_datasets)
    
    # Extract empirical curves computed by the optimizer
    empirical_curves = {}
    for i, condition_name in enumerate(condition_datasets.keys()):
        empirical_curves[condition_name] = {
            'target_bias': global_optimizer.unified_target_bias[i],
            'bias_weights': global_optimizer.unified_bias_weights[i], 
            'target_density': global_optimizer.unified_target_density[i],
            'bias_feat_indices': global_optimizer.unified_feat_indices,
            'density_feat_grid': global_optimizer.feat_diff_grid
        }
    
    # Run optimization for each method
    method_results = {}
    for method in methods_to_run:
        print(f"    Running {method} optimization...")
        start_time = time.time()
        
        optimization_result = global_optimizer.fit_hierarchical_grid(fitting_method=method, verbosity=1,
                                                                     shared_grid_size=40, feat_grid_size=20)
        
        duration = time.time() - start_time
        method_results[method] = {'optimization_result': optimization_result, 'duration': duration}
        
        print(f"    {method.capitalize()} completed in {duration:.2f}s")
        shared = optimization_result['shared_params']
        print(f"    Shared: sd_spat={shared['sd_spat']:.1f}, sd_motor={shared['sd_motor']:.1f}")
        
        [print(f"      {cond}: sd_feat1={res['sd_feat1']:.1f}, sd_feat2={res['sd_feat2']:.1f}, loss={res['loss']:.3f}")
         for cond, res in optimization_result['condition_results'].items()]
    
    # Create extended results for each condition
    condition_extended_results = {}
    
    for condition_name in condition_datasets.keys():
        if is_partial_update and condition_name in existing_results:
            extended_result = existing_results[condition_name].copy()
        else:
            extended_result = {
                'condition': condition_name,
                'data_df': condition_datasets[condition_name],
                'n_trials': len(subject_conditions[condition_name]),
                'empirical_curves': empirical_curves.get(condition_name, {}),
            }
        
        # Add results from each optimization method
        for method, method_data in method_results.items():
            optimization_result = method_data['optimization_result']
            condition_result = optimization_result['condition_results'][condition_name]
            shared_params = optimization_result['shared_params']
            
            method_params = jnp.array([
                condition_result['sd_feat1'],
                condition_result['sd_feat2'],
                shared_params['sd_spat'],
                shared_params['sd_motor']
            ])
            
            # Add method-specific results using dynamic keys
            extended_result.update({
                f'{method}_fitted_params': method_params,
                f'{method}_optimization_time': method_data['duration'],
                f'{method}_loss': condition_result['loss'],  # Method-specific loss (likelihood/expectation/density)
            })
        
        condition_extended_results[condition_name] = extended_result
    
    return condition_extended_results

def run_extended_batch_analysis_v2(
    checkpoint_path: None,
    output_dir: None,
    skip_motor_noise : bool = False,
    data_path: str = 'data_color_comb.csv',
    resume: bool = True,
    max_subjects: Optional[int] = None,
    include_methods: List[str] = None,
    overwrite_methods: List[str] = None,
    include_outliers: bool = False,
    feat_diff_col: str = 'abs_td_dist',
    bias_col: str = 'bias_to_distr_corr',
    outlier_col: str = 'is_outlier',
    experiment: str = None,
    corr_weight = 0.25,
    results_dir: str = "results"
) -> Dict:
    """Run extended batch analysis using GridBasedMultiConditionOptimizer."""
    
    include_methods = include_methods or ['density']
    overwrite_methods = overwrite_methods or []
    
    resolved_output_dir = resolve_results_path(output_dir, results_dir)
    resolved_checkpoint_path = resolve_input_path(checkpoint_path, results_dir)

    print("=== Extended Batch Fit Analysis V2 (Grid-Based Multi-Condition) ===")
    print(f"Model: {resolved_checkpoint_path}")
    print(f"Data: {data_path}")
    print(f"Output: {resolved_output_dir}")
    print(f"Methods: {', '.join(include_methods)}" + 
          (f" | Overwrite: {', '.join(overwrite_methods)}" if overwrite_methods else ""))
    print()
    
    Path(resolved_output_dir).mkdir(exist_ok=True, parents=True)
    
    # Load data and existing results
    df_clean = load_data(data_path, include_outliers=include_outliers, outlier_col = outlier_col)
    extended_results, completed_subjects = load_extended_results(str(resolved_output_dir)) if resume else ({}, set())
    
    # Initialize global optimizer with synthetic dummy data
    print("Initializing global optimizer...")
    dummy_dataset = jnp.asarray(np.random.uniform(low=-180, high=180.0, size=(100,2)))
    global_optimizer = GridBasedMultiConditionOptimizer(str(resolved_checkpoint_path), {'dummy': dummy_dataset}, skip_motor_noise=skip_motor_noise,
                                                                     corr_weight = corr_weight)
    print("Global optimizer initialized.\n")
    
    # Group conditions and process
    subject_groups = group_conditions_by_subject_experiment(df_clean)
    total_subjects = min(len(subject_groups), max_subjects) if max_subjects else len(subject_groups)
    
    print(f"Total: {total_subjects} | Completed: {len(completed_subjects)} | Remaining: {total_subjects - len(completed_subjects)}\n")
    
    subjects_processed = 0
    start_time = time.time()
    
    for subject_id, subject_conditions in subject_groups.items():
        if max_subjects and subjects_processed >= max_subjects:
            break
        if experiment is not None and experiment not in subject_id:
            continue
        # Extract subject+experiment part using the new naming scheme
        # subject_id format: subj_exp#experiment_id or subj_exp#experiment_id_variant
        if '#' in subject_id:
            # New naming scheme: extract exact subject_id as completion key
            completion_key = subject_id
        else:
            # Legacy naming: fallback to old logic
            match = re.match(r'^(S\d+\.[^.]+\.\d+)', subject_id)
            completion_key = match.group(1) if match else subject_id.split('_')[0]
        
        # Check if processing needed
        if completion_key in completed_subjects:
            # Match conditions that belong to this specific analysis
            if '#' in subject_id:
                # New naming: match conditions with same subject#experiment prefix
                subject_existing_results = {cond: result for cond, result in extended_results.items()
                                          if cond.startswith(f"{completion_key}#")}
            else:
                # Legacy naming
                base_subject_id = completion_key
                subject_existing_results = {cond: result for cond, result in extended_results.items()
                                          if cond.startswith(f"{base_subject_id}_")}
            
            if subject_existing_results:
                sample_result = next(iter(subject_existing_results.values()))
                missing_types = check_missing_methods(sample_result, include_methods, overwrite_methods)
                
                if not missing_types:
                    print(f"Skipping {subject_id}: already completed")
                    continue
            else:
                missing_types, subject_existing_results = None, None
        else:
            missing_types, subject_existing_results = None, None
        
        try:
            # Process subject
            subject_extended_results = process_single_subject_multi_condition(
                subject_id, subject_conditions, global_optimizer,
                include_methods, subject_existing_results, missing_types,
                feat_diff_col, bias_col)
            
            extended_results.update(subject_extended_results)
            subjects_processed += 1
            
            # Progress tracking every 5 subjects
            if subjects_processed % 1 == 0:
                save_extended_results(extended_results, str(resolved_output_dir), subject_id)
                elapsed = time.time() - start_time
                remaining_time = (total_subjects - subjects_processed) * (elapsed / subjects_processed)
                print(f"  Progress: {subjects_processed}/{total_subjects} | "
                      f"Elapsed: {elapsed/60:.1f}m | Remaining: {remaining_time/60:.1f}m\n")
        
        except Exception as e:
            print(f"  Error processing {subject_id}: {str(e)}")
            raise e
    
    # Final save and summary
    save_extended_results(extended_results, str(resolved_output_dir), "FINAL")
    total_time = time.time() - start_time
    
    print(f"\nExtended analysis complete!")
    print(f"Time: {total_time/60:.1f}m | Conditions: {len(extended_results)} | Subjects: {subjects_processed}")
    if subjects_processed > 0:
        print(f"Avg per subject: {total_time/subjects_processed:.1f}s")
    
    return extended_results

def run_batch_fitting_loop(nsamples_list=[20], 
                          data_path='data_color_comb.csv',
                          base_checkpoint_path='neural_net_checkpoints',
                          include_outliers_list=[False],
                        skip_motor_noise = True,
                           experiment: str = None,
                           results_dir: str = "results"):
    """
    Run batch fitting in a loop over different configurations using GridBasedMultiConditionOptimizer.
    
    Args:
        nsamples_list: List of sample counts to process
        data_path: Path to data CSV file
        base_checkpoint_path: Base path for checkpoints (will append _{n}samples)
        include_outliers_list: List of boolean values for outlier inclusion
    """
    corr_weight = 0.0

    for nsamples in nsamples_list:
        checkpoint_path = f'{base_checkpoint_path}_{nsamples}samples/model_epoch_1500.pkl'
        resolved_checkpoint_path = resolve_input_path(checkpoint_path, results_dir)

        for include_outliers in include_outliers_list:
            # Create directory structure matching batch_fit_analysis.py pattern
            outlier_str = "with_outliers" if include_outliers else "no_outliers"
            motor_str = '' if not skip_motor_noise else "_no_motor_noise"
            corr_str = f'_cw_{corr_weight:.2f}' if corr_weight!=0.25 else ''

            output_dir = f'model_fit_to_data_results_v2{motor_str}{corr_str}/{nsamples}samples/{outlier_str}'
            resolved_output_dir = resolve_results_path(output_dir, results_dir)
            
            print(f"\n{'='*60}")
            print(f"Processing configuration:")
            print(f"  Samples: {nsamples}")
            print(f"  Include outliers: {include_outliers}")
            print(f"  Output directory: {resolved_output_dir}")
            print(f"{'='*60}")
            
            # Run extended analysis
            print(f"\nRunning extended analysis for {resolved_output_dir}...")
            extended_results = run_extended_batch_analysis_v2(
                checkpoint_path=str(resolved_checkpoint_path),
                output_dir=str(resolved_output_dir),
                skip_motor_noise=skip_motor_noise,
                data_path=data_path,
                resume=True,
                max_subjects=None,
                include_methods=['density'],
                overwrite_methods=[],
                include_outliers=include_outliers,
                corr_weight=float(corr_weight),
                experiment=experiment,
                results_dir=results_dir,
            )
            print(f"Extended analysis completed for {resolved_output_dir}")
            print(f"Processed {len(extended_results)} conditions")


            print(f"\nCompleted configuration: {resolved_output_dir}")
    
    print(f"\n{'='*60}")
    print("All batch fitting configurations completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch fit analysis v2 with configurable parameters."
    )
    parser.add_argument(
        "--mode",
        choices=["loop", "extended"],
        default="loop",
        help="Run the batch loop or a single extended analysis.",
    )
    parser.add_argument(
        "--nsamples",
        nargs="+",
        type=int,
        default=[20],
        help="Sample counts to process (space-separated) for loop mode.",
    )
    parser.add_argument(
        "--data-path",
        default="data_color_comb.csv",
        help="Path to the input data CSV.",
    )
    parser.add_argument(
        "--base-checkpoint-path",
        default="neural_net_checkpoints",
        help="Base path for checkpoints (suffix _{n}samples will be appended) in loop mode.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Checkpoint path for extended mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for extended mode.",
    )
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="Include outlier trials in the analysis.",
    )
    parser.add_argument(
        "--skip-motor-noise",
        dest="skip_motor_noise",
        action="store_true",
        default=True,
        help="Skip motor noise in the optimizer (default).",
    )
    parser.add_argument(
        "--no-skip-motor-noise",
        dest="skip_motor_noise",
        action="store_false",
        help="Include motor noise in the optimizer.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing extended results (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not resume; start a fresh extended analysis.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Limit the number of subjects processed in extended mode.",
    )
    parser.add_argument(
        "--include-methods",
        nargs="+",
        default=["density"],
        help="Optimization methods to include in extended mode.",
    )
    parser.add_argument(
        "--overwrite-methods",
        nargs="+",
        default=[],
        help="Optimization methods to overwrite in extended mode.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Optional experiment filter (substring match on subject_id).",
    )
    parser.add_argument(
        "--corr-weight",
        type=float,
        default=0.25,
        help="Correlation weight for the optimizer in extended mode.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Base directory for outputs (relative paths are placed here).",
    )
    args = parser.parse_args()

    if args.mode == "extended":
        if not args.checkpoint_path or not args.output_dir:
            parser.error("--checkpoint-path and --output-dir are required for extended mode.")
        run_extended_batch_analysis_v2(
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            skip_motor_noise=args.skip_motor_noise,
            data_path=args.data_path,
            resume=args.resume,
            max_subjects=args.max_subjects,
            include_methods=args.include_methods,
            overwrite_methods=args.overwrite_methods,
            include_outliers=args.include_outliers,
            corr_weight=args.corr_weight,
            experiment=args.experiment,
            results_dir=args.results_dir,
        )
    else:
        run_batch_fitting_loop(
            nsamples_list=args.nsamples,
            skip_motor_noise=args.skip_motor_noise,
            data_path=args.data_path,
            base_checkpoint_path=args.base_checkpoint_path,
            include_outliers_list=[args.include_outliers],
            experiment=args.experiment,
            results_dir=args.results_dir,
        )
