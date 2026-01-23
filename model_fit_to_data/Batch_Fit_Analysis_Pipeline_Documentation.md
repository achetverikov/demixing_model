# Batch Fit Analysis Pipeline (V2)

## Overview

This document describes the model fitting pipeline driven by `model_fit_to_data_analysis.py`. It runs grid-based multi-condition optimization across subjects and noise conditions, saving extended fit results per method (default: density only).

---

## Pipeline Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ START: model_fit_to_data_analysis.py                             │
│ Command: python model_fit_to_data_analysis.py                    │
│ Defaults: 20 samples, no outliers, density method                │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. Load & Filter Data                                            │
│                                                                  │
│    - CSV: data_color_comb.csv                                    │
│    - Optional outlier removal (outlier_col='is_outlier')         │
│                                                                  │
│    Function: load_data()                                         │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. Group Conditions                                              │
│                                                                  │
│    - Groups by subject + experiment                              │
│    - Each group must have >=2 conditions                         │
│                                                                  │
│    Function: group_conditions_by_subject_experiment()            │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. Initialize Optimizer                                          │
│                                                                  │
│    Class: GridBasedMultiConditionOptimizer                       │
│    File: grid_based_multi_condition_optimizer_jax_loops.py       │
│                                                                  │
│    - Loads checkpoint: neural_net_checkpoints_{n}samples/...     │
│    - Uses dummy data for initial setup                           │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. Per-Subject Optimization                                      │
│                                                                  │
│    For each subject group:                                       │
│    - Filter data for fitting                                     │
│    - Run hierarchical grid search                                │
│    - Methods: density (default)                                  │
│                                                                  │
│    Function: process_single_subject_multi_condition()            │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. Save Extended Results                                         │
│                                                                  │
│    Output: model_fit_to_data_results_v2_.../                      │
│    Files:                                                        │
│    - extended_fit_results.pkl                                    │
│    - extended_progress.json                                      │
│                                                                  │
│    Function: save_extended_results()                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Default Settings

Defaults configured in `model_fit_to_data_analysis.py`:
- `nsamples_list=[20]`
- `include_outliers_list=[False]`
- `include_methods=['density']`

---

## Outputs

Each condition entry in `extended_fit_results.pkl` includes method-specific fields:
- `{method}_fitted_params`
- `{method}_optimization_time`
- `{method}_loss`

---

## Post-Fit Plots

Use `create_unified_subject_plots.py` to generate unified subject plots and CSV exports from the saved results.

Example:
```bash
python model_fit_to_data/create_unified_subject_plots.py \
  --results-path model_fit_to_data_results_v2_no_motor_noise/20samples/no_outliers/extended_fit_results.pkl \
  --checkpoint-path neural_net_checkpoints_20samples/model_epoch_1500.pkl \
  --summary-plots --csv-exports --no-individual-plots
```

---

## How To Run

From the repo root:
```bash
python model_fit_to_data_analysis.py
```

Outputs will be written under:
```
model_fit_to_data_results_v2_no_motor_noise/20samples/no_outliers/
```

---

## Related Files (Copied for Standalone Use)

The standalone folder `model_fit_to_data/` includes:
- `model_fit_to_data_analysis.py`
- `create_unified_subject_plots.py` (post-fit plots/exports)
- `grid_based_multi_condition_optimizer_jax_loops.py`
- `config.py`
- `utils.py`
- `seed_manager.py`
- `surface_folder_parsing.py`
- `surface_functions.py`
- `plotting.py`
