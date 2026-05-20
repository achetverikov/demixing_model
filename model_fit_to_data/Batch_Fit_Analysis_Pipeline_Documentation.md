# Batch Fit Analysis Pipeline

## Overview

This document describes the model fitting pipeline. There are two entry-point scripts in `model_fit_to_data/`:

- **`fit_model_to_data.py`** — general-purpose fitting script. Use this for any dataset (including Fritsche, Fischer-Whitney, Moors, or your own data). Accepts a CSV via `--data-path` and writes results to `--output-dir`.
- **`model_fit_to_CSH2026_data.py`** — fitting script written specifically for the Chetverikov & Hansmann-Roth (2026) dataset. It handles the multi-experiment, multi-noise-condition structure of that dataset (including the `color_2` first/second-report split) and is not intended for general use.

Both scripts use the same underlying `GridBasedMultiConditionOptimizer` and produce compatible output formats.

---

## Pipeline Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ START: fit_model_to_data.py  (general)                           │
│        model_fit_to_CSH2026_data.py  (CSH 2026 data only)        │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. Load & Filter Data                                            │
│                                                                  │
│    - CSV specified via --data-path                               │
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
│    - Loads checkpoint (default: pretrained/model_epoch1500_8ktrain_20samples.pkl) │
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
│    - Methods: density (default), expectation, likelihood         │
│                                                                  │
│    Function: process_single_subject_multi_condition()            │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. Save Extended Results                                         │
│                                                                  │
│    Output: <output-dir>/                                         │
│    Files:                                                        │
│    - extended_fit_results.pkl                                    │
│    - extended_progress.json                                      │
│                                                                  │
│    Function: save_extended_results()                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Default Settings (`fit_model_to_data.py`)

- `--include-methods density`
- `--checkpoint-path pretrained/model_epoch1500_8ktrain_20samples.pkl`
- `--min-trials 30`
- Outliers excluded by default

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
  --results-path <output-dir>/extended_fit_results.pkl \
  --checkpoint-path pretrained/model_epoch1500_8ktrain_20samples.pkl \
  --summary-plots --csv-exports --no-individual-plots
```

---

## How To Run

### General use (`fit_model_to_data.py`)

```bash
python model_fit_to_data/fit_model_to_data.py \
  --data-path example_data/fritsche_prepared.csv \
  --output-dir results/fritsche
```

### CSH 2026 data (`model_fit_to_CSH2026_data.py`)

```bash
python model_fit_to_data/model_fit_to_CSH2026_data.py \
  --data-path data_color_comb.csv \
  --output-dir results/csh2026
```

Or via its loop mode (iterates over sample counts and outlier settings):

```bash
python model_fit_to_data/model_fit_to_CSH2026_data.py --mode loop
```

---

## Related Files

- `fit_model_to_data.py` — general fitting entry point
- `model_fit_to_CSH2026_data.py` — CSH 2026 specific entry point
- `create_unified_subject_plots.py` — post-fit plots and CSV exports
- `grid_based_multi_condition_optimizer_jax_loops.py` — optimizer core
