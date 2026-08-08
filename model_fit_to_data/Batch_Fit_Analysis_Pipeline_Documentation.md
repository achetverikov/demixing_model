# Batch Fit Analysis Pipeline

## Overview

This document describes the reusable model fitting pipeline in `model_fit_to_data/`:

- **`fit_model_to_data.py`** — general-purpose fitting script. Use this for any dataset (including Fritsche, Fischer-Whitney, Moors, or your own data). Accepts a CSV via `--data-path` and writes results to `--output-dir`.

---

## Pipeline Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ START: fit_model_to_data.py                                     │
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
│    - Conditions need at least --min-trials observations          │
│                                                                  │
│    Function: group_conditions()                                  │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. Initialize Optimizer                                          │
│                                                                  │
│    Class: GridBasedMultiConditionOptimizer                       │
│    File: grid_based_multi_condition_optimizer_jax_loops.py       │
│                                                                  │
│    - Loads checkpoint (default: pretrained/model_epoch1425_10ktrain_20samples.pkl) │
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
│    Function: process_subject()                                   │
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
│    Function: save_results()                                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Default Settings (`fit_model_to_data.py`)

- `--include-methods density`
- `--checkpoint-path pretrained/model_epoch1425_10ktrain_20samples.pkl`
- `--min-trials 30`
- Outliers excluded by default

## Dissimilarity smoothing in the fitting objectives

The NN predicts simulation surfaces that already contain a nominal 6° Gaussian
smoother across dissimilarity (three steps on the 2° grid). Fitting then handles the
empirical and predicted sides as follows:

| Objective | Empirical side | Predicted side |
|---|---|---|
| `likelihood`, `crps` | Raw trials | Pointwise NN column; no added dissimilarity smoother |
| `expectation` | Circular means in 4° bins | NN circular mean at the matching column |
| `smoothed_exp` | Rolling circular moments, nominal 20° Gaussian SD | Pointwise NN mean curve; no added 20° smoother |
| `density` | Density-asymmetry curve, nominal 20° Gaussian trial weights | NN asymmetry curve with a nominal 20° Gaussian convolution |
| `balanced_crps`, `bias_weighted_crps` | Conditional empirical distributions, nominal 20° Gaussian trial weights | Pointwise NN distribution |

Thus the default `density` objective does apply similar 20° dissimilarity smoothing to
both sides. It is not exact: the NN side also inherits the upstream 6° smoother, and its
later discrete 20° kernel has finite support and edge padding, whereas the empirical
kernel is normalized over the available trials. Since 6° is small relative to 20°, this
is expected to be a minor approximation, but the objectives should not all be described
as using matched smoothing.

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
  --checkpoint-path pretrained/model_epoch1425_10ktrain_20samples.pkl \
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

---

## Related Files

- `fit_model_to_data.py` — general fitting entry point
- `create_unified_subject_plots.py` — post-fit plots and CSV exports
- `grid_based_multi_condition_optimizer_jax_loops.py` — optimizer core
