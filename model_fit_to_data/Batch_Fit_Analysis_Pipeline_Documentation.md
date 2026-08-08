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
│    - Search: hierarchical zoom, or an exhaustive scan over a      │
│      precomputed curve cache (density only, --search)            │
│    - Methods: density (default) + 7 others, see below            │
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
│    - extended_run_fingerprint.json                               │
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
- `--search hierarchical`
- Outliers excluded by default

The `--search` default is the **generic CLI default**, not the production
configuration: `bias_model_comparison/pipeline/regenerate_all_fits.sh` passes
`--search exhaustive` (see `DM_SEARCH`). The default is left unchanged so that
every existing invocation keeps its behaviour and the backend that produced a
result is visible in the command line rather than implicit in a version.

## Search backends

| Backend | What it does | When |
|---|---|---|
| `hierarchical` | Zooming grid search, refining between grid points | Every objective; the default |
| `exhaustive` | Scans a precomputed 1-degree lattice of density-asymmetry curves | `density` only |

Dispatch is per **method**: `--search exhaustive` routes `density` to the scan
and leaves every other method hierarchical, because the cache holds density
curves and nothing else.

The scan is exact on its lattice. At a fixed shared parameter the conditions are
independent, so each condition's own minimum can be taken separately and summed
to give the joint optimum — which is what makes a full scan affordable. It cannot
search *between* lattice points, so it is not uniformly better than the zoom:
measured over 51 csh2026 groups the two agree to 0.15% of the loss (median), with
the zoom ahead in 30 groups on sub-degree points and the scan ahead in 21 where
the zoom settled in a worse basin. The scan takes ~4 s per group against ~34 s.

`--curve-cache PATH` selects the cache root; it is built on demand if absent
(single-writer locked), or ahead of time with
`model_fit_to_data/build_curve_cache.py`. The cache is keyed by a digest of
everything that changes a curve — checkpoint, both parameter lattices, the
feat_diff and mu1_bias grids, the density-target settings — so a cache built
under different settings cannot be read by mistake. `--no-skip-motor-noise` is
refused under `exhaustive`: `sd_motor` is a fourth axis the cache does not span.

## Refusing stale results

A run records how it was produced in `<output-dir>/extended_run_fingerprint.json`
— dataset and checkpoint hashes, circular space, grids, objective definitions,
search backend, column mapping, outlier policy. Resuming into a directory whose
fingerprint differs, or which has results but no fingerprint, **raises** and
prints which fields differ, rather than appending fits computed one way onto fits
computed another. `--force-refit` discards those results and refits; it is not
needed to add a method to a run whose fingerprint matches.

## Dissimilarity smoothing in the fitting objectives

The NN predicts simulation surfaces that already contain a nominal 6° Gaussian
smoother across dissimilarity (three steps on the 2° grid). Fitting then handles the
empirical and predicted sides as follows:

| Objective | Empirical side | Predicted side |
|---|---|---|
| `likelihood`, `crps` | Raw trials | Pointwise NN column; no added dissimilarity smoother |
| `expectation` | Circular means in 4° bins | NN circular mean at the matching column |
| `smoothed_exp` | Rolling circular moments, nominal 20° Gaussian SD | Pointwise NN mean curve; no added 20° smoother |
| `density`, `density_legacy` | Density-asymmetry curve, nominal 20° Gaussian trial weights | NN asymmetry curve with a nominal 20° Gaussian convolution |
| `balanced_crps`, `bias_weighted_crps` | Conditional empirical distributions, nominal 20° Gaussian trial weights | Pointwise NN distribution |

Thus the default `density` objective does apply similar 20° dissimilarity smoothing to
both sides. It is not exact: the NN side also inherits the upstream 6° smoother, and its
later discrete 20° kernel has finite support and edge padding, whereas the empirical
kernel is normalized over the available trials. Since 6° is small relative to 20°, this
is expected to be a minor approximation, but the objectives should not all be described
as using matched smoothing.

---

## The density objective

`density` minimises `1 - CCC`, Lin's concordance correlation between the
predicted and empirical density-asymmetry curves. With
`D = var_p + var_t + (mean_p - mean_t)^2`, `1 - CCC = MSE/D` and `CCC = r * C_b`,
so the score carries accuracy (`C_b`: right amplitude and offset) as well as
precision (`r`: right shape).

`density_legacy` is the pre-2026-08 objective, `0.75*MSE/range + 0.25*(1-r)`,
retained so published numbers stay reproducible. Its MSE term divides by `range`
rather than `range**2`, which is not scale-free and left the term too weak to
constrain amplitude: over 204 csh2026 condition fits it accepted 115 curves at
least 5x too small in amplitude, against none under CCC. `--corr-weight` reaches
only `density_legacy`; it is inert for `density`.

A constant empirical target makes both objectives **raise**: CCC is undefined
there and returns its worst value for a perfect match. Conditions are never
silently dropped from a fit.

## Outputs

Each condition entry in `extended_fit_results.pkl` includes method-specific fields:
- `{method}_fitted_params`
- `{method}_optimization_time`
- `{method}_loss`

The CSV export adds `density_ccc`, `density_r` and `density_C_b` for the density
optimizer, so an amplitude failure is visible per condition rather than hidden
inside a single number. They are `NaN` where a component is undefined.

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
