# Batch Fit Analysis Pipeline

## Overview

This document describes the reusable model fitting and result pipeline in `model_fit_to_data/`.

There are two intentionally different input paths:

- **Production WNM:** `fit_model_to_data.py --bundle ...` consumes a compiled
  `contextual_biases_database` bundle. The bundle fixes the cleaned trial
  population, analysis cells, observed-design operators, bandwidths, and target
  arrays before Demixing Model fitting begins.
- **Legacy CSV/surface replay:** `fit_model_to_data.py --data-path ...` derives
  those semantics from a CSV and remains available for historical reproduction,
  recovery tooling, and exploratory work. It is not a production WNM input path.

Both paths write fingerprinted results to `--output-dir`; plotting and rescoring
must recover the surrogate from that fingerprint rather than substituting the
current default.

---

## Pipeline Flow Diagram

Production bundle path:

```text
compiled bundle
      │
      ▼
fit_model_to_data.py
      │
      ▼
ContinuousEngine + packaged WNM
      │
      ▼
extended_fit_results.pkl + extended_run_fingerprint.json
      │
      ├── create_unified_subject_plots.py
      ├── plot_pdf_slices.py
      └── export_wnm_fit_curves.py
```

The historical CSV/surface path is retained as follows:

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

The bare CLI defaults are retained for legacy CSV compatibility:

- `--include-methods density`
- `--checkpoint-path pretrained/model_epoch1425_10ktrain_20samples.pkl`
- `--min-trials 30`
- `--search hierarchical`
- Outliers excluded by default

They are **not** the production WNM configuration. A production run supplies a
compiled `--bundle` and a packaged WNM artifact (for example
`pretrained/wnm_k12_100samples.pkl`). Bundle-native fitting uses the continuous
WNM engine and records that backend, the artifact digest, and bundle identity in
the run fingerprint. The CLI refuses `--data-path` together with
`--search continuous` so CSV-derived empirical semantics cannot be mistaken for
a production WNM run.

## Search backends

| Backend | What it does | When |
|---|---|---|
| `continuous` | Bounded multistart gradient optimization of the analytic WNM objectives | Production bundle-native WNM fitting |
| `hierarchical` | Zooming grid search, refining between grid points | Legacy surface/CSV fitting; generic CLI default |
| `exhaustive` | Scans a precomputed 1-degree lattice of density-asymmetry curves | Legacy surface `density` only |

On the legacy surface path dispatch is per **method**: `--search exhaustive`
routes `density` to the scan and leaves every other method hierarchical,
because the cache holds density curves and nothing else. Bundle-native WNM runs
do not use the surface curve cache.

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

The legacy surface NN predicts simulation surfaces that already contain a nominal
6° Gaussian smoother across dissimilarity (three steps on the 2° grid). The WNM
backend instead evaluates its conditional distribution analytically at the stored
prediction coordinates. Fitting then handles the empirical and predicted sides as
follows:

| Objective | Empirical side | Predicted side |
|---|---|---|
| `likelihood`, `crps` | Raw trials | Family-specific direct distribution evaluation; WNM uses the analytic conditional density/cell probabilities, surface replay uses the matching NN column |
| `expectation` | Circular means in 4° bins | Surrogate circular mean at the matching coordinates |
| `smoothed_exp` | Rolling circular moments, nominal 20° Gaussian SD | Surrogate complex moments pooled through the same observed-design operator |
| `density` | Exact wrapped signed mass after a pooled-SJ bias KDE and nominal 20° Gaussian trial weights | Surrogate density convolved by the same bias KDE, then pooled through the same observed-design operator |
| `density_legacy` | Legacy sampled-KDE density-asymmetry curve | Legacy surface asymmetry curve with a nominal 20° Gaussian convolution |
| `balanced_crps`, `bias_weighted_crps` | Conditional empirical distributions, nominal 20° Gaussian trial weights | Family-specific predicted response distribution |

Thus the current `density` and `smoothed_exp` objectives apply the identical
empirical observed-design operator to either model family's prediction. The
surface NN still inherits the upstream 6° simulation smoother because that is
part of the legacy surrogate itself, not an extra fitting-objective smoother.
The WNM path adds no reconstructed surface or hidden dissimilarity interpolation.

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

The production WNM tabular export is generated by
`export_wnm_fit_curves.py`. It carries fitted parameters, method-specific fit
losses, cross-objective evaluation losses, explicit model/physical angular
units, bundle identity, surrogate identity, and likelihood replay checks. The
retired plotter-specific CCC decomposition columns are no longer part of the
production export contract.

---

## Post-Fit Plots

Use `create_unified_subject_plots.py` to generate unified subject and group-summary
plots from saved results. CSV export from this plotter has been retired. Bundle-native result keys are opaque analysis-cell
IDs; the plotter reads subject, experiment, condition, and report-order labels
from the stored `analysis_cell_values` metadata. The run fingerprint identifies
and verifies the fitted surrogate, and the saved results identify the physical
circular period, so neither checkpoint nor `--circ-space` should normally be
supplied. A conflicting period override raises rather than silently relabeling
curves or fitted SDs.

```bash
python model_fit_to_data/create_unified_subject_plots.py \
  --results-path <output-dir>/extended_fit_results.pkl \
  --output-dir <output-dir> \
  --summary-plots --no-individual-plots
```

The standalone PDF-slice command follows the same fingerprint identity and now
supports both direct WNM and historical surface fits:

```bash
python model_fit_to_data/plot_pdf_slices.py \
  --results-path <output-dir>/extended_fit_results.pkl \
  --output-dir <output-dir> \
  --optimizer density
```

For production WNM exports, `export_wnm_fit_curves.py` is the compact,
bundle-aware output path. It preserves analysis-cell and fit-group identity,
explicit model/physical angular units, direct WNM curves, and trial-likelihood
replay checks.

---

## How To Run

### Production WNM from a compiled bundle

```bash
python model_fit_to_data/fit_model_to_data.py \
  --bundle ../contextual_biases_database/data/bundles/<dataset>/<bundle> \
  --checkpoint-path pretrained/wnm_k12_100samples.pkl \
  --output-dir results/<dataset>
```

### Legacy CSV/surface replay

```bash
python model_fit_to_data/fit_model_to_data.py \
  --data-path example_data/fritsche_prepared.csv \
  --output-dir results/fritsche_legacy
```

---

## Related Files

- `fit_model_to_data.py` — fitting entry point for compiled WNM and legacy CSV runs
- `compiled_bundle.py` — strict reader for bundle-native WNM inputs
- `create_unified_subject_plots.py` — family-aware post-fit subject, summary, and PDF-slice plots
- `plot_pdf_slices.py` — family-aware conditional-density slice plots
- `export_wnm_fit_curves.py` — sole production WNM tabular exporter for curves, parameters, and trial likelihoods
- `grid_based_multi_condition_optimizer_jax_loops.py` — legacy surface optimizer core
