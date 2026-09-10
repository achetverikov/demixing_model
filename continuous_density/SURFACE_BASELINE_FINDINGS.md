# Surface-model baseline: development results

Artifact paths and analysis scripts named below are relative to
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`.
Generated artifacts are external and are not shipped with or expected in a
normal checkout.

This is the legacy surface-NN baseline for the single-condition, actual-DM
recovery panel (`n_samples = 100`). It is a comparator for the WNM recovery
analysis, not a proposed direction for further surface-model development.

## Scope and fitting methods

- Only the eight development tuples were analyzed: 8 tuples × 5 response seeds
  × 3 trial counts (180, 450, and 900) = 120 subject datasets.
- At this development-baseline stage, the four held-out tuples had been fitted
  but their fitted parameters and empirical curves had not been read. They were
  opened only after the objective-specific WNM settings were frozen.
- Likelihood, bias-weighted CRPS, and smoothed exponential bias loss used the
  deployed hierarchical search. Density-asymmetry CCC used exhaustive 1-degree
  curve-cache search.
- Every result below is therefore a fit to sampled behavior. Comparison with
  independent/noiseless actual-DM reference curves remains necessary when the
  WNM baseline is evaluated.

## Parameter recovery

`Joint log RMSE` is `sqrt(mean(log(fit / truth)^2))` over all three parameters
and all 120 datasets. “All within ×1.5” requires every fitted/truth ratio to be
between 2/3 and 1.5.

| Fit objective | Median absolute error: feat1 | feat2 | spatial | Joint log RMSE | All within ×1.5 |
|---|---:|---:|---:|---:|---:|
| Likelihood | 1.0° | 3.5° | 9.4° | 0.453 | 62.5% |
| Bias-weighted CRPS | 1.8° | 5.1° | 10.3° | 0.602 | 46.7% |
| Smoothed exponential bias | 12.0° | 27.5° | 15.0° | 0.899 | 20.8% |
| Density asymmetry | 9.0° | 9.0° | 18.0° | 1.093 | 18.3% |

Joint log RMSE by trial count:

| Fit objective | 180 | 450 | 900 |
|---|---:|---:|---:|
| Likelihood | 0.490 | 0.469 | 0.395 |
| Bias-weighted CRPS | 0.709 | 0.542 | 0.539 |
| Smoothed exponential bias | 0.986 | 0.826 | 0.878 |
| Density asymmetry | 1.138 | 1.100 | 1.038 |

Likelihood is the strongest surface baseline for parameter recovery, followed
by bias-weighted CRPS. Curve-targeted fitting can reproduce its target curve
well without recovering the generating parameters.

## Curve recovery across dissimilarity

The table reports median Lin CCC and median RMSE across the 120 development
datasets. Mean-bias and bias-SD RMSE are in degrees; asymmetry RMSE is unitless.

| Fit objective | Mean-bias CCC | Mean RMSE | Bias-SD CCC | SD RMSE | Asymmetry CCC | Asymmetry RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Likelihood | 0.704 | 0.634 | 0.346 | 2.900 | 0.568 | 0.0646 |
| Bias-weighted CRPS | 0.781 | 0.515 | 0.326 | 2.826 | 0.577 | 0.0629 |
| Smoothed exponential bias | 0.922 | 0.287 | 0.027 | 6.646 | 0.350 | 0.0790 |
| Density asymmetry | 0.357 | 1.523 | 0.017 | 6.933 | 0.887 | 0.0311 |

Median CCC by trial count:

| Fit objective | Trials | Mean bias | Bias SD | Asymmetry |
|---|---:|---:|---:|---:|
| Likelihood | 180 | 0.512 | 0.247 | 0.481 |
| Likelihood | 450 | 0.704 | 0.373 | 0.466 |
| Likelihood | 900 | 0.763 | 0.485 | 0.693 |
| Bias-weighted CRPS | 180 | 0.680 | 0.213 | 0.526 |
| Bias-weighted CRPS | 450 | 0.744 | 0.335 | 0.560 |
| Bias-weighted CRPS | 900 | 0.861 | 0.469 | 0.710 |
| Smoothed exponential bias | 180 | 0.921 | 0.018 | 0.328 |
| Smoothed exponential bias | 450 | 0.917 | 0.014 | 0.302 |
| Smoothed exponential bias | 900 | 0.944 | 0.160 | 0.498 |
| Density asymmetry | 180 | 0.366 | 0.019 | 0.841 |
| Density asymmetry | 450 | 0.391 | 0.011 | 0.892 |
| Density asymmetry | 900 | 0.317 | 0.022 | 0.938 |

The objectives specialize: smoothed exponential bias has the best mean-bias
CCC in 118/120 datasets, while density fitting has the best asymmetry CCC in
120/120. Likelihood has the best bias-SD CCC in 56/120 and bias-weighted CRPS
in 34/120; each curve-specialized method wins 15/120. Likelihood and
bias-weighted CRPS are the more balanced baselines across all three summaries.

## Flat curves and CCC

CCC is

`2 covariance / (predicted variance + target variance + squared mean difference)`.

When a target is nearly flat across dissimilarity, it contains almost no shape
to correlate. A small absolute level or amplitude mismatch can then drive CCC
toward zero even when RMSE is small. An exactly constant curve has no useful
correlation-based score. Consequently CCC must be read with RMSE and with the
target curve's across-dissimilarity SD/range, all of which are saved in the
row-level results.

The clearest instance is `narrow_1`. Across its 15 datasets, the median target
mean-bias SD is only 0.054° (median range 0.183°), and the median target bias-SD
SD is only 0.109° (median range 0.427°). For this case, median mean-bias CCC is
only 0.13–0.56 across methods despite median mean-bias RMSE of 0.05–0.31°.
Likewise, bias-SD CCC is essentially zero. These low CCC values principally say
that there is little across-dissimilarity variation to recover; they should not
be interpreted alone as large absolute curve errors.

## Why some empirical bias-SD bins are missing

These are not empty bins and the files do not contain stored infinite values.
The empirical circular SD uses the small-sample-corrected concentration

`rho_hat_squared = (n * sample_resultant_squared - 1) / (n - 1)`.

Circular SD is `sqrt(-2 log(rho))`. When the corrected concentration is zero or
negative, the observations are consistent with the uniform circular limit,
where this SD parameterization is unbounded. The estimator therefore returns
`NaN` instead of inventing a very large finite SD. For `n = 10`, this happens
when the sample resultant length is at most `sqrt(1/10) = 0.316`.

There are 10 such bins among 2,160 development dataset-bins (0.46%). All have
10 trials and occur in the broadest case, `broad_3`, at 180 trials:

- seed 0: 5 missing bins
- seed 3: 4 missing bins
- seed 4: 1 missing bin

No bias-SD bins are missing at 450 or 900 trials, or in any other development
case. The SD CCC/RMSE calculations use the remaining finite bins (13–18 points
per dataset). This finite-bin restriction should be remembered when comparing
methods on the three affected `broad_3`, 180-trial datasets.

## Saved artifacts

- `surface_baseline_development_metrics.csv`: one row per development dataset
  and fit method (480 rows), with truth/fits, parameter errors, CCC, RMSE,
  target/predicted variability, and missing-bin counts.
- `surface_baseline_development_summary.csv`: overall, by-trial-count, and
  by-regime summaries, including CCC quartiles and target variability.
- `surface_baseline_empirical_sd_bins.csv`: all 2,160 development SD bins with
  counts, raw and corrected resultant squared, SD, and missing-value reason.
- `surface_baseline_analysis_manifest.json`: definitions and SHA-256 provenance.
- `analyze_surface_baseline.py`: the reproducible analysis that generated the
  CSVs and manifest.

These artifacts are development-only. The corresponding held-out surface
metrics have since been analyzed after the WNM fitting/search settings were
frozen; they are recorded in `surface_baseline_held_out_*.csv` and the
objective-specific WNM findings documents.
