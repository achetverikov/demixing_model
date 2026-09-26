# Fit behavioral data

Fit the Demixing Model to trial-level response errors, then export parameters and plot predictions across stimulus dissimilarity. Run these commands from the repository root after following the [installation guide](../INSTALL.md).

## Fit a CSV

```bash
python model_fit_to_data/fit_model_to_data.py \
  --data-path example_data/fritsche_prepared.csv \
  --n-samples 20 \
  --output-dir results/fritsche
```

For orientation data, add `--circ-space 180`. The default, 360°, suits color and motion direction.

The CSV contains one row per trial:

| Column | Meaning | Rename with |
|---|---|---|
| `expName` | Experiment identifier | `--exp-col` |
| `subject` | Participant identifier | `--subject-col` |
| `condition` | Condition identifier | `--condition-col` |
| `abs_td_dist` | Absolute target–competitor difference in study degrees | `--x-col` |
| `bias_to_distr_corr` | Response error signed toward the competitor: positive for attraction, negative for repulsion | `--y-col` |
| `is_outlier` | Optional flag: 1 excludes the trial | `--outlier-col` |

The fitter groups conditions within each participant and experiment. Each condition gets its own `sd_feat1` and `sd_feat2`; the group shares `sd_spat` and, when enabled, `sd_motor`. By default, conditions need at least 30 usable trials and flagged outliers are excluded.

## Choose a fitting criterion

Select one or more criteria with `--include-methods`, for example:

```bash
python model_fit_to_data/fit_model_to_data.py \
  --data-path path/to/trials.csv \
  --output-dir results/my_study \
  --include-methods density smoothed_exp likelihood
```

Each criterion produces a separate fit:

| Criterion | What it matches |
|---|---|
| `density` (default) | Density asymmetry: how much response probability lies toward versus away from the competing item, across dissimilarity |
| `smoothed_exp` | The smoothed circular-mean bias curve |
| `likelihood` | The conditional response density at each observed trial |
| `crps` | Trial-level response distributions, scored using circular distance |
| `balanced_crps` | Conditional response distributions, with equal weight across supported dissimilarities |
| `bias_weighted_crps` | Conditional response distributions, weighted by empirical bias |

Choose the criterion to match the question: `smoothed_exp` for mean bias, `density` for attraction–repulsion asymmetry, or a distribution-based criterion for the full pattern of response errors. Inspect the resulting curves across dissimilarity as well as the loss.

### Density asymmetry and mean bias

`density` minimizes `1 − CCC`, where CCC is Lin's concordance correlation between empirical and predicted asymmetry curves. It measures agreement in shape, amplitude, and offset. A perfect match has loss zero. A constant empirical target cannot be fitted with this criterion and produces an error identifying the affected condition.

`smoothed_exp` minimizes the weighted squared angular difference between empirical and predicted circular means.

Both criteria pool predictions at the observed stimulus differences using the same dissimilarity weights as the empirical estimates. The default Gaussian weighting has a nominal SD of 20° in model units. For `density`, the predicted response density also receives the same bias-axis KDE bandwidth as the empirical density estimate. This makes the plotted and fitted summaries comparable at the observed trial design.

`balanced_crps` and `bias_weighted_crps` likewise pool predicted distributions using the empirical dissimilarity weights. `likelihood` and `crps` score individual trials directly.

## Adjust the fit

| Option | Default | Use |
|---|---|---|
| `--n-samples` | `20` | Choose the 20- or 100-sample observer model |
| `--circ-space` | `360` | Set the angular period of the behavioral data to 180 or 360 |
| `--min-trials` | `30` | Set the minimum usable trials per condition |
| `--include-outliers` | Off | Include trials flagged as outliers |
| `--no-skip-motor-noise` | Off | Estimate response-stage noise shared across conditions |
| `--continuous-starts` | `64` | Set the number of optimization starting points |
| `--continuous-seed` | `0` | Set the seed used to generate starting points |
| `--checkpoint-path` | Selected by `--n-samples` | Load a custom trained model |

The optimizer uses bounded L-BFGS-B from 64 deterministic starting points, evaluated in two sequential batches of 32. It jointly estimates condition-specific and shared parameters and records the outcome of each start. Bounds come from the selected model's supported domain.

Motor noise broadens the response distribution through a circular Gaussian convolution. When enabled, its upper bound is based on the smallest condition error SD, with a 10% margin and a maximum of 50 model degrees. Mean-bias fitting is skipped in motor-noise runs because symmetric motor noise leaves the circular mean unchanged.

## Results and resuming

Relative output directories are placed under `results/`; absolute paths are used as supplied.

| File | Contents |
|---|---|
| `extended_fit_results.pkl` | Fitted parameters, method-specific losses, empirical targets, and metadata; used to resume fitting and generate outputs |
| `extended_run_fingerprint.json` | Data and checkpoint hashes, column mapping, filtering, circular geometry, objective definitions, and optimizer settings |
| `extended_progress.json` | Human-readable progress summary |

Rerun the command to resume. Additional criteria can be added to a run when its fingerprint matches. Use `--no-resume` to recompute with the same settings. For changed data or settings, choose a new output directory, or use `--force-refit` to discard existing results and refit.

Each condition stores `{method}_fitted_params`, `{method}_loss`, and `{method}_optimization_time`. Exports and plots load the model identified by the run fingerprint and use the saved angular period.

## Export tables

```bash
python model_fit_to_data/export_wnm_fit_curves.py \
  --results-dir results/my_study \
  --output-dir results/my_study/csv_exports
```

The exporter creates:

- `fitted_parameters.csv`: noise estimates, fitted losses, and evaluations under the other criteria;
- `fitted_curves.csv`: predictions across stimulus dissimilarity;
- `manifest.json`: model identity and unit definitions.

Use columns ending in `*_model_deg` or `*_model_deg2` for the internal 360° scale, and `*_deg` or `*_deg2` for the study's physical angular scale. Unsuffixed SD columns contain model degrees.

Compiled-bundle fits also produce `trial_loglik_split/` and `trial_loglik_checks.csv`. These associate likelihoods with the original trial identities and verify that their sum reproduces the fitted likelihood objective under the recorded numerical precision.

## Plot results

Create group-summary plots:

```bash
python model_fit_to_data/create_unified_subject_plots.py \
  --results-path results/my_study/extended_fit_results.pkl \
  --output-dir results/my_study \
  --summary-plots --no-individual-plots
```

Use `--individual-plots` to add participant plots. To examine response distributions at selected stimulus differences:

```bash
python model_fit_to_data/plot_pdf_slices.py \
  --results-path results/my_study/extended_fit_results.pkl \
  --output-dir results/my_study \
  --optimizer density
```

Set `--optimizer` to the criterion used for the fit. Labels, circular period, and model identity are read from the saved results.

## Fit a compiled bundle

Use a `contextual_biases_database` bundle when comparing model families or coordinating analyses across datasets. A bundle supplies shared trial selection, angular geometry, bandwidths, empirical targets, and stable row identities.

```bash
python model_fit_to_data/fit_model_to_data.py \
  --bundle ../contextual_biases_database/data/bundles/<dataset>/<analysis> \
  --n-samples 100 \
  --output-dir results/<dataset>
```

Then run the same export and plotting commands shown above. Bundle metadata supplies participant, experiment, and condition labels.

## Notes for developers

The fitting pipeline uses `shared/prediction.py` to evaluate the conditional wrapped-normal mixture and `continuous_fit.py` / `continuous_optimizer.py` for parameter estimation. `fitting_targets.py`, `empirical_targets.py`, and `wnm_scoring.py` define the empirical summaries and their matched model predictions. `compiled_bundle.py` loads bundle inputs.

The surface-network tools accept an explicit surface checkpoint with `--search hierarchical` or `--search exhaustive`. Hierarchical search refines a parameter grid; exhaustive search scans cached density-asymmetry curves on a lattice and requires motor noise fixed at zero. `--curve-cache` selects the cache location; `build_curve_cache.py` prepares it. In exhaustive mode, other criteria use hierarchical search.

The compatibility criteria `expectation` and `density_legacy` provide hard-binned mean fitting and a range-scaled MSE/correlation asymmetry loss, respectively. `--corr-weight` configures `density_legacy`. Use the criteria listed above for new analyses.

Detailed surface-search benchmarks and earlier objective comparisons are preserved with generated results under `$DEMIXING_ARTIFACT_ROOT/exploration_archive/documentation_20260926/`. These artifacts are not shipped and are not expected in a normal checkout.
