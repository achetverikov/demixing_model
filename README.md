# Demixing Model

The Demixing Model explains why a remembered, perceived, or evaluated item can be biased toward or away from another item. The central idea is that the brain must separate two noisy, overlapping representations. Depending on how similar the items are and where the noise occurs, this separation produces attraction or repulsion.

Use the model to **generate theoretical predictions** or **fit behavioral data**. Two trained models, representing 20 and 100 internal evidence samples, are included in `pretrained/` and ready to use.

## Install

Python 3.11+ is required. An NVIDIA GPU is strongly recommended for fitting.

- **Development container:** open the repository in VS Code and choose **Dev Containers: Reopen in Container** to get Python, R, and the GPU libraries.
- **CPU container:** select `.devcontainer/cpu/devcontainer.json` for a machine without an NVIDIA GPU. Fitting takes substantially longer on a CPU.
- **Manual installation:** create a Python environment and run `python -m pip install .`, or `python -m pip install ".[cuda]"` for a host with CUDA 13.

See the [installation guide](INSTALL.md) for setup, verification, and troubleshooting. Run the commands below from the repository root.

## Generate predictions

Start with a question such as: *How does the bias curve change as uncertainty about which item is which increases?* The example input holds target and non-target feature noise fixed and varies identifiability noise:

| Target feature noise | Non-target feature noise | Identifiability noise |
|---:|---:|---:|
| 10° | 30° | 20° |
| 10° | 30° | 60° |
| 10° | 30° | 120° |

```bash
python surface_simulator_for_predictions/surface_simulator.py \
  --input-path example_data/prediction_parameters.csv \
  --n-samples 20 \
  --output-path results/prediction_example.parquet \
  --skip-motor-noise
```

`--n-samples` selects the 20- or 100-sample observer model. The output contains curves for mean bias, density asymmetry, and response variability across stimulus dissimilarity. Parquet preserves each curve as a numeric array; CSV is also supported.

![Prediction curves for three identifiability-noise parameter values](docs/images/prediction_generation_example.png)

See the [prediction guide](surface_simulator_for_predictions/README.md) for input columns, output fields, and the R interface. To explore predictions interactively, launch the [browser](surface_browser/README.md):

```bash
streamlit run surface_browser/main_app.py
```

## Fit behavioral data

### Try the demo

```bash
python demo_fischer_whitney.py
```

The demo downloads the Fischer and Whitney (2014) orientation dataset, prepares a CSV, fits both observer models using five fitting criteria, and creates parameter tables and plots. The first run needs an internet connection. The complete demo can be slow on a CPU.

Results are saved in:

```text
results/fischer_whitney_20samples_circular/
results/fischer_whitney_100samples_circular/
```

![Empirical bias with a representative Demixing Model fit](docs/images/data_fitting_example.png)

The figure shows a mean-bias fit using the 20-sample model.

### Prepare your data

Supply one row per trial with these columns:

| Column | Meaning |
|---|---|
| `expName` | Experiment name or identifier |
| `subject` | Participant identifier |
| `condition` | Experimental condition |
| `abs_td_dist` | Absolute angular difference between the target and competing stimulus |
| `bias_to_distr_corr` | Signed response error, with positive values for attraction toward the competing stimulus and negative values for repulsion |
| `is_outlier` | Optional: 1 to exclude a trial, 0 to include it |

Use degrees in your study's angular scale. Column names can be changed through command-line options. The default minimum is 30 usable trials per participant and condition.

### Run a fit

```bash
python model_fit_to_data/fit_model_to_data.py \
  --data-path path/to/trials.csv \
  --output-dir my_study \
  --n-samples 20 \
  --include-methods density
```

For orientation data, add `--circ-space 180`: orientations repeat after 180°, so stimulus differences span 0–90° and response errors span ±90°. For color or motion direction, use the 360° default.

The fitter estimates target and non-target feature noise for each condition, with identifiability noise shared across a participant's conditions within an experiment. The default `density` criterion matches how response-distribution asymmetry changes with stimulus dissimilarity. Use `--include-methods smoothed_exp` to fit the mean-bias curve, or `likelihood` to fit the full response distribution trial by trial. The [fitting guide](model_fit_to_data/Batch_Fit_Analysis_Pipeline_Documentation.md) explains all criteria and options.

Results go to `results/my_study/`. Interrupted fits resume from saved results when you rerun the command. Each run also records its data, model, and settings so exports and plots use the same configuration.

### Export tables and plots

```bash
python model_fit_to_data/export_wnm_fit_curves.py \
  --results-dir results/my_study \
  --output-dir results/my_study/csv_exports

python model_fit_to_data/create_unified_subject_plots.py \
  --results-path results/my_study/extended_fit_results.pkl \
  --output-dir results/my_study \
  --summary-plots --no-individual-plots

python model_fit_to_data/plot_pdf_slices.py \
  --results-path results/my_study/extended_fit_results.pkl \
  --output-dir results/my_study \
  --optimizer density
```

The first command exports fitted parameters and curves. The second makes group-summary plots; use `--individual-plots` to include participant plots. The third plots predicted response distributions at selected dissimilarities. Set `--optimizer` to the fitting criterion you used.

```text
results/my_study/
├── extended_fit_results.pkl        # fitted parameters, losses, and empirical targets
├── extended_run_fingerprint.json   # data, model, and settings
├── extended_progress.json          # readable progress summary
├── csv_exports/                    # parameter and curve tables
├── pdf_slice_plots/
├── summary_plots/
└── unified_subject_plots/
```

## Understand the parameters

| Parameter | Interpretation |
|---|---|
| Target feature noise (`sd_feat1`) | Uncertainty in the reported feature of the target item |
| Non-target feature noise (`sd_feat2`) | Uncertainty in the other item's feature |
| Identifiability noise (`sd_spat`) | Uncertainty along the dimension that distinguishes the items, such as location or timing; shared across conditions during fitting |
| Motor noise (`sd_motor`) | Variability added at the response stage; fixed at zero by default and fitted with `--no-skip-motor-noise` |
| Internal sample count (`--n-samples`) | Amount of internal evidence available for separating the two representations; choose 20 or 100 |

The sample count is a theoretical assumption about evidence within a simulated trial. Experimental trial counts determine how much behavioral data you have for fitting.

The model uses a 360° internal scale. Feature-noise parameters span 2.5–200° and identifiability noise spans 5–200° on that scale. The fitter converts 180° data into model units, and exports provide explicitly labelled model and study-scale units. See the [pretrained model reference](pretrained/README.md) for supported domains.

Interpret predictions as curves across stimulus dissimilarity: their sign, size, and shape describe when attraction or repulsion occurs.

## How the model is built

The computational pipeline has four steps:

1. **Simulate internal evidence.** Generate noisy samples from two item representations and fit a mixture to recover each item's feature. Repeating this produces a distribution of response errors.
2. **Train a fast predictor.** A neural network predicts the parameters of a conditional wrapped-normal mixture (WNM), approximating the simulated response distribution across noise levels and stimulus dissimilarities.
3. **Predict or fit.** Evaluate the trained model at chosen noise values, or use continuous multistart optimization to estimate them from behavioral data.
4. **Inspect results.** Export curves and parameters, and compare predictions with observations across dissimilarity.

Read the [model pipeline explanation](docs/model_pipeline.md) for the scientific assumptions, simulation, training, and fitting details. To train a custom predictor, start with [simulation and training-data generation](surface_computation/README.md) and the [training guide](surrogate_training/wnm/README.md).

For direct inspection of simulated distributions or predictions in the identifiability dimension (`mu2`), use the [averaged-surface tools](surface_computation/Likelihood_Surface_Pipeline_Documentation.md). Generating a complete simulation grid is a substantial GPU computation.

## Compare models and reproduce project analyses

For analyses that compare model families or datasets, use a compiled `contextual_biases_database` bundle. It fixes trial selection, angular geometry, bandwidths, and empirical targets for all models. The [fitting guide](model_fit_to_data/Batch_Fit_Analysis_Pipeline_Documentation.md#fit-a-compiled-bundle) shows how to use one.

The companion [bias_model_comparison](https://github.com/achetverikov/bias_model_comparison) repository coordinates dataset preparation, fitting, comparison, and report generation. Follow its README for the required sibling repositories, datasets, and reproduction commands.

## Documentation

- [Installation](INSTALL.md)
- [Prediction in Python and R](surface_simulator_for_predictions/README.md)
- [Fitting, exports, and plots](model_fit_to_data/Batch_Fit_Analysis_Pipeline_Documentation.md)
- [Interactive browser](surface_browser/README.md)
- [Scientific pipeline explanation](docs/model_pipeline.md)
- [Pretrained model reference](pretrained/README.md)
- [Documentation index](docs/README.md)

## License, citation, and disclaimer

The code is distributed under the [MIT License](LICENSE).

If you use the model or demo, cite the relevant work:

- <a href="https://doi.org/10.1101/2023.03.26.534226" title="Chetverikov, A. (2023). Demixing model: A normative explanation for inter-item biases in memory and perception. bioRxiv. https://doi.org/10.1101/2023.03.26.534226">Chetverikov, A. (2023). Demixing model: A normative explanation for inter-item biases in memory and perception. <em>bioRxiv</em>.</a>
- <a href="https://doi.org/10.7554/eLife.111380.1" title="Chetverikov, A., &amp; Hansmann-Roth, S. (2026). Noise in competing representations determines the direction of memory biases. eLife, 15, RP111380. https://doi.org/10.7554/eLife.111380.1">Chetverikov, A., &amp; Hansmann-Roth, S. (2026). Noise in competing representations determines the direction of memory biases. <em>eLife, 15</em>, RP111380.</a>
- <a href="https://doi.org/10.1038/nn.3689" title="Fischer, J., &amp; Whitney, D. (2014). Serial dependence in visual perception. Nature Neuroscience, 17(5), 738–743. https://doi.org/10.1038/nn.3689">Fischer, J., &amp; Whitney, D. (2014). Serial dependence in visual perception. <em>Nature Neuroscience, 17</em>(5), 738–743.</a>

The code and documentation are provided as-is without warranty. Some project documentation was produced with AI assistance and may contain errors.

## Notes for developers

`shared/` provides model loading, prediction, circular geometry, and common data helpers. `model_fit_to_data/` handles fitting and result processing; `surface_computation/` and `surrogate_training/wnm/` handle simulation and training. See the [shared module reference](shared/README.md) and [test guide](tests/README.md).

Planned work is in `TODO.md`; completed work is recorded in `HISTORY.md`.
Generated simulation corpora, averaged surfaces, and experiment records belong under `$DEMIXING_ARTIFACT_ROOT`. These artifacts are not shipped and are not expected in a normal checkout.
