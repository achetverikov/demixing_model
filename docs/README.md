# Documentation

## Use the model

| Task | Guide |
|---|---|
| Get started | [Main README](../README.md) |
| Set up Python, R, and GPU support | [Installation](../INSTALL.md) |
| Generate theoretical predictions | [Python and R prediction guide](../surface_simulator_for_predictions/README.md) |
| Fit behavioral data and export results | [Fitting guide](../model_fit_to_data/Batch_Fit_Analysis_Pipeline_Documentation.md) |
| Explore predictions interactively | [Browser](../surface_browser/README.md) |
| Understand the scientific assumptions | [Model pipeline](../docs/model_pipeline.md) |
| Check parameter domains and model details | [Included trained models](../pretrained/README.md) |

## Build and develop

- [Simulation and training-data generation](../surface_computation/README.md)
- [Predictor training and packaging](../surrogate_training/wnm/README.md)
- [Surface averaging and network training](../neural_network_optimization/Neural_Network_Optimization_Pipeline_Documentation.md)
- [Shared model utilities](../shared/README.md)
- [Tests](../tests/README.md)

## Notes for developers

### Regenerate documentation figures

`generate_pipeline_figure.py` creates a standalone model-pipeline illustration. Its construction panels are explanatory; the response-density panel uses a small run of the observer simulator, the network tile uses the included 20-sample predictor, and the prediction and fitting panels use the outputs described below. After preparing those outputs, run:

```bash
python docs/generate_pipeline_figure.py
```

Use `--predictions`, `--trials`, and `--fit-curves` to select other prediction and fitting outputs.

`generate_readme_figures.py` creates the two data-derived figures embedded in the main README. The prediction figure reads the output of the documented prediction example:

```bash
python surface_simulator_for_predictions/surface_simulator.py \
  --input-path example_data/prediction_parameters.csv \
  --n-samples 20 \
  --output-path results/prediction_example.parquet \
  --skip-motor-noise
```

The fitting figure reads the prepared <a href="https://doi.org/10.1038/nn.3689" title="Fischer, J., &amp; Whitney, D. (2014). Serial dependence in visual perception. Nature Neuroscience, 17(5), 738–743. https://doi.org/10.1038/nn.3689">Fischer and Whitney (2014)</a> trials and fitted curves written by `export_wnm_fit_curves.py` under `csv_exports/fitted_curves.csv`. It excludes the pooled `combined` pseudo-subject from both the observations and fitted curve, then averages the four real subjects. The generator defaults to the `smoothed_exp` mean-bias objective. Use `--fit-objective` to select another mean-bias fit. After running `demo_fischer_whitney.py`, regenerate both assets with:

```bash
python docs/generate_readme_figures.py
```

Alternative or isolated fit outputs can be selected with `--fit-curves`; use `--predictions` for an alternative prediction export. Commit the rendered PNG files together with any generator change so GitHub does not need to run the model to display the README.

### Development records

[HISTORY.md](../HISTORY.md) summarizes completed work. [history/wnm_transition/](history/wnm_transition/README.md) preserves model-selection findings and the transition record. Use the guides above for runnable workflows.
