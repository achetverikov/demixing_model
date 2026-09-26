# Documentation index

For current executable workflows, start with:

- [root README](../README.md) - installation overview, prediction, fitting, and repository map;
- [installation guide](../INSTALL.md) - environments, GPU/CPU setup, verification, and troubleshooting;
- [fitting pipeline](../model_fit_to_data/Batch_Fit_Analysis_Pipeline_Documentation.md) - WNM fitting, objectives, outputs, exports, plotting, and historical surface replay;
- [prediction interface](../surface_simulator_for_predictions/README.md) - Python/R prediction generation;
- [simulation and training-data generation](../surface_computation/README.md) - current WNM simulation path and historical surface generation;
- [WNM training and packaging](../surrogate_training/wnm/README.md) - training checkpoints and production artifacts;
- [shared runtime layer](../shared/README.md) - WNM runtime, surrogate identity, prediction operations, and common helpers;
- [prediction/surface browser](../surface_browser/README.md) - WNM on-demand browsing and optional stored-surface views;
- [pretrained artifacts](../pretrained/README.md) - production checkpoint identity, provenance, and supported domains;
- [tests](../tests/README.md) - maintained pytest baseline and smoke pipelines.

The remainder of this file documents README figure generation and the historical transition archive.

## README figure sources

`generate_pipeline_figure.py` creates a current WNM pipeline illustration that is not embedded in the main README. Its construction panels are explanatory; the response-density panel uses a small run of the current WNM simulator, the network tile uses the packaged 20-sample predictor, and the prediction and fitting panels use the outputs described below. After preparing those outputs, run:

```bash
python docs/generate_pipeline_figure.py
```

Use `--predictions`, `--trials`, and `--fit-curves` to select other WNM outputs.

`generate_readme_figures.py` creates the two data-derived figures embedded in the main README. It does not reuse the repository's older diagnostic images. The prediction figure reads the output of the documented prediction example:

```bash
python surface_simulator_for_predictions/surface_simulator.py \
  --input-path example_data/prediction_parameters.csv \
  --n-samples 20 \
  --output-path results/prediction_example.parquet \
  --skip-motor-noise
```

The fitting figure reads the prepared <a href="https://doi.org/10.1038/nn.3689" title="Fischer, J., &amp; Whitney, D. (2014). Serial dependence in visual perception. Nature Neuroscience, 17(5), 738–743. https://doi.org/10.1038/nn.3689">Fischer and Whitney (2014)</a> trials and fitted curves written by `export_wnm_fit_curves.py` under `csv_exports/fitted_curves.csv`. It excludes the pooled `combined` pseudo-subject from both the observations and fitted curve, then averages the four real subjects. The generator defaults to the maintained `smoothed_exp` mean-bias objective. Pass `--fit-objective expectation` only when intentionally reproducing an older hard-binned expectation fit. After running `demo_fischer_whitney.py`, regenerate both assets with:

```bash
python docs/generate_readme_figures.py
```

Alternative or isolated fit outputs can be selected with `--fit-curves`; use `--predictions` for an alternative prediction export. Commit the rendered PNG files together with any generator change so GitHub does not need to run the model to display the README.


## Historical transition archive

`history/wnm_transition/` contains the quantitative findings and decision record from the 2026 WNM transition. Those files are archival evidence, not current run instructions. Paths and commands inside individual archived findings may reflect the transition-era repository layout; use the root README and the current package READMEs for executable workflows.
