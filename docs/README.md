# README figure sources

`generate_pipeline_figure.py` creates an experimental pipeline figure that is not currently embedded in the main README. Its construction panels are illustrations grounded in the model code rather than a literal depiction of one simulation or the exact neural-network architecture. The likelihood surface, prediction curves, behavioral observations, and fitted curve are read from project outputs.

The committed figure uses a broad surface from the full simulation outputs and the 100-sample predictions prepared for CSH2026. Its defaults match the local outputs used for rendering:

```bash
python docs/generate_pipeline_figure.py
```

Because averaged surfaces are not currently shipped, regenerating the figure requires a local `averaged_sf1_*.pkl` surface supplied with `--surface`. The CSH2026 curves default to `../results/csh2026_100samples_circular/sim_model_preds_raw_nn.csv` and can be replaced with `--csh-predictions`. The fitting inputs can likewise be changed with `--trials` and `--fit-curves`.

`generate_readme_figures.py` creates the two data-derived figures embedded in the main README. It does not reuse the repository's older diagnostic images. The prediction figure reads the output of the documented prediction example:

```bash
python surface_simulator_for_predictions/surface_simulator.py \
  --input-path example_data/prediction_parameters.csv \
  --n-samples 20 \
  --output-path results/prediction_example.parquet \
  --skip-motor-noise
```

The fitting figure reads the prepared <a href="https://doi.org/10.1038/nn.3689" title="Fischer, J., &amp; Whitney, D. (2014). Serial dependence in visual perception. Nature Neuroscience, 17(5), 738–743. https://doi.org/10.1038/nn.3689">Fischer and Whitney (2014)</a> trials and the 20-sample `expectation` curves exported by `create_unified_subject_plots.py`. After running `demo_fischer_whitney.py`, regenerate both assets with:

```bash
python docs/generate_readme_figures.py
```

Alternative or isolated fit outputs can be selected with `--fit-curves`; use `--predictions` for an alternative prediction export. Commit the rendered PNG files together with any generator change so GitHub does not need to run the model to display the README.
