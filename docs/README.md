# README figure sources

`generate_readme_figures.py` creates all three figures embedded in the main README. It does not reuse the repository's older diagnostic images.

The workflow overview has no data dependency. The prediction figure reads the output of the documented prediction example:

```bash
python surface_simulator_for_predictions/surface_simulator.py \
  --input-path example_data/prediction_parameters.csv \
  --n-samples 20 \
  --output-path results/prediction_example.parquet \
  --skip-motor-noise
```

The fitting figure reads the prepared <a href="https://doi.org/10.1038/nn.3689" title="Fischer, J., &amp; Whitney, D. (2014). Serial dependence in visual perception. Nature Neuroscience, 17(5), 738–743. https://doi.org/10.1038/nn.3689">Fischer and Whitney (2014)</a> trials and the 20-sample `expectation` curves exported by `create_unified_subject_plots.py`. After running `demo_fischer_whitney.py`, regenerate all assets with:

```bash
python docs/generate_readme_figures.py
```

Alternative or isolated fit outputs can be selected with `--fit-curves`; use `--predictions` for an alternative prediction export. Commit the rendered SVG and PNG files together with any generator change so GitHub does not need to run the model to display the README.
