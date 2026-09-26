# Explore model predictions

Use the Streamlit browser to explore how response distributions change with target noise, non-target noise, and identifiability noise.

Run from the repository root:

```bash
streamlit run surface_browser/main_app.py
```

Open the **WNM on demand** tab, choose the 20- or 100-sample observer model, and select the noise parameters. The browser evaluates the included trained model at those values.

For batch predictions and the R interface, see the [prediction guide](../surface_simulator_for_predictions/README.md).

## Inspect simulation surfaces

Stored-surface tabs show averaged simulation outputs, including the separate identifiability-dimension (`mu2`) predictions. To enable them, set the artifact root to a directory containing your generated averaged surfaces:

```bash
DEMIXING_ARTIFACT_ROOT=/path/to/artifacts \
  streamlit run surface_browser/main_app.py
```

Use a writable directory for compressed surface bundles: the browser extracts requested files there as needed. See the [simulation guide](../surface_computation/README.md) for generating surfaces.

## Notes for developers

`main_app.py` routes the tabs. `core/wnm_view.py` evaluates the trained model through `shared.surrogate` and `shared.prediction`; `core/data_manager.py` discovers and loads averaged surfaces.

Averaged surfaces are generated artifacts under `$DEMIXING_ARTIFACT_ROOT`. They are not shipped and are not expected in a normal checkout.
