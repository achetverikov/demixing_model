# Prediction and surface browser

The Streamlit browser provides two kinds of views:

1. **WNM on demand** evaluates the packaged production surrogate continuously at
   a selected `(sd_feat1, sd_feat2, sd_spat)` triple. This mode does not require
   stored simulation surfaces.
2. **Stored-surface views** inspect locally available averaged simulation
   surfaces and are retained for historical surface-NN reproduction and raw
   `mu2` analyses.

Run from the repository root:

```bash
streamlit run surface_browser/main_app.py
```

The WNM tab loads the 20- or 100-sample packaged artifact through
`shared.surrogate` and evaluates it through `shared.prediction`. The browser
does not implement a second copy of the WNM mathematics.

To enable stored-surface tabs, point the repository at an artifact root that
contains averaged surfaces:

```bash
DEMIXING_ARTIFACT_ROOT=/path/to/artifacts \
  streamlit run surface_browser/main_app.py
```

Without stored surfaces, the WNM tab remains usable and the historical
surface-backed tabs report that no surfaces are available.

## Relevant modules

- `main_app.py` - Streamlit entry point and tab routing.
- `core/wnm_view.py` - Streamlit-independent WNM evaluation wrapper.
- `core/data_manager.py` - discovery/loading of stored averaged surfaces.
- `tabs/` - individual WNM and historical surface views.

For batch prediction without Streamlit, use
[`surface_simulator_for_predictions/`](../surface_simulator_for_predictions/README.md).
