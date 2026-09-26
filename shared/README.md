# Shared runtime and identity layer

The `shared/` package contains maintained code used across fitting, prediction,
export, plotting, and surrogate loading. New WNM code should depend on these
focused modules rather than on `shared/utils.py`.

## Maintained modules

- `wnm.py` - conditional wrapped-normal-mixture runtime, artifact
  serialization, circular density primitives, and motor-noise handling.
- `surrogate.py` - family-aware checkpoint resolution, production artifact
  selection, stored-run checkpoint identity, and supported-domain lookup.
- `prediction.py` - common predictor operations used by fitting, plotting,
  rescoring, the public prediction interface, and the browser.
- `mu1_axis.py` - authoritative periodic bias grid, cell width, wrapping,
  integration, sign masks, and bin indexing.
- `circular.py` - circular curve smoothing.
- `empirical.py` - empirical KDE/bandwidth and bias-curve helpers.
- `behavioral_data.py` - behavioral-trial filtering used by fitting and
  postprocessing.
- `paths.py` - repository/results input and output path resolution.
- `hashing.py` - streaming file SHA-256 used for artifact and run identity.

## Historical compatibility

`utils.py` remains for the retained surface-NN/raw-surface workflow and older
checkpoint utilities. Maintained WNM modules should not add new dependencies on
it.

The public fitting and prediction entry points are documented in the root
[README](../README.md), the fitting guide under
[`model_fit_to_data/`](../model_fit_to_data/Batch_Fit_Analysis_Pipeline_Documentation.md),
and the prediction guide under
[`surface_simulator_for_predictions/`](../surface_simulator_for_predictions/README.md).
