# Shared model utilities

The `shared/` package provides model evaluation, checkpoint loading, circular geometry, and data helpers used by fitting, prediction, exports, and plotting.

## Modules

| Module | Purpose |
|---|---|
| `wnm.py` | Conditional wrapped-normal mixture, circular densities, motor noise, and artifact serialization |
| `surrogate.py` | Checkpoint selection, saved-run model identity, and supported-domain lookup |
| `prediction.py` | Common prediction operations |
| `mu1_axis.py` | Periodic bias grid, wrapping, integration, sign masks, and bin indexing |
| `circular.py` | Circular curve smoothing |
| `empirical.py` | Empirical density estimation, bandwidths, and bias curves |
| `behavioral_data.py` | Behavioral-trial filtering |
| `paths.py` | Input and output path resolution |
| `hashing.py` | File hashes for artifact and run identity |
| `utils.py` | Surface-network and averaged-surface helpers |

For usage examples, see the [fitting guide](../model_fit_to_data/Batch_Fit_Analysis_Pipeline_Documentation.md) and [prediction guide](../surface_simulator_for_predictions/README.md).

## Notes for developers

Use `surrogate.py` to load models and resolve the checkpoint recorded in a saved fit. Build prediction consumers on `prediction.py` so fitting, plotting, and exports share the same calculations. Use the focused modules above when adding model or data functionality.
