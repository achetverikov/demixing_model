# Surface Simulator for Predictions

This folder contains a Python helper used to generate prediction surfaces/curves and an R script that consumes the output.

## Python usage

```bash
python surface_simulator_for_predictions/surface_simulator.py \
  <input_params.csv|parquet|arrow> \
  <n_samples> \
  <output_results.csv|parquet|arrow> \
  --skip-motor-noise
```

Arguments:
- `input_params`: CSV/Arrow/Parquet file with parameter columns.
- `n_samples`: Training sample count used to pick the checkpoint/surfaces (e.g., `20`, `100`).
- `output_results`: Where to save results (CSV or Arrow/Parquet).

Optional flags:
- `--skip-motor-noise` to set `sd_motor=0`.
- `--use-nn-surfaces` is on by default; pass `--use-nn-surfaces=false` to load averaged surfaces from disk.

Notes:
- The script expects checkpoints under `results/neural_net_checkpoints_{n_samples}samples/`.
- Averaged surfaces are read from `results/averaged_surfaces_10k_{n_samples}samples/` when `--use-nn-surfaces=false`.
- Input file must include `sd_feat1`, `sd_feat2`, `sd_spat`, and (unless `--skip-motor-noise`) `sd_motor`.
- You can update the defaults at the top of `surface_simulator_for_predictions/surface_simulator.py`:
  `RESULTS_DIR`, `CHECKPOINT_PREFIX`, `CHECKPOINT_EPOCH`, and `AVERAGED_SURFACES_DIR_TEMPLATE`.

## R usage

`surface_simulator_for_predictions/surface_simulator.R` writes Arrow/Parquet parameter files, invokes the Python simulator, then reads the Arrow/Parquet results, unpacks the curve arrays, and provides plotting helpers for the mu1/mu2 density and expectation curves plus the SD curves across feature differences.

Example (Arrow/Parquet via `arrow`):

```r
source("surface_simulator_for_predictions/surface_simulator.R")

params <- data.frame(
  sd_feat1 = c(10, 20),
  sd_feat2 = c(20, 10),
  sd_spat = c(30, 30),
  sd_motor = c(0, 0)
)

results <- simulate_surfaces(
  parameters = params,
  n_samples = 20,
  skip_motor_noise = TRUE,
  use_nn_surfaces = TRUE
)
```
