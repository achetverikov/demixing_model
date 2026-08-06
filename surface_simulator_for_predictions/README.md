# Prediction generator

Use this tool when you have fitted parameter values—or a theoretical set of noise values—and want to see the bias and response-variability curves predicted by the Demixing Model. The standard route uses the trained models included in `pretrained/` and does not require any additional simulation files. Both Python and R interfaces are available.

## Python: predictions with the included model

Run commands from the repository root. The input may be CSV, Parquet, or Arrow and needs one row per parameter combination:

| Column | Required | Meaning |
|---|---|---|
| `sd_feat1` | yes | Feature noise for the target item |
| `sd_feat2` | yes | Feature noise for the competing item |
| `sd_spat` | yes | Uncertainty about which feature belongs to which item/location |
| `sd_motor` | unless `--skip-motor-noise` | Response-stage motor noise |

```bash
PYTHONPATH=. python surface_simulator_for_predictions/surface_simulator.py \
  --input-path example_data/prediction_parameters.csv \
  --n-samples 20 \
  --output-path results/prediction_example.parquet \
  --skip-motor-noise
```

The `--n-samples 20` selects one of the two theoretical internal-sampling assumptions. It does not refer to the number of experimental trials. The matching trained model (`pretrained/model_epoch1500_10ktrain_20samples.pkl`) is loaded automatically; pass `--checkpoint-path` only to point at a different one.

The three required inputs can also be given positionally — `surface_simulator.py INPUT N_SAMPLES OUTPUT` — which is what the smoke scripts use.

The output contains one row per input combination. Its main prediction columns are:

- `mu1_density_curve`: asymmetry of the predicted response distribution;
- `mu1_expectation_curve`: predicted mean response bias;
- `sd_curve`: predicted response variability;
- `feat_diff_grid` and bias-grid/configuration metadata in the first row.

Each value is a curve across stimulus dissimilarity rather than a single average. Parquet/Arrow preserves these curves as numeric arrays and is recommended. CSV stores them as text.

## Advanced: predictions from raw simulation surfaces

The included trained model predicts the primary (`mu1`) component. Researchers who need predictions for the secondary (`mu2`) mixture component must provide raw averaged simulation surfaces:

```bash
PYTHONPATH=. python surface_simulator_for_predictions/surface_simulator.py \
  --input-path example_data/prediction_parameters.csv \
  --n-samples 20 \
  --output-path results/prediction_example_raw.parquet \
  --surface-source raw \
  --averaged-surfaces-dir /path/to/averaged_surfaces_10k_20samples_circular \
  --skip-motor-noise
```

Raw mode requires the exact requested parameter combinations to exist on the 5° surface grid. It adds `mu2_density_curve` and `mu2_expectation_curve`. Both loose surface files and compressed bundle directories are supported; bundle-backed directories must be writable so requested files can be materialized on demand.

Averaged surfaces are not included in the repository and are not currently published as release assets.

## Complete CLI

```text
surface_simulator.py INPUT N_SAMPLES OUTPUT      # or the named forms below
  [--input-path INPUT] [--n-samples N] [--output-path OUTPUT]
  [--skip-motor-noise]
  [--surface-source {nn,raw}]
  [--checkpoint-path CHECKPOINT.pkl]
  [--averaged-surfaces-dir DIRECTORY]
```

`--averaged-surfaces-dir` is required with `--surface-source raw`.

## R interface

The wrapper requires `arrow`, `stringr`, and `data.table`. It writes a temporary Parquet parameter table, invokes Python, reads the results, and returns one row per parameter combination and feature difference.

```r
source("surface_simulator_for_predictions/surface_simulator.R")

params <- data.frame(
  sd_feat1 = c(10, 10, 10),
  sd_feat2 = c(30, 30, 30),
  sd_spat = c(20, 60, 120)
)

predictions <- simulate_surfaces(
  parameters = params,
  n_samples = 20,
  skip_motor_noise = TRUE,
  use_nn_surfaces = TRUE,
  checkpoint_path = normalizePath(
    "pretrained/model_epoch1500_10ktrain_20samples.pkl"
  )
)
```

For raw mode, set `use_nn_surfaces = FALSE` and provide an absolute `averaged_surfaces_dir`. Helper functions `simulate_unequal_noise2()` and `simulate_equal_noise()` construct common parameter sweeps.
