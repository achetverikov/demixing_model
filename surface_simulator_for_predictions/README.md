# Generate predictions in Python and R

Supply fitted parameters or a theoretical set of noise values to generate bias and response-variability curves across stimulus dissimilarity. The tool loads the trained model for your chosen internal sample count.

## Prepare a parameter table

Use CSV, Parquet, or Arrow with one row per parameter combination:

| Column | Required | Meaning |
|---|---|---|
| `sd_feat1` | Yes | Target-item feature noise |
| `sd_feat2` | Yes | Competing-item feature noise |
| `sd_spat` | Yes | Uncertainty about which evidence belongs to which item |
| `sd_motor` | Unless `--skip-motor-noise` is used | Response-stage motor noise |

Supply noise values in 360° model units. For example, 10° on a 180° orientation scale corresponds to 20 model degrees. See the [model reference](../pretrained/README.md) for supported parameter ranges.

## Run predictions in Python

From the repository root:

```bash
python surface_simulator_for_predictions/surface_simulator.py \
  --input-path example_data/prediction_parameters.csv \
  --n-samples 20 \
  --output-path results/prediction_example.parquet \
  --skip-motor-noise
```

`--n-samples` selects the 20- or 100-sample observer model. This is the amount of internal evidence available in a simulated trial. Use `--checkpoint-path` to select a custom trained model.

The output contains one row per input combination:

| Field | Contents |
|---|---|
| `mu1_density_curve` | Asymmetry of the predicted feature-error distribution |
| `mu1_expectation_curve` | Predicted circular mean bias in model degrees |
| `sd_curve` | Predicted circular response SD in model degrees |
| `feat_diff_grid` | Stimulus differences in model degrees, stored in the first row with grid/configuration metadata |

Positive bias means attraction; negative bias means repulsion. Each prediction is a curve across dissimilarity. Parquet and Arrow preserve curves as numeric arrays; CSV stores them as text.

## Run predictions in R

The wrapper requires `arrow`, `stringr`, and `data.table`. It invokes Python and returns one row per parameter combination and feature difference.

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
  skip_motor_noise = TRUE
)
```

Helpers `simulate_unequal_noise2()` and `simulate_equal_noise()` construct common parameter sweeps.

## Predict from averaged simulation surfaces

Use raw mode to inspect averaged simulation predictions, including bias in the separate identifiability dimension (`mu2`). Provide a directory of generated averaged surfaces:

```bash
python surface_simulator_for_predictions/surface_simulator.py \
  --input-path example_data/prediction_parameters.csv \
  --n-samples 20 \
  --output-path results/prediction_example_raw.parquet \
  --surface-source raw \
  --averaged-surfaces-dir /path/to/averaged_surfaces_10k_20samples_circular \
  --skip-motor-noise
```

The requested parameter combinations must exist on the 5° surface grid. Raw mode adds `mu2_density_curve` and `mu2_expectation_curve`. Both loose surface files and compressed bundles are supported; use a writable directory for bundles so requested files can be extracted.

In R, set `surface_source = "raw"` and provide an absolute `averaged_surfaces_dir`. See the [simulation guide](../surface_computation/README.md) for generating surfaces.

## Command reference

```text
surface_simulator.py INPUT N_SAMPLES OUTPUT      # positional form
  [--input-path INPUT] [--n-samples N] [--output-path OUTPUT]
  [--skip-motor-noise]
  [--surface-source {model,nn,raw}]
  [--checkpoint-path CHECKPOINT.pkl]
  [--averaged-surfaces-dir DIRECTORY]
```

`model` loads the selected trained predictor. `raw` reads averaged surfaces from `--averaged-surfaces-dir`. `nn` selects the surface-network predictor.

## Notes for developers

The prediction interface uses `shared.surrogate` for model identity and `shared.prediction` for evaluation. Output rows carry `surrogate_family` and `surrogate_artifact`.

The R argument `use_nn_surfaces` is a deprecated compatibility alias; use `surface_source` in new code.

Averaged surfaces are generated artifacts under `$DEMIXING_ARTIFACT_ROOT`. They are not shipped and are not expected in a normal checkout.
