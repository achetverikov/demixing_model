# Included trained models

Choose `--n-samples 4`, `20`, `100`, or `500` when fitting or generating predictions. The corresponding K8 checkpoint loads automatically.

| Internal evidence samples | Checkpoint |
|---:|---|
| 4 | `wnm_k8_n4_seed0_packaged.pkl` |
| 20 | `wnm_k8_n20_seed0.pkl` |
| 100 | `wnm_k8_n100_seed0.pkl` |
| 500 | `wnm_k8_n500_seed0_packaged.pkl` |

The sample count sets how much internal evidence the observer has in each simulated trial. These are distinct observer models, not numerical precision settings.

## Parameters and units

The predictor takes `[sd_feat1, sd_feat2, sd_idf, feat_diff]` in 360° model units. It predicts the target item's circular response-error distribution; swap `sd_feat1` and `sd_feat2` for the other item. Positive bias means attraction, and density is per model degree.

| Input | Supported range in model degrees (rounded for display) |
|---|---|
| Target and non-target feature noise | 2.5001016–199.9541016 |
| Identifiability noise | 5.0018077–200 |
| Stimulus difference | 0.5–180 |

The fitter reads exact bounds from each checkpoint's `supported_domain`, measured from training parameters including swapped-item rows. These bounds describe coverage, not an accuracy guarantee. For 180° data, behavioral angles are converted to model units; exports retain both model and study-scale degrees.

## Architecture and training

All four predictors use 8 wrapped-normal mixture components, hidden layers `(128, 256, 256)`, and minimum component scale 0.25 model degrees. The seed-0 models minimize histogram-bin cross-entropy and select the final step, 92,560. They retain a 10% held-out parameter-group test split separately from training and checkpoint selection.

See the [pipeline explanation](../docs/model_pipeline.md) and [training guide](../surrogate_training/wnm/README.md). Validate custom predictors against independent simulations across stimulus dissimilarity.

## Checkpoint identity

Use `--checkpoint-path path/to/model.pkl` for a custom packaged model. Training checkpoints must first pass through `python -m surrogate_training.wnm.package_artifact`.

Saved fits record the checkpoint SHA-256. Exporters and plotters resolve that exact model and reject mismatched overrides. The previous K12 files remain installed solely to reproduce historical fits; fresh fits and predictions select K8. Reusing old fit parameters with a new network requires a refit, not a relabelled export.

## Notes for developers

Use `shared.surrogate.production_checkpoint` for fresh runs and `checkpoint_for_run` for saved fits. The shipped K8 files are byte-for-byte copies of the checkpoints used by the current comparison fits, preserving hashes and version labels.

Source checkpoints are under `$DEMIXING_ARTIFACT_ROOT/fits/networks/`; N20/N100 selection evidence is under `$DEMIXING_ARTIFACT_ROOT/wrapped_em_refits_20261024/`. These external experiment artifacts are not required in a normal checkout. Each shipped artifact records source digest, architecture, objective, training step, sample count, and parameter domain. Previous packaging documentation and promotion verification are under `$DEMIXING_ARTIFACT_ROOT/investigations/k8_defaults_20261107/`.
