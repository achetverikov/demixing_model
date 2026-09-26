# Included trained models

Choose a model with `--n-samples 20` or `--n-samples 100` when fitting or generating predictions. The corresponding checkpoint loads automatically.

| Internal evidence samples | Checkpoint |
|---:|---|
| 20 | `current_wnm_k12_20samples.pkl` |
| 100 | `current_wnm_k12_100samples.pkl` |

The sample count sets how much internal evidence the observer has in each simulated trial. Each sample is drawn from the two-item mixture, with equal probability of coming from either item. More samples provide more evidence for separating the representations.

## Parameters and units

The predictor takes `[sd_feat1, sd_feat2, sd_spat, feat_diff]` in 360° model units. It predicts the target item's circular response-error distribution; swap `sd_feat1` and `sd_feat2` to obtain the other item's prediction. Positive bias means attraction toward the competing item, and density is measured per model degree.

| Input | Supported range in model degrees |
|---|---|
| Target feature noise, `sd_feat1` | 2.5–200 |
| Non-target feature noise, `sd_feat2` | 2.5–200 |
| Identifiability noise, `sd_spat` | 5–200 |
| Stimulus difference, `feat_diff` | 0.5–180 |

The fitter reads parameter bounds from the checkpoint. For 180° data, it converts behavioral angles into model units; exports include columns in both model and study-scale degrees.

## Model architecture and training

Each checkpoint contains a conditional wrapped-normal mixture (WNM) with 12 components, hidden layers `(128, 256, 256)`, and a minimum component scale of 0.25 model degrees. Training minimizes negative log-likelihood (NLL) on simulated response errors. See the [pipeline explanation](../docs/model_pipeline.md) and [training guide](../surrogate_training/wnm/README.md).

| Internal samples | Selected training step | In-sample NLL |
|---:|---:|---:|
| 20 | 90,000 | 3.9785 |
| 100 | 89,000 | 3.1657 |

Both checkpoints use all training trajectories. The stopping step is selected using a 10% in-sample diagnostic slice; the table reports that training diagnostic. Generalization is evaluated separately on independent simulation trajectories.

## Use a custom checkpoint

Pass `--checkpoint-path path/to/model.pkl` to fitting or prediction commands. Package a training checkpoint with `python -m surrogate_training.wnm.package_artifact`; the [training guide](../surrogate_training/wnm/README.md) gives a complete example.

Fitted runs record the checkpoint's SHA-256 digest. Exporters and plotters resolve that recorded model and verify any explicit checkpoint override against it. Fresh predictions select by `--n-samples`, check the requested sample count, and label outputs with the loaded model's identity.

## Notes for developers

Load checkpoints through `shared.surrogate.load_surrogate`. Use `production_checkpoint` for fresh predictions and `checkpoint_for_run` for saved fits. Architecture, sample count, units, and supported-domain metadata are stored in each artifact.

The declared domain rounds the measured training hull outward to the design limits. `meta["corpus_hull"]` records the sampled bounds and `meta["declared_domain_overhang"]` records extrapolation at each edge. The packager allows up to 5° of outward rounding and checks predictions after reloading the packaged file.

The corpus labels are `continuous_density_4.1p` for 20 samples and `continuous_density_4.1o` for 100 samples; independent validation uses `continuous_density_4.1q`. These generated corpora and results live under `$DEMIXING_ARTIFACT_ROOT`. They are not shipped and are not expected in a normal checkout.

`surface_legacy_epoch1425_10ktrain_20samples.pkl` is also included for the surface-network tools. It uses a 128-row periodic decoder, a 128-column training grid, and the `circular_trajectory` objective on 10k-simulation KDE surfaces. Its three noise axes span 5–200 model degrees. See the [surface-network training guide](../neural_network_optimization/Neural_Network_Optimization_Pipeline_Documentation.md).
