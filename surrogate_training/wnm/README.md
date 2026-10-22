# Train a custom predictor

This pipeline trains a conditional wrapped-normal mixture (WNM) on simulated response errors and packages it for fitting and prediction. The [included models](../../pretrained/README.md) provide ready-to-use 20- and 100-sample predictors.

Run commands from the repository root.

## 1. Generate training data

```bash
python -m surface_computation.generate_wnm_training_data \
  --n-samples 20 \
  --out /path/to/training_data
```

Choose the internal sample count for the observer you want to model. See the [simulation guide](../../surface_computation/README.md) for design size, repetitions, and output details.

Custom training and packaging accept any positive integer sample count, including
500. Packaged defaults remain 20 and 100; use an explicit checkpoint path for
other counts, and validate that predictor independently before fitting data.

## 2. Train the mixture

```bash
python -m surrogate_training.wnm.train \
  --source /path/to/training_data \
  --components 12 \
  --out /path/to/run_checkpoint.pkl
```

Training minimizes negative log-likelihood on the raw simulated biases. The checkpoint records the sample count, selected step, architecture, and training parameter bounds, including the swapped-item examples.

Completed directory corpora use memory-mapped bias arrays. Minibatches fetch only
their selected outcomes; parameter-group train/validation splits keep row-index
views, and trajectory-reference evaluation reads 16 rows at a time. All raw
diagnostics remain available in the corpus. Existing NPZ and legacy grid sources
are also supported; an NPZ bias member is loaded into RAM because ZIP members
cannot be memory-mapped directly.

## 3. Package the checkpoint

```bash
python -m surrogate_training.wnm.package_artifact \
  --fit /path/to/run_checkpoint.pkl \
  --n-samples 20 \
  --out /path/to/custom_wnm_20samples.pkl
```

Use the same `--n-samples` as the generated data. The packager copies the weights, records scientific metadata and the training domain, reloads the artifact, and checks its predictions before saving the final file.

Validate the packaged model against independent simulations across its parameter domain, including the shape of bias and variability curves across dissimilarity.

## 4. Use the model

Pass `--checkpoint-path /path/to/custom_wnm_20samples.pkl` and `--n-samples 20` to the [fitter](../../model_fit_to_data/Batch_Fit_Analysis_Pipeline_Documentation.md) or [prediction generator](../../surface_simulator_for_predictions/README.md).

## Notes for developers

- `train.py`: training loop and checkpoint selection.
- `data.py`: source loading, batching, and item-swap augmentation.
- `evaluation.py`: moment and likelihood evaluation.
- `package_artifact.py`: model packaging.
- `shared/wnm.py`: runtime model and serialization format.

Consumers load packaged artifacts through `shared.surrogate`. The packaged domain follows the source checkpoint's training bounds. Selected-fit dictionaries from stage-based experiments use `--corpus-stage` to locate their source manifests.

Corpus labels such as `continuous_density_4.1p` identify generated experiment artifacts under `$DEMIXING_ARTIFACT_ROOT`. These artifacts are not shipped and are not expected in a normal checkout.
