# WNM surrogate training

This package trains and packages the conditional wrapped-normal-mixture (WNM)
surrogate used by the production fitting and prediction paths.

Ordinary users do **not** need to run this pipeline. The packaged 20- and
100-sample production artifacts are already included under `pretrained/`.

## Data flow

1. Generate or load raw simulator outcomes.
   - Continuous/off-grid designs: `python -m surface_computation.generate_wnm_training_data`
   - Existing raw grid corpus: loaded through `surface_computation.legacy_sample_import`
2. Train the conditional mixture:
   ```bash
   PYTHONPATH=. python -m surrogate_training.wnm.train \
     --source /path/to/training_data.npz \
     --components 12 \
     --out /path/to/run_checkpoint.pkl
   ```
3. Package a selected training checkpoint into a self-contained production
   artifact:
   ```bash
   python -m surrogate_training.wnm.package_artifact \
     --fit /path/to/selected-best.pkl \
     --corpus-stage /path/to/corpus-stage \
     --out pretrained/wnm_k12_20samples.pkl
   ```
4. Production consumers load artifacts through `shared.surrogate`, not by
   opening training checkpoints directly.

The packager copies weights without retraining, records architecture and
scientific metadata, verifies the declared supported domain against the corpus,
reloads the artifact through the production loader, and checks a fixed parameter
panel before replacing the destination file.

## Files

- `train.py` - WNM training loop and checkpoint selection.
- `data.py` - raw-outcome stores, batching, mirror augmentation, and source loading.
- `evaluation.py` - training-time moment and NLL evaluation helpers.
- `package_artifact.py` - conversion from selected training checkpoint to
  production artifact.
- `shared/wnm.py` - maintained runtime model and artifact serialization format.
- `pretrained/README.md` - identity, domain, and provenance of the packaged
  production artifacts.

Historical corpus-stage names such as `continuous_density_4.1p` are immutable
provenance labels. They are not current source-code paths.
