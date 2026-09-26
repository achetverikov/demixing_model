# WNM surrogate training

This package trains and packages the conditional wrapped-normal-mixture (WNM)
surrogate used by the production fitting and prediction paths.

Ordinary users do **not** need to run this pipeline. The packaged 20- and
100-sample production artifacts are already included under `pretrained/`.

## Data flow

1. Generate or load raw simulator outcomes.
   - Continuous/off-grid designs:
     ```bash
     python -m surface_computation.generate_wnm_training_data \
       --n-samples 20 --out /path/to/training_data.npz
     ```
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
     --fit /path/to/run_checkpoint.pkl \
     --n-samples 20 \
     --out /path/to/custom_wnm_20samples.pkl
   ```
4. Production consumers load artifacts through `shared.surrogate`, not by
   opening training checkpoints directly.

Use the same `--n-samples` as the generated training data (the generator defaults
to 100). The trainer records sample identity, selected step, and the training
parameter hull, including mirrored rows. The packager refuses a different sample
count. Its declared domain is that hull, not the production checkpoints' full
domain; coverage alone does not establish surrogate accuracy. Validate a custom
model independently before using it for scientific inference.

Historical selected-fit dictionaries still require `--corpus-stage`; that option
is not used for checkpoints produced by the maintained trainer.

The packager copies weights without retraining, records architecture and
scientific metadata, preserves the recorded training domain, reloads the artifact,
and checks predictions at a parameter panel before replacing the destination file.
Historical stage-based packaging additionally checks the original corpus manifests.

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
