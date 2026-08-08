# Neural Network Optimization Pipeline

## Overview

Two-step pipeline for training the mirror-aware neural network:

1. Create averaged surfaces from simulated samples.
2. Train the mirror-aware network on those surfaces.

`combine_mirrored_surfaces.py` is a leftover from an older three-step version where
mirroring was a separate pass. The current `create_averaged_surfaces_from_samples.py`
already loads both `(sf1, sf2, sp)` and `(sf2, sf1, sp)` sample files, merges them
with appropriate component flipping, and builds the KDE surfaces in one step.

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: create averaged surfaces                                │
│                                                                 │
│ python neural_network_optimization/create_averaged_surfaces_from_samples.py │
│   --input-folder  results/<samples_folder>                      │
│   --output-folder results/<averaged_folder>                     │
│                                                                 │
│ Inputs : samples_sf1_*_sf2_*_sp_*.pkl.gz  (L1 or L1+L2 grid)   │
│ For each unique (sf1 ≤ sf2, sp):                                │
│   - load (sf1, sf2, sp) and mirror (sf2, sf1, sp) sample files  │
│   - combine samples with component flipping                     │
│   - smooth across feat_diff with SD 3 grid steps = 6°           │
│   - fit KDE → mu1_comp1_surface, mu1_comp2_surface, mu2_surface │
│ Output : averaged_sf1_*_sf2_*_sp_*.pkl                         │
│          4,200 canonical files for L1; 32,800 for full 5° grid │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: train mirror-aware NN                                   │
│                                                                 │
│ python neural_network_optimization/mirror_aware_training.py     │
│   --surfaces-folder results/<averaged_folder>                   │
│   --epochs 1500                                                 │
│   --batch-size 32                                               │
│   --learning-rate 2e-3                                          │
│   --weight-decay 1e-4                                           │
│   --loss-profile circular_trajectory                            │
│   --native-mu1-rows 128                                         │
│   --training-feat-cols 128                                      │
│   --save-dir results/<checkpoints_folder>                       │
│                                                                 │
│ Data prep:                                                      │
│   canonical case  : inputs [sf1, sf2, sp] → mu1_comp1_surface  │
│   mirrored case   : inputs [sf2, sf1, sp] → mu1_comp2_surface  │
│     (only added when sf1 ≠ sf2)                                 │
│ Output: checkpoints + training logs                             │
└─────────────────────────────────────────────────────────────────┘
```

The flags above reproduce the architecture and objective of the production
20-observation run. The CLI defaults remain `circular`, 64 native mu1 rows, and
90 training feature columns so older experiments stay reproducible; omitting
the three production flags therefore starts a different model.

## Production 20-observation surrogate

The production checkpoint is
`pretrained/model_epoch1425_10ktrain_20samples.pkl`. It was selected from a
1500-epoch run trained on the 10k-simulation surfaces using:

- a native 128-row periodic decoder;
- 128 feature-difference columns in the training loss, resized to 90 only for
  inference;
- the `circular_trajectory` objective: equal-weight forward KL, circular
  energy, first-circular-moment vector error, density-asymmetry error, and
  second-difference error of the circular-moment trajectory across feature
  dissimilarity;
- AdamW, batch size 32, peak learning rate 0.002, weight decay 1e-4, and a
  1000-step warmup followed by cosine decay.

The targets remain the regular 10k-simulation KDE surfaces. Two independent
100k-simulation reference sets were used only for validation. Epoch 1425 gave
the best balanced validation score; the lower training loss at epoch 1500 was
not used as the selection criterion.

The loaded source currently produces 64,005 augmented rows because five
diagonal records are duplicated in the bundles. Each epoch uses 2,000 complete
batches, or 64,000 rows; the five unused rows have no meaningful speed effect.
See `pretrained/README.md` for the versioned checkpoint provenance.

---

## Disk Space: Stubbing Raw Sample Files

Raw sample files are ~6 MB each (~50 GB for a full L1 run). Once averaged, they can be
replaced with tiny stubs (~1 KB) that preserve the filename/hash so
`simulated_samples_grid.py` still counts those combinations as done and won't recompute
them.  Pass `--stub-samples` to enable this:

```bash
python neural_network_optimization/create_averaged_surfaces_from_samples.py \
  --input-folder  results/<samples_folder> \
  --output-folder results/<averaged_folder> \
  --stub-samples
```

Without `--stub-samples` (the default) sample files are left untouched.  The script is
safe to re-run either way — it skips combinations whose averaged surface already exists.

---

## Output Format (averaged surface file)

```python
{
    'parameters': {
        'sd_feat1': float,
        'sd_feat2': float,
        'sd_spat': float
    },
    'surface': AveragedSurface(
        mu1_comp1_surface,   # shape (180, 90) — bias × feat_diff grid (half-open circle)
        mu1_comp2_surface,
        mu2_surface,
        ...
    ),
    'creation_timestamp': 'YYYY-MM-DDTHH:MM:SS'
}
```

Filename: `averaged_sf1_{sf1}_sf2_{sf2}_sp_{sp}.pkl`  
Convention: `sf1 ≤ sf2` (canonical order).

---

## Key Configuration

```python
config.param_range_low  = 10
config.param_range_high = 200
config.param_step       = 10
config.param_grid_low          # derived Level-2 lower bound = 5
config.mu1_surface_shape = (180, 90)   # (bias_points, feat_diff_points)
```

`--feat-bandwidth` is expressed in feature-difference **grid steps**, not degrees.
The production grid advances by 2°, so the default `--feat-bandwidth 3` gives a
nominal 6° Gaussian SD across neighboring simulated dissimilarities. This smoothing
is part of each training target and is consequently baked into the trained NN output.

The mu1 axis is a half-open periodic grid, `[-180, 180)` in 2° cells. The
production model is trained through a 128-row native decoder but always returns
the configured 180-row periodic density. Feature difference is bounded rather
than circular.

## Warm-starting a completed run

`--init-checkpoint` loads parameters into a fresh optimizer schedule, while
`--epoch-offset` continues checkpoint numbering and the deterministic shuffle
stream. This is a warm restart, not an exact optimizer-state resume. For
example, a 500-epoch continuation of epoch 1500 uses `--epochs 500
--epoch-offset 1500 --init-checkpoint .../model_epoch_1500.pkl`. A completed
cosine schedule has zero learning rate, so any meaningful continuation must
choose and document a new learning-rate schedule.

## Notes for developers

The full experiment inventory, figures, CSV files, 100k references, and
intermediate checkpoints are generated artifacts, not repository contents. In
the current development workspace they are indexed at
`$DEMIXING_ARTIFACT_ROOT/mu1_experiments/README.md`, with the historical
protocol in `OBJECTIVE_ABLATION.md` beside it. A normal checkout is not expected
to contain these paths.
