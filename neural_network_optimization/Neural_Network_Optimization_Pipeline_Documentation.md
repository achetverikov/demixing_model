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
│ Inputs : samples_sf1_*_sf2_*_sp_*.pkl.gz  (L1 grid only)       │
│ For each unique (sf1 ≤ sf2, sp):                                │
│   - load (sf1, sf2, sp) and mirror (sf2, sf1, sp) sample files  │
│   - combine samples with component flipping                     │
│   - fit KDE → mu1_comp1_surface, mu1_comp2_surface, mu2_surface │
│ Output : averaged_sf1_*_sf2_*_sp_*.pkl  (4,200 files for L1)   │
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
│   --save-dir results/<checkpoints_folder>                       │
│                                                                 │
│ Data prep:                                                      │
│   canonical case  : inputs [sf1, sf2, sp] → mu1_comp1_surface  │
│   mirrored case   : inputs [sf2, sf1, sp] → mu1_comp2_surface  │
│     (only added when sf1 ≠ sf2)                                 │
│ Output: checkpoints + training logs                             │
└─────────────────────────────────────────────────────────────────┘
```

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
        mu1_comp1_surface,   # shape (181, 90) — bias × feat_diff grid
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
config.mu1_surface_shape = (181, 90)   # (bias_points, feat_diff_points)
```
