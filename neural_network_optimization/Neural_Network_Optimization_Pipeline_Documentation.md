# Average simulation surfaces and train a surface network

Use these tools to turn simulated samples into averaged density surfaces and train a network that predicts those surfaces. For the conditional-mixture predictor used by the main fitting and prediction tools, see the [training guide](../surrogate_training/wnm/README.md).

There are two steps:

1. Create averaged surfaces from simulated samples.
2. Train the mirror-aware network on those surfaces.

`create_averaged_surfaces_from_samples.py` loads both `(sf1, sf2, sp)` and `(sf2, sf1, sp)` sample files, exchanges the matching components, and builds the KDE surfaces in one pass.

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

The example selects `circular_trajectory`, 128 native bias rows, and 128 training feature columns. The CLI defaults are `circular`, 64 rows, and 90 columns; pass the example flags to use the configuration of the included surface checkpoint.

## Reduce raw-sample storage

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

By default, raw sample files are preserved. With `--stub-samples`, their simulated outcomes are deleted after averaging, so keep a copy if you need them for further analysis. Rerunning skips combinations whose averaged surface already exists.

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

`--feat-bandwidth` is expressed in feature-difference **grid steps**.
The surface grid advances by 2°, so the default
`--feat-bandwidth 3` gives a nominal 6° Gaussian SD across neighboring
simulated dissimilarities. This smoothing
is part of each training target and is consequently baked into the trained NN output.

The mu1 axis is a half-open periodic grid, `[-180, 180)` in 2° cells. The included surface model uses a 128-row native decoder and returns the configured 180-row periodic density. Feature difference has a bounded axis.

## Warm-starting a completed run

`--init-checkpoint` loads parameters into a fresh optimizer schedule, while
`--epoch-offset` continues checkpoint numbering and the deterministic shuffle
stream. The optimizer starts with a fresh state and learning-rate schedule. For
example, a 500-epoch continuation of epoch 1500 uses `--epochs 500
--epoch-offset 1500 --init-checkpoint .../model_epoch_1500.pkl`. A completed
cosine schedule has zero learning rate, so any meaningful continuation must
choose and document a new learning-rate schedule.

## Notes for developers

The included surface checkpoint, `pretrained/surface_legacy_epoch1425_10ktrain_20samples.pkl`, was selected at epoch 1425 of a 1500-epoch run on 10k-simulation KDE surfaces. It uses a 128-row periodic decoder, 128 training feature columns, and the `circular_trajectory` objective: equal-weight forward KL, circular energy, first-moment error, density-asymmetry error, and second-difference error of the circular-moment trajectory. Training uses AdamW, batch size 32, peak learning rate 0.002, weight decay `1e-4`, and a 1000-step warmup followed by cosine decay.

Experiment inventories, validation references, and intermediate checkpoints are generated artifacts under `$DEMIXING_ARTIFACT_ROOT/mu1_experiments/`, indexed by `README.md` and `OBJECTIVE_ABLATION.md`. They are not shipped and are not expected in a normal checkout. Additional source-bundle and checkpoint-selection details are preserved under `$DEMIXING_ARTIFACT_ROOT/exploration_archive/documentation_20260926/`.
