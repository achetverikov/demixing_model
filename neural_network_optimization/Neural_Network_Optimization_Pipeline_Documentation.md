# Neural Network Optimization Pipeline

## Overview

This document describes the current pipeline for mirror-aware neural network optimization. It has two stages:
1) Create averaged surfaces from simulated samples.
2) Train the mirror-aware network on the averaged surfaces.

The workflow starts from `create_averaged_surfaces_from_samples.py` and continues with `mirror_aware_training.py`.

---

## Pipeline Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ START: neural_network_optimization/create_averaged_surfaces_from_samples.py │
│ Command: python neural_network_optimization/create_averaged_surfaces_from_samples.py │
│ Args: --input-folder <samples> --output-folder <averaged>        │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. Load L1 sample files                                          │
│                                                                  │
│    Inputs: samples_sf1_*_sf2_*_sp_*.pkl.gz                        │
│    Filter: L1 (coarse) grid based on config.param_step           │
│                                                                  │
│    Symmetry rule:                                                │
│    mu1_comp1(sf1,sf2,sp) = mu1_comp2(sf2,sf1,sp)                  │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. Build averaged surfaces from samples                          │
│                                                                  │
│    For each unique (sf1, sf2, sp):                               │
│    - mirror/merge samples for mu1_comp1 and mu1_comp2             │
│    - create KDE-based surfaces for mu1/mu2                        │
│                                                                  │
│    Output: averaged_sf1_*_sf2_*_sp_*.pkl                           │
│    Default folder: combined_mirrored_surfaces_10k                 │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. Train mirror-aware model                                      │
│                                                                  │
│    Script: mirror_aware_training.py                               │
│    Command: python mirror_aware_training.py                       │
│    Args: --surfaces-folder <averaged> --save-dir <checkpoints>    │
│                                                                  │
│    Data prep:                                                     │
│    - canonical case: use mu1_comp1_surface                        │
│    - mirrored case: use mu1_comp2_surface with swapped inputs     │
│                                                                  │
│    Output: checkpoints + training logs                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Create Averaged Surfaces From Samples

### Command
```bash
python neural_network_optimization/create_averaged_surfaces_from_samples.py \
  --input-folder <samples_folder> \
  --output-folder averaged_surfaces_10k_20samples
```

### Output Format (averaged surface)
```python
{
    'parameters': {
        'sd_feat1': float,
        'sd_feat2': float,
        'sd_spat': float
    },
    'surface': AveragedSurface(...),
    'creation_timestamp': 'YYYY-MM-DDTHH:MM:SS'
}
```

Notes:
- Output filename format: `averaged_sf1_{sf1}_sf2_{sf2}_sp_{sp}.pkl`
- `sf1`/`sf2` are stored in canonical order (sf1 <= sf2)

---

## Stage 2: Mirror-Aware Training

### Command
```bash
python mirror_aware_training.py \
  --surfaces-folder averaged_surfaces_10k_20samples \
  --epochs 1500 \
  --batch-size 32 \
  --learning-rate 2e-3 \
  --weight-decay 1e-4 \
  --save-dir neural_net_checkpoints_20samples
```

### Data Preparation
- Each averaged surface yields one canonical training example:
  - inputs: [sf1, sf2, sp]
  - targets: mu1_comp1_surface
- If sf1 != sf2, a mirrored example is added:
  - inputs: [sf2, sf1, sp]
  - targets: mu1_comp2_surface

### Outputs
- Checkpoints saved to `--save-dir` (via `utils.save_checkpoint`)
- Training log entries saved by `utils.save_training_log_smart`

---

## Related Files (Copied for Standalone Use)

The standalone folder `neural_network_optimization/` includes:
- `create_averaged_surfaces_from_samples.py`
- `combine_mirrored_surfaces.py`
- `mirror_aware_training.py`
- `mirror_aware_model.py`
- `loss_functions.py`
- `config.py`
- `utils.py`
- `seed_manager.py`
- `surface_folder_parsing.py`
- `surface_functions.py`
- `plotting.py`

---

## Key Configuration

Relevant fields from `config.py`:
```python
config.param_range_low = 10
config.param_range_high = 200
config.param_step = 10
config.mu1_surface_shape = (181, 89)
```
