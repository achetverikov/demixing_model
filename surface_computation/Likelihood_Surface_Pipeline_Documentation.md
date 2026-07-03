# Simulated Samples Grid Pipeline

## Overview

This document describes the current pipeline for `simulated_samples_grid.py`, which generates simulated bias samples across a 3D parameter grid with dynamic chunking and multi-machine coordination.

---

## Pipeline Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ START: simulated_samples_grid.py (main entry point)             │
│ Command: python3 simulated_samples_grid.py                      │
│ Args: --machine-id PC1 --grid-level 1                           │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. ChunkedGridComputer.__init__()                               │
│                                                                  │
│    Creates 3D parameter grid:                                   │
│    - sd_feat1: Standard deviation for feature component 1       │
│    - sd_feat2: Standard deviation for feature component 2       │
│    - sd_spat: Spatial standard deviation                        │
│                                                                  │
│    Grid configuration:                                           │
│    - Level 1 (coarse): step = config.param_step                 │
│    - Level 2 (fine): step = config.param_step / 2               │
│    - Range: config.param_range_low to param_range_high          │
│    - Total combinations: n_param_values³                        │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. Dynamic Chunk Assignment                                     │
│                                                                  │
│    Process:                                                      │
│    1. Scans for incomplete parameter combinations               │
│    2. Creates chunk locks to coordinate multiple machines       │
│    3. Randomly selects from available chunks                    │
│                                                                  │
│    Chunk sizing:                                                 │
│    - Large chunks: 50 parameter combinations                    │
│    - Small chunks: 5 combinations (near completion)             │
│    - Threshold: switches when ≤10 large chunks remain           │
│                                                                  │
│    Lock file format: computing_L{level}_chunk_{start}_{end}.lock│
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. Chunk Processing Loop                                        │
│                                                                  │
│    FOR EACH parameter combination (sd_feat1, sd_feat2, sd_spat):│
│    │                                                             │
│    ├─ Check if samples already exist (skip if yes)              │
│    ├─ Simulate bias samples via jax_fit_main                    │
│    └─ Save result via save_samples_checkpoint()                 │
│                                                                  │
│    Progress tracking:                                            │
│    - Samples computed/skipped counters                           │
│    - Timing per combination                                     │
│    - Chunk completion time                                       │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. Sample Generation (jax_fit_main.py)                          │
│    Function: simulate_dual_component_bias_distribution()        │
│                                                                  │
│    INPUT PARAMETERS:                                             │
│    ├─ sd_feat1, sd_feat2, sd_spat (from grid)                   │
│    ├─ feat_diff (from config.create_grid('feat_diff'))          │
│    ├─ n_simulations (per run; split across 10 scans)            │
│    ├─ n_samples (from runtime args)                             │
│    ├─ algorithm (EM only; VBEM/VBEM_MIX NOT implemented → raise)│
│    ├─ diagonal_covariance (True only), fix_weights              │
│                                                                  │
│    COMPUTATION LOOP:                                             │
│    - 10 scan loops with jax.lax.scan()                           │
│    - Each scan simulates num_sims_per_loop                      │
│    - Results concatenated to reach n_simulations total          │
│                                                                  │
│    RETURNS:                                                      │
│    - mu_1_bias: (n_feat_diff, n_simulations, 2)                 │
│    - mu_2_bias: (n_feat_diff, n_simulations, 2)                 │
│    - full_results (optional, when save_full_results=True)       │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. Save Samples Checkpoint                                      │
│    Function: save_samples_checkpoint()                          │
│                                                                  │
│    FILENAME FORMAT:                                              │
│    samples_sf1_{X.X}_sf2_{Y.Y}_sp_{Z.Z}_{HASH}.pkl.gz           │
│    Example: samples_sf1_60.0_sf2_115.0_sp_82.0_a3f2c1d8.pkl.gz  │
│                                                                  │
│    STORAGE LOCATION:                                             │
│    config.samples_folder (set by configure_samples_folder)      │
│                                                                  │
│    FILE FORMAT: gzip-compressed pickle                          │
│                                                                  │
│    SAVED DATA STRUCTURE:                                         │
│    {                                                             │
│        'parameters': {                                           │
│            'sd_feat1': float,                                    │
│            'sd_feat2': float,                                    │
│            'sd_spat': float,                                     │
│            'param_name': str,                                    │
│            'param_hash': str,                                    │
│            'machine_id': str,                                    │
│            'platform': str                                       │
│        },                                                        │
│        'mu1_samples': Array (float16),                           │
│        'mu2_samples': Array (float16),                           │
│        'full_results': Array (float32, optional),                │
│        'computation_time': float,                                │
│        'timestamp': float                                        │
│    }                                                             │
└───────────────┬──────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 6. Progress Tracking & Coordination                             │
│    Function: _save_progress_summary()                           │
│                                                                  │
│    PROGRESS FILE:                                                │
│    progress_summary_{machine_id}.json                            │
│                                                                  │
│    Lock file format:                                             │
│    computing_L{level}_chunk_{start:05d}_{end:05d}.lock           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Grid Level System

### Level 1: Coarse Grid
- **Step size**: 10° (config.param_step)
- **Purpose**: Initial broad coverage of parameter space
- **Example range**: [10, 20, 30, ..., 200]
- **Total combinations**: (n_values)³

### Level 2: Fine Grid
- **Step size**: 5° (config.param_step / 2)
- **Purpose**: Fill in gaps from Level 1
- **Important**: Only computes combinations not already in Level 1
- **Auto-advance**: Can automatically start after Level 1 completes

---

## Multi-Machine Coordination

### How It Works
1. Each machine runs with unique `--machine-id` (PC1, PC2, PC3, PC4, PC5)
2. Machines randomly select from available chunks
3. Lock files prevent duplicate work
4. Progress tracked independently per machine
5. Stale lock cleanup recovers from crashes

### Coordination Files
```
sim_samples_*_*/  # config.samples_folder
├── samples_sf1_60.0_sf2_115.0_sp_82.0_a3f2c1d8.pkl.gz
├── samples_sf1_70.0_sf2_120.0_sp_90.0_b4e3d2f9.pkl.gz
├── ...
├── progress_summary_PC1.json
├── progress_summary_PC2.json
├── computing_L1_chunk_00000_00050.lock
└── computing_L1_chunk_00050_00100.lock
```

---

## Data Formats

### Samples PKL.GZ File Contents
```python
{
    'parameters': {
        'sd_feat1': 60.0,
        'sd_feat2': 115.0,
        'sd_spat': 82.0,
        'param_name': 'sf1_60.0_sf2_115.0_sp_82.0',
        'param_hash': 'a3f2c1d8',
        'machine_id': 'PC1',
        'platform': 'hostname'
    },
    'mu1_samples': array([...], dtype=float16),   # always float16
    'mu2_samples': array([...], dtype=float16),   # always float16
    'full_results': array([...], dtype=float32),  # optional, only when enabled
    'computation_time': 28.4,
    'timestamp': 1728394425.123
}
```

### Loading Samples
```python
import gzip
import pickle
from pathlib import Path

samples_file = Path("samples_sf1_60.0_sf2_115.0_sp_82.0_a3f2c1d8.pkl.gz")
with gzip.open(samples_file, 'rb') as f:
    data = pickle.load(f)

mu1 = data['mu1_samples']
mu2 = data['mu2_samples']
params = data['parameters']
```

---

## Key Configuration

### From config.py
```python
config.samples_folder = './samples_10k'
config.n_samples = 100
config.param_step = 10
config.param_range_low = 10
config.param_range_high = 200
```
Note: `config.samples_folder` is overridden during runtime; sample generation uses the folder set by `configure_samples_folder()`.

### Runtime Folder Selection
```python
configure_samples_folder(
    n_simulations, n_samples, algorithm,
    diagonal_covariance, fix_weights
)
```

### Samples Folder Naming
The output folder name is constructed by `build_samples_folder_name()` in `simulated_samples_grid.py`:
```python
sim_samples_{sim_part}_{n_samples}samples_{geometry}_{algorithm}_{covariance}_{weights}
```
Where:
- `sim_part` is `n_simulations` formatted as `1k`, `2k`, etc. when divisible by 1000 (otherwise raw integer)
- `geometry` is `circular` if `jf.wrap_1st` is True, else `linear`
- `algorithm` is `em` (lowercased). **Only `em` is implemented.** `vbem` / `vbem_mix` exist
  as CLI choices, docstrings, and zero-filled result columns (`weight_mix`/`mu1_mix`/`mu2_mix`)
  but have **no inference code** — requesting them now raises `NotImplementedError`
  (pinned by `tests/test_flag_dispatch.py`).
- `covariance` is `diagcov` (the default and **only implemented mode**). `fullcov`
  (`--no-diagonal-covariance`) now raises `NotImplementedError` — full covariance is not
  implemented for the circular model. (Pre-2026-07-03 artifacts labeled `fullcov` were
  actually diagonal; the flag used to be silently ignored.)
- `weights` is `fix_weights` if `fix_weights` is True, else `free_weights`. `fix_weights`
  is implemented (holds mixture weights at 1/K); `free_weights` estimates them (floored to
  [0.1, 0.9]).

### Sample Generation Parameters
```python
feat_diff_step = 2
mu1_bias_step = 2
mu2_bias_step = 6
n_simulations = 1000
n_samples = 100
```

---

## Performance Characteristics

### Timing Estimates (approx)
- **Per parameter combination**: depends on n_simulations × n_samples × feat_diff grid size
- **Per chunk**: 50 combinations per large chunk, 5 per small chunk
- **Multi-machine**: linear speedup with number of machines

### Computational Bottleneck
- `simulate_dual_component_bias_distribution()`: repeated EM fits per simulation (225-init multistart; EM is the only implemented algorithm)
- JAX JIT compilation provides substantial speedup for large runs

### Memory Usage
- `mu1_samples` and `mu2_samples` are stored as float16 to reduce disk usage
- Optional `full_results` uses float32 and increases file size
