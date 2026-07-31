# Pretrained checkpoints

Both checkpoints use the same architecture (`MirrorAwareMu1Predictor`, hidden dims 64→128→256, CNN channels 128→64→32→1, latent size 16) and training setup (1500 epochs, batch 32, lr 0.002, weight decay 1e-4, 3M total steps with 1000 warmup, `combined_probabilistic` loss, mirror-aware augmentation).  They differ only in how many simulation samples were drawn per item when generating each training surface.

| File | Epoch | Training surfaces | Sim samples / item | Final loss |
|---|---|---|---|---|
| `model_epoch1500_10ktrain_20samples.pkl`  | 1500 | 10k-sim surfaces, 5° param grid, circular geometry | 20  | 0.02234 |
| `model_epoch1500_10ktrain_100samples.pkl` | 1500 | 10k-sim surfaces, 5° param grid, circular geometry | 100 | 0.06681 |

**Sim samples / item**: in the demixing model, the brain runs its mixture-fitting inference over a set of noisy internal samples it has of each presented item.  This number parameterizes how rich that internal representation is — 20 samples means a noisy / lower-evidence regime, 100 samples means a sharper, more-evidence regime.  Different values produce qualitatively similar bias curves but reallocate where the noise lives (more assumed internal samples → more inferred internal spatial noise when fit to the same data).

**Circular geometry**: the first feature dimension is wrapped (360° model space), matching how the surfaces are computed in `surface_computation/jax_fit_functions.py` (`wrap_1st = True`).

**Training surfaces**: drawn from `results/averaged_surfaces_10k_{20,100}samples_circular/` ("10k" = 10,000 simulations per surface, not grid points). The parameter grid is the combined Level-1+2 grid: sd_feat1, sd_feat2, sd_spat each on a **5° step over [5, 200]** (40 values per axis). The folder stores the canonical half (sd_feat1 ≤ sd_feat2); mirror augmentation expands it to 40³ = 64,000 training examples (2 per off-diagonal canonical pair + 1 per diagonal), giving the 3M total steps at 1500 epochs × batch 32 recorded in both checkpoints.

**Parameter range**: sd_feat1, sd_feat2, sd_spat all swept over [5, 200] degrees (model space). Fitting-time queries down to sd_spat = 5 are therefore inside training coverage.
