# Pretrained checkpoints

Both checkpoints use the same architecture (`MirrorAwareMu1Predictor`, hidden dims 64→128→256, CNN channels 128→64→32→1, latent size 16) and training setup (1500 epochs, batch 32, lr 0.002, weight decay 1e-4, combined_probabilistic loss).  They differ only in how many simulation samples were averaged to produce each training surface.

| File | Epoch | Training surfaces | Sim samples / surface | Final loss |
|---|---|---|---|---|
| `model_epoch1500_8ktrain_20samples.pkl`  | 1500 | 8 000 (from 10k-point param grid) | 20  | 0.01849 |
| `model_epoch1500_8ktrain_100samples.pkl` | 1500 | 8 000 (from 10k-point param grid) | 100 | 0.05976 |

**Sim samples / surface**: each averaged surface was computed by running the simulator that many times at a given parameter combination and averaging the resulting empirical distributions.  More samples → smoother ground-truth surfaces but longer simulation time.

**Training surfaces**: 8 000 of the 10 000 simulated parameter combinations were used for training; the remainder formed the validation set.

**Parameter range**: sd_feat1, sd_feat2, sd_spat all swept over [10, 200] degrees (model space).

Source checkpoints in `/gmm2/checkpoints_mirror_aware_20samples/` and `/gmm2/checkpoints_mirror_aware_100samples/`.
