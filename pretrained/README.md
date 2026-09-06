# Pretrained checkpoints

The production 20-observation checkpoint is
`model_epoch1425_10ktrain_20samples.pkl`. It was selected from the full
1500-epoch feature-smoothing experiment because its downstream bias curves were
smoother and closer to the simulated-surface curves than the former production
model. The stable defaults in the simulator and fitting pipeline point to it.

| File | Epoch | Native training grid | Objective | Sim samples / item | Loss |
|---|---:|---|---|---:|---:|
| `model_epoch1425_10ktrain_20samples.pkl` | 1425 | 128 mu1 rows × 128 feature columns | KL + circular energy + circular moment + density asymmetry + moment-trajectory curvature | 20 | 0.00008997 |
| `model_epoch1500_10ktrain_100samples.pkl` | 1500 | 64 mu1 rows × 90 feature columns | legacy `combined_probabilistic` | 100 | 0.06681 |

Both use `MirrorAwareMu1Predictor`, mirror-aware augmentation, batch size 32,
peak learning rate 0.002, and weight decay 1e-4. The 20-observation model is
resized to the standard periodic 180 × 90 surface only at inference; the
higher native training grid and the trajectory term reduce column-wise
oscillation without imposing a generic surface-flattening penalty. The old
`model_epoch1500_10ktrain_20samples.pkl` is retained only as a historical
checkpoint and is no longer selected automatically.

**Sim samples / item**: in the demixing model, the brain runs its mixture-fitting inference over a set of noisy internal samples it has of each presented item.  This number parameterizes how rich that internal representation is — 20 samples means a noisy / lower-evidence regime, 100 samples means a sharper, more-evidence regime.  Different values produce qualitatively similar bias curves but reallocate where the noise lives (more assumed internal samples → more inferred internal spatial noise when fit to the same data).

**Circular geometry**: the first feature dimension is wrapped (360° model space), matching how the surfaces are computed in `surface_computation/jax_fit_functions.py` (`wrap_1st = True`).

**Training surfaces**: "10k" means 10,000 simulations per surface, not grid points. The parameter grid is the combined Level-1+2 grid: sd_feat1, sd_feat2, sd_spat each on a **5° step over [5, 200]** (40 values per axis). The folder stores the canonical half (sd_feat1 ≤ sd_feat2); mirror augmentation produces 64,005 stored training rows because five diagonal cases are duplicated in the source bundles. This negligible duplication does not change batch geometry or training speed.

**Parameter range**: sd_feat1, sd_feat2, sd_spat all swept over [5, 200] degrees (model space). Fitting-time queries down to sd_spat = 5 are therefore inside training coverage.

## Conditional wrapped-normal mixture (WNM)

The replacement surrogate. Where the surface network emits a 180 x 90 log-density
surface, these artifacts emit the parameters of a K-component wrapped-normal
mixture over the component-1 bias, continuous in both the bias and the feature
difference. They are not load-compatible with `shared/utils.py:load_checkpoint`;
reach every checkpoint through `shared/surrogate.py:load_surrogate`, which reads
the family from the file's own content rather than from its name.

| File | K | Hidden | min_scale | Sim samples / item | Selected step | Corpus | NLL |
|---|---:|---|---:|---:|---:|---|---:|
| `wnm_k12_20samples.pkl` | 12 | (128, 256, 256) | 0.25 | 20 | 90000 | continuous_density_4.1p | 3.9785 |
| `wnm_k12_100samples.pkl` | 12 | (128, 256, 256) | 0.25 | 100 | 89000 | continuous_density_4.1o | 3.1657 |

**These artifacts are not yet the production default.** `shared/surrogate.py`
still defaults to the surface network; the transition plan
(`results/continuous_density_4.1q/TRANSITION_PLAN.md`) flips it after acceptance.
`fit_model_to_data.py` currently refuses a WNM checkpoint outright, because no
search backend consumes one yet.

**The NLL column is in-sample.** Both artifacts come from the `alldata` training
variant, which trains on every trajectory and picks its stopping step on an
in-sample 10% slice -- deliberately mirroring the surface network, which has no
validation split at all. The number is a training diagnostic and must never be
reported as a held-out score. Held-out accuracy is measured separately, on the
4.1q benchmark, whose trajectories are guaranteed disjoint from the corpus.

**Parameter order** is `[sd_feat1, sd_feat2, sd_spat, feat_diff]` in model
degrees, with `d' = 42 / sd_spat`. The model predicts component 1; component 2
comes from swapping `sd_feat1` and `sd_feat2`, not from a sign flip. Bias is
positive for attraction toward the other item, density is per model degree, and
the period is 360. Supported domain: SD in [5, 200], feature difference in
[0, 180]. Each artifact carries all of this in its own `meta` dict -- read it
from there rather than from this file.

**Regenerating one**: `continuous_density/package_wnm_artifact.py` packages a research fit
from a training stage's `run_cache/`. It copies weights and verifies the
packaged artifact reproduces the research checkpoint exactly before writing.

## Notes for developers

The source surfaces are untracked generated artifacts, not files shipped with
the repository. Their conventional names under `$DEMIXING_ARTIFACT_ROOT` are
`averaged_surfaces_10k_{20,100}samples_circular/`. These names document
checkpoint provenance; a normal checkout is not expected to contain them.
