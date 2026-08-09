# Continuous conditional-density prototype

Experimental replacement for the numerical representation of the Demixing Model
likelihood:

```
raw EM outcomes → conditional wrapped-normal mixture
```

The production pipeline is untouched. This prototype models
`p(b | sd_feat1, sd_feat2, sd_ident, feat_diff)` directly and trains by raw-sample
negative log likelihood—never from a histogram, KDE, bias grid, or surface.

## Model and simulator

`sim_interface.py` calls the existing
`jax_fit_main.simulate_dual_component_bias_distribution`, preserving finite
internal samples, circular EM, component matching, bias definitions, diagonal
covariance, and fixed/free mixture-weight configuration. Each simulation yields
component-1 training data at `(sd_feat1, sd_feat2)` and component-2 training data
at the swapped input. Train/validation grouping is mirror-aware, so swapped
triples cannot leak across the split.

The emulator is a configurable `K`-component wrapped-normal mixture predicted by
a 64→128→128 MLP. SD inputs are log-scaled; `feat_diff` is supplied linearly and
with sine/cosine features. Means use `(cos,sin)`, weights use softmax, and scales
use softplus plus a 0.25° numerical floor. Wrapped-normal likelihoods use a
spatial log-sum-exp at narrow scales and a Fourier representation at broad
scales, remaining normalized even for scales far above 360°.

Motor noise is analytic: every component variance becomes
`sigma_dm**2 + sigma_motor**2`.

## Data and validation

Training can use fresh scrambled-Sobol continuous samples or remaining raw files
from the old 100-observation corpus. The latter corpus contains about 66k
filenames but most are processed stubs; `existing_samples.py` filters stubs
before selecting `--corpus-files`. It is sparse and on-grid, so continuous fresh
training data is the preferred scientific run.

Validation has two complementary raw-simulation designs:

- `points`: scattered, stratified off-grid combinations covering low/high noise,
  unequal feature SDs, and small/intermediate/large dissimilarity;
- `trajectories`: off-grid perturbations of previously identified peaked, flat,
  seam, asymmetric, and multimodal SD regimes, each evaluated along a fixed-SD
  `feat_diff` trajectory including 2° and 180°.

Earlier `results/mu1_experiments` work showed why both are needed: feature-axis
KDE pooling was the dominant old-pipeline distortion, especially at the
dissimilarity boundaries, while mean bias aggregated over dissimilarity could
hide structured errors. The archived 100k references contain only KDE surface
objects—the raw outcomes were discarded—so they are regression context, not
ground truth here. Generate fresh raw off-grid references. A second independent
seed can be passed to `evaluate.py` to quantify Monte Carlo repeat uncertainty.

The 20- and 100-observation simulators are distinct estimands. Checkpoint and
reference metadata are checked for `n_samples` mismatches; use epoch 1425 for the
20-observation production competitor and epoch 1500 for the 100-observation one.

## Reproducing

Run from this repository root. Set the artifact root explicitly before expanding
paths—if the variable is unset, `$DEMIXING_ARTIFACT_ROOT/...` would otherwise
become an unintended root-level path.

```bash
export DEMIXING_ARTIFACT_ROOT=/workspaces/demixing_model/results
export PYTHONPATH=.
PY=/workspaces/.venv/bin/python
mkdir -p "$DEMIXING_ARTIFACT_ROOT/continuous_density"

# Moderate continuous training set (raise sizes after the smoke run succeeds).
$PY continuous_density/generate_training_data.py \
  --n-points 4096 --n-simulations 200 --n-samples 100 --block-rows 1 \
  --simulation-chunk 200 \
  --shard-rows 128 --resume \
  --out "$DEMIXING_ARTIFACT_ROOT/continuous_density/train.npz"

# Fit each requested mixture size directly to raw outcomes.
for K in 2 4 8 12; do
  $PY continuous_density/train.py \
    --source "$DEMIXING_ARTIFACT_ROOT/continuous_density/train.npz" \
    --components "$K" \
    --out "$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k${K}.pkl"
done

# Optional selected low-spatial-d' augmentation. It mixes unequal, similar, and
# general feature-SD pairs so improving one UEV trough does not move the failure.
$PY continuous_density/generate_training_data.py \
  --training-design low-dprime --n-points 8192 --n-simulations 500 \
  --n-samples 100 --block-rows 8 --simulation-chunk 500 \
  --shard-rows 32 --resume --seed 27182 --design-seed 27182 \
  --out "$DEMIXING_ARTIFACT_ROOT/continuous_density/train_low_dprime_balanced_8k_500.npz"

$PY continuous_density/train.py \
  --source "$DEMIXING_ARTIFACT_ROOT/continuous_density/train_16k_500.npz" \
  --augmentation "$DEMIXING_ARTIFACT_ROOT/continuous_density/train_low_dprime_balanced_8k_500.npz" \
  --augmentation-fraction .5 \
  --init-model "$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k12_16k500.pkl" \
  --components 12 --steps 5000 --batch-size 8192 --lr .0003 --warmup 100 \
  --seed 29 \
  --out "$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k12_lowdprime_balanced_aug.pkl"

# Primary and independent-repeat scattered off-grid raw references.
for SEED in 314159 271828; do
  $PY continuous_density/generate_training_data.py --validation \
    --validation-design points --per-stratum 12 --n-simulations 100000 \
    --n-samples 100 --block-rows 1 --simulation-chunk 250 \
    --shard-rows 1 --resume \
    --design-seed 424242 --seed "$SEED" \
    --out "$DEMIXING_ARTIFACT_ROOT/continuous_density/validation_points_${SEED}.npz"
done

# Fixed-SD difficult trajectories, also fresh and off-grid in the SD dimensions.
for SEED in 314159 271828; do
  $PY continuous_density/generate_training_data.py --validation \
    --validation-design trajectories --trajectory-curves 15 --trajectory-points 24 \
    --n-simulations 100000 --n-samples 100 --block-rows 1 \
    --simulation-chunk 250 --shard-rows 1 --resume \
    --design-seed 424242 --seed "$SEED" \
    --out "$DEMIXING_ARTIFACT_ROOT/continuous_density/validation_trajectories_${SEED}.npz"
done

# Structured unequal-encoding-variability curves. Spatial d' = 40 / sd_ident;
# canonical feature-SD pairs retain both component curves without duplication.
$PY continuous_density/generate_training_data.py --validation \
  --validation-design uev --uev-feature-step 2 \
  --n-simulations 100000 --n-samples 100 --block-rows 4 \
  --simulation-chunk 5000 --shard-rows 32 --resume \
  --seed 161803 \
  --out "$DEMIXING_ARTIFACT_ROOT/continuous_density/validation_uev_100k_seed161803.npz"

$PY continuous_density/plot_uev.py \
  --model "$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k12_16k500.pkl" \
  --checkpoint pretrained/model_epoch1500_10ktrain_100samples.pkl \
  --reference "$DEMIXING_ARTIFACT_ROOT/continuous_density/validation_uev_100k_seed161803.npz" \
  --out-dir "$DEMIXING_ARTIFACT_ROOT/continuous_density/uev_raw100k_vs_models"

# Diagnose whether the worst UEV trough is limited by K=12 or global training.
# Fits on half the raw outcomes and reports only held-out-half likelihoods.
$PY continuous_density/diagnose_local_capacity.py \
  --model "$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k12_16k500.pkl" \
  --checkpoint pretrained/model_epoch1500_10ktrain_100samples.pkl \
  --reference "$DEMIXING_ARTIFACT_ROOT/continuous_density/validation_uev_100k_seed161803.npz" \
  --summary "$DEMIXING_ARTIFACT_ROOT/continuous_density/uev_raw100k_vs_models/uev_raw100k_vs_models.csv" \
  --out-dir "$DEMIXING_ARTIFACT_ROOT/continuous_density/uev_local_capacity"

# Raw-reference metrics and repeat uncertainty; repeat for K=2,4,8,12 and both designs.
$PY continuous_density/evaluate.py \
  --model "$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k8.pkl" \
  --reference "$DEMIXING_ARTIFACT_ROOT/continuous_density/validation_points_314159.npz" \
  --repeat-reference "$DEMIXING_ARTIFACT_ROOT/continuous_density/validation_points_271828.npz" \
  --out "$DEMIXING_ARTIFACT_ROOT/continuous_density/eval_k8.csv"

# Head-to-head against the matching 100-observation production surrogate.
$PY continuous_density/compare_existing_model.py \
  --model "$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k8.pkl" \
  --checkpoint pretrained/model_epoch1500_10ktrain_100samples.pkl \
  --reference "$DEMIXING_ARTIFACT_ROOT/continuous_density/validation_points_314159.npz" \
  --out "$DEMIXING_ARTIFACT_ROOT/continuous_density/comparison.csv"

# Five plan figures from real artifacts.
$PY continuous_density/report.py \
  --eval "$DEMIXING_ARTIFACT_ROOT"/continuous_density/eval_k*.csv \
  --comparison "$DEMIXING_ARTIFACT_ROOT/continuous_density/comparison.csv" \
  --model "$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k8.pkl" \
  --checkpoint pretrained/model_epoch1500_10ktrain_100samples.pkl \
  --reference "$DEMIXING_ARTIFACT_ROOT/continuous_density/validation_points_314159.npz" \
  --out-dir "$DEMIXING_ARTIFACT_ROOT/continuous_density/figures"

# Independently optimise each likelihood on one empirical condition.
$PY continuous_density/fit_demo.py \
  --model "$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k12_16k500.pkl" \
  --data example_data/fischer_whitney_prepared.csv \
  --condition-col condition --condition orientation \
  --exclude-outside-domain \
  --checkpoint pretrained/model_epoch1500_10ktrain_100samples.pkl

# Standardised warmed-up inference timing and checkpoint size.
$PY continuous_density/benchmark.py \
  --model "$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k8.pkl" \
  --trials 1000 --repeats 100 \
  --out "$DEMIXING_ARTIFACT_ROOT/continuous_density/benchmark_k8.json"

# CPU-only unit tests. The opt-in simulator test may use the configured JAX device.
JAX_PLATFORMS=cpu $PY -m pytest continuous_density/tests -q
```

Generating all 100k references is deliberately expensive. The production
simulator vmaps over `n_simulations`, so passing 100k to one device call can
exhaust GPU memory even with one design row. `--simulation-chunk` bounds that
axis; `--block-rows 1` is the conservative default for the other axis. Run a
small smoke test before increasing the simulation chunk, and never launch
multiple GPU simulator jobs at once. Raw, uncompressed NPZ is the default
because compression can dominate runtime; pass `--compress` only when that
tradeoff is worthwhile. Every `(row shard, simulation chunk)` is atomically
written, and `--resume` verifies its design and shape before reuse. Chunk keys
are derived deterministically from their absolute row and simulation offsets.
The final NPZ is assembled only after all chunks exist; keep the shard directory
until the run is accepted. Progress is reported every 100 chunks by default;
change this with `--progress-every` without affecting the resumable manifest.

## Evaluation outputs

`evaluate.py` reports raw held-out NLL (primary), circular mean/resultant/SD,
density asymmetry, reporting-grid KL and L1 error, and reference/predicted
secondary-mode count, location, and mass. All large calculations are chunked;
non-finite EM outcomes are excluded rather than changed to zero. Trajectory
references additionally produce complex circular-moment curve RMSE and residual
second-difference summaries, so seam crossings and weak resultants do not make
the smoothness diagnostic unstable. The evaluator prints empirical behaviour
coverage; a hand-picked stratum label is not treated as proof that the raw
reference actually contains attraction, repulsion, multimodality, or seam mass.

`compare_existing_model.py` evaluates the conditional mixture and production NN
against the same raw outcomes. Production log-density interpolation is
renormalized before every likelihood or diagnostic.

`fit_demo.py` evaluates arbitrary trial-level dissimilarities and keeps fitted SDs
inside `[5,200]`. Prepared example files default to `abs_td_dist` and
`bias_to_distr_corr`; an optional production checkpoint is optimised
independently on the same trials. `benchmark.py` reports warmed-up prediction and
1,000-trial likelihood timings, device, and checkpoint size in JSON.

## Status

The prototype has been trained and evaluated on two independent fresh 100k
off-grid raw-reference corpora. The K=12 conditional density improves raw
held-out NLL over the production surrogate on 97.7% of scattered cases and
89.7% of difficult trajectory cases; a trial-level empirical fit was within
0.00133 NLL per trial of the independently optimized production model. See
[`RESULTS.md`](RESULTS.md) for the quantitative results, limitations, and
conclusion. Generated checkpoints, references, and CSVs remain in the external
artifact directory.

## Notes for developers

Generated NPZs, checkpoints, CSVs, and figures belong under
`$DEMIXING_ARTIFACT_ROOT/continuous_density/`; they are not tracked. The prototype
remains separate from production fitting code.
