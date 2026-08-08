# Mu1 objective ablation

This experiment tests training objectives against independent, high-simulation
reference surfaces. The `20` and `100` labels are observations available to the
simulated observer; they are separate estimands and are never compared as
low- versus high-quality versions of one another.

Run the stages separately from the repository root:

```bash
./neural_network_optimization/run_objective_ablation.sh 20 select
./neural_network_optimization/run_objective_ablation.sh 20 truth
./neural_network_optimization/run_objective_ablation.sh 20 train
./neural_network_optimization/run_objective_ablation.sh 20 evaluate
```

Replace `20` with `100` for the other observer condition. `truth` generates
100,000 independent simulation iterations for 15 selected surfaces: three each
from peaked, flat, seam-enriched, asymmetric, and multimodal regimes. Selection
uses a deterministic subset spread across stored bundles and enforces parameter
spacing within each regime.

The default training comparison is KL; KL plus circular energy; the full
circular distributional objective; that objective plus Hellinger; that objective
plus the legacy log-gradient penalty; and that objective plus a 10,000-weight
probability-curvature penalty. Curvature profiles at weights 1,000 and 100,000
are also available. For example:

```bash
PROFILES="circular_curvature_1k circular_curvature_10k circular_curvature_100k" \
  ./neural_network_optimization/run_objective_ablation.sh 20 train
```

The ablation defaults to a 25-epoch, batch-32 screening run. Measurements on an
RTX 5080 found essentially identical full-epoch time at batches 32, 64, and 128;
batch 128 additionally approached the 16 GB memory limit. Twenty-five epochs
are 50,000 updates over the full 64k-example grid, versus three million updates
in the historical production schedule.
After selecting an objective, reproduce the old production update schedule only
for the winner with:

```bash
EPOCHS=1500 BATCH_SIZE=32 RUN_TAG=production PROFILES="<winning-profile>" \
  ./neural_network_optimization/run_objective_ablation.sh 20 train
```

`truth-repeat` optionally generates a second independent 100k reference. If it
exists, `evaluate` reports reference-to-reference disagreement as the Monte
Carlo floor.

Results are written under
`results/mu1_objective_ablation_<N>samples/`. The detailed evaluation CSV keeps
every feature-dissimilarity column. The summary is stratified by scenario; do
not use a mean bias collapsed across feature dissimilarity to select a loss.

Useful overrides are `REFERENCE_SIMULATIONS`, `PER_SCENARIO`, `EPOCHS`,
`BATCH_SIZE`, `RUN_TAG`, `PROFILES`, `SOURCE_SURFACES`, and `EXPERIMENT_ROOT`.
