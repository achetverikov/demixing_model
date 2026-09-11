# Continuous conditional-density prototype: audit summary

**Snapshot:** 2026-08-09, branch `dev/continuous-density-fixes`, code state
`ae261b0` plus this report.  The governing plan is
the historical workspace brief `Task_ Prototype a continuous conditional-density
representation for the Demixing Model.md` (not shipped with this repository).

**Current-status note (2026-09-10):** this report remains the authoritative
record of the trajectory-model experiment at its snapshot, not the live
transition checklist. The focused actual-DM recovery and paired surface-NN
comparison have since completed for likelihood, bias-weighted CRPS, density,
and smoothed expectation. Current gate status is in `TRANSITION_AUDIT.md`; the
objective-specific `*FINDINGS.md` files contain the results. Representative
real-data fits, selected-search routing, comparison bounds, and direct WNM
prediction remain before default promotion.

## Executive conclusion

The prototype succeeds at the plan's main representation test: a conditional
wrapped-normal mixture trained directly on raw EM outcomes gives better
held-out raw-sample likelihood than the production KDE-surface NN, is continuous
in all four conditioning variables, and is much smaller.  The current retained
checkpoint is:

```text
$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k12_trajectory_design_large.pkl
```

It is a K=12 wrapped-normal mixture with a 128-256-256 parameter network,
112,048 network parameters, and a 453,794-byte checkpoint.  Its measured GPU
times are 0.138 ms for one-condition prediction and 0.318 ms for a 1,000-trial
likelihood after warm-up (100 repeats).  The matching production competitor is:

```text
pretrained/model_epoch1500_10ktrain_100samples.pkl
```

That production checkpoint is 9,184,330 bytes, about 20 times larger.

The direct representation is not uniformly accurate.  The remaining important
failure is structured underestimation of bias magnitude and response width in
high feature-noise regimes.  On newly simulated SD=90/120 development curves,
the retained model reaches a 3.68-degree mean-bias error and a 4.39-degree
response-SD error.  It is nevertheless much less fragile there than production,
whose corresponding maxima are 15.78 and 20.97 degrees.  The model should not
replace production yet without broader empirical fits and targeted work on
these structured tails.

## Which results are authoritative

The experiments use three levels of evidence that must not be conflated:

1. **Untouched final test:** independent off-grid low-spatial-d-prime
   trajectories, simulated only after model selection was frozen.  This is the
   authoritative selection result.
2. **Selection set:** 24 independent low-d-prime trajectories with 2,000
   outcomes per row, used repeatedly during training.  It is not a final test.
3. **UEV development regression:** canonical unequal-encoding-variability
   curves and a high-feature-noise extension, both with 100k raw outcomes per
   row.  These data drove diagnosis and plotting and are not untouched tests.

At this snapshot the complete CPU test suite reported **72 passed, 1 skipped**;
this is not a current branch-wide test count. The skipped test was the opt-in
simulator/GPU test.

## Canonical artifacts

| Purpose | Artifact |
|---|---|
| retained checkpoint | `$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k12_trajectory_design_large.pkl` |
| original global K=12 baseline | `$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k12_16k500.pkl` |
| trajectory training data | `$DEMIXING_ARTIFACT_ROOT/continuous_density/train_trajectories_2k_45x500.npz` |
| checkpoint-selection data | `$DEMIXING_ARTIFACT_ROOT/continuous_density/selection_lowd_24x45_2k.npz` |
| untouched final raw test | `$DEMIXING_ARTIFACT_ROOT/continuous_density/test_lowd_16x90_100k.npz` |
| baseline/retained final metrics | `$DEMIXING_ARTIFACT_ROOT/continuous_density/test_lowd_16x90_100k_density_comparison.csv` |
| retained/production final comparison | `$DEMIXING_ARTIFACT_ROOT/continuous_density/comparison_trajectory_large_on_test_lowd_16x90_100k.csv` |
| base UEV raw reference | `$DEMIXING_ARTIFACT_ROOT/continuous_density/validation_uev_100k_seed161803.npz` |
| high-noise UEV raw extension | `$DEMIXING_ARTIFACT_ROOT/continuous_density/validation_uev_highfeat_dprime2_100k_seed271828.npz` |
| current three-way UEV table | `$DEMIXING_ARTIFACT_ROOT/continuous_density/uev_objective_ablation/uev_raw100k_vs_models.csv` |
| current UEV figures | `$DEMIXING_ARTIFACT_ROOT/continuous_density/uev_objective_ablation/` |
| tightened local K=12 diagnostic | `$DEMIXING_ARTIFACT_ROOT/continuous_density/uev_local_capacity_asym_sd10_60_dprime2_comp1_b2048/` |
| retained-checkpoint benchmark | `$DEMIXING_ARTIFACT_ROOT/continuous_density/benchmark_k12_trajectory_design_large.json` |

The artifact root is external to the repository and is not shipped or expected
in a normal checkout. Raw samples are saved in the NPZ files; the references are
not merely precomputed surfaces.

## Model and data construction

- The existing JAX Demixing Model simulator and EM implementation are reused;
  the prototype does not replace EM.
- Each simulation supplies component-1 bias at the original input and
  component-2 bias at the mirrored input.  Train/validation grouping is
  mirror-aware.
- The density is a conditional mixture of wrapped normals trained by raw-sample
  NLL.  No histogram, KDE, bias grid, or surface enters the training loss.
- SD inputs are log-scaled; feature dissimilarity is represented linearly and
  periodically.  Mixture weights use softmax, scales use softplus plus a
  0.25-degree floor, and circular means use sine/cosine outputs.
- The retained training design contains 2,048 Sobol SD triples crossed with 45
  complete feature-dissimilarity locations: 92,160 parameter rows x 500 raw
  simulations.  Common random numbers are shared within each trajectory.
- The selection set contains 24 x 45 rows x 2,000 raw simulations.
- The final test contains 16 x 90 rows x 100,000 raw simulations, with 0
  non-finite outcomes.

Simulation arrays are chunked and resumable.  The final 100k test used four
design rows per device block and 5,000 simulations per chunk after an earlier
larger run caused a machine-level OOM crash.

## Untouched final-test results

### Retained trajectory model versus original global K=12

| metric | global K=12 | retained trajectory K=12 |
|---|---:|---:|
| mean raw-sample NLL | 3.838696 | **3.838553** |
| paired NLL difference | reference | **-0.000143 +/- 0.000028 (95% CI half-width)** |
| pointwise wins | 43.4% | **56.6%** |

The difference is small but reproducible on the frozen final test.

Maximum errors depend strongly on whether the reference circular mean is
identifiable.  Unfiltered angle maxima near a uniform distribution are not
scientifically interpretable, so resultant thresholds are reported explicitly.

| minimum reference resultant | global max mean | retained max mean | global max SD | retained max SD |
|---:|---:|---:|---:|---:|
| 0.2 | 9.292 deg | **6.474 deg** | **3.701 deg** | 4.170 deg |
| 0.3 | **2.413 deg** | 3.880 deg | 3.620 deg | **2.385 deg** |
| 0.5 | **2.413 deg** | 2.679 deg | 3.620 deg | **2.385 deg** |

Trajectory training therefore improves likelihood and some worst cases, but it
does not dominate every moment maximum.

### Retained trajectory model versus production NN

| metric | retained density | production NN |
|---|---:|---:|
| mean raw-sample NLL | **3.838553** | 3.867196 |
| pointwise NLL wins | **96.9%** | 3.1% |
| max mean error, reference R >= 0.2 | **6.474 deg** | 9.509 deg |
| max response-SD error, reference R >= 0.2 | **4.170 deg** | 16.255 deg |
| max mean error, reference R >= 0.3 | 3.880 deg | **2.732 deg** |
| max response-SD error, reference R >= 0.3 | **2.385 deg** | 3.936 deg |

The primary full-distribution metric favors the retained density decisively.
Production retains one advantage: the lower maximum mean error after excluding
all cases below R=0.3.

## Earlier global prototype and mixture-size test

Before the revised trajectory experiment, the 16,384 x 500 global Sobol model
already beat production on independent 100k off-grid references:

| validation design | global K=12 NLL | production NLL | density wins |
|---|---:|---:|---:|
| 108 scattered points | **3.530510** | 3.627022 | 97.7% |
| 15 difficult trajectories | **3.866589** | 3.997955 | 89.7% |

On the common scattered reference, mixture capacity showed diminishing returns:

| K | mean raw NLL | excess over K=12 |
|---:|---:|---:|
| 2 | 3.536577 | 0.006068 +/- 0.002594 |
| 4 | 3.531723 | 0.001213 +/- 0.000491 |
| 8 | 3.530681 | 0.000171 +/- 0.000118 |
| 12 | **3.530510** | reference |

K=12 was retained for accuracy, although K=8 is close.

## Corrected UEV development regression

The base UEV corpus contains feature-SD levels `{10,20,30,60}`, 90 feature
dissimilarities, both components, and nominal spatial d-prime `{0.5,1,2}`: 5,400
comparisons per method.  The high-noise extension adds SD 90 and 120 at nominal
d-prime 2: 1,980 comparisons per method.

The labels are historical: UEV design uses `40 / sd_ident`, but simulator
metadata records an actual spatial separation of 42 degrees.  Thus nominal
d-prime 0.5, 1, and 2 correspond to actual simulator values 0.525, 1.05, and
2.10.

Errors below are prediction minus raw.  Mean errors are circularly wrapped.
The maximum is the largest absolute pointwise error, with its sign retained.

### Base UEV levels

| method / metric | MAE | RMSE | signed maximum and location |
|---|---:|---:|---|
| retained mean bias | **0.104 deg** | **0.181 deg** | -1.220 deg at `(20,60,d'=0.5,diff=38,c2)` |
| production mean bias | 0.137 deg | 0.229 deg | -2.828 deg at `(30,60,d'=1,diff=2,c2)` |
| retained response SD | **0.184 deg** | **0.309 deg** | -1.554 deg at `(10,10,d'=0.5,diff=150,c2)` |
| production response SD | 0.309 deg | 0.436 deg | -3.121 deg at `(60,60,d'=0.5,diff=174,c2)` |
| retained density asymmetry | **0.0094** | **0.0126** | -0.0618 at `(10,60,d'=1,diff=122,c1)` |
| production density asymmetry | 0.0213 | 0.0292 | -0.1673 at `(10,60,d'=0.5,diff=132,c1)` |

The retained model is better on all three aggregate metrics and all three base
pointwise maxima.

### High-feature-noise extension

| method / metric | MAE | RMSE | signed maximum and location |
|---|---:|---:|---|
| retained mean bias | **0.418 deg** | **0.676 deg** | -3.677 deg at `(90,120,d'=2,diff=28,c2)` |
| production mean bias | 0.424 deg | 1.079 deg | -15.779 deg at `(10,120,d'=2,diff=168,c2)` |
| retained response SD | **0.406 deg** | **0.630 deg** | -4.389 deg at `(10,120,d'=2,diff=178,c2)` |
| production response SD | 0.937 deg | 1.685 deg | -20.968 deg at `(10,120,d'=2,diff=178,c2)` |
| retained density asymmetry | **0.0089** | **0.0117** | +0.0380 at `(90,90,d'=2,diff=126,c1)` |
| production density asymmetry | 0.0137 | 0.0176 | +0.0623 at `(10,120,d'=2,diff=72,c2)` |

The extension reveals worsening retained-model bias/SD extrapolation, but much
larger production outliers.  In averaged higher-noise-item curves, retained
mean-bias undershoot grows from -0.61 degrees for the SD-low=30 group to -1.92
at SD-low=60 and -3.68 at SD-low=90.  The corresponding density-asymmetry
undershoots are -0.016, -0.018, and -0.036.

The canonical current figures and their source CSV contain only raw 100k,
retained trajectory density, and production NN:

```text
$DEMIXING_ARTIFACT_ROOT/continuous_density/uev_objective_ablation/
```

The directory name is historical; it no longer contains objective-ablation
curves.

## Density-asymmetry correction

Density asymmetry is now evaluated as the direct raw sign mass

```text
P(bias > 0) - P(bias < 0)
```

with 0 and the circular antipode excluded from both sides.  An earlier plot
applied a fixed-kappa reporting KDE to raw samples but integrated model density
without the same blur.  At `(10,60,d'=2,c1,diff=116)`, that mismatch changed the
raw statistic from 0.339 to 0.079 and created a spurious 0.22 discrepancy.  All
current UEV asymmetry plots and `evaluate.py` use direct raw sign mass.

Consequences for auditing old artifacts:

- Do not use `ref_density_asym`, `pred_density_asym`, or
  `production_density_asym` in older comparison CSVs generated before commit
  `8a8d6b9`.
- Their raw NLL, circular mean, resultant, and response-SD fields remain valid.
- Use `uev_objective_ablation/uev_raw100k_vs_models.csv` for the corrected UEV
  asymmetry comparison.

## Local capacity tests

Two different diagnostics were run and answer different questions.

### Original low-d-prime mean-bias trough

At `(30,60,d'=0.5,component=2)`, a local fine-tune of the existing K=12 network
on the first 50k raw outcomes and evaluation on the untouched other 50k reduced:

- maximum mean-bias error from 2.317 to 0.763 degrees;
- maximum response-SD error from 1.916 to 1.053 degrees;
- held-out NLL from 4.692995 to 4.692046, improving 88/90 dissimilarities.

This showed that the original trough was not forced by K=12 mixture capacity;
global training allocation or parameter-map smoothing was implicated.  This
finding motivated the trajectory-structured experiment.

### Corrected asymmetry condition

Independent local K=12 mixtures were initialized from the retained model for
every dissimilarity on `(10,60,d'=2,component=1)`, fit on 50k outcomes, and
evaluated on a held-out 50k.  The tightened run used 3,000 steps and minibatches
of 2,048.

| method | maximum asymmetry error | mean held-out NLL |
|---|---:|---:|
| retained conditional density | **0.0479** | **1.999069** |
| independent local K=12 | 0.0584 | 1.999177 |
| local network fine-tune | 0.0682 | 1.999251 |
| production NN | 0.1042 | 2.045627 |

At feature difference 116 degrees, held-out raw asymmetry was 0.3390 versus
0.3010 retained, 0.3028 local K=12, and 0.2925 production.  The local K=12 test
does not identify mixture capacity as the bottleneck on this trajectory.  It
does not test the separate high-noise `(90,120)` mean-bias failure.

## Training and objective attempts

### What helped

| attempt | result |
|---|---|
| Direct raw-outcome likelihood | Large NLL improvement over production on both original off-grid 100k designs; establishes the representation's main value. |
| Increasing K from 2 to 4 to 8/12 | Clear likelihood gains through K=8; K=8 to 12 is small but measurable. |
| Balanced low-d-prime augmentation | Reduced development UEV max mean error 2.317 -> 1.385 deg and max SD error 1.916 -> 1.790 deg; slightly improved broad 100k NLL. It was an intermediate checkpoint, not the final retained model. |
| Local fine-tune at the original trough | Recovered much of the missing trough and showed that global allocation/smoothing, not K=12 alone, caused that failure. |
| Complete trajectory training design | Reduced base UEV maximum mean error to 1.220 deg and retained a reproducible final-test NLL gain over the original global model. |
| Larger 128-256-256 parameter network | Improved selection trajectory NLL from 3.851743 for the 64-128-128 trajectory model to 3.851568.  This is the retained architecture. |
| Conservative simulation chunking | Completed the 100k final/UEV corpora without another OOM; it changes runtime, not the statistical target. |

### What did not help enough

| attempt | result |
|---|---|
| More global Sobol data (`64k x 500` or `16k x 2k`) | Changed K=8 raw NLL only around the fourth decimal place, inconsistently across cases; comparable to a seed change. More isolated samples were not the bottleneck. |
| Narrow low-d-prime augmentations | Improved the targeted trough but moved response-SD or mean failures to uncovered regimes. The balanced version was safer but remained a targeted patch. |
| Independent local K=12 at corrected asymmetry condition | Did not improve maximum asymmetry error or held-out NLL over the retained conditional network. |
| Production NN on the corrected local trajectory | Worse maximum asymmetry error (0.1042) and held-out NLL (2.045627) than the retained density. |

### Grouped moment/CVaR objective ablation

All objective models used the same trajectory corpus, K=12 large network, seed,
20,000 steps, and selection reference.  The checkpoint was still selected by
trajectory raw NLL.

| objective | selection NLL | max mean, R>=0.2 | max SD, R>=0.2 | decision |
|---|---:|---:|---:|---|
| retained trajectory NLL | **3.851568** | 6.031 deg | **5.357 deg** | retain |
| grouped NLL | 3.851662 | 5.317 deg | 6.876 deg | reject |
| grouped + moment weight 2 | 3.851657 | **4.689 deg** | 6.709 deg | reject |
| grouped + moment weight 10 | 3.851789 | 4.960 deg | 5.664 deg | reject |
| moment 10 + CVaR 0.1 | 3.851936 | 4.464 deg | 7.879 deg | reject |

Moment weight 2 paid a paired selection-NLL cost of +0.000089 +/- 0.000036;
moment weight 10 paid +0.000221 +/- 0.000044.  On development UEV curves,
moment weight 2 reduced maximum mean error from 1.220 to 1.141 degrees but
increased maximum SD error from 1.554 to 2.493 degrees.  Moment weight 10 gave
1.221 and 2.047 degrees.  These objectives moved the first circular moment but
did not improve the full density or selection likelihood enough; they functioned
as band-aids and were rejected.

Rejected checkpoints remain available for audit:

```text
$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k12_grouped_nll.pkl
$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k12_grouped_moment2.pkl
$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k12_grouped_moment10.pkl
$DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k12_grouped_moment10_cvar10.pkl
```

## Plan coverage and remaining gaps

| Plan item | Status |
|---|---|
| reuse simulator/EM | implemented and tested |
| continuous off-grid design | implemented with Sobol points and complete trajectories |
| mirror symmetry | augmentation/grouping implemented and tested |
| direct wrapped-mixture likelihood | implemented without KDE/surfaces in training |
| K=2/4/8/12 | completed |
| independent 100k off-grid references | completed, including frozen final test |
| production comparison | completed for NLL, moments, UEV, and local diagnostic |
| density asymmetry | corrected to matched direct raw/model sign mass for current UEV/evaluator |
| secondary modes and grid density discrepancy | evaluated in older reports, but affected by reporting-KDE choices and not rerun for the retained checkpoint after the asymmetry correction |
| analytic motor noise | implemented and unit-tested; not validated in a full behavioral-fitting comparison |
| empirical fitting demo | full CSH2026 comparison completed for the retained trajectory checkpoint and objective-matched surface NN |
| focused actual-DM recovery | completed for all four retained objectives, including paired surface-NN and held-out comparisons |
| density/smoothed-exp target alignment | completed; matched operators improve target semantics but both objectives remain weak for parameter identification |
| selected-search routing, direct WNM prediction, bounds policy, and representative real-data panel | completed; the native curve-objective comparison does not clear WNM promotion |
| replacement of production | deliberately not done |

## Known limitations and audit warnings

1. **UEV is development data.** Its improvements and high-noise failures cannot
   be used as untouched model-selection evidence.
2. **Historical UEV d-prime labels are nominal.** Actual simulator separation is
   42 degrees, not 40.
3. **Older asymmetry columns are stale.** See the correction section above.
4. **`continuous_density/RESULTS.md` is historical forward-model evidence.** It
   now labels the balanced augmentation as an intermediate selection and points
   to this report and the transition findings for current status.
5. **High feature noise remains weak.** The retained model is much better than
   production on extreme maxima but still has structured bias/SD undershoot.
6. **The corrected local K=12 test covered `(10,60)`, not `(90,120)`.** It does
   not establish whether the high-noise failure is parameter-network training or
   local density-family capacity.
7. **The empirical fit result is not current-model evidence.** The earlier
   global K=12 was within 0.00133 NLL/trial of production on one 6,526-trial
   condition, but the retained trajectory checkpoint has not been refit there.
8. **Checkpoint-selection gains are small in absolute NLL.** Their paired
   uncertainty is reported; moment maxima do not uniformly improve.

## Snapshot recommendations and current next work

Items 1--4 below are forward-model experiments recommended at the audit
snapshot; they remain optional research rather than the current transition
critical path. Item 5 reflects the current R5--R6 work.

1. Run the held-out local-capacity protocol directly on the high-noise failures,
   especially `(90,120,nominal d'=2,component=2)` around feature difference 28
   degrees and `(10,120,nominal d'=2,component=2)` near 178 degrees.  This must
   precede another architecture or objective change.
2. If local K=12 recovers those curves, add targeted high-noise complete
   trajectories to a new independent training design and select strictly by raw
   trajectory NLL.  Do not add another mean-only penalty.
3. If local K=12 cannot recover them under held-out NLL, compare larger K and the
   plan's periodic Fourier log-density alternative locally before global
   retraining.
4. Regenerate retained-checkpoint off-grid density-asymmetry and secondary-mode
   metrics with the corrected raw statistic; do not reuse stale comparison CSV
   asymmetry fields.
5. Keep the completed CSH2026 matched-objective comparison as a transition gate.
   The selected WNM optimizer is adequate under common-WNM rescoring, but the
   surface NN has lower native density and smoothed-expectation loss on average;
   do not promote WNM or retire the surface backend on the current evidence.

## Reproduction entry points

The exact retained-data generation, training, UEV plotting, local-capacity, and
benchmark commands are in `continuous_density/README.md`.  The main scripts are:

```text
continuous_density/generate_training_data.py
continuous_density/train.py
continuous_density/evaluate.py
continuous_density/compare_existing_model.py
continuous_density/plot_uev.py
continuous_density/diagnose_local_capacity.py
continuous_density/benchmark.py
```

The retained experiment's shorter result record is also at:

```text
$DEMIXING_ARTIFACT_ROOT/continuous_density/trajectory_experiment_results.md
```

## Suggested audit checks

1. Load each checkpoint with `wrapped_mixture_model.load_model` and verify
   `hidden`, `components`, source metadata, selection score, and parameter count.
2. Recompute the frozen baseline/retained NLL means and resultant-filtered maxima
   from `test_lowd_16x90_100k_density_comparison.csv`; do not average mean bias
   over feature dissimilarity.
3. Recompute retained/production final NLL from
   `comparison_trajectory_large_on_test_lowd_16x90_100k.csv`, ignoring its stale
   density-asymmetry columns.
4. Confirm that the current UEV CSV has exactly the methods `raw 100k`,
   `trajectory density`, and `production NN`.  Recompute circular mean error as
   `wrap(predicted - raw)`, and take maxima over points rather than MAE alone.
5. Check `empirical_density_asymmetry` and its unit test: raw asymmetry must be
   computed from sign counts without passing through `reference_density`.
6. Confirm that training NLL calls raw mixture likelihood and never consumes a
   KDE or surface object, while grid/KDE operations are confined to reporting.
7. Verify that the final-test design seed differs from both training and
   selection seeds and that the final corpus metadata reports 100,000
   simulations per row and zero non-finite outcomes.
