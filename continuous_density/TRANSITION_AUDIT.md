# K12 transition audit and revised work plan — 2026-09-07

Current status updated: 2026-09-11.

## Purpose and evidence

This memo records an audit of `TRANSITION_PLAN.md` and the inner
`demixing_model` transition through `f47f3af`, followed by the corrections made
in the subsequent discussion. It replaces the audit's initial ordering of the
remaining work. It does not replace the detailed transition plan or claim
production clearance.

Generated evidence referenced through `$DEMIXING_ARTIFACT_ROOT` is external and
is not shipped with or expected in a normal checkout.

The audit used the repository guidance in `AGENTS.md`, `CLAUDE.md`, and
`bias_model_comparison/analysis/CODING_GUIDE.md`. The original audit did not
rerun tests or GPU jobs; statements about its test totals and discovered defects
came from the commit record. Claims about current support were checked against
source. The 2026-09-07 update fixed and reran the density start-sweep selection
and its targeted recovery tests. A later correction found that the first
single-condition likelihood comparison used incomplete hierarchy and BADS
comparators. Corrected objective-specific searches and paired surface
comparisons have now completed for all four retained objectives on the focused
n=100 panel, including their held-out stages. Detailed protocol and generated
results remain under `$DEMIXING_ARTIFACT_ROOT`.

## Executive summary

K12 itself remains a credible replacement for the surface NN. Development
comparisons against the simulation corpus show that it is the more faithful
forward representation. The focused actual-DM recovery panel now supports WNM
under likelihood and bias-weighted CRPS; density and smoothed expectation give
weak parameter recovery for both families even after target-alignment checks.
The full representative real-data comparison, public routing for the selected
common WNM search, the comparison-bounds decision, and direct fitted-WNM curve
exports are now complete. The matched native density and smoothed-expectation
comparisons favor the surface NN on average, although common-WNM rescoring favors
the WNM parameter solutions. Broader production rollout is therefore not
cleared.

The transition built much of the low-level WNM machinery across loaders,
prediction operations, every historical objective, optimization, provenance,
caches, plotting estimators, and recovery infrastructure before the key recovery
comparisons were complete. From the first transition
commit (`bd8a0f1`) through the last implementation commit covered by this update
(`25448c3`), 27 commits changed 49 files, adding 9,710 and deleting 1,159 lines;
4,200 inserted lines are under `tests/`. The next commit, `f47f3af`, added 1,479
lines by bringing the four transition documents into version control and is not
counted as implementation growth. At least 49 defects are explicitly recorded
across the audit/fix commits, several of which changed fitted or reported
numbers.

R5 and the production-report part of R6 are complete. The WNM direct export has
73,440 curve rows and 204 condition plots. Presentation-only browser work,
rollout, regeneration, and removal of the surface NN remain deferred.

## Current implementation status

### Completed or substantially completed

- The selected n=20 and n=100 K12 weights are packaged as self-contained
  artifacts with architecture, observer-sample identity, domain, and corpus
  provenance.
- A central surrogate loader and checkpoint-by-run resolution exist.
- WNM supplies analytic point density, moments, circular SD, signed-arc mass,
  cell probabilities, and analytic motor convolution.
- Empirical fitting-target construction has been separated from the surface
  optimizer.
- Bounded multistart continuous WNM fitting is connected to the public fitting
  command, with fingerprints and search diagnostics.
- WNM scoring exists for every objective currently written by the fitting
  command.
- Likelihood postprocessing can resolve and rescore a WNM fit.
- Initial n=20 WNM closed-loop density recovery, random-range panels, and a
  noise-free start-count sweep over hard range-panel cases have run.
- The first actual-DM data panel is frozen and generated: n=100, single
  condition, no motor noise, with uniformly allocated dissimilarities and nested
  180/450/900-trial datasets. Objective-specific WNM searches and the deployed
  surface baseline fitted the complete development and held-out panels for
  likelihood, bias-weighted CRPS, density, and smoothed expectation. Common
  recovery, loss, mean-bias, bias-SD, asymmetry, and dissimilarity-band summaries
  are saved with the runs. Density and smoothed-expectation target-alignment
  diagnostics are also complete.

### Partial or not yet completed

- The rejected/diagnostic objective-specific hierarchy, cache, and JAX-BADS
  searches remain recovery-only. The selected common 64-start batched port is
  integrated into the public fitter through `--search continuous`.
- The direct prediction command identifies WNM artifacts but still constructs
  the surface-only optimizer, so it does not yet provide a working WNM prediction
  path.
- On the actual-DM n=100 panel, paired development/held-out surface comparisons
  are complete for all four objectives. A four-condition, 2,330-trial
  representative color-2 real-data comparison is also complete. Broader n=20,
  multi-condition recovery, motor extensions, and additional real-data panels
  remain.
- Unified subject plots and PDF slices explicitly reject WNM. This is acceptable
  during the recovery stage and is deferred below.
- `surface_browser`, demos, external fit scripts, the comparison pipeline, model
  defaults, production regeneration, and surface-NN retirement have not migrated.

## Audit findings that remain applicable

### 1. Broad infrastructure preceded the decisive analyses

The branch has built horizontal layers for many consumers and edge cases before
establishing which WNM search works for the objectives that determine the model
comparison. This enlarged the review surface and contributed to repeated
result-changing audit fixes. New work should now be justified by a named recovery
comparison or by the direct prediction/fitting path needed to run it.

### 2. The shared prediction abstraction is incomplete

`shared/prediction.py` is described as a common contract, but `BiasPredictor`
declares only identity and motor-noise methods. The WNM and surface implementations
have incompatible operation signatures, surface motor noise remains owned by the
old optimizer, and `predictor_from_surrogate` constructs only WNM predictors.
Callers therefore still branch by family, and WNM-specific scoring exists beside
the surface scoring path.

This need not be redesigned before recovery. It does mean that further generic
abstraction should stop unless it directly removes duplication from a recovery or
production call site.

### 3. Compatibility and provenance machinery is heavier than necessary

The identity fixes address real failures, but implementation remains partially
duplicated: WNM loading reconstructs the model separately from `load_model`, and
SHA-256 file readers exist in the surrogate loader, run fingerprinting, and WNM
packager. Some identity and shared-CLI helpers have no non-test consumers yet.

The golden-record fixtures and recorder scripts are explicitly scheduled for
deletion with the legacy backend. They also failed to catch the 37-degree pooled-SD
regression because the fixture did not exercise the clamp. Stable mathematical
contracts and public-call-site tests are more valuable than expanding temporary
equivalence scaffolding.

### 4. The change/audit cycle required a frozen boundary

Recent fixes include scoring on the wrong feature grid, accepting failed optimizer
starts as winners, silently fitting motor-enabled runs at zero motor noise, an
unreachable public WNM rescoring branch, a reduction mismatch larger than the
reproduction tolerance, a 37-degree surface-path change, an ineffective float64
fix, inverted explicit-zero motor semantics, condition-minor truth-column ordering
in the first port of the start sweep, and its subsequent selection of the first 40
thresholded rows rather than the claimed worst 40. The first regression test also
reimplemented the column parser instead of executing the runner, so it could not
protect the production path. The audits caught these, but the pattern shows that
features were being layered before a small set of canonical entry points was
frozen and exercised.

The focused recovery code and settings were subsequently frozen and the paired
analyses completed. Preserve that boundary: avoid reopening optimizer or target
selection from held-out results while moving to the real-data and prediction
stages.

### 5. Status documentation is already drifting

`pretrained/README.md` first says several consumers still resolve checkpoints
independently and retain known gaps, then later says the same identity defects are
fixed. Recent commits also corrected three earlier implementation claims. Commit
`f47f3af` fixed a separate provenance failure: earlier commit messages claimed to
carry transition-document changes that were outside the repository and therefore
untracked. The plan and findings are now tracked beside the code for the active
transition, with generated CSV/JSON artifacts remaining under
`$DEMIXING_ARTIFACT_ROOT/`. Status
prose should be updated only at frozen milestones and should be derived from
executing call sites where possible. The detailed result narrative remains
temporary tracked transition material and should return to the result artifacts
when the transition closes, as required by `AGENTS.md`.

## Corrections to the initial audit

### Search porting and caches are required

Porting hierarchical search and the relevant cache machinery to WNM is not legacy
feature parity and is not optional overengineering. It supplies independent search
strategies and practical reference solutions for WNM. Without them, a failure of
continuous WNM fitting cannot be separated from a poor optimizer.

The surface NN does not need analogous new development. It remains the deployed
baseline and should be run with its established fitting strategy.

### Recovery is a core attribution analysis

The density result available at the initial audit concerned the combined WNM +
empirical density target + continuous-search pipeline. Increasing from 16 to 64 starts did not materially
improve empirical-target recovery. A separate noise-free sweep shows that this
does not mean the search is adequate: even large multistart budgets miss known
optima on some hard cases. The empirical fits nevertheless achieve losses below
the loss at truth, proving that the realized empirical target can prefer a
non-truth parameter vector even if a still-better basin remains undiscovered.
Hierarchical, cached exhaustive, or other affordable reference searches were
therefore necessary to quantify the optimization gap and choose a practical
search; completed R2 below records the result.

On the corrected 40 hardest density cases, 16, 64, and 256 starts missed the
known noise-free optimum in 30, 18, and 7 cases, at median costs of 2.2, 7.1,
and 27.2 seconds per fit. No tested multistart count is a global-search oracle.

Recovery must separate:

1. **Forward surrogate fidelity:** WNM predictions versus independent,
   high-precision DM simulation at known parameters.
2. **Optimization quality:** competing searches evaluated by exactly the same WNM
   scorer and target.
3. **Objective/target identifiability:** whether the best established optimum is
   at the generating parameters.
4. **Surrogate contribution:** the additional error when data come from the actual
   DM observer rather than from WNM itself.
5. **Deployment comparison:** WNM with its selected search versus the deployed
   surface-NN pipeline on paired data.

### Optimization quality is objective-specific

`bias_model_comparison/analysis/support_functions.R` retains four demixing fit
objectives:

- `density`
- `smoothed_exp`
- `bias_weighted_crps`
- `likelihood`

These define the minimum optimizer-validation matrix. The raw `expectation`
objective is superseded and excluded from the analysis artifact. `balanced_crps`
is still generated by the external pipeline but is not a retained comparison
candidate. `density_legacy` needs replay parity, not a new recovery campaign.

Search quality cannot be inferred from density alone because the four costs have
different landscapes. The selected search may legitimately differ by objective.
A density curve cache does not need to be generalized to every cost merely for
uniformity; each objective needs an appropriate independent reference.

### Curve objectives have different recovery expectations

`density` and `smoothed_exp` compress a conditional response distribution to a
curve. They should be tested for parameter recovery, but poor recovery is not
surprising even when their fitted curves and predictions are accurate.

Likelihood is the primary parameter-recovery objective because it retains
trial-level bias and feature-difference information. Bias-weighted CRPS is the
secondary distributional recovery objective, while recognizing that its weighting
deliberately deemphasizes regions with little signed bias. For the two curve
objectives, predictive-curve recovery is primary and parameter recovery is a
diagnostic.

### The surface NN must be included in the paired analyses

The purpose is not to develop the surface NN or force it to use WNM's search. For
every matched synthetic and real dataset, compare:

- WNM using the search selected for that objective; and
- the surface NN using its existing deployed search.

Evaluate both on common downstream scores and independent simulator references.
This determines whether the complete WNM pipeline is at least as effective as the
legacy production pipeline, while the within-WNM search comparisons determine
whether a WNM failure is merely an optimization failure.

## Completed recovery analyses and current critical path

R1--R4 below record the completed focused recovery stage. R5--R6 are now the
current critical path.

### R1. Freeze the first paired recovery data design — completed

- Start with n=100, one condition, and no motor noise.
- Use 12 fixed truth tuples: three each in narrow, ordinary asymmetric, reversed
  asymmetric, and broad/weak regimes. Eight are development tuples and one from
  each regime is held out.
- At every tuple, allocate trials uniformly over feature differences 2:2:180,
  with global orientation fixed at zero. Generate five response seeds and nested
  180/450/900-trial datasets (2/5/10 responses per dissimilarity).
- Reuse the identical generated datasets across search and surrogate arms.
- The exact protocol, simulator/code hashes, checkpoint hashes, design, and raw
  responses are under
  `$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`.

Shared evaluation metrics and practical optimization-gap/runtime criteria are
part of R2. The likelihood held-out fits were already inspected before the
comparator correction, so they remain descriptive rather than a new prospective
confirmation. n=20, multi-condition, and motor-noise extensions are deferred
until this focused panel works.

### R2. Establish optimization quality separately by objective — completed

All candidate solutions for an objective must be reevaluated through one common
WNM scorer. Record best loss, gaps to the best available reference, boundary hits,
convergence, start variability, evaluator calls, wall time, and peak memory.

| Objective | Minimum useful comparison |
|---|---|
| `density` | Cached exhaustive, hierarchical, continuous multistart, and grid-seeded continuous polish. |
| `likelihood` | Hierarchical and continuous search, plus an affordable dense or profiled reference on tractable cases. |
| `bias_weighted_crps` | Hierarchical and continuous search, plus a small explicit-grid/profile reference. |
| `smoothed_exp` | Hierarchical and continuous search; no fitted-motor arm because symmetric motor noise leaves the mean unchanged. |

The goal is not to declare one universal optimizer. Freeze the simplest reliable
strategy separately for each objective.

For likelihood, the corrected comparison used the production surface hierarchy,
32-start JAX-BADS with BBZ's three-parameter search budget, and a 24-dataset
PyBADS check. The three affordable arms were extended to all 120 development
datasets and rescored separately on CPU and GPU. The JAX-BADS versus L-BFGS-B
NLL order reversed with the scoring device, while paired parameter recovery was
effectively tied. L-BFGS-B had the smaller worst-case gap on both devices, at
half the median time and about one ninth of the evaluations. Likelihood was
initially frozen on serial SciPy L-BFGS-B with 32 deterministic log-space
Latin-hypercube starts. A subsequent faithful-port comparison isolated default
GPU TF32 matmuls as the source of the remaining device discrepancy. With
float32 arrays and `highest` matmul precision, the 32-start batched JAX port met
the 0.001 gate on all 120 development datasets at 9.65-fold median paired
speedup. That configuration now supersedes serial SciPy and is frozen for
held-out confirmation. PyBADS was not extended because it tied L-BFGS-B on CPU
at roughly 30 times the runtime. The production hierarchy was worse under both
scoring devices; its catastrophic truth-curve failures were concentrated in the
narrow cases because its one-degree absolute feature lattice is too coarse near
the lower bound. Detailed traces and thresholds are recorded in the external
recovery artifact.

BWCRPS followed the same implementation transition. The 32-start float32 port
with `highest` matmul precision met the 0.001 union gate on all 120 development
and all 60 held-out datasets. On held-out data its worst gap was 0.000721 and
its median paired speedup over serial SciPy was 5.26-fold, so it now supersedes
the serial implementation.

For the legacy density operator, an 80-by-64 log curve cache plus one continuous
polish was selected. The current density target is matched KDE; its 32-start
`highest`-matmul port reproduced all 60 frozen held-out SciPy diagnostics within
1.79e-7 at 25.75-fold median speedup. The subsequent proper development
comparison rebuilt the 80-by-64 lattice per empirical bandwidth. A 64-start
port met the 0.001 gate on all 120 datasets with maximum gap 4.17e-7, was never
materially worse than the lattice, and was 65.9 times faster at the median; it
is now the selected matched-density search. The current smoothed-expectation
operator uses observed-design matched complex-moment smoothing. Its rebuilt
cache remained necessary in the proper development comparison, which also
included JAX-BADS. Replacing the 32-start SciPy component with the port preserved
the adapted composite's 116/120 coverage and 0.249 worst gap while reducing
median time from 7.20 to 1.99 seconds. That substitution is frozen, but the
conditional selection rule remains unresolved because four union winners are
still missed. A complete 64-start port follow-up improved the port-only gate
rate from 86/120 to 91/120 but left all four composite misses unchanged, so the
remaining defect is the trigger rather than the continuous start budget.

The final cross-objective policy supersedes those objective-specific selections.
By executive decision, every retained objective uses the 64-start batched JAX
L-BFGS-B port with float32 arrays and `highest` matmul precision; no cache,
SciPy union, or conditional hierarchy is selected. The reason is operational
consistency, not dominance on every empirical objective. In particular, the
decision accepts matched-smoothed coverage of 91/120 and worst gap 44.40 rather
than the cache/composite's 116/120 and 0.249. Paired parameter recovery and
whole/banded truth-curve CCC did not show a systematic 64-start deficit. The
64-start policy improved likelihood's rare deep-basin misses, was neutral for
BWCRPS recovery, and fully covered the matched-density development union.

### R3. Complete WNM closed-loop recovery — completed

- Run likelihood and bias-weighted-CRPS recovery first; these are the strongest
  parameter-recovery tests.
- Run density and smoothed-expectation recovery as well, emphasizing recovery of
  their fitted curves and independent predictions.
- Use the search comparisons from R2 to label failures as optimization gaps or as
  optima that genuinely prefer parameters away from the truth.
- Where a well-established optimum remains away from truth, use a very-large-sample
  or expected-target diagnostic to separate finite-sample variation from target
  construction and intrinsic non-identifiability.
- Retain the existing density range panel as evidence about density fitting only;
  do not generalize its `sd_feat` result to WNM likelihood or distributional
  recovery.

The actual-DM likelihood development fits provide the first result: 72.5% of
datasets recovered all three parameters within a factor of 1.5, with median
per-dataset joint log-RMSE 0.150. Broad/weak cases remained hardest. This is a
finite-sample result from the selected 32-start search, whose optimization
quality is now independently supported by the corrected development comparison.
The corresponding BWCRPS development and held-out analyses are now complete.
On the 60 held-out datasets, WNM had the lower common WNM BWCRPS score than the
surface-NN fit on all cases; median joint log-RMSE was 0.269 versus 0.345 and
factor-1.5 recovery was 50.0% versus 43.3%. Curve metrics remained mixed, so
the result supports the primary distributional and parameter criteria without
claiming a uniform curve-level win.

Density and smoothed-expectation recovery are also complete. Density-summary
parameter recovery is weak for both WNM and the surface pipeline. Aligning both
branches with the same KDE operator reduces the WNM's large-error tail but does
not shift typical recovery; held-out global joint log-RMSE is 1.305 for matched
WNM and 1.306 for the existing surface pipeline. Prediction-equivalence checks
show genuine non-identifiability in narrow and reversed regimes, but materially
wrong selected distributions in parts of the broad/weak and ordinary regimes.
Smoothed expectation supports WNM mean-bias prediction away from near-zero
dissimilarity, while its parameters and spread remain weakly identified. Its
empirical curve applies a 20-degree feature kernel but its model curve is
pointwise. The completed development alignment diagnostic applies the same
operator to model complex moments, improves median log-RMSE from 0.730 to 0.637,
and removes the paired surface advantage. All-parameter factor-1.5 recovery
remains only 23.3%, so weak identification persists. The frozen 60-dataset
held-out matched-WNM run without truth starts is complete. Matched WNM has
median/global log-RMSE 1.130/1.239, versus 0.981/1.248 for current WNM under the
same 32-start SciPy search and 0.970/1.149 for the existing surface NN. The
paired differences are inconclusive, and the development parameter gain does
not generalize. Matched truth is nevertheless closer to the empirical curve on
73.3% of held-out datasets, supporting the matched operator for curve semantics
while leaving smoothed expectation a weak parameter-recovery objective. An
annotated check of the two intermediate held-out cases at 450 trials clarifies
why. For both density and smoothed expectation, the recovered-parameter curve is
closer to the realized empirical target than the generating-parameter curve in
all 10 datasets for both WNM and the surface NN. The fitting is good, but with
only five responses per dissimilarity the empirical curves fluctuate too much
around truth. The optimizer follows those fluctuations, producing poor
parameters and, especially for reversed smoothed-expectation cases, materially
displaced curves.

On the already-inspected held-out tuples, 70.0% recovered all parameters within
a factor of 1.5, with median per-dataset joint log-RMSE 0.140. Recovery rose from
25% at 180 trials to 90% at 450 and 95% at 900. These are descriptive results for
the selected 32-start search, not a fresh prospective confirmation: the tuples
had already been inspected before the corrected development comparison.

### R4. Run paired recovery from the actual DM observer — completed

Generate the main synthetic panel with the actual DM observer simulator, not WNM
or the surface NN. Fit every dataset with both:

1. WNM and the objective-specific search selected in R2; and
2. the surface NN and its current deployed search.

For likelihood and bias-weighted CRPS, prioritize parameter bias, log-ratio RMSE,
condition contrasts, boundary/failure rates, and predictive distributions. For
density and smoothed expectation, report the same parameter diagnostics but judge
primarily by banded curve and predictive recovery.

Evaluate both surrogate pipelines against independent DM references using common
quantities: parameter errors, bias/asymmetry/spread errors by dissimilarity band,
shared density loss, shared BWCRPS, common likelihood-mass scores, and operational
cost. Do not compare only each model's native training loss.

The first pass is limited to the frozen n=100 single-condition panel from R1.
Broader condition structures and motor recovery are not prerequisites for this
first comparison.

For the 120 likelihood development datasets, WNM beat the deployed surface
pipeline on common-grid NLL in 78.3% of pairs. Like-for-like global joint
log-RMSE was 0.410 for WNM versus 0.453 for surface; median per-dataset values
were 0.150 versus 0.189. Bias-SD and asymmetry curve summaries favored WNM;
mean-bias CCC favored WNM while its median RMSE was slightly higher.

The one-time 60-dataset held-out comparison found the same directional result.
WNM beat surface on common-grid NLL in 70.0% of pairs; global joint log-RMSE was 0.391
versus 0.587, median per-dataset joint log-RMSE was 0.140 versus 0.315, and
factor-of-1.5 recovery was 70.0% versus 43.3%. Curve metrics remained mixed,
especially for nearly flat targets, rather than showing uniform dominance. It
cannot serve as prospective optimizer confirmation because it was inspected
before the corrected development-only search selection.

### R5. Run paired representative real-data fits — completed, promotion gate not met

The first panel used subject S10 from `data_color_comb_color2_two_subjects.csv`,
four conditions and 2,330 retained trials, with motor noise fixed at zero. WNM
used the K12 n=20 artifact and the selected 64-start batched port; the surface
baseline used `model_epoch1425_10ktrain_20samples.pkl` and its deployed
hierarchical search. Both completed all four retained objectives.

Both sets of fitted parameters were then evaluated through the same selected
WNM scorer. WNM had the lower common loss for likelihood (9812.5701 versus
9814.9429), BWCRPS (73.0669 versus 73.2619), and matched smoothed expectation
(11.3554 versus 11.8579). The surface-derived density parameters were lower on
matched density (1.7438 versus 1.7883). The latter is a real residual search
qualification: the WNM density fit had 61/64 converged starts and a wide spread,
so the production search must retain all endpoints and cannot be described as a
global optimizer. Full parameter and cross-score tables are written by
`compare_wnm_surface_fits.py`.

The full CSH2026 comparison then refitted the epoch-1500 surface NN on the same
56,511 retained trials and the same subject-by-experiment groups as WNM, using
the matched pooled-SJ density target and observed-design complex-moment target.
There are 51 paired fit groups and 204 conditions. Surface-minus-WNM native loss
averaged -0.0500 for density (surface won 39/51 groups) and -1.3157 for smoothed
expectation (surface won 45/51). The direction held in every experiment.

This is not an objective or optimizer mismatch. Both native curve comparisons
use the same empirical targets, observed-design operators, and loss definitions.
When both parameter sets are instead evaluated through WNM, WNM has the lower
aggregate score for all four objectives; it wins 51/51 groups for likelihood,
BWCRPS, and density, and 41/51 for smoothed expectation. At the surface-fitted
parameters, changing only the forward representation from surface NN to WNM
raises total density loss by 12.0824 and smoothed-expectation loss by 621.7887.
The largest condition-level smoothed-expectation gap was then checked against
100,000 actual GMM/EM solutions at every point of the 2:2:180 dissimilarity
grid. At those surface-selected parameters, observed-design-pooled mean-curve
RMSE was 0.388 degrees for WNM and 9.748 degrees for the surface NN; WNM was
also closer for circular SD, density asymmetry, held-out NLL, and density L1.
Thus this large score reversal is a surface forward-approximation error, not
evidence that its parameterized curve is closer to the mechanistic model. One
case does not establish corpus-average fidelity, and the selected 64-start
optimizer retains its separately recorded local-minimum qualification. Detailed
generated tables and plots are under
`$DEMIXING_ARTIFACT_ROOT/csh2026_20samples_wnm/comparison_current_objectives/`.
The actual-GMM check is under
`$DEMIXING_ARTIFACT_ROOT/csh2026_20samples_wnm/forward_truth_check_s15_color_hv_1_high_low/`.

- Fit the same representative datasets, rows, conditions, outlier policy, and
  motor policy with WNM and with the deployed surface-NN baseline.
- Cover all four objectives retained by `bias_model_comparison`.
- Evaluate both through the common metrics consumed by that analysis: smoothed
  empirical-curve RMSE, shared density loss, shared BWCRPS, and common rescored
  likelihood/AIC/BIC where applicable.
- Check failures, boundary solutions, reproducibility, cold/warm runtime, and
  memory in addition to fit scores.
- Reuse legacy NN results only when their data, preprocessing, objective semantics,
  and evaluation conventions match exactly; otherwise rerun the paired NN arm.
- Before launching the curve-objective arms, expose the recovery-selected WNM
  cache/search dispatch through the fitting entry point. The focused panel did
  not settle multi-condition/shared-`sd_spat` search, so either limit the first
  paired panel accordingly or validate that extension separately.
- Resolve `OPEN_DECISIONS.md` item 3 and state whether both families share one
  fitting box or retain their artifact-specific bounds.

### R6. Validate the direct prediction path needed by the analyses — completed

The recovery and real-data comparisons must use WNM predictions directly rather
than route them through a sampled surface. Complete and test the direct WNM branch
of the prediction command for bias, asymmetry, circular SD, and distributional
outputs, including observer-sample identity and motor noise. Preserve the separate
averaged-surface/mu2 path.

This is prediction correctness needed for the scientific comparison. Presentation
features and browser integration remain deferred.

For the representative run, `export_wnm_fit_curves.py` predicts bias, matched
asymmetry, and circular SD directly from the fitted mixture and writes CSV plus
condition plots after validating the fit/checkpoint fingerprint.
`create_unified_subject_plots.py` now uses the same analytic path for standard
subject plots, summary plots and CSVs, exact report-order pooling, and PDF
slices, including fitted motor noise. It restores density-curve and matmul
settings from the fingerprint, preserves surrogate identity in exports, and
processes observed-design operators in bounded chunks. Its 1,440 representative
curve rows agree bit for bit with the dedicated exporter and are accepted by
the downstream comparison parser.

## Deferred stages

The following work remains deferred because the completed R5 comparison did not
meet the promotion rule.

### Presentation and exploratory interfaces

- The separate legacy `plot_pdf_slices.py` entry point (the production PDF-slice
  path in `create_unified_subject_plots.py` already supports WNM).
- `surface_browser` on-demand K12 views.
- Demo and publication-specific plotting scripts.
- General plot/export refactors unrelated to recovery outputs.

### Full downstream rollout

- Rewriting all external `bias_model_comparison` fit scripts and shell orchestration.
- Full production fit regeneration and comparison-artifact/report rebuilding.
- Default-family promotion in DM.
- Installation and user-facing documentation changes beyond concise current-status
  corrections.

These begin only after recovery selects the objective-specific WNM searches and
the paired real-data panel shows that the candidate pipeline is not worse than the
surface-NN baseline on the relevant shared scores.

### Surface-NN retirement

- Removing the surface backend, training code, checkpoints, and NN-only tests.
- Removing the transitional family selector.
- Deleting temporary surface-equivalence goldens and their recorder scripts.
- Archiving legacy artifacts and recording the final replay tag.

The surface NN remains load-bearing until downstream results have been regenerated
and verified on WNM.

### Additional research not required for this transition gate

- Joint mu1--mu2 modeling and corpus regeneration for paired outcomes.
- New WNM architectures, additional component-count studies, or new training seeds
  unless recovery identifies a material forward-surrogate failure.
- Recovery campaigns for excluded `expectation`, non-selected `balanced_crps`, or
  `density_legacy` beyond compatibility checks.
- General cache support for every objective when a smaller independent reference
  suffices.
- Global x64 adoption. The targeted likelihood diagnostic instead selected
  float32 arrays with `highest` GPU matmul precision; default TF32 changed the
  surrogate objective materially, while global x64 is unnecessary.

## Decision rule for resuming the migration

Resume presentation and rollout work when the evidence supports all of the
following:

1. Each of the four retained objectives has a frozen, independently checked WNM
   search strategy.
2. Remaining recovery failures can be attributed to optimization,
   objective/target identifiability, or forward-surrogate error rather than left as
   a combined unexplained result.
3. WNM is not worse than the deployed surface-NN pipeline on the paired actual-DM
   recovery panel and the representative real-data panel under the predeclared
   common metrics.
4. Direct WNM predictions and fit rescoring agree with their recorded artifact,
   observer-sample count, objective semantics, and motor-noise settings.

Parameter recovery need not be equally strong for every objective. In particular,
poor feature-SD recovery from a curve objective is a scientific limitation to
report, not automatically a K12 rejection, when optimization is adequate and the
corresponding predictive curves recover well.
