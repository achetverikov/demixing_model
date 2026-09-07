# K12 transition audit and revised work plan — 2026-09-07

## Purpose and evidence

This memo records an audit of `TRANSITION_PLAN.md` and the inner
`demixing_model` transition through `f47f3af`, followed by the corrections made
in the subsequent discussion. It replaces the audit's initial ordering of the
remaining work. It does not replace the detailed transition plan or claim
production clearance.

The audit used the repository guidance in `AGENTS.md`, `CLAUDE.md`, and
`bias_model_comparison/analysis/CODING_GUIDE.md`. The original audit did not
rerun tests or GPU jobs; statements about its test totals and discovered defects
came from the commit record. Claims about current support were checked against
source. The 2026-09-07 update fixed and reran the density start-sweep selection
and its targeted recovery tests. A later update records the completed
single-condition development likelihood comparison; its detailed protocol and
generated results remain under `$DEMIXING_ARTIFACT_ROOT`.

## Executive summary

K12 itself remains a credible replacement for the surface NN. Development
comparisons against the simulation corpus show that it is the more faithful
forward representation. The unresolved question is whether the complete WNM
fitting pipeline can exploit that fidelity reliably and perform at least as well
as the deployed surface-NN pipeline on recovery and real-data fit.

The transition has built much of the low-level WNM machinery, but implementation
has spread across loaders, prediction operations, every historical objective,
optimization, provenance, caches, plotting estimators, and recovery infrastructure
before the key recovery comparisons are complete. From the first transition
commit (`bd8a0f1`) through the last implementation commit covered by this update
(`25448c3`), 27 commits changed 49 files, adding 9,710 and deleting 1,159 lines;
4,200 inserted lines are under `tests/`. The next commit, `f47f3af`, added 1,479
lines by bringing the four transition documents into version control and is not
counted as implementation growth. At least 49 defects are explicitly recorded
across the audit/fix commits, several of which changed fitted or reported
numbers.

The immediate work should therefore be recovery and the minimum search,
prediction, and fitting support required to attribute recovery failures. Plotting,
browser work, rollout, regeneration, and removal of the surface NN are deferred.

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
  180/450/900-trial datasets. The deployed surface baseline and the selected WNM
  likelihood search fitted all 120 development datasets; the frozen WNM search
  was then confirmed once on all 60 held-out datasets. Common recovery,
  likelihood, mean-bias, bias-SD, and asymmetry summaries are saved with the runs.

### Partial or not yet completed

- The likelihood benchmark can evaluate WNM with an independent batched
  hierarchy and with continuous search, but the public fitter still couples WNM
  to `--search continuous`; objective-specific hierarchy/cache integration
  remains for the other retained objectives where the comparison requires it.
- The direct prediction command identifies WNM artifacts but still constructs
  the surface-only optimizer, so it does not yet provide a working WNM prediction
  path.
- On the actual-DM n=100 panel, likelihood search selection and paired
  development/held-out surface comparisons are complete. Bias-weighted CRPS,
  density, smoothed expectation, and representative real-data comparisons remain.
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

### 4. The change/audit cycle has not reached a frozen boundary

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

Batch the recovery-enabling fixes, freeze their code and settings, and then run
the paired analyses. Avoid alternating large new layers with audit repairs while
the benchmark itself is changing.

### 5. Status documentation is already drifting

`pretrained/README.md` first says several consumers still resolve checkpoints
independently and retain known gaps, then later says the same identity defects are
fixed. Recent commits also corrected three earlier implementation claims. Commit
`f47f3af` fixed a separate provenance failure: earlier commit messages claimed to
carry transition-document changes that were outside the repository and therefore
untracked. The plan and findings are now tracked beside the code for the active
transition, with generated CSV/JSON artifacts remaining under `results/`. Status
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

The current density result concerns the combined WNM + empirical density target +
continuous-search pipeline. Increasing from 16 to 64 starts did not materially
improve empirical-target recovery. A separate noise-free sweep shows that this
does not mean the search is adequate: even large multistart budgets miss known
optima on some hard cases. The empirical fits nevertheless achieve losses below
the loss at truth, proving that the realized empirical target can prefer a
non-truth parameter vector even if a still-better basin remains undiscovered.
Hierarchical, cached exhaustive, or other affordable reference searches remain
necessary to quantify the optimization gap and choose a practical search.

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

## Currently needed work: recovery analyses

Only code required to execute and interpret the following analyses is on the
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
part of R2 and must be fixed before inspecting the held-out fits. n=20,
multi-condition, and motor-noise extensions are deferred until this focused
panel works.

### R2. Establish optimization quality separately by objective — likelihood selected

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

For likelihood, the development comparison selected serial 32-start SciPy
L-BFGS-B with deterministic log-space Latin-hypercube starts. Sixteen starts had
material misses; 64 changed only sub-threshold gaps at nearly twice the CPU time.
The initially promising hierarchical-plus-polish search failed on an expanded
panel because its zoom was sensitive to lattice alignment. This settles the
likelihood search for the focused held-out confirmation, not the other three
objectives. Detailed traces and thresholds are recorded in the external recovery
artifact.

### R3. Complete WNM closed-loop recovery — likelihood complete

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
finite-sample recovery result under an independently checked search; it does not
replace the pending BWCRPS and curve-objective analyses.

On the held-out tuples, 70.0% recovered all parameters within a factor of 1.5,
with median per-dataset joint log-RMSE 0.140. Recovery rose from 25% at 180 trials
to 90% at 450 and 95% at 900, confirming that the smallest datasets are the
limiting condition for this strict criterion.

### R4. Run paired recovery from the actual DM observer — likelihood complete

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

The one-time 60-dataset held-out comparison confirmed the main result. WNM beat
surface on common-grid NLL in 70.0% of pairs; global joint log-RMSE was 0.391
versus 0.587, median per-dataset joint log-RMSE was 0.140 versus 0.315, and
factor-of-1.5 recovery was 70.0% versus 43.3%. Curve metrics remained mixed,
especially for nearly flat targets, rather than showing uniform dominance.

### R5. Run paired representative real-data fits

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

### R6. Validate the direct prediction path needed by the analyses

The recovery and real-data comparisons must use WNM predictions directly rather
than route them through a sampled surface. Complete and test the direct WNM branch
of the prediction command for bias, asymmetry, circular SD, and distributional
outputs, including observer-sample identity and motor noise. Preserve the separate
averaged-surface/mu2 path.

This is prediction correctness needed for the scientific comparison. Presentation
features and browser integration remain deferred.

## Deferred stages

The following work should not expand while R1--R6 remain unresolved, except for a
small fix required to run a recovery analysis safely.

### Presentation and exploratory interfaces

- Unified subject-plot WNM routing.
- PDF-slice plotting.
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
- Global x64 adoption. Run a targeted x64 diagnostic only if a key likelihood or
  BWCRPS optimization comparison shows that float32 precision materially changes
  the selected solution; otherwise defer it until the surface backend is retired.

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
