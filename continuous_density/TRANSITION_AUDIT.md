# K12 transition audit and revised work plan — 2026-09-07

## Purpose and evidence

This memo records a read-only audit of `TRANSITION_PLAN.md`, the current inner
`demixing_model` tree at `142a4af`, and the transition commits made on
2026-09-06--07. It also records the corrections agreed in the subsequent
discussion and replaces the audit's initial ordering of the remaining work. It
does not replace the detailed transition plan or claim production clearance.

The audit used the repository guidance in `AGENTS.md`, `CLAUDE.md`, and
`bias_model_comparison/analysis/CODING_GUIDE.md`. Tests and GPU jobs were not
rerun for this audit; statements about test totals and discovered defects come
from the commit record. Claims about current support were checked against the
current source.

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
commit (`bd8a0f1`) through `142a4af`, 26 commits changed 49 files, adding 9,517
and deleting 1,159 lines; 4,171 inserted lines are under `tests/`. Seven explicit
audit-fix commits enumerate at least 43 defects, with three more fixes recorded
in the latest recovery commit. Several defects changed fitted or reported
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
- Initial n=20 WNM closed-loop density recovery and random-range panels have run.

### Partial or not yet completed

- WNM cannot yet use the hierarchical or cached exhaustive search paths. The
  public fitter currently couples WNM to `--search continuous` and the surface NN
  to the lattice searches.
- The direct prediction command identifies WNM artifacts but still constructs
  the surface-only optimizer, so it does not yet provide a working WNM prediction
  path.
- Only a WNM-generated, n=20, density-objective recovery campaign has been run.
  The actual DM observer, n=100, likelihood, bias-weighted CRPS, smoothed
  expectation, paired surface-NN baseline, and full real-data comparisons remain.
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
fix, and inverted explicit-zero motor semantics. The audits caught these, but the
pattern shows that features were being layered before a small set of canonical
entry points was frozen and exercised.

Batch the recovery-enabling fixes, freeze their code and settings, and then run
the paired analyses. Avoid alternating large new layers with audit repairs while
the benchmark itself is changing.

### 5. Status documentation is already drifting

`pretrained/README.md` first says several consumers still resolve checkpoints
independently and retain known gaps, then later says the same identity defects are
fixed. The transition plan describes both training corpora as living under 4.1p,
whereas the packaged n=100 artifact records 4.1o. Recent commits also corrected
three earlier implementation claims. Status prose should be updated only at
frozen milestones and should be derived from executing call sites where possible.

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
improve recovery, but multistart alone is not a global-search oracle. Hierarchical,
cached exhaustive, or other affordable reference searches are still needed before
attributing the failure to the objective or target.

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

### R1. Freeze the paired recovery design

- Use both observer sample counts, n=20 and n=100.
- Use matched truth vectors, simulated datasets, response seeds, and trial counts
  across every search and surrogate arm.
- Include single- and multi-condition designs, asymmetric feature noise in both
  orders, low and high spatial d-prime, narrow and broad responses, and motor-off
  and motor-on cases where the objective identifies motor noise.
- Separate development cases used to choose search settings from held-out cases
  used to report performance.
- Predeclare the shared evaluation metrics and practical optimization-gap and
  runtime criteria before inspecting held-out results.

### R2. Establish optimization quality separately by objective

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

### R3. Complete WNM closed-loop recovery

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

### R4. Run paired recovery from the actual DM observer

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
