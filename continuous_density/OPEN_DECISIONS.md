# Open decisions — K12 transition

Unresolved choices and evidence thresholds, recorded here instead of burying them
in implementation notes. Items 1 and 3 affect the active recovery comparisons;
item 2 is conditional on those results. Items 4--6 are deferred and do not block
the recovery work.

Ordered by when the plan needs them settled.

---

## 1. Objective-specific WNM search and start budgets (active in recovery R2)

**Status: the focused single-condition likelihood search is reopened; the other
retained objectives are also still open.** The earlier held-out run used
serial SciPy L-BFGS-B with 32 deterministic log-space Latin-hypercube starts,
seed 0, artifact bounds, 500 iterations, `ftol=1e-9`, `gtol=1e-6`, and JAX
float32. The generic continuous optimizer's default of 8 starts remains a
placeholder and has not become a universal production default.

On the likelihood development budget panel, 16 starts had two material misses,
32 reached the 0.001-NLL gate on every dataset, and 64 only reduced
sub-threshold gaps while nearly doubling median CPU time. That gate result held
only against a union best without a 32-start BADS arm; on the corrected panel
32-start SciPy reaches the gate on none of the 24 datasets (see below).
However, the competing arms were not faithful production comparisons. The WNM
hierarchy used a new 9/13-point log lattice with a coupled nearest-cell zoom,
rather than the surface pipeline's 40-point shared grid, 20-by-20 feature grid,
fixed feature-step schedule, 0.5 spatial zoom, and 1-degree stopping rule. The
JAX-BADS pilot used only four starts on the representative panel and a budget
of 500 evaluations / 100 iterations; BBZ uses eight curated starts and, for
three parameters, defaults to 1,500 evaluations / 300 iterations. PyBADS was
not tested at all. Therefore the earlier comparison does not rule out any of
those three approaches.

Use the same 32 deterministic dispersed starts for PyBADS, JAX-BADS, and
L-BFGS-B in the primary comparison, so optimizer family is not confounded with
start coverage. Retain the eight-start JAX-BADS run separately as a
BBZ-production-count diagnostic. The hierarchy has no start-count setting, so
it alone is compared at a fixed configuration while every rival was equalized;
its failure mode is lattice alignment, and the analogue of a multistart budget
would be jittered lattice offsets. No such arm has been built.

The corrected five-arm panel has since been run and rescored on one device, in
`wnm_optimizer_corrected_panel_v1/`, under the results directory named below.
Thirty-two-start JAX-BADS nominally leads on optimization gap (median 0.0000,
83.3% within 0.001) and parameter recovery (median joint log-RMSE 0.096,
factor-of-1.5 rate 66.7%) against 0.0115 / 0.0% and 0.139 / 62.5% for 32-start
SciPy, but neither lead is real. The NLL ordering is an artifact of the rescore
device: rescored on CPU instead of the GPU the panel used, JAX-BADS is lower on
0 of 24 rather than 23 of 24, a sign flip on 23 datasets, because each arm
converged to its own device's float32 objective. The recovery gap is a
marginal-median artifact: paired, it is -0.0001, with one dataset of 24
accounting for it. PyBADS is likewise not worse: against L-BFGS-B, its own
device-mate, it is tied (median difference 0.00000 on a CPU rescore), and its
apparent deficit to JAX-BADS is the same device artifact, 22 sign flips of 24.
It spends fewer evaluations per start than JAX-BADS, 117.3 against 378.5, but
that is a difference of stopping rules rather than of budget: JAX-BADS's
1500-evaluation cap never binds, the largest observed being 508.3 per start, so
both implementations self-terminate on their own convergence criteria. Its
30-fold runtime is 26.1 ms per evaluation of Python-level call overhead, not
extra search. All three 32-start arms find the same optimum here.
They differ only in cost, 6.28 against 3.20 seconds, and the production surface
search geometry, driving the same WNM likelihood rather than the surface
network, is worst on both. PyBADS is excluded from further comparisons (user
decision, 2026-09-08): tied with L-BFGS-B on its own device, it adds no
discrimination, and a 120-dataset arm would cost about 3.3 hours against
roughly 13 minutes for JAX-BADS. Its purpose was to check that the JAX port
matches the reference implementation, which it does; the existing 24-dataset
results stay as the record, and the exclusion is on cost, not on quality.

The decision between the three 32-start arms is not settled by these
gaps: rescoring one winner on another device moves the loss by up to 0.0177,
and most of their gaps fall below that, so the 0.001 gate is finer than the
scoring reproducibility. Curve agreement does not break the tie either: by
dissimilarity band no arm exceeds 0.4 median CCC in any band, and the
hierarchy's apparently better whole-curve CCC is a marginal-median artifact
that disappears when datasets are paired. Scored against a noise-free curve at
the generating parameters instead of the noisy empirical one, the likelihood
arms reach 0.93 median CCC and the hierarchy 0.08, its failure concentrated on
the six `narrow_1` datasets where the production feature lattice cannot
represent a true `sd_feat1` of 5.0 and returns 4.50. Selecting among the
32-start arms needs a criterion that is resolvable in float32, not a further
tightening of the gate.

The complete 120-dataset development L-BFGS-B run finished with at least 24
converged starts per dataset. Detailed protocol and results are stored under
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`.
The already-inspected held-out data showed WNM beating the surface pipeline on
common-grid NLL in 70.0% of pairs and improving global joint log-RMSE from 0.587
to 0.391. Those numbers remain descriptive, but they are no longer evidence of a
prospectively frozen optimizer choice. Do not inspect held-out data while
reselecting the search, and do not describe a later reuse of those tuples as a
first confirmation.

`continuous_optimizer.minimize_continuous` defaults to a placeholder `n_starts`.
The budget materially decides the answer:

- three-condition synthetic fixture, 6 starts, all converged: loss spread 1.03
  (start losses 0.96 to 2.00)
- real subject (`data_color_comb_color2`, 2 conditions), 4 starts, all
  converged: spread 0.81 against a winning loss of 0.435
- same subject, 8 starts: spread 0.99, two coordinates railed at the low bound

So on real data three of four starts land in worse basins. The completed density
panels add two results that must not be conflated: moving from 16 to 64 starts
barely changes empirical-target parameter recovery, while the noise-free
hard-case sweep shows that continuous search still misses a known optimum. The
former says the observed density-recovery failure is not explained by starts;
the latter says the optimizer is not yet adequate.

On the corrected 40 hardest `range_n10000` density cases, refitted against a
noise-free target, 16, 64, and 256 starts missed the known optimum in 30, 18,
and 7 cases respectively. Median time per fit rose from 2.2 to 7.1 to 27.2
seconds. Even the largest tested budget is therefore not a reference solution,
and the generic 8-start default is below every measured arm.

There should be no universal budget chosen from density alone. On development
groups, compare each retained objective through a common WNM scorer against an
appropriate hierarchical, cached, dense, or profiled reference, then freeze the
simplest search that meets predeclared optimization-gap, reliability, and runtime
criteria. Likelihood and bias-weighted CRPS are the priority parameter-recovery
cases; curve-search recovery is still tested but is expected to be weaker. The
surface NN keeps its deployed search and is compared as a legacy pipeline; it does
not need matching WNM search implementations.

## 2. float64 for the continuous search (conditional during recovery R2)

Recorded in `TODO.md` item 3. The repo runs JAX at float32, which floors
convergence near 1e-8 relative; recovery of a known optimum lands at 1.5e-7.
Whether that costs anything scientifically is only answerable by the recovery
panel, and adopting x64 would change the surface backend's arithmetic too, so it
cannot be switched on unilaterally.

**Decision needed only if a key likelihood or BWCRPS comparison shows a
precision-sensitive solution**: whether to run and potentially adopt a targeted
x64 configuration, and if adoption precedes NN retirement, whether to re-verify
the surface parity fixtures under x64. Otherwise this stays deferred.

## 3. Fitting bounds versus the corpus hull (needed before any production WNM fit)

The mixture is trained over `sd_feat` down to 2.5, and the WNM search bounds
already use that: `continuous_fit` derives them from the artifact's declared
domain, so a `--search continuous` run does search down to 2.5 today. What is
undecided is the *surface* backend's box, which the plan still describes as the
production fitting box, and whether the two should be stated as one policy.

The production box and the corpus hull are not nested:

```
sd_feat1   box [5, 200]   hull [2.5001, 198.1130]   box exceeds by 1.887
sd_feat2   box [5, 200]   hull [2.5351, 199.9541]   box exceeds by 0.046
sd_spat    box [5, 200]   hull [5.0018, 200.0000]   box below by 0.0018
```

The declared domain accepts this (user decision, 2026-09-06: "that tiny
extrapolation is fine on both ends"), and the packager caps outward overhang at
5 degrees. What is *not* decided is whether the production fitting bounds should
now open `sd_feat` down to 2.5 for real fits. Doing so is what makes the
narrow-density coverage usable; not doing so leaves the surrogate's main
advantage unreachable in production.

**Decision needed**: whether the surface backend's [5, 200] and the mixture's
[2.5, 200] should be described as one production policy or two, and whether any
comparison between the families must hold the box fixed. As it stands a
head-to-head run gives the mixture a wider search than the network, which is a
real advantage of the surrogate but not a like-for-like comparison of searches.
The plan says wider bounds are separate work, but it was written before the
corpus hull was measured.

## 4. mu2 and the (mu1, mu2) joint (deferred, unscheduled)

Recorded in full in `TRANSITION_PLAN.md`, "Deferred: mu1 x mu2 covariation".
Not blocking: averaged surfaces stay, so mu2 keeps working. The decision is
*when*, and it has a cheap-now/expensive-later shape: the per-trial data is
discarded at accumulation time, so any corpus re-run made before the change
produces another mu2-less corpus. If a re-run happens for another reason,
retaining column 22 costs almost nothing then.

## 5. Plot outputs for a mixture fit (deferred until the recovery gate)

`create_unified_subject_plots` and `plot_pdf_slices` recompute curves, moments
and SDs through the surface optimizer. They now refuse a mixture artifact rather
than crashing; step 4c is intended to route them through the shared prediction
layer after recovery.

**Status**: the pooled-SD estimator is shared and the surface path is verified
bit-identical, including at the clamp. The mixture reaches the same estimator
analytically; the two agree to about 9e-05 degrees on the golden fixture, the
residual being the 2-degree grid's approximation error rather than the analytic
value's. Recorded here in case a reviewer expects one implementation.

**Still open, but intentionally deferred**: neither plotting entry point can yet
*run* on a mixture fit. Both refuse a WNM artifact rather than crashing. Direct
prediction needed to score recovery remains in scope; presentation wiring does
not.

The curve computations a mixture plot needs now exist:
`shared.prediction.mixture_plot_curves` produces all four curve families (bias,
smoothed asymmetry, circular SD, pooled SD), and `pooled_bias_weighted_crps`
takes probabilities from either family.

**One thing the wiring must supply that the helper cannot check.** The asymmetry
curve is smoothed here exactly as the density objective smooths its target, so
`emp_density_weights_sd` and `density_smoothing_sigma` have to be forwarded from
the *fit's own settings*. A fit run at one sigma and plotted at the helper's
default shows a curve the fit never optimised, and nothing in the plot says so.
The caller holds the fit record; the helper has no way to detect the mismatch,
which makes this a call-site obligation rather than something a test of the
helper can cover.

What remains is wiring, and it is
awkward for a reason worth recording rather than discovering again:
`prepare_all_subjects_data` is a 400-line function of which only ~15 lines are
surface-specific. Those 15 sit in the middle, so a family branch around them
means either re-indenting the block or extracting it, and the rest of the
function -- the mapping back to (condition, optimizer), the empirical curves, the
report-order handling -- is shared and must not move.

**Approach when it is picked up**: extract the surface-specific span into a
helper verbatim, golden-record its outputs on a real subject *first*, add the
mixture sibling, then require the surface outputs unchanged. Not an in-place
re-indent. An earlier routing commit in this stage changed a clamp while
re-writing surrounding code and moved a plotted number by 37 degrees with every
golden test still green; the same shape of mistake is available here and the
function is much larger.

## 6. The pooled-SD clamp is inert at its upper bound (found 2026-09-06)

`compute_predicted_sd_curves_batch_pooled` clamps the resultant to
`[1e-10, 1 - 1e-10]`, but `1 - 1e-10` is exactly `1.0` in float32 (eps is
1.19e-7), so the upper bound has never bound. A distribution concentrated in a
single reporting cell therefore gives `r == 1`, `log(1) == 0`, and a pooled SD of
`-0.0` degrees.

This is longstanding deployed behaviour, not something the routing introduced --
verified identical against the pre-routing implementation, and now pinned by
`test_the_upper_clamp_is_inert_in_float32_and_that_is_the_deployed_behaviour`.

**Decision needed**: leave it, or make the upper clamp effective. An SD of zero
for a distribution with one cell of support is arguably correct; arguably a floor
of one cell width is wanted, since the model cannot resolve below that. Either
way it changes a plotted number, so it belongs in a decision rather than in a
routing commit. Low urgency: it only fires on degenerate surfaces that real fits
do not produce.

---

## Not decisions — resolved, kept for the record

- **Averaged surfaces**: retained (2026-09-06), so mu2 and `surface_browser`
  keep their existing path.
- **Surface NN**: retired at step 7, after every downstream artifact has been
  regenerated on K12.
- **Declared domain**: `sd_feat [2.5, 200]`, `sd_spat [5, 200]`,
  `feat_diff [0.5, 180]`.
- **First actual-DM recovery data design**: n=100, single condition, zero motor
  noise, 12 fixed truth tuples, five response seeds, and nested 180/450/900 trials
  uniformly allocated over 2:2:180 dissimilarity. Broader panels wait for this one.
- **`n_samples` is the parameter**: exactly two production artifacts at a time,
  one per observer model, selected through `shared.surrogate`; other checkpoints
  remain reachable by name.
