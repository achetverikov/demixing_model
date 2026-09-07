# Open decisions — K12 transition

Unresolved choices and evidence thresholds, recorded here instead of burying them
in implementation notes. Items 1 and 3 affect the active recovery comparisons;
item 2 is conditional on those results. Items 4--6 are deferred and do not block
the recovery work.

Ordered by when the plan needs them settled.

---

## 1. Objective-specific WNM search and start budgets (active in recovery R2)

**Status: no production search or start count has been selected.** The generic
continuous optimizer's default of 8 starts remains a placeholder, not a validated
configuration.

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
- **`n_samples` is the parameter**: exactly two production artifacts at a time,
  one per observer model, selected through `shared.surrogate`; other checkpoints
  remain reachable by name.
