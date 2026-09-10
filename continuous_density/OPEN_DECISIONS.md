# Open decisions — K12 transition

Current status updated: 2026-09-10.

Unresolved choices and evidence thresholds, recorded here instead of burying
them in implementation notes. The focused recovery decisions in items 1--3 are
resolved and retained for provenance; items 4--6 remain deferred or low
priority.

Generated evidence referenced through `$DEMIXING_ARTIFACT_ROOT` is external and
is not shipped with or expected in a normal checkout.

Ordered by when the plan needs them settled.

---

## 1. Objective-specific WNM search and start budgets (resolved for focused recovery)

**Status: resolved by executive decision for focused recovery.** All four
retained objectives use one search policy: the batched JAX L-BFGS-B port with
64 deterministic dispersed starts, float32 arrays, and `highest` matmul
precision. Objective-specific caches, SciPy unions, and conditional hierarchy
rules are not part of the selected path. This deliberately favors one
maintainable optimizer over the last increment of objective-specific search
coverage.

The tradeoff is explicit. Sixty-four starts removed the three material
likelihood misses of the 32-start port and gave complete 32/64-union coverage;
BWCRPS recovery and loss were effectively unchanged; matched-KDE density met
the full development gate and made its non-reusable cache unnecessary. For
matched smoothed expectation, however, the port alone reached the 0.001 gate on
91/120 datasets with worst gap 44.40, versus 116/120 and 0.249 for the matched
cache/composite. The executive decision accepts those misses for cross-
objective consistency. These are search losses against empirical objectives,
not corresponding deficits in paired truth-curve CCC or parameter recovery.

This policy is now routed through the public `--search continuous` fitter for
all four retained objectives. The public run records all 64 endpoints and their
statuses, executes two sequential batches of 32, and fingerprints the pinned
port version, batch size, dtype, and matmul policy. The first four-condition
real-data fit completed successfully; its matched-density start spread and the
paired result below preserve the remaining multi-condition sensitivity rather
than treating 64 starts as proof of a global optimum.

**Historical selection path.** On the likelihood development budget panel, 16 starts had two material misses,
32 reached the 0.001-NLL gate on every dataset, and 64 only reduced
sub-threshold gaps while nearly doubling median CPU time. That gate result held
only against a union best without a 32-start BADS arm,
and is device-conditional: on the 24-dataset panel rescored on GPU, 32-start
SciPy reaches the gate on none of the 24, while on the 120-dataset panel
rescored on CPU it reaches it on 98.3% (see below).
However, the competing arms were not faithful production comparisons. The WNM
hierarchy used a new 9/13-point log lattice with a coupled nearest-cell zoom,
rather than the surface pipeline's 40-point shared grid, 20-by-20 feature grid,
fixed feature-step schedule, 0.5 spatial zoom, and 1-degree stopping rule. The
JAX-BADS pilot used only four starts on the representative panel and a budget
of 500 evaluations / 100 iterations; BBZ uses eight curated starts and, for
three parameters, defaults to 1,500 evaluations / 300 iterations. PyBADS was
not tested at all. Therefore the earlier comparison did not rule out any of
those three approaches.

The corrected comparison used the same 32 deterministic dispersed starts for
PyBADS, JAX-BADS, and L-BFGS-B, so optimizer family was not confounded with
start coverage. It retained the eight-start JAX-BADS run separately as a
BBZ-production-count diagnostic. The hierarchy has no start-count setting, so
it alone was compared at a fixed configuration while every rival was equalized;
its failure mode is lattice alignment, and the analogue of a multistart budget
would be jittered lattice offsets. No such arm has been built.

The comparison has since been run on all 120 development datasets for the three
surviving arms, assembled twice so the ranking could be computed under each
scoring device (`wnm_development_panel_gpu_v1/` and
`wnm_development_panel_cpu_v1/`). The result is that the NLL ranking between
32-start JAX-BADS and 32-start L-BFGS-B is entirely an artifact of the scoring
device: paired over 120 datasets JAX-BADS is lower on 102 of 120 when scored on
GPU and on 0 of 120 when the identical winners are scored on CPU, because each
arm converged to its own device's float32 objective. Parameter recovery, which
does not depend on the scoring device, is a tie: paired difference -0.00008
with JAX-BADS better on 63 of 120, against marginal medians of 0.1423 and
0.1495 that suggest a difference that is not there. What does separate them is
worst-case reliability and cost, both favouring L-BFGS-B: its largest gap over
120 datasets is 0.067 against JAX-BADS's 0.659, at half the time and a ninth of
the evaluations. **Thirty-two-start serial L-BFGS-B is therefore the choice for
the likelihood**, with the earlier JAX-BADS lead recorded as a scoring
artifact. PyBADS was excluded from this panel on cost (user decision,
2026-09-08): it is tied with L-BFGS-B on its own device, and its 30-fold
runtime is Python-level call overhead at 26.1 ms per evaluation, not extra
search. The production surface search geometry, driving the same WNM likelihood
rather than the surface network, is worst under both devices, which is the one
device-robust ranking. Its catastrophic mean-bias truth-curve degradation is
confined to 13 of 15 `narrow_1` datasets and one `narrow_3`, where a 1.0-degree
feature lattice anchored at 2.5 cannot represent a true `sd_feat1` of 5.0 and
returns 3.5 or 4.5. It also has smaller optimization gaps outside those cases,
so this statement does not claim that all hierarchy error is confined to the
narrow regime. Quantization error on a scale parameter is relative, so a fixed
absolute step cannot serve a range spanning 5 to 160 degrees.

The implementation choice was superseded on 2026-09-10 after benchmarking the
faithful batched JAX L-BFGS-B port. Default GPU float32 matmuls used TF32 and
changed the likelihood surface enough to alter basins. With
`JAX_DEFAULT_MATMUL_PRECISION=highest`, the 32-start port was within 0.001 of
the selected SciPy/port union on all 120 development datasets, with maximum gap
0.000977 and 9.65-fold median paired speedup. This configuration is now frozen;
held-out evaluation is confirmatory. Detailed traces and the cross-objective
precision comparison live under
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`.

Bias-weighted CRPS was selected the same way on 2026-09-08 and initially chose
32-start serial L-BFGS-B, recorded in `WNM_BWCRPS_OPTIMIZER_FINDINGS.md`. Its
median ordering against JAX-BADS is
again device-decided, but at differences of about 7e-05 that sit below the
device shift, while JAX-BADS's worst case of 0.062 and the hierarchy's 0.0034
are device-robust against L-BFGS-B's 0.0005. Recovery is tied. The more
important BWCRPS result is that the objective prefers the fitted parameters
over the truth on essentially every dataset, by about 153 times the spread
between arms, so for this objective the search choice is close to irrelevant
beside the objective's own displacement from truth. The faithful batched port
subsequently met the 0.001 union gate on all 120 development and all 60 held-out
datasets with `highest` matmul precision. On held-out data its median time was
1.037 seconds versus 5.490 for SciPy, and it also had the lower common-rescored
loss by more than 0.001 on the three cases where the SciPy arm missed the gate.
The port therefore supersedes serial SciPy for BWCRPS.

The frozen BWCRPS choice was confirmed once on the 60 held-out datasets. WNM
beat the surface-NN parameter fit after both were rescored through the common
WNM evaluator on all 60 datasets; median common-score difference was -0.00794.
Median joint log-RMSE was 0.269 versus 0.345 for the surface fit, and factor-
1.5 recovery was 50.0% versus 43.3%. Curve metrics were mixed, so this is
supporting evidence for the primary distributional and parameter criteria, not
a claim of uniform curve superiority. Full results are in
`WNM_BWCRPS_OPTIMIZER_FINDINGS.md` and the external recovery artifact it names.

For the legacy density operator, the selected cache-plus-polish search reached
the common-score gate on the complete development panel. The target-alignment
panel showed that matched KDE
does not shift typical parameter recovery relative to the current operator, but
does reduce large-error tails. On held-out data its global joint log-RMSE was
1.305, versus 1.306 for the existing surface pipeline; the paired median
difference was inconclusive. Matched KDE was selected for semantic consistency,
and is now the validation-selected density operator. The proper development
comparison rebuilt the 80-by-64 lattice for each dataset because the fitted
curve depends on its empirical bandwidth. The 64-start `highest`-matmul port
covered all 120 development datasets at the 0.001 gate, with maximum gap
4.17e-7, versus 0.000960 for the lattice. Median time was 0.282 versus 18.546
seconds. The port was materially better on three datasets and the lattice was
never materially better, so the 64-start port is selected and the matched-KDE
cache is abandoned. Detailed results remain with the external recovery artifact.

For the legacy smoothed-expectation operator, the cache/eight-polish/conditional-hierarchy rule was
within 0.001 of the union best on every development and held-out dataset. WNM
improved held-out mean-bias prediction across most non-near-zero dissimilarity
bands, but parameter recovery was worse than for the surface NN and spread
recovery was weak, as expected for a mean-only objective. JAX-BADS was included
in the development panel and added no production win beyond the selected
combination. This recovery comparison uses the current mismatched operator: the
empirical curve is feature-smoothed, while the fitted model curve is pointwise.
The completed development alignment diagnostic improves WNM median log-RMSE
from 0.730 to 0.637 and removes the paired surface advantage, but all-parameter
factor-1.5 recovery remains only 23.3%. The frozen held-out matched-WNM run
without truth starts is complete. Its median/global log-RMSE is 1.130/1.239,
versus 0.981/1.248 for current WNM under the same 32-start SciPy search and
0.970/1.149 for the existing surface NN. Neither paired difference is
conclusive; the development parameter gain therefore does not generalize, even
though matched truth is closer to the empirical curve on 73.3% of held-out
datasets. Use the matched operator for WNM target consistency, but retain
smoothed expectation as a weak parameter-recovery objective. A matched surface
refit is needed only for strict model-family attribution.

The complete 120-dataset development L-BFGS-B run finished with at least 24
converged starts per dataset. Detailed protocol and results are stored under
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`.
The already-inspected held-out data showed WNM beating the surface pipeline on
common-grid NLL in 70.0% of pairs and improving global joint log-RMSE from 0.587
to 0.391. Those numbers remain descriptive, but they are no longer evidence of a
prospectively frozen optimizer choice. Do not inspect held-out data while
reselecting the search, and do not describe a later reuse of those tuples as a
first confirmation. The matched operator remains current for semantic
consistency. The proper development comparison rebuilt the complex-moment
cache and included the port, SciPy, hierarchy, and JAX-BADS arms. The cache
remains necessary. Substituting the port for SciPy in the adapted conditional
composite preserves its 116/120 coverage and 0.249 worst gap at 3.62-fold lower
median runtime, but the four misses mean the selection rule remains open. A
complete 64-start port run did not fix any composite miss; all four evade the
cache-versus-continuous disagreement trigger.

`continuous_optimizer.minimize_continuous` now defaults to the selected 64
starts. The evidence that made the budget scientifically material remains:

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
the latter says continuous multistart alone is not an adequate density reference.
This motivated the cached search selected above.

On the corrected 40 hardest `range_n10000` density cases, refitted against a
noise-free target, 16, 64, and 256 starts missed the known optimum in 30, 18,
and 7 cases respectively. Median time per fit rose from 2.2 to 7.1 to 27.2
seconds. Even the largest tested budget is therefore not a reference solution,
and the generic 8-start default is below every measured arm.

The completed comparisons confirm that the common 64-start budget does not make
every landscape equally easy. That selected consistency policy is frozen for
the focused panel. The surface NN keeps its deployed search as the legacy
comparator; it does not need matching WNM search implementations.

## 2. Numerical precision for the continuous search (resolved)

Originally recorded in `TODO.md` item 3. The repo runs JAX at float32, which
floors convergence near 1e-8 relative; recovery of a known optimum lands at
1.5e-7.
Whether that costs anything scientifically is only answerable by the recovery
panel, and adopting x64 would change the surface backend's arithmetic too, so it
cannot be switched on unilaterally.

**Decision needed only if a key likelihood or BWCRPS comparison shows a
precision-sensitive solution**: whether to run and potentially adopt a targeted
x64 configuration, and if adoption precedes NN retirement, whether to re-verify
the surface parity fixtures under x64.

**That condition fired on 2026-09-08.** The likelihood comparison is
precision-sensitive in exactly the sense this item names: rescoring the same
120-dataset panel of stored winners on CPU rather than GPU reverses the
JAX-BADS versus L-BFGS-B order, from lower on 102 of 120 to lower on 0 of 120,
because in float32 the two devices are slightly different objective functions.

The recorded rationale above is also too optimistic about the scale. It
justifies deferral by a convergence floor near 1e-8 relative, but the effect
that actually decided an arm comparison is a device-dependent reduction
difference of up to 0.04 NLL, four orders of magnitude larger. The question is
not optimizer convergence; it is that two devices disagree about the objective
by more than the arms disagree with each other. A targeted matched x64
diagnostic is therefore due, reported separately and not as global adoption:
JAX's x64 flag is global, so enabling it would change the surface backend's
arithmetic as well, and that stays deferred until the surface backend is
retired. The diagnostic must also verify that float64 is genuinely in force
rather than assumed. The scoring path casts parameters and trials to float32
explicitly and loads float32 checkpoint weights, so setting the global flag
alone does not promote the computation, and this repository has already
recorded one ineffective float64 fix (`TRANSITION_AUDIT.md`, finding 4).  **The
diagnostic was run on 2026-09-08 and the question is now answered.** Rescoring
the 120-dataset likelihood panel at every combination of device and precision,
float64 reduces the cross-device disagreement from a median of 5.95e-03 to
5.08e-07, and under float64 both devices agree that 32-start L-BFGS-B attains
the lower likelihood on all 120 datasets. The GPU float32 result that put
JAX-BADS ahead on 102 of 120 reflected device arithmetic, not optimizer-family
quality. That scoring-only test confirmed L-BFGS-B but did not yet isolate the
responsible float32 operation; the later port diagnostic below did.

The faithful-port diagnostic on 2026-09-10 refined that explanation. The large
likelihood discrepancy came primarily from the surrogate's default GPU TF32
matrix multiplications, not from the L-BFGS-B implementation or ordinary sum
reduction order. Keeping float32 arrays but setting matmul precision to
`highest` restored the CPU likelihood basin and passed the 0.001 gate on all
120 development datasets. Global float64 adoption is therefore unnecessary;
the frozen GPU likelihood rule is full-float32 (`highest`) matmuls.

## 3. Fitting bounds versus the corpus hull (resolved)

**Decision, 2026-09-10:** retain family-specific validated domains as the
production policy. WNM uses feature-SD bounds `[2.5, 200]`; the deployed
surface NN retains `[5, 200]`; both use spatial-SD bounds `[5, 200]`. Comparisons
must name the boxes and report boundary hits. They do not clip WNM to the NN's
box, because access to the validated narrow-density range is a capability of
the replacement model rather than an optimizer confound.

The production box and the corpus hull are not nested:

```
sd_feat1   box [5, 200]   hull [2.5001, 198.1130]   box exceeds by 1.887
sd_feat2   box [5, 200]   hull [2.5351, 199.9541]   box exceeds by 0.046
sd_spat    box [5, 200]   hull [5.0018, 200.0000]   box below by 0.0018
```

The declared domain accepts the small hull overhang (user decision, 2026-09-06:
"that tiny extrapolation is fine on both ends"), and the packager caps outward
overhang at 5 degrees. The representative color-2 fit exercised the new policy:
WNM likelihood and BWCRPS selected a 2.5-degree feature boundary in one
condition. Clipping both families to 5 degrees would therefore change the
deployed WNM answer, not merely make a cosmetic comparison. The paired report
records each family's own boundary solutions and evaluates both parameter sets
through one common WNM scorer.

## 4. mu2 and the (mu1, mu2) joint (deferred, unscheduled)

Recorded in full in `TRANSITION_PLAN.md`, "Deferred: mu1 x mu2 covariation".
Not blocking: averaged surfaces stay, so mu2 keeps working. The decision is
*when*, and it has a cheap-now/expensive-later shape: the per-trial data is
discarded at accumulation time, so any corpus re-run made before the change
produces another mu2-less corpus. If a re-run happens for another reason,
retaining column 22 costs almost nothing then.

## 5. Direct prediction and plot outputs for a mixture fit (partly open)

`create_unified_subject_plots` and `plot_pdf_slices` recompute curves, moments
and SDs through the surface optimizer. They now refuse a mixture artifact rather
than crashing; step 4c is intended to route them through the shared prediction
layer after recovery.

**Status**: the pooled-SD estimator is shared and the surface path is verified
bit-identical, including at the clamp. The mixture reaches the same estimator
analytically; the two agree to about 9e-05 degrees on the golden fixture, the
residual being the 2-degree grid's approximation error rather than the analytic
value's. Recorded here in case a reviewer expects one implementation.

**Representative comparison path completed:**
`model_fit_to_data/export_wnm_fit_curves.py` reads a fingerprinted WNM fit and
exports direct analytic bias, matched density-asymmetry, and circular-SD curves
plus per-condition plots. It validates the checkpoint digest and never
reconstructs an NN surface. The color-2 representative run exported 1,440 curve
rows and four PNGs alongside the paired comparison.

**Still open:** the older general-purpose `create_unified_subject_plots` and
`plot_pdf_slices` entry points do not yet run on a mixture fit. That broader
presentation wiring remains deferred; it is no longer a blocker for inspecting
the representative production comparison.

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
- **Surface NN**: scheduled for retirement at step 7, after every downstream
  artifact has been regenerated on K12; it remains load-bearing today.
- **Declared domain**: `sd_feat [2.5, 200]`, `sd_spat [5, 200]`,
  `feat_diff [0.5, 180]`.
- **First actual-DM recovery data design**: n=100, single condition, zero motor
  noise, 12 fixed truth tuples, five response seeds, and nested 180/450/900 trials
  uniformly allocated over 2:2:180 dissimilarity. This focused panel is complete;
  broader panels remain deferred.
- **`n_samples` is the parameter**: exactly two production artifacts at a time,
  one per observer model, selected through `shared.surrogate`; other checkpoints
  remain reachable by name.
