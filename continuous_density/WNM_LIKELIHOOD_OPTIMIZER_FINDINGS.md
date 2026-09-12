# WNM likelihood optimizer findings

Date: 2026-09-07; corrected panel, banded curve scoring and the full
120-dataset panel added 2026-09-08; 64-start common policy added 2026-09-10

The frozen implementation is now the faithful batched JAX L-BFGS-B port with
64 starts, float32 arrays, and `highest` matmul precision, following the final
cross-objective executive decision. On the 120 development datasets it removed
three material misses of the 32-start port; against their two-arm union, 64
starts covered 120/120 with maximum gap 0.000732. Median time increased from
0.306 to 0.466 seconds. Parameter recovery and whole/banded truth-curve CCC
were effectively unchanged. Against the wider tested union, the 64-start arm
had one 0.001221 gap to SciPy, which is accepted in exchange for a single
optimizer policy across objectives. The earlier optimizer-family and 32-start
comparisons below remain as selection provenance.

All run directories and CSV file names in this document are relative to
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`,
where the analysis scripts named here also live. A compact snapshot of the JAX
port manifests, summaries, and comparison tables is versioned in
[`validation_outputs/jax_lbfgsb_port_2026-09-10/`](validation_outputs/jax_lbfgsb_port_2026-09-10/README.md).
Full per-task records and other generated artifacts remain external.

## Comparator correction and final resolution

The first selection overclaimed the comparator coverage. The 9/13-point WNM
hierarchy was a simplified log-lattice search, not the production surface-NN
hierarchy. The representative JAX-BADS panel used four starts, a 500-evaluation
budget and 100 iterations, whereas BBZ production uses eight curated starts and
defaults to 1,500 evaluations and 300 iterations for a three-parameter problem.
PyBADS was not tested. Selection was therefore reopened for a development-only
comparison against faithful PyBADS, JAX-BADS, and surface-hierarchy
implementations. The primary BADS arms used the same 32 dispersed starts as
L-BFGS-B; an eight-start JAX-BADS run separately represented BBZ's production
start count. Both JAX arms ran on GPU. The corrected 120-dataset comparison below
settled the optimizer family on 32-start L-BFGS-B; the later faithful-port
comparison selected its batched JAX implementation with full float32 matmuls.

## Full 120-dataset development panel (2026-09-08)

The 24-dataset panel was extended to all 120 development datasets for the three
surviving arms, and assembled twice: once rescored on GPU and once on CPU
(`assemble_development_panel.py`, `wnm_development_panel_gpu_v1/` and
`wnm_development_panel_cpu_v1/`). PyBADS is excluded on cost.

| Arm | GPU: median gap / within 0.001 | CPU: median gap / within 0.001 | Max gap (CPU) | Median joint log-RMSE | Median s |
|---|---:|---:|---:|---:|---:|
| scipy-lbfgsb-32 | 0.0079 / 15.8% | **0.0000 / 98.3%** | **0.067** | 0.1495 | **3.09** |
| bbz-jax-bads-32 | **0.0000 / 85.0%** | 0.0019 / 27.5% | 0.659 | 0.1423 | 6.24 |
| surface-production-hierarchical | 0.0361 / 5.0% | 0.0305 / 6.7% | 7.448 | 0.1875 | 4.67 |

**The device reversal reproduces at five times the data.** Paired across 120
datasets, JAX-BADS has the lower NLL on 102 of 120 when scored on GPU and on
**0 of 120** when the identical winners are scored on CPU. Each arm wins on the
device it searched on; neither ordering is a property of the optimizers.

**Recovery is a tie.** Parameter recovery does not depend on the scoring device.
Paired, the difference between JAX-BADS and L-BFGS-B is -0.00008 median with
JAX-BADS better on 63 of 120 -- a coin flip -- against marginal medians of
0.1423 and 0.1495 that again suggest a difference that is not there. Thirteen of
120 datasets differ by more than 0.01, and they split both ways: JAX-BADS is far
better on `broad_3_seed2_n180` (0.415 against 1.207) and far worse on
`broad_3_seed4_n900` (0.631 against 0.192). Six of those thirteen are
`broad_3`; the other seven span narrow, ordinary-asymmetric, and
reversed-asymmetric cases. The two largest differences occur in the weakly
identified `broad_3` regime, but the smaller disagreements are not confined to
it.

**What does separate them is worst-case reliability, and it favours L-BFGS-B.**
Its largest gap over 120 datasets is 0.067, while JAX-BADS misses
`broad_3_seed2_n180` by 0.659 -- an order of magnitude worse, under either
device. That dataset is also one where JAX-BADS recovers the parameters much
better, so the miss is a genuinely different basin, not a failure to converge.
With recovery tied, cost favouring L-BFGS-B by a factor of two in time and nine
in evaluations, and its worst case ten times smaller, **32-start serial
L-BFGS-B is the defensible choice**, and the earlier JAX-BADS lead should be
treated as a scoring artifact.

**The surface hierarchy is worst under both devices**, which is the one ranking
that is device-robust: median gap 0.031-0.036, within-gate rate 5-7%, worst case
7.45.

### Correction to the 24-dataset curve result

On 120 datasets the hierarchy's median mean-bias CCC against the truth curve is
0.860, not the 0.080 measured on the 24-dataset subset, against 0.931 for
L-BFGS-B. The earlier figure was a composition artifact: `narrow_1` is a quarter
of the 24-dataset subset but an eighth of the full panel, and the marginal median
happened to fall inside the collapsed cluster.

The material hierarchy-specific collapse is real and now precisely located. A
CCC deficit greater than 0.2 relative to L-BFGS-B occurs in 13 of 15
`narrow_1` datasets and one `narrow_3`, and none of the other 105. In
`narrow_1`, median CCC against the truth curve is 0.006 for the hierarchy
against 0.943 for both likelihood arms. Flat truth curves can give low absolute
CCC in other regimes too, but without a comparable hierarchy-specific deficit.

The reason is that lattice quantization error is *relative*. The finest feature
step is 1.0 degree on nodes anchored at 2.5, so the attainable values are 3.5,
4.5, 5.5 and so on. That is 20% of a true `sd_feat1` of 5.0, which `narrow_1`
fits can only bracket at 3.5 or 4.5, but 10% of `narrow_3`'s 10.0 (fitted 8.5 to
10.5) and 0.8% of `broad_3`'s 120.0. A fixed absolute step on a scale parameter
that spans 5 to 160 degrees cannot be uniformly adequate, which is why the
failure appears only at the bottom of the range.

## Corrected multi-arm panel (2026-09-08)

All five arms were run on the same 24 development datasets -- `broad_3`,
`narrow_1`, `ordinary_1` and `reversed_1`, seeds 0 and 1, at 180, 450 and 900
trials. That is a fifth of the 120-dataset development split, and it omits four
of its eight cases entirely (`broad_1`, `narrow_3`, `ordinary_3`, `reversed_3`);
only 32-start SciPy has been run on all 120. Every multi-arm statement below
therefore rests on 24 datasets, which is why a result carried by a single one of
them is reported as such rather than as a rate. Every stored winner was rescored
together in one GPU process, because each arm had first
scored its own winner on whichever device it ran on. All five optimize the same
WNM point likelihood through the same K12 predictor and differ only in search;
`surface-production-hierarchical` is the production surface *search geometry*
driving that likelihood, not the surface network, and not the 9/13-point WNM log
lattice that this panel replaced. The surface NN appears only in the separate
legacy-pipeline comparison further below. Artifacts live in
`wnm_optimizer_corrected_panel_v1/`: `summary.csv`, `paired_metrics.csv`,
`method_summary.csv` and `arm_recovery_summary.csv`,
pinned by `assembly_manifest.json` and `arm_recovery_manifest.json`.

| Arm | Median NLL gap | Max gap | Within 0.001 | Median s | Median evaluations | Median joint log-RMSE | Within factor 1.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bbz-jax-bads-32 | 0.0000 | 0.118 | 83.3% | 6.28 | 12,112.5 | 0.096 | 66.7% |
| bbz-jax-bads-8 | 0.0062 | 0.619 | 16.7% | 4.87 | 3,235.5 | 0.139 | 66.7% |
| scipy-lbfgsb-32 | 0.0115 | 0.033 | 0.0% | 3.20 | 1,381 | 0.139 | 62.5% |
| bbz-pybads-32 | 0.0130 | 0.050 | 12.5% | 98.00 | 3,753 | 0.140 | 62.5% |
| surface-production-hierarchical | 0.1176 | 5.657 | 12.5% | 4.65 | 336,007 | 0.274 | 45.8% |

This supersedes the statement that 32 SciPy starts reached the 0.001 gate on
every development dataset. That was measured against a union best that did not
contain a 32-start BADS arm. Against the corrected union, 32-start SciPy reaches
the gate on none of the 24, and 32-start JAX-BADS nominally leads on both
optimization and recovery at about twice the time and nine times the
evaluations -- but neither lead survives the checks below.
PyBADS matches L-BFGS-B exactly, at roughly thirty times the runtime for
reasons that are per-call overhead rather than search effort (see below). The production
surface hierarchy is worst on both axes and spends about 243 times as many
evaluations as L-BFGS-B to get there.

**JAX-BADS and L-BFGS-B are not separated: each wins on its own device.** The
common rescore removed the defect of every arm scoring itself, but it still runs
on *a* device, and that device was the GPU, where JAX-BADS searched. Rescoring
the same stored winners on CPU reverses the result completely
(`check_device_sensitivity.py`, `device_sensitivity_cpu.csv`):

| Rescore device | 32-start JAX-BADS lower NLL | Median difference |
|---|---:|---:|
| GPU (the panel) | 23 of 24 | -0.01105 |
| CPU | 0 of 24 | +0.00168 |

The sign flips on 23 of 24 datasets. Per fit the two devices disagree by a
median of 0.0079 and up to 0.0401 NLL, which is larger than the arm gap itself.
Each optimizer converged to the minimum of its own device's float32 objective,
so scoring on the GPU credits the GPU-run arms and scoring on the CPU credits
the CPU-run ones; the panel's zero rescore adjustment for JAX-BADS and the
surface hierarchy is that coincidence, not a sign of a better optimum. The
ranking of the 32-start arms by NLL is therefore an artifact of the rescore
device and must not be reported as an optimizer result. Separating them needs
either float64 scoring or a decision rule coarser than 0.04 NLL.

Parameter recovery does not separate them either, and its marginal medians
mislead the same way the curve medians do: 0.096 against 0.139 median joint
log-RMSE is a **paired** difference of -0.0001, with JAX-BADS better on 14 of
24 and only **one** dataset differing by more than 0.01. That dataset,
`broad_3_seed1_n900`, is a genuine dissociation rather than noise: JAX-BADS
recovers it far better (0.052 against 0.318) while scoring 0.10 NLL *worse*,
above the device scale. Strict factor-of-1.5 recovery flips on one dataset in
JAX-BADS's favour, 15 datasets satisfying both arms and 8 neither.

The two arms also land on nearly the same parameters. Between them the median
absolute log ratio is 0.0010 for `sd_feat1`, 0.0043 for `sd_feat2` and 0.0036
for `sd_spat`, that is 0.1% to 0.4%, or 0.017 to 0.14 degrees. Nineteen of 24
datasets agree within 1% on all three parameters and 23 of 24 within 5%. That
disagreement is roughly twenty-five times smaller than either arm's own median
distance from the generating parameters (0.027 to 0.11 in log ratio), so the
arms are far closer to each other than either is to the truth.

The single exception is `broad_3_seed1_n900`, the same dataset that carries the
recovery difference: JAX-BADS returns `sd_spat` 13.75 against SciPy's 8.78 with
a truth of 15, and `sd_feat2` 163.66 against 141.54 with a truth of 160. That is
one genuinely different basin in the broad/weak regime, where the parameters are
weakly identified, not a systematic difference in search quality.

What does separate them is cost: 6.28 against 3.20 median seconds and 12,112.5
against 1,381 median evaluations, for a difference in the objective that does
not survive changing device.

**The 0.001 gate is finer than the scoring reproducibility.** Rescoring the same
winner on a different device moved the loss by up to 0.0177 (median 0.0000, p90
0.0099). Below that scale a gap is float32 reduction noise, not a worse optimum:
83.3% of the 32-start SciPy gaps, 75.0% of PyBADS's and 95.8% of 32-start
JAX-BADS's fall under it, against 33.3% for the surface hierarchy. So the arm
ordering is trustworthy at the hierarchy's scale and at JAX-BADS-8's 0.619 miss,
but the separation between the 32-start arms is not established by these float32
gaps. A gate below the device reproducibility cannot be the selection criterion.
The targeted float64 diagnostic below supplies the tighter scoring path this
called for, and it does settle the question.

### PyBADS is not worse than JAX-BADS, and is not budget-matched

PyBADS is the reference implementation that the JAX BADS port follows, so a real
difference in solution quality between them would be surprising. There is none.
Three separate things made PyBADS look weak, and all are artifacts of how the
panel was assembled.

1. **Device.** PyBADS searched on CPU and JAX-BADS on GPU, and the panel rescored
   on GPU. Rescored on CPU, JAX-BADS goes from lower on 22 of 24 to lower on 0 of
   24 against PyBADS, a sign flip on 22 datasets -- the same confound that
   inflated it over L-BFGS-B (`device_sensitivity_cpu_bads_pair.csv`).
2. **Against its own device-mate it is exactly tied.** PyBADS and L-BFGS-B both
   searched on CPU, so their comparison carries no device tilt: PyBADS is lower on
   11 of 24 on the GPU rescore and 10 of 24 on the CPU rescore, with a median
   difference of +0.00020 and +0.00000 respectively.
3. **Runtime is per-call overhead, not more search.** PyBADS spends 26.1 ms per
   likelihood evaluation against 0.52 ms for JAX-BADS and 2.3 ms for L-BFGS-B,
   because each evaluation is a Python-level call into JAX rather than a
   vectorized batch. Its 98-second median is that factor, not extra work: it uses
   *fewer* evaluations than JAX-BADS, 3,753 against 12,112.5.

The arms spend different numbers of evaluations -- 378.5 per start for JAX-BADS
against 117.3 for PyBADS -- but this is not a budget mismatch. JAX-BADS's
`budget` of 1500 never binds: the largest observed is 508.3 evaluations per
start, and no dataset in any BADS arm reaches even a third of the cap. Both
implementations stop on their own convergence criteria (`f_rtol` 1e-5,
`max_stall` 25, `mesh_min` 1e-8 for the JAX port; library defaults for PyBADS),
so raising or equalizing the cap would change nothing. The difference in
evaluations is a property of the stopping rules being compared, not a confound to
be removed. On its own rule PyBADS reaches the same optimum with a third of the
evaluations, and its worst-case gap is smaller than JAX-BADS's, 0.050 against
0.118.

The defensible reading is that all three 32-start arms find the same optimum on
this panel, and that no BADS-versus-L-BFGS-B conclusion should be drawn from it
until budgets and scoring device are matched.

**PyBADS is dropped from further comparisons** (user decision, 2026-09-08). It is
tied with L-BFGS-B on its own device, so extending it buys no discrimination,
and at 26.1 ms per evaluation a 120-dataset arm costs about 3.3 hours against
roughly 13 minutes for JAX-BADS. Its role was to test whether the JAX port
behaves like the reference implementation; on this panel it does, and that
question is answered. The existing 24-dataset PyBADS results stay in the panel as
the record of that check. This is an exclusion from future arms, not a retraction
of a measurement, and it does not license the reverse claim that PyBADS is worse
-- it is not.

### Why the hierarchy's curves looked better

`truth_curve_metrics.csv` and `truth_curve_summary.csv`
(`compare_arm_curves_to_truth.py`, pinned by `truth_curve_manifest.json`) score
every arm against a noise-free curve evaluated at the generating parameters, in
addition to the empirical curve. The result reverses the reading above.

| Arm | Median mean-bias CCC vs empirical | vs truth curve |
|---|---:|---:|
| scipy-lbfgsb-32 | 0.601 | **0.934** |
| bbz-jax-bads-32 | 0.592 | **0.935** |
| bbz-pybads-32 | 0.601 | **0.935** |
| surface-production-hierarchical | 0.753 | **0.080** |

The likelihood arms recover the truth curve at CCC 0.93 and only appear
mediocre because the empirical target is mostly noise at these trial counts:
its median range is 2.60 degrees against a truth-curve range of 1.03. Scoring a
fit against a curve whose amplitude is more than half sampling noise is a weak
test, and it flatters whichever arm happens to track that realization.

The collapse is concentrated, not diffuse. On 18 of 24 datasets the two arms
agree to within 0.05 CCC of the truth curve. All six `narrow_1` datasets, where
the generating parameters are `sd_feat1` 5, `sd_feat2` 10, `sd_spat` 15, carry
the failure: SciPy reaches 0.79-0.99 there and the hierarchy 0.06 or below, and
those same datasets carry the panel's largest NLL gaps, 0.12 to 5.64. One
dataset, `ordinary_1_seed1_n180`, moves the other way by +1.01.

The cause is grid quantization, and it is visible in the fitted values. The
production feature schedule is `[10.39, 10, 6, 4, 2, 1]` on a 20-point centered
grid anchored at the lower bound 2.5, so the finest attainable feature nodes are
2.5, 3.5, 4.5, 5.5 and so on. A true `sd_feat1` of 5.0 falls between two nodes,
and all six `narrow_1` fits return exactly 4.50. There is no polish step: the
loop terminates on the *spatial* spacing reaching 1 degree while the feature SDs
are still on the 1.0-step lattice. With `sd_feat1` pinned off-node, the weakly
identified second component is then stranded far away -- fitted `sd_feat2` of
28.5, 43.5, 100.5, 103.0, 164.5 and 167.5 against a truth of 10.

This is the same failure class as the 9/13-point WNM lattice: which basin the
search enters is decided by where the lattice falls, not by the objective. It
recurs here inside the production geometry, on the cases with the narrowest
feature SDs, which are exactly the cases the lattice resolves worst.

## Curve scoring by dissimilarity band (2026-09-08)

A single CCC per curve pools every feature difference, and the bias curve
changes sign and magnitude with dissimilarity, so a whole-curve concordance can
be carried entirely by the across-band trend. Each curve is now also scored
inside the project's four dissimilarity bands, read from
`continuous_density_4.1c/config/experiment.json` so all banded reporting cuts
the axis identically. CCC and MAE are reported together with the band's
empirical target range. Artifacts: `curve_band_metrics.csv`,
`curve_band_summary.csv` and `curve_band_manifest.json` in the corrected panel
and in both `wnm_scipy32_*` runs.

Median CCC for 32-start SciPy, whole curve against bands:

| Curve | All | [0,18) | [18,60) | [60,120) | [120,180] |
|---|---:|---:|---:|---:|---:|
| mean bias | 0.601 | -0.002 | 0.267 | 0.238 | 0.234 |
| bias SD | 0.471 | not scorable | 0.388 | 0.018 | 0.003 |
| asymmetry | 0.586 | 0.002 | 0.093 | 0.251 | 0.175 |

### Float64 diagnostic: the ranking is a precision artifact (2026-09-08)

`OPEN_DECISIONS.md` item 2 made a targeted x64 diagnostic conditional on a key
likelihood or BWCRPS comparison proving precision-sensitive. This one is, so the
diagnostic was run: `check_precision_sensitivity.py` rescores every stored winner
of the 120-dataset panel at each combination of device and precision, writing
`precision_{cpu,gpu}_{f32,f64}.csv`.

Setting JAX's global x64 flag alone would not have promoted anything -- the
production path casts parameters and trials to float32 and loads float32
checkpoint weights -- so the diagnostic rebuilds the predictor with float64
variables and feeds float64 inputs, and records the enabled flag, the realized
dtype, and the cross-device agreement rather than assuming any of them. This
repository already contains one ineffective float64 fix, which is why the
agreement is reported as evidence and not the flag.

Disagreement between the two devices, scoring identical parameters:

| Precision | Median | p90 | Max |
|---|---:|---:|---:|
| float32 | 5.95e-03 | 1.65e-02 | 6.20e-02 |
| float64 | 5.08e-07 | 1.89e-06 | 5.58e-06 |

Float64 removes it: four orders of magnitude smaller, and below anything that
could reorder the arms. The arm comparison then resolves, paired over 120
datasets, as JAX-BADS minus L-BFGS-B:

| Scoring | JAX-BADS lower | Median difference |
|---|---:|---:|
| CPU float32 | 0 of 120 | +0.001816 |
| GPU float32 | **102 of 120** | -0.007050 |
| CPU float64 | 0 of 120 | +0.001522 |
| GPU float64 | 0 of 120 | +0.001522 |

Under float64 the two devices agree to six decimal places and both say the same
thing: **32-start L-BFGS-B attains the lower likelihood on all 120 datasets.**
The GPU float32 result that put JAX-BADS ahead on 102 of 120 was an artifact of
reduction order, not a property of either optimizer. CPU float32 already agreed
with float64 on the ordering; it was the GPU float32 path that disagreed.

This strengthens rather than changes the selection. L-BFGS-B was chosen on
worst-case reliability and cost with recovery tied; it now also has the lower
objective value everywhere, once the objective is evaluated precisely enough for
the question to have an answer.

The scope of the conclusion is the *scoring*, not the search: every arm still
searched in float32, so this does not show what a float64 *search* would find.
Global x64 adoption remains deferred, since JAX's flag is global and would change
the surface backend's arithmetic as well.

### Banded against the truth curve

Every banded number above scores the fit against the *empirical* curve. Banding
makes that target mechanically harder: restricting the dissimilarity range cuts
the signal variance while leaving the sampling noise, so a low banded CCC is
expected even from a correct fit. Each band is therefore now scored against both
targets, the empirical curve and a noise-free curve evaluated at the generating
parameters (`curve_band_metrics.csv`, column `target`). For 32-start L-BFGS-B on
the 120-dataset panel:

| Curve | Target | All | [0,18) | [18,60) | [60,120) | [120,180] |
|---|---|---:|---:|---:|---:|---:|
| mean bias | empirical | 0.748 | 0.000 | 0.302 | 0.417 | 0.181 |
| mean bias | **truth** | 0.931 | **0.746** | 0.587 | 0.615 | **0.840** |
| bias SD | empirical | 0.509 | n/a | 0.179 | 0.026 | 0.001 |
| bias SD | **truth** | 0.912 | n/a | **0.765** | 0.313 | 0.031 |
| asymmetry | empirical | 0.614 | 0.010 | 0.102 | 0.322 | 0.096 |
| asymmetry | **truth** | 0.938 | **0.471** | 0.427 | 0.741 | 0.800 |

This overturns the reading that no band reaches 0.4. Against the empirical curve
that is true; against the truth curve the model tracks the local shape in every
band, and the worst case of the empirical scoring -- mean bias at `[0,18)`, a
median CCC of exactly 0.000 -- becomes 0.746. Median MAE falls with it, from
0.836 to 0.303 degrees in that band. Most of what the banded empirical numbers
measured was the noise in the target, not error in the fit.

### A low banded CCC usually means a flat band, not a bad fit

Both bands that stay low against the truth curve are bands where the truth curve
barely varies, and the same explanation covers the asymmetry curve's weakest
band. Each case's asymmetry structure sits at a dissimilarity commensurate with
its feature SDs: the narrow cases (SDs 5 to 20 degrees) peak at 30 to 36 degrees
and have decayed by 60, while the broad cases (SDs 60 to 160) have almost
nothing below 18 degrees and peak at 112 to 134. Truth-curve amplitude, in
asymmetry units:

| Case | [0,18) | [18,60) | [60,120) | [120,180] | Peak at |
|---|---:|---:|---:|---:|---:|
| narrow_1 | 0.0257 | **0.0170** | 0.0314 | 0.0048 | 36 deg |
| narrow_3 | 0.0361 | **0.0125** | 0.0727 | 0.0653 | 30 deg |
| broad_1 | **0.0008** | 0.0459 | 0.0465 | 0.0693 | 112 deg |
| broad_3 | 0.0006 | 0.0021 | 0.0026 | 0.0006 | 134 deg |

`[18,60)` is therefore the one band in which no regime is at full amplitude: it
catches the narrow cases after their peak and the broad cases before theirs. For
`narrow_1` and `narrow_3` it is the minimum-amplitude band outright, with median
absolute error at 1.13 and 1.28 times the band's entire range, so their CCC there
is 0.169 and 0.085 while the same fits score 0.918 and 0.595 at `[0,18)` and
0.755 and 0.851 at `[60,120)`. That is what pulls the pooled `[18,60)` asymmetry
median down to 0.427.

`broad_3` is a second and separate contributor, and it is not band-specific: its
asymmetry curve is flat across the whole axis, amplitude 0.0006 to 0.0026 with
error 4.7 to 13.2 times the range, so its 15 datasets score about zero in every
band and put a floor under every pooled median. None of this is a search
difference -- all three arms agree to within 0.01 CCC in this band.

Two things follow. The band edges are inherited from `continuous_density_4.1c`
and are not aligned to this panel's structure: the boundary at 18 degrees cuts
through the narrow cases' peak region while `[18,60)` straddles their decay.
And a banded CCC should not be read without the band's target range next to it;
where the range is small the honest summary is the absolute error, which is why
both are stored. Gating CCC on a minimum truth amplitude, the analogue of the
gate already applied to unestimable circular moments, would express this
directly and has not been implemented.

Two bands stay low against truth for the same reason, being nearly flat rather
than badly fitted: circular SD at `[60,120)` and `[120,180]`
scores 0.313 and 0.031 on truth-curve ranges of 0.691 and 0.378 degrees, with
median absolute errors of 0.401 and 0.442. When a curve varies by less than half
a degree across a band, there is no concordance to measure and the error is the
size of the range; the honest statement is that the model gets the level of the
SD curve roughly right at large dissimilarity and that its shape there carries
no information, not that it fits badly. This is the case that requires CCC, MAE
and target range to be read together.

The whole-curve concordances reported above are therefore not evidence of
within-band agreement against the empirical target: no band of any curve
reaches 0.4, and the two smallest
bands of mean bias and asymmetry are indistinguishable from zero. What the
whole-curve number measures is that both curves rise with dissimilarity. The
`[0,18)` band also carries the largest mean-bias MAE, 1.157 degrees against
0.169 at `[60,120)`, on a band whose median empirical range is only 0.204
degrees, which is exactly the regime where CCC cannot be read alone. Circular SD
has only two of its eighteen bins below 18 degrees, so its CCC there is left
missing by construction rather than estimated on two points; its MAE, 2.33
degrees, is still reported. Bins dropped for non-finite values are counted in
`dropped_points` rather than silently skipped.

Two consequences for the optimizer comparison:

1. **The arms are indistinguishable on curves.** Across every curve and band the
   four likelihood arms agree to within about 0.01 CCC and 0.01 degrees MAE. The
   NLL differences between them do not reach the empirical curves.
2. **The hierarchy's apparent curve advantage is a marginal-median artifact.**
   Its marginal median mean-bias CCC against the empirical curve is 0.753 while
   32-start SciPy's is 0.601, which reads as the worst optimizer producing the
   best curves. Paired per dataset it is not an advantage at all: median
   difference +0.0006, better on 12 of 24. The two marginals sit in different
   parts of a bimodal distribution, so their medians are not comparable. Do not
   quote the marginal pair. Its advantage is in the pooled trend, not within bands. Curve
   agreement and parameter recovery are separate outcomes here and rank the arms
   differently; neither alone selects an optimizer.

## Superseded provisional optimizer decision

All arms used the same continuous WNM point likelihood and common rescore. The
small pilot's apparent winner, log-hierarchical search plus SciPy polishing, did
not generalize: its nine-point version reached the 0.001-NLL gate on 66.7% of a
24-dataset panel and missed one basin by 68.20. A 13-point version repaired the
selected failures but reached the gate on only 75% of the expanded panel, with
a maximum gap of 2.407. The zoom is too sensitive to lattice alignment.

Thirty-two dispersed SciPy L-BFGS-B starts reached the 0.001 gate on every
development budget-panel dataset. Sixteen starts had two material misses, while
64 starts changed only sub-threshold gaps and nearly doubled median CPU time.
The frozen held-out configuration is therefore serial SciPy, 32 deterministic
log-space Latin-hypercube starts with seed 0, float32, artifact bounds, 500
iterations, `ftol=1e-9`, and `gtol=1e-6`.

The full 120-dataset development run completed without a dataset-level failure;
every fit had at least 24 converged starts. Some failed starts ended at losses up
to 0.000488 below the selected successful winner, consistent with float32
termination noise rather than a missed material basin.

## Recovery and surface comparison

WNM recovered all three truth parameters within a factor of 1.5 in 72.5% of
datasets. Median per-dataset joint log-RMSE was 0.150 and global pooled joint
log-RMSE was 0.410. Recovery improved with trial count: the corresponding
factor-of-1.5 rate was 57.5%, 77.5%, and 82.5% for 180, 450, and 900 trials.
Broad/weak cases were hardest (40%), reflecting weak identification as well as
finite-sample displacement of the likelihood optimum. The best likelihood basin
need not be the parameter tuple closest to truth.

On the same development data, WNM had lower common-grid NLL than the legacy
surface pipeline in 78.3% of datasets, with a median difference of -2.09 NLL.
WNM also improved the like-for-like recovery summaries: global joint log-RMSE
0.410 versus 0.453, median per-dataset joint log-RMSE 0.150 versus 0.189, and
factor-of-1.5 recovery 72.5% versus 62.5%.

The empirical-curve comparison was mixed for mean bias but favored WNM overall:

| Curve | Median CCC, WNM / surface | Median RMSE, WNM / surface |
|---|---:|---:|
| mean bias | 0.748 / 0.704 | 0.680 / 0.634 degrees |
| bias SD | 0.509 / 0.346 | 2.544 / 2.900 degrees |
| asymmetry | 0.614 / 0.568 | 0.0597 / 0.0646 |

WNM won the paired RMSE comparison on 57.5%, 66.7%, and 55.0% of datasets for
mean bias, bias SD, and asymmetry respectively. CCC is not interpretable by
itself on nearly flat broad/weak curves; RMSE and empirical target range are
saved alongside it. Ten development bias-SD bins, all in three `broad_3`
180-trial datasets, have a nonpositive Kutil-corrected resultant. This is the
finite-sample uniform limit, where circular SD is unbounded. They are stored as
missing, not as infinity or an imputed value; the bins are not empty.

## Operational result and exclusions

One versus two balanced CPU processes gave 0.280 versus 0.345 datasets/s, an
18% throughput gain, while peak aggregate RSS increased from 643 MB to 1.31 GB.
Serial CPU execution is frozen for the scientific held-out run. GPU execution
was subsequently authorized and used for the JAX-BADS and surface-hierarchy
panels and for the common rescore of every stored winner. GPU utilization,
memory, and threaded contention remain unmeasured; this is an operational
question, not a reason to reopen the search quality decision.

Do not use the first CPU optimizer pilot's JAXopt/BADS timing or the first CPU
throughput run: the former rebuilt solvers per dataset and the latter assigned
imbalanced trial counts. Their corrected `*_v2` successors are the timing
references.

## Previously inspected held-out results

These results describe the provisional 32-start configuration. Because the
comparator defect was discovered afterwards, they must not be described as a
prospectively frozen confirmation or used to tune the corrected comparison.

The frozen configuration was applied once to all 60 held-out `_2` datasets. All
fits completed and every dataset had at least 26 converged starts. Compared with
the frozen surface likelihood baseline:

| Metric | WNM | Surface NN |
|---|---:|---:|
| lower common-grid NLL | 70.0% of pairs | 30.0% of pairs |
| median per-dataset joint log-RMSE | 0.140 | 0.315 |
| global joint log-RMSE | 0.391 | 0.587 |
| all parameters within factor 1.5 | 70.0% | 43.3% |

The median WNM-minus-surface common-grid NLL was -1.33. WNM's strict
factor-of-1.5 recovery rate rose from 25% at 180 trials to 90% at 450 and 95% at
900; both pipelines were at 25% on the smallest datasets.

Held-out curve recovery remained mixed rather than uniformly dominant. WNM
versus surface median CCC was 0.337 versus 0.307 for mean bias, 0.322 versus
0.224 for bias SD, and 0.192 versus 0.182 for asymmetry. Median RMSE was 0.325
versus 0.314 degrees, 0.843 versus 0.804 degrees, and 0.0589 versus 0.0611,
respectively. Paired WNM RMSE win rates were 55.0%, 68.3%, and 45.0%; this is why
both paired differences and marginal medians are retained. The held-out truth
tuples produce particularly flat curves, so CCC remains secondary to RMSE and
target range. No held-out empirical SD bin reached the uniform-limit missing
state.

## Follow-up status and remaining cleanup

1. Search selection is complete for all retained objectives. The final
   executive policy is the same 64-start batched port for each; see their
   objective-specific findings documents.
2. CPU/GPU rescoring, the targeted float64 diagnostic, and the faithful-port
   comparison are complete. They identify default GPU TF32 surrogate matmuls as
   the remaining cross-device discrepancy. Search remains float32 but requires
   `highest` matmul precision; global x64 adoption is unnecessary.
3. No jittered-lattice hierarchy arm was built. This is recorded rather than
   left as an open recovery requirement: the selected 64-start port is cheaper
   and more reliable than the historical hierarchy on the focused likelihood
   panel.
4. The research and transition banded-metric analyses still use separate frame
   schemas. Collapsing them is engineering cleanup for a shared production
   consumer, not a blocker for the completed recovery result.
