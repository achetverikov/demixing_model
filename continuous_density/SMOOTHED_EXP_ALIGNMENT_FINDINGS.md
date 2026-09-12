# Smoothed-expectation target-alignment findings

Date: 2026-09-09; held-out and intermediate-curve update 2026-09-10

Artifact paths and analysis scripts named below are relative to
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`.
A compact snapshot of the JAX port manifests, summaries, and comparison tables
is versioned in
[`validation_outputs/jax_lbfgsb_port_2026-09-10/`](validation_outputs/jax_lbfgsb_port_2026-09-10/README.md).
Full per-task records and other generated artifacts remain external.

## Result

The development-only run completed 360 fits: 120 datasets under each of the
current smoothed-expectation, matched-smoothed-expectation, and binned-
expectation objectives. Every fit used the same 32 dispersed starts, with the
generating parameters replacing one start. All 11,520 candidates are recorded.

Applying the empirical feature smoother to the WNM model branch improves
parameter recovery:

| Objective | Median per-dataset log-RMSE | Global log-RMSE | All parameters within factor 1.5 |
|---|---:|---:|---:|
| current smoothed expectation | 0.730 | 0.992 | 14.2% |
| matched smoothed expectation | **0.637** | **0.951** | **23.3%** |
| binned expectation | 0.706 | 0.963 | 21.7% |

Matched minus current has a paired median log-RMSE difference of -0.030, with a
dataset-bootstrap interval of [-0.057, -0.011]; matched is better on 64.2% of
datasets. Binned expectation improves by -0.016 at the paired median, but its
interval includes zero and it is better on 55.8% of datasets. The fully matched
operator is the strongest of the tested definitions.

The improvement is not uniform across parameter regimes. It is largest in the
reversed cases and is also present in narrow-3 and ordinary-1. Matched smoothing
is worse in narrow-1 and ordinary-3, and nearly tied in broad-1. It therefore
removes a systematic mismatch but does not create a universally identifying
mean objective.

## Target-construction effect

At the generating parameters, applying the feature smoother changes the truth
mean curve by 0.120 degrees median MAE across dissimilarity. The empirical curve
is closer to the matched truth than the raw truth on 65.8% of datasets. At 900
trials, median empirical-versus-truth curve MAE falls from 0.361 degrees with
the raw truth to 0.244 degrees with the matched truth. The largest construction
effect is in the reversed regime: the truth curves differ by 0.651 degrees, and
the empirical curve's median error falls from 0.867 to 0.641 degrees.

The fitted prediction result points the same way. The current fit's raw mean
curve has median MAE 0.426 degrees against the raw truth curve. The matched fit's
matched mean curve has median MAE 0.360 degrees against the matched truth curve.
Thus the recovery gain is accompanied by better recovery of the corresponding
noise-free curve rather than being only a parameter-space relabeling.

## Truth-start diagnosis

Alignment does not solve the larger basin and identification problem:

| Objective | Truth-basin median log-RMSE | Global-winner median log-RMSE | Truth start selected |
|---|---:|---:|---:|
| current smoothed expectation | 0.421 | 0.730 | 7.5% |
| matched smoothed expectation | **0.333** | **0.637** | 10.8% |
| binned expectation | 0.374 | 0.706 | 5.8% |

For matched smoothing, the global winner improves the empirical loss over the
truth-start basin by only 0.00112 at the median while substantially worsening
parameter recovery. The truth basin improves from log-RMSE 0.448 at 180 trials
to 0.305 at 900, whereas unrestricted winners do not improve monotonically.
Finite-sample target chasing and weak parameter identification therefore remain
after the operator is corrected.

## Existing surface NN

Under the current mismatched operator, the diagnostic WNM fit has a paired
median log-RMSE disadvantage of +0.029 against the existing surface fit, with a
bootstrap interval [0.009, 0.052]. With matched smoothing, the difference
changes to -0.025 and the interval becomes [-0.061, 0.023]; WNM is better on
55.8% of datasets. The apparent typical surface advantage therefore disappears.

This is not a target-matched model-family comparison: the surface fit still uses
the current pointwise model operator, and the WNM diagnostic was allowed a truth
start. It shows that the previous surface advantage is not robust to correcting
the WNM target alignment. A matched surface refit is needed only to isolate
model family while holding the operator fixed.

## Frozen held-out confirmation

The prospectively frozen run completed all 60 held-out datasets with the matched
operator, 32 dispersed SciPy L-BFGS-B starts, and no truth start:

| Arm | Median per-dataset log-RMSE | Global log-RMSE | All parameters within factor 1.5 |
|---|---:|---:|---:|
| matched WNM, 32-start SciPy | 1.130 | 1.239 | 5.0% |
| current WNM, same 32-start SciPy | 0.981 | 1.248 | 10.0% |
| current WNM, existing union pipeline | 1.318 | 1.333 | 8.3% |
| existing surface NN | **0.970** | **1.149** | **11.7%** |

Matched versus the same-search current WNM has a paired median difference of
+0.017, interval [-0.013, +0.048], and matched wins 40.0% of datasets. Thus the
development parameter-recovery improvement does not generalize to the held-out
cases. The current union pipeline is worse than either single 32-start arm
because choosing the smallest empirical loss across additional searches often
selects a different, less truth-like solution in this weakly identified
objective. Matched versus that union is tied: -0.004 at the paired median,
interval [-0.090, +0.028], with matched winning 51.7%.

Matched WNM versus the existing surface NN is +0.058 at the paired median,
interval [-0.036, +0.243], and WNM wins 40.0%. This removes the clear held-out
disadvantage of the current WNM union (+0.099, interval [+0.028, +0.366]), but it
does not demonstrate WNM superiority: the surface retains the best absolute
median and global recovery. This remains a practical-pipeline comparison across
different model operators, not strict model-family attribution.

The operator diagnosis itself does reproduce. At the generating parameters the
matched truth curve is closer to the empirical curve on 73.3% of held-out
datasets; median curve MAE is 0.246 degrees for matched truth versus 0.278 for
raw truth. The matched fit's median curve MAE is 0.187 degrees against its
matched truth and 0.102 degrees against the empirical target. Correcting the
operator therefore improves semantic alignment and curve targeting without
making the parameters identifiable.

Candidate convergence was adequate but not perfect: 92.1% of 1,920 candidates
reported normal convergence, and 52 of 60 selected winners did so. Restricting
selection to successful candidates changes median/global log-RMSE to
1.186/1.264 from 1.130/1.239; it does not change any comparison conclusion.

### Intermediate 450-trial curve check

The held-out `ordinary_2` and `reversed_2` cases isolate intermediate feature
scales, excluding the narrow and broad/weak regimes. At 450 trials, or five
responses per dissimilarity, the recovered-parameter mean curve was closer than
the generating-parameter curve to the empirical target in all 10 datasets for
both WNM and the surface NN. Across the two cases, median
recovered-versus-empirical RMSE was 0.118 degrees for matched WNM and 0.132
degrees for the deployed surface operator, compared with 0.284 and 0.332 degrees
at the generating parameters.

The optimizer is therefore fitting the realized empirical curve well, while the
empirical curve itself fluctuates too much around truth to support reliable
parameter recovery. This is most visible for `reversed_2`: median recovered-
versus-generating curve RMSE is 0.614 degrees for WNM and 0.607 degrees for the
surface NN. For `ordinary_2` it is only 0.128 and 0.148 degrees. The held-out
failure is consequently a finite-sample curve-target problem with strong regime
dependence, not evidence that the optimizer generally fails to fit its target.

## Batched optimizer follow-up

The faithful JAX L-BFGS-B port was tested on the same 60 frozen held-out
matched-operator datasets with 32 seed-0 starts, float32 arrays,
`JAX_DEFAULT_MATMUL_PRECISION=highest`, and no truth start. It reached the
0.001 union gate on 59/60 datasets and reduced median time from 6.084 to 0.261
seconds, a 22.10x paired speedup. The miss was `broad_2_seed0_n450`: common CPU
loss 8.29395 for the port versus 5.61148 for SciPy. Median/global joint log-RMSE
was 1.1466/1.2334 versus 1.1297/1.2386, respectively.

A post-hoc hard-case probe produced loss 9.8699 with 64 non-nested starts and
4.6848 with 128. This establishes start-design sensitivity but cannot justify
selecting 128 starts from held-out performance. This comparison tests only the
32-start component; it does not select a matched-objective production search.
The selected cache/eight-polish/32-start-SciPy/conditional-hierarchy rule was
developed for the original pointwise model operator and must be rebuilt with
matched complex-moment curves before the port can be compared with it.

That proper development comparison is now complete. The matched 80-by-64
complex-moment cache contains 409,600 curves and was followed by eight
cache-seeded polishes. The panel also included 32-start SciPy, the 32-start
port, the production hierarchy, and 32-start JAX-BADS. The cache materially
improves empirical-objective coverage: it was within 0.001 of the full union on 116/120 datasets with worst
gap 0.249, whereas the port alone covered 86/120 with worst gap 44.40.
Substituting the port for SciPy inside the adapted cache/continuous/conditional-
hierarchy strategy preserved its 116/120 coverage and 0.249 worst gap while
reducing median strategy time from 7.20 to 1.99 seconds, a 3.62-fold speedup.
The inherited conditional rule still misses four development union winners.

A complete 64-start float32/`highest` port follow-up improved the standalone
gate rate from 86/120 to 91/120 but left its worst gap at 44.40. Using 64 starts
inside the composite still covered 116/120 with worst gap 0.249 and was slightly
slower than the 32-start version (2.032 versus 1.986 median seconds). Combining
both start designs also left the same four misses. In all four, cache and port
agreement prevented the hierarchy trigger; the full-panel winner was the
hierarchy in three and JAX-BADS in one.

The final executive decision nevertheless selects the standalone 64-start
float32/`highest` port and drops the cache/composite, so all retained objectives
share one optimizer. This knowingly accepts 29/120 development fits outside the
0.001 empirical-loss gate and a worst gap of 44.40. The justification is
consistency rather than smoothed-objective dominance: parameter recovery was
effectively tied between 32 and 64 starts, and paired whole-curve and four-band
truth CCC differences were approximately zero. The empirical-loss misses must
remain visible in downstream reporting rather than being described as rare or
numerically negligible.

## Conclusion

Smoothed expectation had the same class of feature-alignment problem as density.
Correcting it materially improves development WNM parameter and truth-curve
recovery and removes the development surface advantage. The parameter gain does
not reproduce held out, although the target-alignment and curve result does. It
does not make smoothed expectation a strong parameter-recovery objective: even
matched, only 23.3% of development and 5.0% of held-out datasets recover all
three parameters within a factor of 1.5.

Use the observed-design matched complex-moment operator for WNM because it
matches the empirical target and improves curve targeting, not because it has a
held-out parameter-recovery advantage. On this balanced design the observed-
design and regular-grid operators are identical to float precision, so the
redundant grid arm did not require fitting. Smoothed expectation remains useful
for mean-curve fitting, not as evidence that WNM can replace the surface NN for
parameter recovery.

## Artifacts

The development run and derived tables are under
`wnm_smoothed_exp_alignment_development_gpu_v1/`; its analysis is in
`analyze_smoothed_exp_alignment.py`. The held-out run and tables are under
`wnm_smoothed_exp_matched_heldout_gpu_v1/`; its analysis is in
`analyze_smoothed_exp_matched_heldout.py`. Annotated intermediate-case curves
and their data are under `intermediate_curve_recovery_n450/`, reproduced by
`plot_intermediate_curve_recovery.py`.

## Notes for developers

The detailed matched-search panel is in the external
`wnm_matched_smoothed_exp_development_cpu_v1/` artifact. Its component and
strategy rows are development evidence; the earlier held-out 64/128-start
probe remains post-hoc and must not be used to tune the final conditional rule.
The complete development 64-start follow-up is in
`wnm_matched_smoothed_exp_port32_port64_development_cpu_v2/`.
