> **Archived transition record.** This file preserves the evidence and terminology used during the 2026 WNM cutover. Commands, paths, and module names may reflect the transition-era repository layout and are not current run instructions. See the root README and current package READMEs for executable workflows.

# WNM smoothed-exp search and surface comparison

Artifact paths and analysis scripts named below are relative to
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`.
Generated artifacts are external and are not shipped with or expected in a
normal checkout.

## Decision

Use the same reusable WNM log curve lattice identified for density: 80 feature
points over 2.5--200 degrees and 64 spatial points over 5--200 degrees. For
`smoothed_exp`, retain the eight best distinct lattice minima and polish all
eight with SciPy L-BFGS-B. Also run 32-start SciPy. Run the production hierarchy
only when the cache and SciPy canonical losses differ by more than 0.01, then
take the lowest canonical score among the permitted arms.

This rule was selected on development and applied unchanged held out. On the
CPU common rescore it was within 0.001 of all searched arms on 120/120
development datasets (worst gap 0.000151) and 60/60 held-out datasets (worst
gap 0.000584). It triggered the hierarchy for 17/120 development and 15/60
held-out datasets.

## Optimizer comparison

The development panel included 32-start serial SciPy L-BFGS-B, 32-start BBZ
JAX-BADS, and the production hierarchy. The curve-cache arm was evaluated on
the same data and all four arms were rescored on CPU with the canonical WNM
`smoothed_exp` evaluator.

| Arm | Within 0.001 of union | Worst gap | Median seconds per dataset |
|---|---:|---:|---:|
| SciPy L-BFGS-B | 93/120 | 34.236 | 5.60 |
| BBZ JAX-BADS | 82/120 | 34.236 | 5.87 |
| Production hierarchy | 68/120 | 0.956 | 12.38 |
| Log cache + eight polishes | 113/120 | 1.183 | 1.78 |

The failures were complementary. SciPy and JAX-BADS repeatedly missed very
large-loss broad/weak basins; the hierarchy found those basins but was less
precise on many ordinary cases. A single cached minimum covered only 91/120 on
the GPU comparison. Polishing eight distinct cached minima raised the CPU
coverage to 113/120. Cache + SciPy + conditional hierarchy attained the
120/120 result above. JAX-BADS was necessary in the comparison panel but added
no production win beyond that selected combination.

The shared 409,600-curve GPU build took about 19.4 seconds and the raw float32
curves occupy about 141 MiB. The per-dataset times exclude this shared build.

## Existing surface-NN comparison

The surface arm is the existing production-hierarchy `smoothed_exp` fit and was
not retuned. Both arms were evaluated against the same empirical curves.

| Split | Arm | Joint log RMSE | Mean-bias CCC | Mean-bias RMSE | Bias-SD RMSE | Asymmetry RMSE |
|---|---|---:|---:|---:|---:|---:|
| Development | WNM | 1.113 | 0.918 | 0.324 | 8.767 | 0.0819 |
| Development | surface NN | 0.899 | 0.923 | 0.286 | 6.657 | 0.0789 |
| Held out | WNM | 1.333 | 0.902 | 0.142 | 34.210 | 0.0760 |
| Held out | surface NN | 1.149 | 0.742 | 0.145 | 5.470 | 0.0690 |

Parameter recovery itself favors the surface NN:

| Split | Arm | Global joint log-RMSE | Median per-dataset log-RMSE | All parameters within factor 1.5 |
|---|---|---:|---:|---:|
| Development | WNM | 1.113 | 0.725 | 11.7% |
| Development | surface NN | **0.899** | **0.668** | **20.8%** |
| Held out | WNM | 1.332 | 1.318 | 8.3% |
| Held out | surface NN | **1.149** | **0.970** | **11.7%** |

The held-out surface factor rate counts an estimate exactly at the reciprocal
factor boundary (5.0 fitted versus 7.5 true) as recovered; the earlier stored
Boolean excluded it through floating-point roundoff.

The paired WNM-minus-surface median log-RMSE difference is +0.038 on
development and +0.099 held out; WNM has lower parameter error on only 35% of
datasets in either split. The surface advantage therefore appears in the paired
comparison as well as the marginal summaries.

### Target-alignment follow-up

These numbers describe the current fitting pipelines, but they do not isolate a
model-family advantage. The empirical smoothed-expectation target is a
20-degree Gaussian-weighted circular mean over the observed feature
dissimilarities. Both model branches instead predict the unsmoothed conditional
mean at each feature-grid point. Consequently the empirical and model branches
do not apply the same feature operator. The surface NN's approximation may
accidentally regularize its mean curves toward the smoothed empirical target.

The appropriate matched model curve is not an arithmetic smoothing of predicted
angles. At every observed dissimilarity, obtain the model's complex first moment
`r * exp(i * mean)`, apply exactly the empirical Gaussian feature weights, sum
those moments, and take the argument.

That development diagnostic is now complete; see
`SMOOTHED_EXP_ALIGNMENT_FINDINGS.md`. Matched smoothing improves WNM median
log-RMSE from 0.730 to 0.637 and factor-1.5 recovery from 14.2% to 23.3%. The
paired surface comparison changes from a clear +0.029 WNM disadvantage under
the current operator to an inconclusive -0.025 under the matched operator. Thus
the previous development surface advantage is not robust to target alignment.

The frozen 60-dataset held-out matched-WNM run without truth starts is also
complete. Matched WNM has median/global joint log-RMSE 1.130/1.239. Against the
same 32-start SciPy current-WNM fit, the paired median difference is +0.017 with
an interval crossing zero, so the development parameter gain does not
generalize. Against the existing surface NN it is +0.058, interval [-0.036,
+0.243], with WNM better on 40.0%; the surface's absolute median/global errors
remain lower at 0.970/1.149. The alignment correction does reproduce at the
curve level: matched truth is closer to the empirical target on 73.3% of held-
out datasets. Use the matched WNM definition for semantic consistency, while
retaining the conclusion that smoothed expectation weakly identifies the
parameters.

Development is a slight surface-NN win on the fitted mean-bias curve. Held out,
WNM has the higher median mean-bias CCC and lower paired RMSE in 73% of
datasets. The dissimilarity-band results qualify that result: WNM has lower
mean-bias RMSE in 62%, 76%, and 55% of development datasets in the 18--60,
60--120, and 120--180 degree bands, but only 31% in 0--18 degrees. Held out,
the corresponding rates are 75%, 78%, 72%, and 43%.

This is a mean-only objective. Parameter recovery can still be measured, and it
is worse for WNM under the current operator; the objective simply gives no
direct constraint on spread or the full distribution, which helps explain the
weak identification. Matched smoothing improves recovery but still recovers all
three parameters within a factor of 1.5 on only 23.3% of development datasets.
The supported positive conclusion for WNM is therefore about mean-bias
prediction across dissimilarity, not generally strong parameter or
distributional recovery.

## Artifacts

- `wnm_smoothed_exp_development_gpu_v1/`: SciPy, JAX-BADS, and hierarchy panel.
- `wnm_smoothed_exp_log80x64_multimin8_development_gpu_v1/`: development cache
  with eight local polishes.
- `wnm_smoothed_exp_development_cpu_v1/`: common rescore, frozen selection, and
  paired surface comparison.
- `wnm_smoothed_exp_heldout_gpu_v1/` and
  `wnm_smoothed_exp_log80x64_multimin8_heldout_gpu_v1/`: frozen held-out search.
- `wnm_smoothed_exp_heldout_cpu_v1/`: held-out selection and paired surface
  comparison.
- `wnm_smoothed_exp_matched_heldout_gpu_v1/`: frozen held-out matched-operator
  fit and analysis.
