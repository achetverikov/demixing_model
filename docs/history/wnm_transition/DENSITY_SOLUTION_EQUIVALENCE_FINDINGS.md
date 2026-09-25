# Density solution-equivalence findings

Date: 2026-09-09

Artifact paths and analysis scripts named below are relative to
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`.
Generated artifacts are external and are not shipped with or expected in a
normal checkout.

## Question and method

The density-alignment recovery panel often selected parameters far from the
generator even though a local optimization initialized at the generator stayed
much closer. This check asks whether the two fitted parameter vectors are merely
alternative parameterizations of effectively the same prediction.

For every one of the 600 dataset/objective fits, the frozen K12 WNM predictor
was evaluated at:

- the generating parameters (`truth`);
- the lowest-loss candidate among all 32 starts (`winner`);
- the result of the start initialized exactly at the generator (`truth_basin`).

Predictions were evaluated on the complete 2-degree feature-dissimilarity grid.
The comparison includes circular mean, resultant, circular SD, raw signed mass,
20-degree feature-smoothed signed mass, the curve actually used by each
objective, and the complete 180-cell response distribution. Distributional
difference is summarized as total variation at each feature difference and then
averaged over feature differences. All cell-probability vectors were checked to
sum to one.

## Overall result

The global winners are not generally prediction-equivalent to the generator.
Median winner-versus-truth errors are:

| Objective | Objective-curve MAE | Mean-bias MAE | Circular-SD MAE | Mean full-distribution TV |
|---|---:|---:|---:|---:|
| current KDE CCC | 0.0447 | 0.952° | 5.01° | 0.260 |
| binned sign CCC | 0.0718 | 1.014° | 5.46° | 0.331 |
| kernel sign CCC | 0.0475 | 0.991° | 5.04° | 0.284 |
| matched KDE CCC | **0.0445** | 1.024° | 5.08° | **0.253** |
| sign likelihood | 0.0568 | 1.107° | 7.69° | 0.355 |

A mean total variation of 0.25--0.35 is substantial: at a typical feature
difference, one quarter to one third of the response probability must be moved
between bias cells to turn the fitted distribution into the generating one.
The poor parameter recovery therefore cannot be dismissed globally as harmless
reparameterization.

The truth-basin solutions are closer to the generating distribution, with
median total variation of 0.131--0.163 depending on objective. They are not
identical to truth either: finite-sample targets pull even a local fit initialized
at truth away from the generating distribution. The unrestricted global search
usually adds another, larger basin-selection effect.

## The result is regime-specific

The most informative comparison is between the global winner and the
truth-basin fit. If their full distributions agree, the large parameter
difference is non-identifiability; if they disagree, the global loss minimum has
selected a genuinely different predictive solution.

For the current density objective:

| Case | Truth basin vs truth TV | Winner vs truth TV | Winner vs truth basin TV |
|---|---:|---:|---:|
| broad_1 | 0.296 | 0.385 | 0.328 |
| broad_3 | 0.116 | **0.899** | **0.847** |
| narrow_1 | 0.163 | 0.167 | **0.0002** |
| narrow_3 | 0.348 | 0.376 | **0.0004** |
| ordinary_1 | 0.108 | 0.387 | **0.425** |
| ordinary_3 | 0.254 | 0.181 | **0.366** |
| reversed_1 | 0.093 | 0.088 | **0.0005** |
| reversed_3 | 0.108 | 0.107 | **0.0005** |

This separates three situations:

1. **Clear parameter non-identifiability.** In the narrow and reversed cases,
   current-density winners and truth-basin fits usually produce virtually the
   same complete distribution despite different parameters. The same pattern
   holds for kernel sign and matched KDE. Across objectives, 58% of narrow fits
   and 85% of reversed fits have winner-versus-truth-basin mean TV below 0.01;
   the median is 0.0021 and 0.0007 respectively. The residual error shared by
   both solutions relative to truth is finite-sample curve error, not a
   consequence of choosing the remote parameter vector.
2. **Genuinely different, worse predictive minima.** `broad_3` and `ordinary_1`
   are the clearest examples. Under current density, the truth-basin fit is only
   0.116 and 0.108 TV from truth, while the global winner is 0.899 and 0.387
   away. The empirical density summary strongly rewards a different response
   distribution in these cases.
3. **Parameter error and predictive error can disagree.** In `ordinary_3`, the
   global winner and truth-basin fit differ substantially, but the global winner
   is closer to the generating distribution in median TV (0.181 versus 0.254).
   Calling it a recovery failure solely because its parameters are farther from
   the generator would be misleading.

Matched KDE produces the same qualitative separation. Its winner-versus-truth-
basin TV is approximately zero for narrow and reversed cases, but 0.763 for
`broad_3` and about 0.41--0.45 for the ordinary cases. Alignment therefore does
not remove the prediction-level basin problem.

Binned sign and sign likelihood discard still more of the distribution. Their
selected solutions can match the fitted sign summary while changing mean,
dispersion, and full density substantially. This explains why neither is a good
three-parameter recovery objective in the completed panel.

## Dissimilarity-dependent curves

The current winner and truth-basin fit have median objective-curve MAE of only
0.0055 overall, much smaller than either one's error against the generating
curve. That overall number conceals the same regime split: winner-versus-truth-
basin objective-curve MAE is about 0.0001--0.0002 in narrow and reversed cases,
0.023 in ordinary cases, and 0.026 in broad/weak cases. Mean-bias and circular-SD
curves show the corresponding split rather than only a marginal mean over
dissimilarity.

Prediction recovery improves with sample size for the useful curve objectives.
Winner-versus-truth mean TV for current density falls from 0.380 at 180 trials
to 0.167 at 900; matched KDE falls from 0.374 to 0.168. Sign likelihood remains
weaker (0.371 to 0.294), consistent with the information it discards.

## Conclusion

Both mechanisms are present:

- In narrow and reversed regimes, much of the alarming parameter error is real
  non-identifiability: distant parameters often generate the same observable
  response distribution.
- In broad/weak and ordinary regimes, many global winners are predictively
  different. The density/sign summaries do not reliably select the generating
  distribution from finite samples, even when the truth basin is explicitly
  available.

Consequently, one global parameter-recovery number is not an adequate selection
criterion. Future comparisons should report parameter recovery together with
full-distribution or multi-curve recovery by regime and dissimilarity. Target
alignment alone does not fix the problem. Of the tested alignments, matched KDE
has the best full-distribution tail behavior, but not enough of an advantage to
justify replacing the current target without the target-matched surface refit.

## Artifacts

- `check_density_solution_equivalence.py`: reproducible prediction evaluator;
- `wnm_density_alignment_development_gpu_v1/solution_equivalence_metrics.csv`:
  per-dataset, per-comparison, per-curve metrics;
- `wnm_density_alignment_development_gpu_v1/solution_equivalence_summary.csv`:
  grouped summaries by objective, trial count, and regime;
- `wnm_density_alignment_development_gpu_v1/solution_equivalence_manifest.json`:
  input, checkpoint, code, and output hashes.
