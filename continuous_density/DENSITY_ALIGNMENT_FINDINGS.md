# Density-target alignment findings

Date: 2026-09-09

Artifact paths and analysis scripts named below are relative to
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`.

## Question

The production density target is an empirical wrapped-KDE signed-mass curve.
The surface NN derives its fitted prediction through a discretized density
surface, while WNM computes signed interval mass analytically. This diagnostic
asked whether aligning the empirical and WNM branches changes parameter
recovery.

The 120 development datasets were each fitted from the same 32 dispersed starts,
with one start replaced by the known generating parameters. Five objectives
were compared:

- `current_kde_ccc`: current empirical KDE target against analytic WNM mass;
- `binned_sign_ccc`: fixed feature bins and hard sign on both branches;
- `kernel_sign_ccc`: the same observed-design feature kernel and hard sign on
  both branches;
- `matched_kde_ccc`: the same observed-design feature kernel and the same
  pooled-SJ circular-Gaussian smoothing on both branches;
- `sign_likelihood`: trial-level Bernoulli likelihood for response sign.

The run completed all 600 fits and 19,200 optimizer candidates without duplicate
dataset/objective rows.

## Recovery result

| Objective | Median per-dataset log-RMSE | Global log-RMSE | All parameters within factor 1.5 |
|---|---:|---:|---:|
| current KDE CCC | 0.938 | 1.189 | 18.3% |
| binned sign CCC | 1.145 | 1.394 | 15.8% |
| kernel sign CCC | 0.881 | 1.217 | 17.5% |
| matched KDE CCC | 0.901 | **1.108** | 18.3% |
| sign likelihood | 1.085 | 1.362 | 14.2% |

The marginal medians make kernel sign and matched KDE look slightly better than
the current target. The paired comparisons do not support that interpretation:

| Alternative minus current | Median paired log-RMSE difference | Alternative better |
|---|---:|---:|
| binned sign CCC | +0.058 | 35.0% |
| kernel sign CCC | +0.008 | 45.0% |
| matched KDE CCC | +0.006 | 45.0% |
| sign likelihood | +0.041 | 37.5% |

Thus none of the aligned alternatives improves the typical dataset. Matched KDE
does reduce the global RMS error, which means it avoids some large misses, but
that is a tail improvement rather than a paired shift. Binned sign is clearly
inferior here. Kernel sign and matched KDE remain plausible definitions, but
this panel supplies no parameter-recovery reason to replace the current target.

Recovery differs much more by generating regime than by alignment choice.
Median log-RMSE under matched KDE is 0.240 for reversed-asymmetric cases, 0.775
for narrow cases, 1.214 for ordinary-asymmetric cases, and 1.433 for broad/weak
cases. The density-derived summaries weakly identify three scale parameters in
large parts of the design.

## What the truth start shows

The optimizer was deliberately given the exact generating point as one start.
Following that local basin gives much better recovery than selecting the lowest
loss across all 32 starts:

| Objective | Truth-start basin median log-RMSE | 32-start winner median log-RMSE | Truth start selected |
|---|---:|---:|---:|
| current KDE CCC | 0.375 | 0.938 | 6.7% |
| binned sign CCC | 0.462 | 1.145 | 4.2% |
| kernel sign CCC | 0.340 | 0.881 | 0.8% |
| matched KDE CCC | 0.330 | 0.901 | 1.7% |
| sign likelihood | **0.327** | 1.085 | 2.5% |

This is not primarily a failure to enter the truth basin: it was explicitly
entered. Other basins usually fit the finite empirical target better while
recovering the generating parameters worse. For sign likelihood, the global
winner improves on the truth-start basin by only 0.00068 median loss, despite a
large recovery difference. For the CCC objectives the median gain is about
0.005--0.013. The sign objective therefore gives the clearest evidence of a
nearly flat or non-identifiable parameter tradeoff; the curve objectives also
permit finite-sample target chasing.

The truth-start basin improves with sample size. For matched KDE its median
log-RMSE falls from 0.526 at 180 trials to 0.241 at 900; for sign likelihood it
falls from 0.418 to 0.169. The corresponding unrestricted 32-start winners
remain much farther from truth. Selecting the numerically lowest empirical loss
is therefore not the same thing as selecting the best-recovering finite-sample
solution.

## Comparison with the existing surface NN

Only `current_kde_ccc` has a target-matched existing surface fit. Under that
comparison, marginal summaries favor the surface:

| Metric | WNM | Surface NN |
|---|---:|---:|
| Median per-dataset log-RMSE | 0.938 | 0.752 |
| Global log-RMSE | 1.189 | 1.093 |

But paired by dataset, WNM wins 58 and the surface wins 62. The median paired
difference (WNM minus surface) is only +0.002, with a dataset-bootstrap 95%
interval of [-0.036, +0.034]. The global RMS difference is produced by large
misses in both directions rather than a stable surface advantage. The earlier
claim that the surface is generally better for density was therefore too strong:
the defensible conclusion is that typical parameter recovery is tied, with a
somewhat worse WNM error tail under this target.

The existing surface fits cannot fairly evaluate the four alternative targets,
because those surface parameters were optimized against the current density
target. Comparing their recovery to WNM fits optimized against another target
would confound model branch and objective. A surface comparison for matched KDE
or hard sign requires refitting the surface with exactly the same empirical and
prediction operators.

## Conclusion and prediction-equivalence follow-up

Target alignment does not explain a general surface-NN recovery advantage,
because there is no general paired advantage to explain. It also does not solve
the larger recovery problem. The dominant result is weak identification and
finite-sample basin selection: parameters far from the generator can give a
slightly lower empirical objective than the truth basin.

That prediction-space comparison is now complete; see
`DENSITY_SOLUTION_EQUIVALENCE_FINDINGS.md`. Both mechanisms occur. Narrow and
reversed cases often have virtually identical full response distributions at
the truth-basin and global-winner parameters, demonstrating genuine parameter
non-identifiability. Broad/weak and ordinary cases often select predictively
different distributions, so their misses cannot be dismissed as harmless
reparameterization.

The frozen held-out matched-KDE WNM run is also complete; see
`DENSITY_MATCHED_KDE_HELDOUT_FINDINGS.md`. Its global recovery error is 1.305,
compared with 1.306 for the existing surface pipeline. The paired comparison is
inconclusive (median WNM-minus-surface difference +0.034, bootstrap interval
[-0.020, +0.094]). This is enough for the practical pipeline comparison. A
surface refit with the matched operator is needed only to isolate model-family
effects while holding the fitting target fixed.

## Artifacts

The completed run and derived tables are under
`wnm_density_alignment_development_gpu_v1/`. Reproducible analysis is in
`analyze_density_alignment.py`. Key derived files are:

- `alignment_recovery_summary.csv`;
- `paired_vs_current_target.csv`;
- `truth_start_summary.csv`;
- `surface_current_target_pairs.csv`;
- `surface_current_target_summary.csv`.
