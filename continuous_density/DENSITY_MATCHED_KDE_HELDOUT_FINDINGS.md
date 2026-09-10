# Held-out matched-KDE density recovery

Date: 2026-09-09; intermediate-curve update 2026-09-10

Artifact paths and analysis scripts named below are relative to
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`.
Generated artifacts are external and are not shipped with or expected in a
normal checkout.

## Result

The frozen held-out run completed 60/60 datasets with 32 starts per dataset
(1,920 candidates). Generating parameters were not supplied as starts. Seven
fits ended at a parameter bound; every fit had finite recovery metrics.

The practical pipeline comparison is:

| Arm | Median per-dataset log-RMSE | Global log-RMSE | All parameters within factor 1.5 |
|---|---:|---:|---:|
| matched-KDE WNM | **1.134** | **1.305** | 3.3% |
| existing density surface NN | 1.290 | 1.306 | 8.3% |

The equal global errors do not mean the same datasets are recovered equally.
Paired by dataset, the median WNM-minus-surface difference is +0.034 and WNM is
better on 38.3% of datasets. A descriptive dataset-bootstrap 95% interval for
the median difference is [-0.020, +0.094]. Thus the point estimate favors the
surface on the typical paired comparison, but the held-out panel does not show
a clear, stable difference between the pipelines.

Recovery remains poor for both arms. The parameter-specific results show a
tradeoff rather than uniform superiority:

| Parameter | WNM median absolute log error | Surface median absolute log error | WNM global log-RMSE | Surface global log-RMSE |
|---|---:|---:|---:|---:|
| `sd_feat1` | 1.033 | **0.722** | **1.246** | 1.274 |
| `sd_feat2` | 0.843 | **0.661** | 1.494 | **1.324** |
| `sd_spat` | **0.817** | 1.036 | **1.150** | 1.320 |

The WNM has a better spatial-scale error distribution, the surface has better
typical feature-scale errors, and both have large misses.

## Relation to the previous WNM target

The prior current-target WNM had median/global log-RMSE 1.226/1.472. The matched
run has 1.134/1.305. However, their paired median difference is only -0.001
(matched better on 53.3%; bootstrap interval [-0.026, +0.011]). The improvement
is in the error tail, not in typical recovery. This repeats the development
pattern, where matched KDE reduced global RMS without a paired median shift.

The held-out old-versus-matched WNM numbers are not a clean objective ablation:
the old arm used the selected lattice-scan-plus-polish search, whereas the
matched arm used 32 SciPy starts. The development alignment panel, where both
targets used the same starts, is the clean target comparison.

## Intermediate 450-trial curve check

The held-out `ordinary_2` and `reversed_2` cases isolate intermediate feature
scales, excluding the narrow and broad/weak regimes. At 450 trials there are
only five responses per dissimilarity. Across their five response seeds, both
optimizers fit the realized density curve well: the recovered-parameter curve
was closer than the generating-parameter curve to its empirical target in all
10 datasets for both WNM and the surface NN. Median recovered-versus-empirical
RMSE was 0.029 for matched-KDE WNM and 0.023 for the deployed surface operator,
compared with 0.058 and 0.059 at the generating parameters.

This changes the interpretation of poor parameter recovery. The fitting is
working; the empirical density curves themselves fluctuate too much around the
generating-parameter curves at this trial count, and the optimizer follows those
fluctuations. The resulting curve is not always prediction-equivalent to truth:
median recovered-versus-generating curve RMSE was 0.054 for WNM and 0.059 for
the surface NN. Thus this panel shows finite-sample target variation and
consequential curve displacement, not merely optimizer failure or harmless
parameter relabeling.

## Batched optimizer follow-up

The faithful JAX L-BFGS-B port was tested on the same 60 frozen held-out
matched-KDE datasets with the same 32 seed-0 starts, float32 arrays,
`JAX_DEFAULT_MATMUL_PRECISION=highest`, and no truth start. After common CPU
rescoring, it was within 0.001 of the SciPy/port union on all 60 datasets; the
maximum gap was 1.79e-7. Median time fell from 5.323 to 0.214 seconds, a 25.75x
paired speedup. Median/global joint log-RMSE was 1.1338/1.3047 for the port and
1.1339/1.3045 for SciPy. The port is therefore endpoint-equivalent to the
32-start SciPy diagnostic, but this does not select a production search. The
actual cache-plus-one-polish winner was developed for the original analytic-
sign-mass operator; matched KDE makes the model curve depend on the empirical
KDE bandwidth, so its cache economics and search quality require a separate
development comparison.

That proper development comparison is now complete. The same 80-by-64 log
lattice plus one polish was rebuilt for every one of the 120 development
datasets because their pooled-SJ bandwidths are all distinct; unlike the legacy
operator, the matched-KDE lattice cannot be shared across datasets. A 32-start
port search reached the 0.001 common-score gate on 117/120 datasets. Increasing
the development budget to 64 starts reached it on 120/120, with maximum gap
4.17e-7, while the per-dataset lattice also reached 120/120 with maximum gap
0.000960. The 64-start port was materially better on three datasets and the
lattice was never materially better. Median time was 0.282 versus 18.546
seconds, a 65.9-fold advantage. The selected matched-KDE search is therefore
the 64-start float32 JAX port with `highest` matmul precision; the cache is not
part of this selected path.

## Conclusion

For the practical replacement question, matched-KDE WNM is recovery-competitive
with the existing surface pipeline on held-out data: global error is equal and
there is no clear paired difference. It is not more accurate in a general
sense, and neither method reliably recovers all three generating scales from
the density summary.

This comparison deliberately changes the whole fitting pipeline. It does not
isolate model family, because the surface was fitted with the existing density
operator rather than the matched-KDE operator. A target-matched surface refit
would still be required only if the scientific question is whether WNM itself,
rather than the WNM-plus-matched-objective pipeline, causes the difference.

The factor-1.5 calculation here treats the exact lower boundary `fit/truth =
2/3` as included. This gives 8.3% for the surface rather than the 5.0% in the
older comparison script, whose log-threshold calculation excluded two exact
boundary cases through floating-point rounding.

## Artifacts

The run and derived results are under
`wnm_density_matched_kde_heldout_gpu_v1/`. The key files are:

- `fits.csv` and `candidates.csv`;
- `matched_recovery_summary.csv`;
- `pipeline_arm_summary.csv`;
- `pipeline_parameter_summary.csv`;
- `pipeline_paired_metrics.csv` and `pipeline_paired_summary.csv`;
- `heldout_analysis_manifest.json`.
- `intermediate_curve_recovery_n450/`: annotated true, recovered, and empirical
  curves for the two intermediate held-out cases.

The held-out summary is reproduced by `analyze_density_matched_heldout.py`; the
curve figure is reproduced by `plot_intermediate_curve_recovery.py`.

## Notes for developers

The detailed development search comparison is in the external
`wnm_matched_kde_cache_vs_port32_port64_development_cpu_v1/` artifact. The
64-start selection postdates inspection of the held-out 32-start diagnostic,
so any 64-start held-out reuse must be described as descriptive rather than a
first prospective confirmation.
