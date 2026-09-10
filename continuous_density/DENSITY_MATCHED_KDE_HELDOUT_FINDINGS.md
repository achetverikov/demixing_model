# Held-out matched-KDE density recovery

Date: 2026-09-09

Artifact paths and analysis scripts named below are relative to
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`.

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

The analysis is reproduced by `analyze_density_matched_heldout.py`.
