# WNM bias-weighted-CRPS optimizer findings

Date: 2026-09-08

Run directories and CSV names are relative to
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`,
where the analysis scripts named here also live. This document is versioned with
the code; the external artifacts it cites are not shipped with or expected in a
normal checkout.

Selection used only the 120 development datasets, following
`WNM_BWCRPS_OPTIMIZER_PLAN.md`. At selection time the held-out tuples had not
been opened. Three arms were compared: 32-start serial SciPy L-BFGS-B, 32-start
JAX-BADS, and the production surface search geometry driving the same BWCRPS
objective. PyBADS was run only on one disputed dataset, as predeclared, rather
than as a fourth arm.

## Result: 32-start serial L-BFGS-B, as for likelihood

Every arm's winner was rescored through one common scorer, on each device
separately (`wnm_bwcrps_development_cpu_v1/`, `wnm_bwcrps_development_gpu_v1/`).

| Arm | CPU: median gap / in gate | GPU: median gap / in gate | Max gap (CPU/GPU) | Median joint log-RMSE | Median s |
|---|---:|---:|---:|---:|---:|
| scipy-lbfgsb-32 | 0.0000 / 100.0% | 0.0001 / 98.3% | **0.0005 / 0.0018** | **0.2586** | **20.4** |
| bbz-jax-bads-32 | 0.0001 / 90.8% | 0.0000 / 92.5% | 0.0617 / 0.0613 | 0.2627 | 21.0 |
| surface-production-hierarchical | 0.0004 / 81.7% | 0.0004 / 83.3% | 0.0034 / 0.0034 | 0.2984 | 22.2 |

The median ordering between L-BFGS-B and JAX-BADS is again decided by the
scoring device -- JAX-BADS is lower on 1 of 120 scored on CPU and on 100 of 120
scored on GPU -- but the differences involved, about 7e-05, are below the
device shift itself (median 1.0e-04, p90 7.6e-04, max 2.4e-03), so that ordering
carries no information either way.

What is device-robust is the worst case. JAX-BADS misses by up to 0.062 under
both devices, twenty-five times the largest device shift, while L-BFGS-B's worst
case is 0.0005 on CPU and 0.0018 on GPU. Parameter recovery is a tie (paired
difference -0.0006, JAX-BADS better on 63 of 120). L-BFGS-B is also the cheapest,
at 1,615 median evaluations against 11,605 and 288,006.

## The objective, not the search, is what limits BWCRPS recovery

`score_bwcrps_at_truth.py` evaluates the deployed objective at the generating
parameters. This distinguishes a search that failed to reach the minimum from an
objective whose minimum is not at the truth, which optimizer traces alone cannot.

The objective prefers the fitted parameters over the truth on essentially every
dataset -- 100% for L-BFGS-B at all three trial counts, 92.5% to 100% for the
hierarchy -- so the searches are succeeding and the optimum is displaced. The
displacement shrinks with data, from a median 0.090 at 180 trials to 0.019 at 450
and 0.018 at 900, which reads as finite-sample rather than as a fixed property of
the target construction.

The scales are what matter for how much optimizer choice can buy:

```
objective displacement from truth   0.019    (median)
spread between optimizer arms       0.0001   (median gap)
device rescore shift                0.0024   (max)
```

The displacement is about 153 times the typical difference between arms. For
BWCRPS the search choice is close to irrelevant beside the objective's own
displacement from truth, which is the opposite emphasis from likelihood, where
the arms differed materially and the target was better behaved.

The disputed case codex identified, `broad_3_seed0_n450`, is an instance rather
than an exception: the objective scores 76.503 at the truth (120, 160, 15) and
75.502 at the fitted parameters, a full 1.0 better at parameters whose joint
log-RMSE is 1.46. L-BFGS-B, the hierarchy, and an independent PyBADS run all
reach 75.5008 to four decimals, so three implementations agree on that basin.
JAX-BADS stops 0.025 short of it, at parameters that happen to be closer to truth.

## Correction to the 24-dataset pilot reading

The pilot reported the hierarchy recovering parameters much worse than the other
arms, median joint log-RMSE 0.8195 against 0.29. That does not survive the full
panel: on 120 datasets it is 0.2984 against 0.2586, a paired difference of
+0.0015 with the hierarchy worse on 66 of 120. The pilot's 24 datasets
over-weighted the cases where the hierarchy fails, exactly as the 24-dataset
likelihood subset did. The hierarchy is still last on every axis here, but by a
small margin rather than a categorical one.

## Held-out confirmation

After the development-only choice was frozen, the selected 32-start serial
L-BFGS-B run was executed once on all 60 held-out datasets. Every fit had at
least 29 of 32 converged starts (the median was 32), and the largest per-start
rescore discrepancy was `4.7e-4`.

The WNM winner was then compared with the already-fitted surface-NN arm by
rescoring both parameter vectors through the same WNM BWCRPS evaluator. WNM had
the lower common score on all 60 datasets, with median WNM-minus-surface
difference `-0.00794` (Q25 `-0.0421`, Q75 `-0.00437`). The advantage was present
at 180, 450, and 900 trials and in all four regimes, although it was smallest
in the broad cases.

Parameter recovery moved in the same direction: median joint log-RMSE was
`0.2693` for WNM versus `0.3454` for the surface fit, and factor-1.5 recovery
was 50.0% versus 43.3%. Curve recovery remained mixed. WNM's median CCC/RMSE
was `0.423/0.271` versus `0.375/0.277` for mean bias, `0.223/0.860` versus
`0.102/0.991` for bias SD, and `0.351/0.0544` versus `0.369/0.0518` for
asymmetry. Thus the held-out result supports WNM on the primary distributional
and parameter criteria without claiming a uniform curve-level win.

The generated rows and summaries are under
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/wnm_bwcrps_scipy32_heldout_cpu_v1/`.
The surface comparison uses the WNM scorer for both arms; the surface model's
native BWCRPS values are not used as a cross-family score.

## What this does not establish

The comparison covers one frozen n=100 single-condition panel with motor noise
fixed at zero. The finite-sample reading of the objective's displacement rests
on three trial counts at one sample size and would be better supported by a
large-sample or expected-target diagnostic, which has not been run. Multi-
condition, motor-noise, and real-data confirmation remain open.
