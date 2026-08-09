# Continuous-density prototype results

Results below are from the 100-observation simulator and production checkpoint,
using fresh off-grid raw simulations. Artifacts are in
`/workspaces/demixing_model/results/continuous_density/`; they are not committed.

## Primary result

The direct conditional wrapped-normal mixture represents the Demixing Model
likelihood at least as accurately as the current simulated-surface → KDE → NN
pipeline on held-out simulator outcomes. The K=12 model trained on 16,384
parameter points × 500 outcomes achieved:

| validation design | conditional density NLL | production NN NLL | cases won | mean NLL difference |
|---|---:|---:|---:|---:|
| 108 scattered points, 100k outcomes each | 3.530510 | 3.627022 | 97.7% | -0.096512 |
| 15 difficult trajectories, 100k outcomes each | 3.866589 | 3.997955 | 89.7% | -0.131366 |

Lower NLL is better. On trajectories, wins were 100% for peaked, multimodal,
and seam cases, 96.5% for asymmetric cases, and 52.1% for essentially flat
cases. Independent raw-reference repeats put the mean absolute per-case Monte
Carlo NLL variation at 0.00264 (points) and 0.00283 (trajectories), much smaller
than the overall advantage over the production surrogate.

Reporting-grid KDE KL/L1 can favor the production model even when raw held-out
NLL strongly favors the conditional mixture. This is expected: the production
target is itself KDE-smoothed, and the reporting KDE visibly broadens narrow raw
distributions. Raw-outcome NLL is therefore the primary accuracy measure.

## Capacity and data scale

On the common 100k scattered reference, mean NLL was 3.536577, 3.531723,
3.530681, and 3.530510 for K=2, 4, 8, and 12. Paired mean excess NLL relative to
K=12 was 0.006068 (95% CI ±0.002594), 0.001213 (±0.000491), and 0.000171
(±0.000118) for K=2, 4, and 8. K=12 is the accuracy choice; K=8 is a
near-equivalent smaller choice.

At K=8, increasing training from 16,384 × 500 outcomes to either 65,536 × 500
or 16,384 × 2,000 changed raw NLL only at about the fourth decimal place,
with inconsistent case-wise wins. Those changes were comparable to changing
the training seed. The 16,384 × 500 corpus is adequate for this architecture;
larger raw corpora are not the current bottleneck.

The K=12 checkpoint is 129 kB, versus 9.18 MB for the production checkpoint
(about 71× smaller). Warmed GPU timing was 0.294 ms for one-condition prediction
and 0.300 ms for a 1,000-trial likelihood evaluation (100 repeats).

## Empirical fit demonstration

The Fischer/Whitney orientation condition contains 122 trials below the planned
2° lower training bound. They were explicitly excluded rather than clamped or
silently extrapolated, leaving 6,526 trials. Eight-restart independent fits gave:

| representation | total NLL | fitted `(sd_feat1, sd_feat2, sd_ident)` |
|---|---:|---|
| K=12 conditional density | 23937.348 | (34.237, 123.475, 47.636) |
| production surface NN | 23928.656 | (32.538, 199.823, 56.194) |

The conditional density was worse by 8.692 total NLL, or 0.00133 per trial:
near parity, but not a win. The production fit also placed `sd_feat2` almost on
the 200° bound, indicating a weak or boundary parameter direction. This single
demonstration establishes that continuous trial-level fitting works; it is not
a parameter-recovery or model-selection study.

## Unequal-variability mean-bias curves

A structured raw reference crossed the canonical feature-SD pairs from
`{10,20,30,60}°`, 90 dissimilarities, and spatial d′ `{0.5,1,2}` at 100k EM
outcomes per parameter row (270 million simulations total). Spatial d′ follows
`40 / sd_ident`, hence these levels correspond to `sd_ident={80,40,20}°`.

Across both components and all 5,400 curves, the K=12 conditional density had
0.110° mean absolute and 0.204° RMS circular-mean error. Mean absolute error
was 0.156°, 0.111°, and 0.063° at d′ 0.5, 1, and 2. The largest localized
miss was 2.32° for the higher-noise component at `(sd_feat1,sd_feat2,d′) =
(30°,60°,0.5)`, where the model underestimates the raw repulsive trough.

Against the same raw curves, the production NN had 0.137° mean absolute and
0.229° RMS error. The conditional density had lower pointwise absolute error
in 63.4% of comparisons and lower trajectory-average error for 46/60 complete
component curves. Production was better in the main localized failure above:
trajectory MAE was 0.241° for production versus 0.501° for the conditional
density. Thus the direct model is better overall but does smooth away part of a
specific strong repulsive trough. Separate three-way grid and averaged figures
for every d′ level are under
`results/continuous_density/uev_raw100k_vs_models/`.

Response variability was compared as circular SD from the same first circular
moment. The conditional density had 0.223° MAE and 0.364° RMSE, versus 0.309°
and 0.436° for production. It had lower pointwise absolute error in 68.8% of
comparisons and lower trajectory-average error for 48/60 component curves.
Both representations were least accurate at d′=0.5: conditional-density MAE
was 0.414° for component 1 and 0.448° for component 2, falling to 0.051° and
0.105° at d′=2. The same results directory contains separate response-SD grid
and averaged plots for all three d′ levels.

### Local capacity diagnostic

The worst shared trajectory `(sd_feat1,sd_feat2,d′,component) =
(30°,60°,0.5,2)` was refit using the first 50k raw outcomes at every
dissimilarity and evaluated on the untouched other 50k. A local fine-tune of
the existing K=12 MLP reduced maximum mean-bias error from 2.317° to 0.763°
and maximum response-SD error from 1.916° to 1.053°. Held-out NLL improved
from 4.692995 to 4.692046 and improved at 88/90 dissimilarities. At the original
36° failure, mean bias moved from -10.416° to -13.180° against raw -12.733°.

The unchanged mixture size and unchanged network architecture can therefore
represent the trough. The failure is attributable to global training
allocation/optimization smoothing over this localized regime, rather than a
K=12 family-capacity limit. Independent per-dissimilarity mixtures also
improved held-out NLL, but their moment curves were noisy because they lack the
smooth parameter map and were fit from only 50k outcomes each.

## Remaining approximation error

- **Simulator Monte Carlo:** independently measured by two 100k references;
  per-case NLL uncertainty remains a few thousandths and dominates very small
  distinctions such as K=8 versus K=12 in some cases.
- **Conditional-family capacity:** the reproducible K=2/4 penalties and the
  small K=8→12 gain show finite mixture capacity matters, but little remains at
  K=12 on the tested designs.
- **Parameter-to-density network/training:** residual structured calibration
  errors, especially on high-dispersion trajectories, can come from the MLP and
  optimization. Increasing the training corpus did not materially improve raw
  NLL, so more simulations alone are unlikely to fix them.
- **EM:** the prototype deliberately emulates outcomes of the existing finite-
  sample circular EM and matching procedure. Any bias, instability, or model
  approximation introduced by EM remains part of the target; it is not removed
  by changing the density representation.

The prototype therefore succeeds as a direct continuous likelihood
representation, but the current production model should remain untouched until
empirical fits and parameter recovery are compared across datasets and the
2° lower-domain limitation is resolved if zero/one-degree trials must be used.
