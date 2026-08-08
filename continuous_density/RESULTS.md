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
