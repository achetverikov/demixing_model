# WNM density search and surface comparison

Artifact paths and analysis scripts named below are relative to
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`.
Generated artifacts are external and are not shipped with or expected in a
normal checkout.

## Decision

For the frozen n=100 single-condition panel, use an 80-point log feature axis
over 2.5--200 degrees and a 64-point log spatial axis over 5--200 degrees.
Build the 409,600 WNM density-asymmetry curves once, scan them exhaustively, and
polish the lattice winner with one SciPy L-BFGS-B start. A finite point returned
after a failed convergence status remains eligible by its canonical rescore;
the failed status is retained in the candidate diagnostics.

The cache is useful for batches, not for a single fit. The GPU build took about
21 seconds, the raw float32 curves occupy about 141 MiB, and median scan plus
polish time was 0.63 seconds per development dataset. Rebuilding for every
single subject would be slower than 32-start L-BFGS-B; reuse across a panel
amortizes the build rapidly. Persistent-cache loading was not needed to answer
the search question and has not yet been benchmarked.

## Optimizer panel

The development comparison included 32-start serial SciPy L-BFGS-B, 32-start
BBZ JAX-BADS, and the existing production hierarchy, all rescored by the same
WNM density evaluator on CPU. SciPy was within 0.001 of the three-arm union on
117/120 datasets, JAX-BADS on 112/120, and the hierarchy on 82/120. Their worst
gaps were 0.116, 0.123, and 0.077 respectively. The failures were complementary.

A 2.5-degree linear curve lattice plus polish was inadequate (106/120 within
0.001; worst gap 0.040), because low-noise optima can be much sharper than that
spacing. Log allocation resolved this without increasing the curve count. The
selected log lattice plus polish was within 0.001 of the optimizer-panel union
on 120/120 development datasets, with worst gap 0.000735 and p90 gap 0.000020.
This is the protocol used unchanged on the 60 held-out datasets.

## Existing surface-NN comparison

Both arms were scored against the same sampled-data empirical curves. The
surface arm is the existing one-degree exhaustive density-cache fit; it was not
retuned. Whole-curve medians are:

| Split | Arm | Joint log RMSE | Mean-bias RMSE | Bias-SD RMSE | Asymmetry CCC | Asymmetry RMSE |
|---|---|---:|---:|---:|---:|---:|
| Development | WNM | 1.187 | 1.401 | 6.672 | 0.881 | 0.0337 |
| Development | surface NN | 1.093 | 1.524 | 6.947 | 0.887 | 0.0311 |
| Held out | WNM | 1.472 | 1.820 | 14.688 | 0.828 | 0.0296 |
| Held out | surface NN | 1.306 | 1.854 | 11.567 | 0.844 | 0.0262 |

The density objective is weakly identifying for both model families. WNM had
lower bias-SD RMSE in 60% of development datasets, but that advantage did not
replicate held out (45%). Surface NN had lower asymmetry RMSE in 69% of
development and 72% of held-out datasets. Thus the WNM density fit is usable and
its search is settled, but it does not improve on the existing surface density
fit in this panel.

The four dissimilarity bands tell the same qualified story. On development,
WNM improves median bias-SD RMSE in all four bands and improves mean-bias RMSE
outside 18 degrees, while surface is better for asymmetry in three of four
bands. Held out, the small bias-SD advantage disappears and surface is clearly
better for asymmetry in the 120--180 degree band. No conclusion here is based
on mean bias pooled across dissimilarities.

## Target-alignment follow-up

The empirical density curve is constructed through a wrapped KDE, whereas the
current WNM branch uses analytic signed mass. A later diagnostic applied the
same KDE operator to both branches. On development this reduced large-error
tails without shifting typical parameter recovery. On the frozen held-out
panel, matched-KDE WNM and the existing surface pipeline had global joint
log-RMSE 1.305 and 1.306; their paired difference was inconclusive. The matched
variant is therefore viable but not a demonstrated recovery improvement. It
remains diagnostic until a production search is selected for that operator; see
`DENSITY_ALIGNMENT_FINDINGS.md` and
`DENSITY_MATCHED_KDE_HELDOUT_FINDINGS.md`.

## Artifacts

- `wnm_density_development_panel_cpu_v1/`: common-rescored SciPy, JAX-BADS,
  and hierarchy panel.
- `wnm_density_log80x64_development_cpu_v3/`: selected development fits and
  paired surface comparison, including per-band metrics.
- `wnm_density_log80x64_heldout_cpu_v1/`: frozen held-out fits and paired
  surface comparison, including per-band metrics.
- `probe_wnm_density_lattice.py`, `assemble_density_lattice.py`, and
  `compare_wnm_surface_density.py`: pinned runners and analyses.

The subsequent smoothed-expectation optimizer and alignment stages are complete;
see `WNM_SMOOTHED_EXP_FINDINGS.md` and
`SMOOTHED_EXP_ALIGNMENT_FINDINGS.md`.
