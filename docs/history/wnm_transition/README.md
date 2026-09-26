# WNM transition archive

This directory preserves the quantitative findings and decision record from the
2026 transition from the surface neural-network surrogate to the production
conditional wrapped-normal mixture (WNM).

These files are **historical evidence, not current run instructions**. Commands,
module names, artifact paths, and repository locations inside the archived
findings may reflect the transition-era tree, including the former
`continuous_density/` workspace. For current executable workflows, use:

- the root `README.md` for fitting and prediction;
- `surface_computation/README.md` for simulation and WNM training-data generation;
- `surrogate_training/wnm/README.md` for WNM training and packaging;
- `model_fit_to_data/Batch_Fit_Analysis_Pipeline_Documentation.md` for fitting,
  exports, plotting, and historical surface replay;
- `pretrained/README.md` for production artifact identity and domains.

## Archived findings

- `RESULTS.md` - transition-level quantitative summary.
- `RECOVERY_FINDINGS.md` - parameter-recovery findings and limitations.
- `SURFACE_BASELINE_FINDINGS.md` - historical surface baseline.
- `DENSITY_ALIGNMENT_FINDINGS.md` - density-target alignment work.
- `DENSITY_MATCHED_KDE_HELDOUT_FINDINGS.md` - matched-KDE held-out checks.
- `DENSITY_SOLUTION_EQUIVALENCE_FINDINGS.md` - equivalent-solution diagnostics.
- `SMOOTHED_EXP_ALIGNMENT_FINDINGS.md` - smoothed-expectation alignment.
- `WNM_DENSITY_FINDINGS.md` - WNM density-objective findings.
- `WNM_SMOOTHED_EXP_FINDINGS.md` - WNM smoothed-expectation findings.
- `WNM_LIKELIHOOD_OPTIMIZER_FINDINGS.md` - likelihood optimizer study.
- `WNM_BWCRPS_OPTIMIZER_FINDINGS.md` - bias-weighted CRPS optimizer study.

The implemented architecture decisions are summarized in the root `HISTORY.md`.
The original transition branches and removed workspace files remain recoverable
from git history.
