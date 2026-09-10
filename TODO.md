# Demixing Model — TODO

Consolidated 2026-07-17 (single source file, `TODO.md`, rewritten in place — nothing
else to merge in for this repo). Fully-implemented mitigations are summarized tersely;
only genuinely open follow-ups are kept in detail.

## 1. Motor-noise likelihood floor (mitigated; cleaner fixes still open)

`apply_motor_noise_with_precomputed_kernel()` convolves the NN log-density surface in
probability space, clips negative FFT artifacts to zero, and returns
`log(prob + 1e-10) + log_max`. Deep-tail cells pin to a hard density floor, so near that
floor tiny backend/reduction differences can flip a trial across the floor and shift
reproduced `eval_likelihood_loss` by whole nats even with identical fitted parameters.

**Done:** `model_fit_to_data/postprocess_fitted_likelihoods.py` counts near-floor trials
and writes `n_floor_trials`/`floor_tolerance`/`within_tolerance` to
`trial_loglik_checks.csv`; the strict `0.01` reproduction gate still applies when there
are no floor-region trials; `bias_model_comparison/pipeline/regenerate_all_fits.sh`
gates demixing-likelihood resumability checks on `within_tolerance` so floor-explained
diffs don't repeatedly fail the gate. (Verified current in code, 2026-07-17.)

**Open — potential cleaner fixes** (none started):

- Export or recompute exact fitted per-trial likelihoods during fitting itself, before
  backend-sensitive floor ambiguity enters a separate postprocessing pass.
- Replace the hard `log(prob + eps)` floor with a smoother, better-documented density
  floor, and refit affected motor-noise models.
- Store enough per-fit diagnostics to distinguish real likelihood mismatches from
  floor-region trials without relying on a bounded slack rule.

## 2. `expectation` objective chases ill-defined circular means at high `sd_feat`

Diagnostic finding only — no code change yet.

The `expectation` objective fits subject-level binned mean-bias curves by extracting the
circular mean angle from the predicted response surface. This breaks down for two
compounding reasons: subject-level bins can be trial-sparse (noisy empirical circular
means even when pooled bins are well sampled), and for high fitted `sd_feat` the model's
surface slices go broad/near-uniform, so resultant length `R = sqrt(C^2+S^2) -> 0` and
`atan2(S,C)` becomes numerically undefined — the fitted mean-angle curve can jump sharply
from tiny surface asymmetries rather than tracking a stable bias.

Concrete case (CSH2026 `color_hv_1 / high - low`, 20-sample checkpoint, `expectation`
optimizer): S11 (`sd_feat1=200`, `sd_feat2=139`, `sd_spat=5`) has model `R` around
0.002-0.004 near `feat_diff=28/30` with derived mean angles ~24° apart, and pooled first
moments near `(C,S)=(0,0)` while empirical bins have much larger resultant length — i.e.
the model predicts an almost-uniform response, not a stable curve. Contrast: S13
(low `sd_feat1=10`, `sd_feat2=25.5`, `sd_spat=66.25`) has model `R` around 0.83-0.94 and
an interpretable curve. The pathology tracks broad/high-`sd_feat` predictions, not
plotting or subject-averaging artifacts.

**Open — potential fixes/follow-ups** (none started; no `C`/`E[cos]`/`E[sin]`/`R` export
exists yet anywhere in `model_fit_to_data/`, confirmed 2026-07-17):

- Export model first moments (`C = E[cos(theta)]`, `S = E[sin(theta)]`, `R`) for fitted
  curves, not just `mu_bias`.
- For curve objectives, score empirical vs. model first moments directly (e.g.
  `(C_model-C_emp)^2 + (S_model-S_emp)^2`, model moments pooled across the same
  feature-difference bins as the empirical target) — this naturally shrinks near-uniform
  predictions toward `(0,0)` instead of chasing an arbitrary angle.
- At minimum, flag/downweight low-`R` model predictions when plotting or scoring
  mean-bias curves.
- Until resolved, treat `expectation`/mean-bias RMSE results as less reliable than
  likelihood- or CRPS-style distributional objectives.

## 3. Parameter recovery and search selection for the K12 transition (resolved)

The focused development and held-out recovery panels are complete. Production
WNM fitting uses the pinned JAX L-BFGS-B port at commit `0350da1`: 64
deterministic starts, two sequential batches of 32, float32 arrays,
`highest` matmul precision, and seed 0. Every endpoint, convergence status,
boundary hit, and optimizer setting is persisted and fingerprinted.

The precision diagnostic identified default GPU TF32 matmuls—not nominal
float32 itself—as the material source of device-dependent likelihood basins.
`highest` matmul precision recovered the selected result without global x64,
which would also alter the surface backend.

The first paired representative real-data fit is complete for all four retained
objectives. WNM uses feature bounds `[2.5, 200]`, the deployed surface NN keeps
`[5, 200]`, and both use spatial bounds `[5, 200]`; boundary hits are reported.
Common WNM rescoring favored the WNM fits on likelihood, BWCRPS, and matched
smoothed expectation, while the surface-derived density parameters were modestly
better on matched density. This preserves the known density basin sensitivity
rather than overstating 64-start search as globally complete. Detailed rationale
and remaining rollout work are in `continuous_density/OPEN_DECISIONS.md` and
`continuous_density/TRANSITION_AUDIT.md`.

## Notes for developers

The detailed plan is `continuous_density/TRANSITION_PLAN.md`.
It is an external development artifact, not shipped or expected in a normal checkout.
