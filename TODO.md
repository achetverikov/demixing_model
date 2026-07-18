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
