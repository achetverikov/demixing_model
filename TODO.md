# Demixing Model TODO

## Motor-noise likelihood reproduction is ill-conditioned at the density floor

Status: B1 mitigation implemented in `model_fit_to_data/postprocess_fitted_likelihoods.py`.

When `sd_motor > 0`, `apply_motor_noise_with_precomputed_kernel()` convolves the
NN log-density surface in probability space, clips negative FFT artifacts to zero,
and returns `log(prob + 1e-10) + log_max`. Deep-tail cells can therefore pin to a
hard density floor. Near that floor, tiny backend/reduction differences decide
whether a trial is just above the floor or clipped to it, so reproducing the stored
`eval_likelihood_loss` can differ by whole nats per affected trial even with the
same fitted parameters.

Current mitigation:

- `postprocess_fitted_likelihoods.py` counts scored trials close to the motor-noise
  floor and writes `n_floor_trials`, `floor_tolerance`, and `within_tolerance` to
  `trial_loglik_checks.csv`.
- The strict `0.01` reproduction gate is preserved when there are no floor-region
  trials.
- `bias_model_comparison/pipeline/regenerate_all_fits.sh` uses the
  `within_tolerance` gate for demixing likelihood exports so resumability checks
  do not repeatedly fail on floor-explained differences.

Potential cleaner fixes:

- Export or recompute the exact fitted per-trial likelihoods during fitting, before
  backend-sensitive floor ambiguity enters a separate postprocessing pass.
- Replace the hard `log(prob + eps)` floor with a smoother, better-documented
  density floor and refit affected motor-noise models.
- Store enough per-fit diagnostics to distinguish real likelihood mismatches from
  floor-region trials without relying on a bounded slack rule.

## Expectation/mean-bias curve fitting can chase ill-defined circular means

Status: diagnostic finding; no code change yet.

The `expectation` objective fits subject-level binned mean-bias curves. Empirical
trials are binned by feature difference, and the model prediction is the circular
mean angle extracted from the predicted response surface. This can be fragile on
both sides:

- Subject-level target bins can have very few trials, so their empirical circular
  means are noisy even when report-level pooled bins are well sampled.
- For high fitted `sd_feat` values, model surface slices can be broad or nearly
  uniform. Their resultant length `R = sqrt(C^2 + S^2)` is then close to zero,
  making `atan2(S, C)` effectively undefined. The plotted/fitted mean angle can
  jump sharply because of tiny surface asymmetries rather than a stable predicted
  bias direction.

Concrete diagnostic from CSH2026 `color_hv_1 / high - low`, 20-sample checkpoint,
`expectation` optimizer:

- S11 has fitted `sd_feat1 = 200`, `sd_feat2 = 139`, `sd_spat = 5`. Neighboring
  model slices around `feat_diff = 28`/`30` have `R` around `0.002-0.004` and
  circular SD around `190-200` degrees, while their derived mean angles differ by
  about 24 degrees. The single-subject exported curve is therefore already very
  wiggly before any report-level averaging.
- Pooling model surface slices within the same 8-degree empirical bin reduces the
  visual wiggle, but `R` remains near zero, so the binned mean angle is still not
  interpretable.
- In first-moment space, the same fitted S11 model is near `(C, S) = (0, 0)`
  across bins, while the empirical bins often have much larger resultant lengths.
  This indicates that the model is predicting an almost uniform/broad response
  distribution, not a stable bias curve.
- A low-`sd_feat` comparison subject (S13: `sd_feat1 = 10`, `sd_feat2 = 25.5`,
  `sd_spat = 66.25`) has model `R` around `0.83-0.94`; its mean-angle curve is
  therefore much more interpretable. The pathology tracks broad/high-`sd_feat`
  model predictions rather than plotting or subject averaging alone.

Potential fixes / follow-ups:

- Export model first moments (`C = E[cos(theta)]`, `S = E[sin(theta)]`, `R`) for
  fitted curves, not only `mu_bias`.
- For curve objectives, compare empirical and model first moments with a resultant
  vector loss, e.g. `(C_model - C_emp)^2 + (S_model - S_emp)^2`, preferably using
  model moments pooled across the same feature-difference bins as the empirical
  target. This shrinks near-uniform predictions toward `(0, 0)` instead of
  chasing arbitrary angles.
- At minimum, flag or downweight low-`R` model predictions when plotting/scoring
  mean-bias curves.
- Treat `expectation`/mean-bias RMSE results as less reliable than likelihood or
  CRPS-style distributional objectives until this is resolved.
