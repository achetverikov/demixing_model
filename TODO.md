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
