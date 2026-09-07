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

## 3. Parameter recovery and search selection for the K12 transition

Open: implement replicated parameter recovery from independent DM simulator
responses, using the existing empirical fitting targets. Forward-prediction CCC/MAE
checks do not establish recovery of the generating noise parameters.

Compare hierarchical, cached exhaustive density, multistart gradient, and a simple
grid-seeded gradient search on identical K12 problems before selecting a default.
Report parameter recovery, identifiability, held-out dissimilarity-dependent
predictions, search failures, and cold/warm costs including cache amortization.
Separate optimizer comparisons from K12-versus-surface-NN comparisons.

The proposed recovery design and transition sequence are recorded with the generated
experiment artifacts; no production optimizer replacement is selected yet.

**Select the gradient start budget here, not before** (decision, 2026-09-06).
`continuous_optimizer.minimize_continuous` defaults to a placeholder `n_starts`
and it must not be read as a tuned value. The budget matters: on a
three-condition fixture the density objective's loss spread across six converged
starts was 1.03, start losses running 0.96 to 2.00, so five of six starts landed
in worse basins and the budget, not the objective, chose the answer. Tuning it on
whatever cases are to hand would select it on its own benchmark. Choose it on the
development groups of the recovery panel, freeze it, then score the held-out
groups -- and record the frozen value with the results, since a fit's parameters
mean something different at a budget that reliably finds the basin than at one
that does not.

**Test float64 as part of the recovery panel.** The repo runs JAX in its default
float32, so `continuous_optimizer` computes values and gradients at float32 and
widens them only at the SciPy boundary. That floors the achievable convergence
tolerance at roughly 1e-8 relative: recovery of a known optimum on a synthetic
objective lands at 1.5e-7, which is the arithmetic limit rather than a search
failure. Whether that floor costs anything *scientifically* is unknown and is a
question only recovery can answer -- a parameter whose recovery RMSE is dominated
by finite-trial variation will not care, while a weakly identified one on a flat
ridge might.

Run at least one recovery arm twice, identical but for `JAX_ENABLE_X64=1`, and
compare recovered parameters, per-parameter bias/RMSE, convergence status and
boundary hits, plus wall time and memory. Report it as its own axis, not folded
into the search comparison.

Two constraints on acting on the result. x64 is a global JAX flag, not a
per-module one, so adopting it changes the surface backend's arithmetic too --
either the surface parity fixtures get re-verified under x64, or adoption waits
until that backend is retired (transition plan step 7). And if x64 does improve
recovery, that is evidence about the objective's conditioning as much as about
the optimizer: a parameter that only becomes recoverable at double precision is
weakly identified, and the scientific claim resting on it should say so rather
than quietly relying on the extra digits.

## Notes for developers

The detailed plan is `continuous_density/TRANSITION_PLAN.md`.
It is an external development artifact, not shipped or expected in a normal checkout.
