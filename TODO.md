# Demixing Model — TODO

This is the sole live work plan for this repository. Completed and superseded
work is recorded in `HISTORY.md`; result-specific evidence remains in the
`continuous_density/*FINDINGS.md` files.

## 1. Finish retained WNM validation follow-ups

The runtime cutover is complete: WNM is the default surrogate for fresh
prediction and ordinary CSV fitting, compiled bundles remain available for
controlled model-comparison work, fitted-result plots/exports are WNM-aware, and
the public demo plus an end-to-end WNM smoke use the new path. `andriushchenko`
is tracked with the compiled comparison in `bias_model_comparison/TODO.md`, not
here.

- Close the retained WNM BWCRPS validation gaps while the production bundles
  are built: confirm the common 64-start policy on held-out cases, inspect its
  multi-condition, motor-noise, and real-data behavior, and use a large-sample
  or expected-target check to separate finite-sample objective displacement
  from model error. Assess curves by dissimilarity rather than pooled mean bias.
- Unify the research and transition banded-metric frame schemas if banded
  metrics become a maintained production product.

Pipeline and report consolidation is owned by
`bias_model_comparison/TODO.md`; canonical data and bundle work is owned by
`contextual_biases_database/TODO.md`.

## 2. Consolidate repository architecture after the WNM cutover

The detailed audit and proposed target structure are in
[`ARCHITECTURE_AUDIT.md`](ARCHITECTURE_AUDIT.md). In particular,
`continuous_density/` was a temporary transition workspace and should be
dissolved by moving its production pieces into the repository's functional
layers: simulator/training-data generation, surrogate training/packaging,
shared WNM runtime math, fitting/scoring, and explicitly historical research.

The audit's pre-test source hygiene is implemented. Before declaring the
baseline green, simplify and speed the suite according to
[`TEST_SUITE_IMPROVEMENT_PLAN.md`](TEST_SUITE_IMPROVEMENT_PLAN.md): remove
confounded/duplicated tests, collapse repeated fits into shared fixtures, reduce
oversized numerical fixtures, and add `tests/AGENTS.md` with rules for how new
tests are classified and added. Then run the maintained pytest suite and
`tests/run_smoke_wnm.sh`. Only after that should the architectural moves begin,
with tests between stages.

## 3. Decide the zero-width pooled-SD convention

The float32 upper clamp in pooled-SD reporting is currently inert, so a
distribution concentrated in one reporting cell yields exactly zero degrees.
Decide whether exact zero is the intended statistic or whether reported SD
should have a resolution floor. This is low priority because current real fits
do not reach the degenerate case, but changing it changes exported values and
therefore requires a contract/version bump and regression fixture.

## 4. Remove the repository-local secret exposure

The ignored `.env` file's local permissions were restricted from `0777` to
`0600` without reading or copying its values. Move runtime secrets out of the
repository and rotate credentials that may have been exposed. This is an
operational security task, independent of the legacy surface pipeline.

## Explicitly not planned

- Further surface-NN retraining, circular-axis migration, search refinement,
  browser integration, parity work, or promotion analysis. WNM is the default
  backend; the surface NN remains only for explicit historical reproduction.
- Revival of the retired hard-binned `expectation` objective.
- Cleanup of the surface-NN motor-noise density floor unless needed solely to
  reproduce a historical artifact.
- Joint `(mu1, mu2)` modelling. This remains unscheduled research, not migration
  work.
