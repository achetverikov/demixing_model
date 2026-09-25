# Demixing Model — TODO

This is the sole live work plan for this repository. Completed and superseded
work is recorded in `HISTORY.md`; result-specific transition evidence is archived under
`docs/history/wnm_transition/`.

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
- Unify the development and transition banded-metric frame schemas if banded
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
shared WNM runtime math, fitting/scoring, and explicitly development-only workflows.

The pre-refactor test cleanup and maintained WNM baseline have been completed.

Phase B is implemented:

- neutral curve losses, BWCRPS scoring, and empirical BWCRPS/binned-bias target
  builders live in `model_fit_to_data/objectives.py` and
  `model_fit_to_data/empirical_targets.py`;
- WNM engine construction no longer imports the historical surface optimizer;
- maintained path, behavioral-data, circular-smoothing, and empirical KDE/
  bandwidth helpers have been split out of `shared/utils.py`;
- `model_fit_to_data` is now a package and the maintained continuous-fitting
  path uses absolute package imports;
- result-key identity, streaming file hashing, and WNM likelihood
  export/rescoring each have one maintained implementation.

Phase C is now in progress. The production WNM runtime lives in `shared/wnm.py`;
training and packaging live in `surrogate_training/wnm/`; simulation,
continuous design, training-data generation, and legacy raw-sample import live
in `surface_computation/`. Their former `continuous_density/` modules are
compatibility shims only.

Next Phase C items:

- move WNM validation/transition experiments under `development/wnm_transition/`;
- move local representation experiments under `development/wnm_representation/`;
- redistribute or remove `continuous_density/tests/`;
- remove tracked generated validation outputs;
- delete `continuous_density/` once only compatibility/history content remains.

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
