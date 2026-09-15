# Demixing Model — TODO

This is the sole live work plan for this repository. Completed and superseded
work is recorded in `HISTORY.md`; result-specific evidence remains in the
`continuous_density/*FINDINGS.md` files.

## 1. Complete the bundle-native WNM production cutover

- Finish the catalogued WNM production fits and exports needed by the compiled
  cross-family comparison.
- Pass the full compiled-product coverage, checksum, row-identity, likelihood,
  and objective-replay gates for every production bundle.
- Once the compiled comparison has passed, remove or explicitly historicalize
  DM input paths that still construct empirical semantics from prepared CSVs.
  Production fitting must consume compiled bundles; any retained CSV importer
  must first produce and validate such a bundle.
- Update user-facing run documentation when the compiled runner becomes the
  only production entry point.
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

## 2. Decide the zero-width pooled-SD convention

The float32 upper clamp in pooled-SD reporting is currently inert, so a
distribution concentrated in one reporting cell yields exactly zero degrees.
Decide whether exact zero is the intended statistic or whether reported SD
should have a resolution floor. This is low priority because current real fits
do not reach the degenerate case, but changing it changes exported values and
therefore requires a contract/version bump and regression fixture.

## 3. Remove the repository-local secret exposure

The ignored `.env` file is still mode `0777` and contains credential-shaped
configuration. Without reading or copying its values: move runtime secrets out
of the repository, restrict permissions, and rotate credentials that may have
been exposed. This is an operational security task, independent of the legacy
surface pipeline.

## Explicitly not planned

- Further surface-NN retraining, circular-axis migration, search refinement,
  browser integration, parity work, or promotion analysis. The surface NN is a
  legacy backend being replaced by WNM.
- Revival of the retired hard-binned `expectation` objective.
- Cleanup of the surface-NN motor-noise density floor unless needed solely to
  reproduce a historical artifact.
- Joint `(mu1, mu2)` modelling. This remains unscheduled research, not migration
  work.
