# Project context

This file records the current repository-level context for the ChatGPT project working on the demixing model refactor.

## Repository

- Repository: `achetverikov/demixing_model`
- Active refactor branch: `dev/continuous-density-fixes`
- Architecture plan: `ARCHITECTURE_AUDIT.md`

## Current architecture status

Phase B is complete:

- neutral fitting objectives and target construction were extracted from the historical surface optimizer;
- maintained WNM fitting no longer imports the historical surface optimizer;
- WNM-facing helpers were split out of `shared/utils.py`;
- maintained imports use package-qualified `model_fit_to_data.*` paths;
- result identity, file hashing, and WNM likelihood evaluation are centralized.

Phase C is in progress:

- production WNM runtime: `shared/wnm.py`;
- WNM prediction: `shared/prediction.py`;
- WNM fitting/scoring: `model_fit_to_data/`;
- WNM training and packaging: `surrogate_training/wnm/`;
- simulation/design/training-data generation: `surface_computation/`;
- transition findings/history: `docs/history/wnm_transition/`;
- development-only transition validation: `development/wnm_transition/`;
- development-only representation experiments: `development/wnm_representation/`.

## Architectural convention

Do not use `research/` as a repository layer. The entire repository is research software.

Use `development/` for code that exists only to develop, compare, validate, or diagnose model/surrogate alternatives and is not intended to be part of the production PR. Maintained runtime, fitting, prediction, simulation, and training code stays outside `development/`.

The pytest marker for these non-production checks is `development`, not `research`.

## Compatibility namespace

`continuous_density/` has been removed. Production WNM code now lives entirely in the maintained functional packages, and development-only transition/representation work lives under `development/`.

Generated `continuous_density/validation_outputs/` was removed before the namespace deletion.

## CI checkpoint

The maintained test suite was green on `b4e31b94c7a841e099e0e42a78f79d63147a58cd` before the subsequent Phase C development-file moves.

The Tests workflow is also present on `main` so `workflow_dispatch` can manually run the workflow against `dev/continuous-density-fixes`.
