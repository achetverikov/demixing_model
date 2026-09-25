# Test authoring instructions

These instructions apply to all tests in this repository.

Before adding a test, ask:

> What production bug would this catch that the existing lower-level suite would not?

## Classify the test first

State which layer owns the behavior: unit, contract, maintained integration, cross-repository integration, or legacy surface. Put primitive mathematics at the lowest authoritative layer and let higher layers test routing, persistence, and user-facing contracts rather than re-proving the same equations.

Use the repository markers consistently:

- unmarked/default: maintained WNM/product behavior;
- `integration`: sibling-repository or compiled-product integration;
- `legacy_surface`: historical surface-NN behavior;
- `slow`: intentionally expensive maintained checks;
- `gpu`: GPU-specific tests.

New historical-surface or CDB/cross-repository tests must not silently enter the maintained default suite. Development-only experiments belong on a development branch, not in the production suite.

## Cost discipline

A new test may run an optimizer only when the claim cannot be tested below the fitting layer. If a module already has a fitted artifact, compiled predictor, or other expensive fixture, consume that fixture instead of fitting again.

Prefer the smallest discriminating fixture: the minimum trial count, parameter count, feature-grid size, sample count, or numerical reference resolution that still separates correct behavior from the bug. One expensive run should support many assertions, including result schema, fingerprint, diagnostics, objective exports, and resume/refusal checks copied from the same result directory.

Invalid-family, stale-fingerprint, unknown-objective, condition-order, trial-count, and parameter-domain refusals should reach the validation branch before optimization.

Any test expected to take materially longer than ordinary unit tests must be marked `slow` and document why that cost is necessary.

## Numerical tests

Do not duplicate primitive mathematics. In particular:

- wrapped-normal mathematics belongs in `test_wrapped_mixture.py`;
- predictor API/domain/routing belongs in `test_prediction_contract.py`;
- scoring dispatch belongs in `test_wnm_scoring.py`;
- fit integration belongs in `test_continuous_fit.py`;
- CLI/persistence belongs in `test_continuous_cli.py`.

Prefer deterministic analytic or numerical references to Monte Carlo checks. If sampling is essential, document why, fix the seed, and use the smallest sample size compatible with a robust tolerance.

Calibration sweeps and negative controls used while designing a fixture normally belong in development notes outside the production branch, not in the maintained suite. Do not keep tests whose main purpose is to test the test oracle.

## Persistence and source inspection

Persistent reader/parser tests may deliberately receive malformed artifacts when a real trust boundary justifies it. Do not invent impossible states for pure functions, and keep one corruption test per meaningful persistent boundary rather than many hand-edited variants no supported writer can produce.

Do not use source-grep tests by default. Source inspection is allowed only when behavior cannot enforce a repository-wide rule, such as the dedicated mu1-axis lint.

Do not add transition-history guards to the maintained baseline. Once a migration is complete, preserve the supported behavior, not the implementation path that was used to reach it.
