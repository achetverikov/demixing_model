# Test Suite Improvement Plan

**Date:** 2026-09-25  
**Branch:** `dev/continuous-density-fixes`  
**Purpose:** establish a small, fast, high-signal maintained test baseline before the post-WNM architecture refactor.

This plan consolidates the repository-wide test audit, the architecture audit,
and the follow-up review of repeated fitting and oversized numerical fixtures.
The larger code reorganization remains defined by
[`ARCHITECTURE_AUDIT.md`](ARCHITECTURE_AUDIT.md).

The goal is:

> **The default suite should catch regressions in supported WNM behavior quickly,
> without repeatedly refitting the same model, re-testing the same mathematics at
> multiple layers, or spending normal CI time on transition-only and historical
> surface contracts.**

---

# 1. Test-suite principles

## 1.1 Keep tests for supported contracts

A test belongs in the maintained default suite when at least one is true:

- a supported user/API input can reach the branch;
- a persisted result/artifact can reach it;
- changing the behavior changes a fitted/exported/plotted/predicted quantity;
- it protects a scientific estimator definition;
- it verifies a current public entry point, result schema, fingerprint, or
  prediction contract.

Do not keep a test only because it once justified a migration.

## 1.2 Current curated datasets are not the whole input domain

Ordinary CSV input is supported. Therefore constant responses, identical values
inside a bandwidth pool, zero BWCRPS bias weights, antipodal observations in an
SD bin, and corrupt persisted fingerprints are still valid edge cases to guard.

The deletion question is:

> Can a supported input or persistent state reach this branch, and is the same
> behavior already tested at a lower authoritative layer?

## 1.3 One expensive run, many assertions

Expensive fitting should be fixture-based:

```text
one fit
  ├─ result schema
  ├─ fingerprint
  ├─ objective-score exports
  ├─ diagnostics
  └─ resume/refusal checks from copied result directory
```

Do not rerun the same fit once per field assertion.

## 1.4 Refusals should stop before optimization

Invalid family, stale fingerprint, unknown objective, condition-order mismatch,
trial-count mismatch, or invalid parameter-domain tests should reach the
validation branch directly rather than running L-BFGS-B first.

## 1.5 Primitive mathematics is tested once

Preferred ownership:

- wrapped-normal mathematics: `test_wrapped_mixture.py`;
- predictor API/domain/routing: `test_prediction_contract.py`;
- scoring dispatch: `test_wnm_scoring.py`;
- fit integration: `test_continuous_fit.py`;
- CLI/persistence: `test_continuous_cli.py`.

Higher layers test delegation and contracts, not the same equations again.

## 1.6 Historical/development tests stay opt-in

Markers remain:

- default: maintained WNM/product;
- `integration`: sibling-repository / compiled-product;
- `development`: transition, optimizer-comparison, recovery development checks;
- `legacy_surface`: historical surface-NN;
- `slow`: intentionally expensive maintained checks;
- `gpu`: GPU-specific.

---

# 2. Add test-authoring instructions for agents

Create **`tests/AGENTS.md`** before or during this cleanup and keep it under
version control.

It should instruct coding agents how new tests are added and reviewed. At
minimum it must require:

1. **Choose the layer first.** State whether the test is unit, contract,
   maintained integration, cross-repo integration, development, or legacy surface.
2. **Use the correct marker.** New development/surface/CDB tests must not silently
   enter the maintained default suite.
3. **Justify any real fit.** A new test may run an optimizer only when the claim
   cannot be tested below the fitting layer.
4. **Reuse expensive fixtures.** If a module already has a fitted artifact or
   compiled predictor fixture, new assertions should consume it instead of
   fitting again.
5. **Prefer the smallest discriminating fixture.** Use the minimum trial count,
   parameter count, feature-grid size, sample count, or numerical reference
   resolution that still separates correct from faulty behavior.
6. **No duplicate mathematics.** A predictor/plot/export test should not
   independently re-prove wrapped-normal mathematics already owned by the
   primitive suite.
7. **No source-grep tests by default.** Source inspection is allowed only when
   behavior cannot enforce the repository-wide rule, such as the dedicated
   mu1-axis lint.
8. **No tests of the test oracle.** Calibration sweeps and negative controls
   used while designing a fixture should normally be removed after the fixture
   is established.
9. **Persistent corruption tests need a real trust-boundary reason.** Do not
   invent impossible states for pure functions, but reader/parser code may
   deliberately receive malformed persisted artifacts.
10. **Monte Carlo tests need a reason and a budget.** Prefer deterministic
    analytic/numerical references. If sampling is essential, document why,
    select a seed, and use the smallest sample size compatible with a robust
    tolerance.
11. **Do not add transition-history guards to the maintained baseline.** Once a
    migration is complete, preserve behavior, not the implementation path used
    to get there.
12. **Document runtime-sensitive tests.** Any test expected to take materially
    longer than ordinary unit tests should be marked `slow` and say what makes
    the cost necessary.

The instructions should also recommend asking, before adding a test:

> "What production bug would this catch that the existing lower-level suite
> would not?"

---

# 3. Delete or rewrite low-value tests

## 3.1 Confounded/unrealizable backend fingerprint tests

Delete:

- `test_continuous_cli.py::test_a_hierarchical_run_cannot_resume_into_a_continuous_one`
- `test_run_fingerprint.py::test_the_two_backends_cannot_share_a_digest`

The former changes both backend and checkpoint family and accepts broad errors.
The latter constructs a WNM continuous fingerprint using the historical surface
checkpoint, which the supported fitter cannot run.

Retain realistic fingerprint checks: backend field, continuous spec, absent
lattice fields, changed start count/seed/settings, and actual same-family resume
refusal.

## 3.2 Degenerate density targets

Keep:

- one neutral target refusal, preferably
  `test_check_targets_fittable_names_every_offender`;
- one deterministic WNM-level refusal showing density rejects a flat target
  while a non-density objective remains usable.

Rewrite the currently skipping randomized WNM degenerate fixture to be
deterministic.

Delete duplicate constant-target/threshold/source-grep tests in
`test_density_ccc_objective.py` and duplicate refusal tests elsewhere.

If `test_density_degenerate_target_refusal.py` is retained, mark it
`legacy_surface` and reduce it to at most one surface refusal plus one
non-density-unaffected test.

## 3.3 Impossible density-asymmetry fixtures

Rewrite `test_density_objective_parity.py` fixtures to stay in `[-1,1]`.

Keep the float32 centering/cancellation regression, but use a high-mean,
low-variance legal curve such as approximately:

```text
0.95 + 0.01 * sin(...)
```

rather than `base + 50`.

## 3.4 Transition/source-text guards

Delete:

- `test_surrogate.py::test_promotion_is_one_switch`;
- cache source-inspection guards that duplicate behavior tests;
- source greps in `test_mu1_circular_axis.py` when direct behavior is already
  covered;
- `test_continuous_optimizer.py::test_the_finite_difference_step_sits_in_its_flat_region`.

Keep `test_mu1_axis_lint.py` as the deliberate repository-wide exception.

---

# 4. Keep reachable user-input edge cases

Retain:

- zero BWCRPS bias-weight refusal;
- identical-value bandwidth fallback/floor;
- antipodal empirical-SD bin behavior;
- corrupt fingerprint JSON refusal;
- meaningful persisted-artifact integrity checks.

These are reachable through supported CSV/persistent-state interfaces even when
current curated datasets do not exhibit them.

---

# 5. Fix motor-noise validation, then consolidate its tests

A production gap remains: concrete `sd_motor=` overrides and public prediction
CSV values do not all pass through the same validation as
`with_motor_noise()`.

## 5.1 Production fix

Validate concrete motor SD overrides and public prediction input:

- negative -> refuse;
- NaN/Inf -> refuse;
- zero -> legal;
- traced optimizer values remain governed by bounded optimization.

## 5.2 Test consolidation

Keep one authoritative predictor/public-input validation test.

After that, delete redundant negative-motor tests from plotting/recovery wrappers
unless those wrappers transform the value in a distinct way.

---

# 6. Eliminate repeated fitting

## 6.1 `test_continuous_cli.py`

Current file repeats the same two-condition WNM fit for many assertions.

Create one module-scoped baseline run, probably with **2 starts** and a much
smaller dataset.

Reuse it for:

- result existence and parameter shape/bounds;
- cross-objective score presence;
- fingerprint fields;
- fixed-zero motor;
- search diagnostics;
- recorded search configuration.

For resume/refusal tests, copy the result directory rather than refit it.

Keep separate real runs only for:

1. one normal baseline WNM fit;
2. one motor-enabled fit;
3. one 1-start subprocess CLI smoke.

Reduce the synthetic fixture from 400 trials/condition to roughly 40–60, provided
all target-building paths remain identified.

Target: approximately **3 fits instead of ~12**.

## 6.2 `test_continuous_fit.py`

Create one module-scoped `density_fit` fixture.

Use it for:

- result shape;
- per-condition fields;
- sum of condition losses;
- start diagnostics;
- best-start relation;
- bounds/search-box metadata.

Do not vary start count merely to inspect metadata.

Delete the fit-level feature-grid test because
`test_wnm_scoring::test_the_scorer_uses_the_grid_it_is_given` already tests the
actual contract directly.

Condition-order, trial-list, and unknown-objective refusals should remain
pre-optimization.

Motor application may require one additional fit.

Reproducibility-by-running-the-entire-fit-twice should either be deleted or
marked `slow`; deterministic start generation is already tested separately.

Target: **2–4 fits instead of ~15**.

## 6.3 `test_continuous_optimizer.py`

Never run 64 optimization starts to inspect defaults.

Replace `test_the_selected_production_configuration_is_recorded` with direct
inspection of constants/function defaults.

Reduce toy start counts:

- convex optimum: ~2;
- solver-cache argument update: ~1–2 each;
- bound hit: 1–2;
- start recording: 3;
- multimodal spread: smallest fixed count that still discriminates;
- NaN-region search: smallest count that hits both needed regions.

Simplify gradient coverage:

- one fixed direction;
- representative points below, around, and above the wrapped-normal branch
  switch;
- retain one motor-gradient case and one circular-moment gradient case.

---

# 7. Separate recovery development checks from default runtime coverage

`test_recovery.py` mixes cheap contracts with full recovery experiments.

## 7.1 Default baseline

Keep cheap tests for:

- `RecoveryCase` layout;
- deterministic data generation;
- diagnosis classification;
- summary/range-summary logic;
- parameter-family grouping;
- bound/railing bookkeeping;
- one very small `run_replicate` integration check if needed.

## 7.2 Mark as `development` or `slow`

Move/mark:

- noise-free 16-start recovery;
- panel runner that performs fits;
- serial-vs-parallel numerical recovery;
- start-budget sweeps;
- range-panel recovery experiments.

The maintained `standardized_recovery.py` should own current recovery
integration coverage; old panel workflows should not define ordinary pytest
runtime.

## 7.3 Reduce sampling fixtures

Current sampler tests use 20,000 draws repeatedly.

- statistical moment check: reduce to roughly 4,000–5,000 or mark `slow`;
- deterministic seeded equality (e.g. zero-motor override): tens of samples are
  enough;
- motor-widens-resultant check: use a few thousand, not 20,000.

---

# 8. Remove historical surface execution from default WNM tests

## 8.1 `test_pooled_bwcrps.py`

The scorer consumes probability surfaces; it does not require the surface NN.

Replace real surface-model predictions with deterministic synthetic normalized
circular distributions.

This preserves:

- pool-before-score semantics;
- unequal support;
- disjoint coverage;
- zero bias weights;
- seam wrapping;
- positional pairing;

while eliminating a historical checkpoint load/compile.

## 8.2 `test_plot_estimators.py`

This is primarily migration/golden parity against the historical surface path.

Mark the module `legacy_surface`.

Move any genuinely unique current WNM assertion to
`test_mixture_plot_curves.py` or `test_sd_estimators.py`; remove duplicates.

## 8.3 `test_public_prediction.py`

Keep WNM-default prediction in the maintained suite.

Mark the historical
`test_historical_surface_prediction_requires_explicit_source` test
`legacy_surface`.

## 8.4 Surface-specific surrogate tests

Move/mark as `legacy_surface`:

- surface checkpoint registry identity;
- surface search bounds;
- historical checkpoint reachability.

Keep generic resolver and current WNM artifact contracts in the maintained
suite.

---

# 9. Remove oversized/duplicate numerical references

## 9.1 `test_wrapped_mixture.py`

Delete the 400,000-sample Monte Carlo confirmation in
`test_motor_noise_adds_variance`.

Keep:

- exact variance addition;
- deterministic numerical circular-convolution comparison.

The deterministic reference is stronger and faster.

## 9.2 `test_prediction_contract.py`

Delete duplicated dense primitive tests:

- numerical motor convolution;
- analytic asymmetry against 0.02-degree quadrature.

Those belong to and are already covered by `test_wrapped_mixture.py`.

The predictor suite should test predictor routing, domain, identity, smoothing
contract, and operation delegation.

## 9.3 `test_sd_estimators.py`

Reduce Monte Carlo counts approximately:

- 1500 -> 300–500 for small-n correction;
- 400 -> ~100 for large-n invariance;

then set tolerances based on the resulting fixed-seed variance while keeping a
wide margin between correct and faulty behavior.

## 9.4 `test_curve_smoother.py`

`LENGTHS = [90, 91]` is currently unused.

Actually test odd/even padding, but do not form a full Cartesian product of every
sigma × every property × every length.

Use representative combinations:

- both lengths for length preservation;
- one length for constant/gain invariance;
- existing spike test for sigma-width ordering.

---

# 10. Speed up artifact/fingerprint tests

## 10.1 `test_postprocess_identity.py`

Digest resolution does not require real multi-megabyte checkpoints.

Use tiny temporary files and monkeypatch the installed checkpoint registry.

This removes repeated hashing and dependency on historical checkpoint presence
while testing the exact same identity logic.

## 10.2 Persistent reader corruption

Keep one test per meaningful trust boundary:

- unknown/stale format;
- checksum corruption;
- manifest/array shape disagreement;
- corrupt fingerprint.

Do not keep many hand-edited states that no supported format ever wrote unless
the reader explicitly promises to defend against them.

---

# 11. Consolidate overlapping test modules

## 11.1 CCC

Primitive CCC math in one module.

Fitter/scorer tests only check routing/refusal.

## 11.2 Wrapped mixture vs predictor contract

`test_wrapped_mixture.py` owns formulas.

`test_prediction_contract.py` owns API semantics.

## 11.3 SD/plotting

Separate:

- empirical/model SD estimator unit tests;
- WNM plotting bundle contract;
- historical surface parity as `legacy_surface`.

Avoid checking the same pooling mathematics in three files.

## 11.4 Recovery

Separate reusable recovery data/result primitives from actual recovery
experiments.

---

# 12. Execution phases

## Phase 1: remove obvious waste before first baseline

1. Delete confounded fingerprint/backend tests.
2. Delete source-grep/transition-only tests.
3. Rewrite density parity fixtures into legal range.
4. Fix motor validation and consolidate its tests.
5. Share the repeated CLI and continuous-fit fixtures.
6. Remove the 64-start metadata-only optimizer test.
7. Remove the 400k Monte Carlo check.
8. Mark surface-only plotting/public-prediction tests appropriately.
9. Create `tests/AGENTS.md`.

Then run:

```bash
PYTHONPATH=. python -m pytest
```

## Phase 2: profile actual runtime

Run pytest with durations:

```bash
PYTHONPATH=. python -m pytest --durations=30
```

Record the slowest maintained tests in a short section of `tests/README.md`.

Do not optimize tests that are already negligible.

## Phase 3: reduce remaining top runtime costs

Based on real durations:

- lower Monte Carlo sample counts;
- lower optimizer start counts;
- replace real model execution with deterministic fixtures where semantics allow;
- move genuinely expensive development validations to `slow` or `development`.

Run the maintained suite after each logical batch.

## Phase 4: architecture refactor

Only after the maintained baseline is green and reasonably fast, begin
`ARCHITECTURE_AUDIT.md` Phase B/C restructuring.

As modules move, migrate tests with their functional layer and apply
`tests/AGENTS.md` to prevent the suite from growing back into transition-style
duplication.

---

# 13. Acceptance criteria

Before starting the large architecture refactor:

- `PYTHONPATH=. python -m pytest` passes;
- `tests/run_smoke_wnm.sh` passes;
- default pytest performs no historical surface-NN training/search work;
- default pytest requires no sibling repository;
- no test runs the production 64-start optimizer merely to inspect metadata;
- repeated identical WNM fits in a module have been collapsed into shared
  fixtures;
- no maintained test uses an impossible density-asymmetry value;
- no maintained test duplicates wrapped-mixture numerical mathematics already
  covered by the primitive suite;
- recovery development checks is not part of ordinary runtime unless explicitly marked;
- the slowest maintained tests are known from `--durations`;
- `tests/AGENTS.md` exists and defines how future tests are classified,
  justified, minimized, and marked.

A useful outcome would be a maintained suite that is small enough to run
routinely during the upcoming refactor, while the slower integration/development/
legacy suites remain available for targeted verification.
