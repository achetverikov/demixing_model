# Architecture Audit: WNM Cutover and Repository Consolidation

**Date:** 2026-09-25  
**Audited branch:** `dev/continuous-density-fixes`  
**Audited head at start:** `8a3c05104d5ee3b5a6ae5a40a38fc9d4aca407b5`

This document records an architecture audit of the Demixing Model repository
after the wrapped-normal-mixture (WNM) runtime cutover. It is an audit and design
reference, not a second live work plan. Accepted execution items should be tracked
in the root `TODO.md`. The concrete pre-refactor test cleanup, deletion, and
runtime-reduction work is specified separately in
[`TEST_SUITE_IMPROVEMENT_PLAN.md`](TEST_SUITE_IMPROVEMENT_PLAN.md).

## Executive summary

The scientific/runtime WNM transition is substantially complete: WNM is the
default surrogate for fresh predictions and ordinary CSV fitting, compiled
bundles remain available for controlled model-comparison work, fitted-result
plots and exports are WNM-aware, and the public demo and WNM smoke path use the
new model.

The repository architecture has not yet caught up with that runtime state.
Three eras coexist in the active tree:

1. the original simulator + averaged-surface + surface-NN pipeline;
2. the temporary `continuous_density/` WNM research/transition workspace;
3. the current WNM fitting, scoring, prediction, export, and recovery path.

The principal architectural problem is therefore not the WNM mathematics. It is
that the new runtime was added beside the old architecture instead of being
fully redistributed through the repository's functional layers. Production WNM
code still reaches into historical surface modules for neutral objective helpers,
the maintained WNM model class still lives in a package that labels itself an
experimental prototype, tests mix product, integration, research, and historical
surface contracts, and several numerical/utilitarian functions have multiple
implementations.

The intended end state should have **no production import from
`continuous_density/`**. That directory was temporary and should disappear once
its components have been moved to functional homes.

## Architectural principles for the cleanup

1. **Organize by function, not by transition history.** Simulation, surrogate
   training, fitting/scoring, prediction, visualization, and research validation
   should be separate layers, as they were conceptually for the surface approach.
2. **One production implementation per scientific quantity.** Circular moments,
   SD, density asymmetry, likelihood, result identity, bandwidth rules, and
   fingerprints should not have consumer-specific copies.
3. **WNM runtime must not depend on legacy surface machinery.** Historical
   surface replay may depend on shared objective/target code, but current WNM
   fitting should not import a large surface optimizer merely to obtain neutral
   loss functions.
4. **Historical reproduction is explicit.** Surface-NN and transition experiments
   remain reachable, but they should be labeled and isolated so they do not
   define the normal import graph or default test suite.
5. **Core installation must support ordinary fitting/prediction without sibling
   repositories.** CDB-dependent tests and workflows are integration layers.
6. **A test should validate one layer.** Mathematical primitives, predictor
   contracts, fitting integration, plotting integration, cross-repository
   integration, and historical reproduction should not repeatedly test the same
   mathematics.
7. **Generated research artifacts do not belong in the source tree** unless they
   are deliberately small golden fixtures or compact provenance/findings files.

---

# 1. Items to fix before interpreting a full test run

These are pre-existing architecture/test defects, not numerical failures that a
test run should be used to discover.

## 1.1 Stale CCC test import [resolved before baseline run]

`tests/test_density_ccc_objective.py` imports
`_ccc_components` from `create_unified_subject_plots.py`. That helper was
removed when plotter-side CSV export was retired.

The neutral implementation is already
`model_fit_to_data/density_objective.py:ccc_components`, and
`tests/test_density_objective_parity.py` already tests it.

**Action:** change the test to the neutral implementation or remove the
duplicated CCC-component test.

## 1.2 Default tests depend on a sibling repository [resolved before baseline run]

At least these product-tree tests assume
`../contextual_biases_database` exists:

- `tests/test_bwcrps_observed_design.py`
- `tests/test_production_compiled_bundles.py`

The latter also imports `yaml`, which is not a declared DM dependency.

**Action:** mark these as integration tests and skip them cleanly when CDB is not
available. `pytest tests` should validate a standalone Demixing Model checkout.

## 1.3 Pytest has no explicit collection boundary [resolved before baseline run]

There are two substantial test roots:

- `tests/`: current runtime plus historical surface tests
- `continuous_density/tests/`: WNM research/prototype tests

Without a `testpaths` policy, a bare `pytest` can collect both and make
transition research part of the implicit product contract.

**Action:** configure the default suite as `tests/`, then explicitly mark or
separately invoke:

- `integration`
- `legacy_surface`
- `research`
- `slow`
- `gpu`

The temporary `continuous_density/tests/` directory should disappear as part of
the redistribution described below.

---

# 2. Production architecture issues

## 2.1 WNM fitting still imports neutral functionality from the surface optimizer

Current WNM construction imports several scientifically neutral helpers from the
84 KB historical surface optimizer
`model_fit_to_data/grid_based_multi_condition_optimizer_jax_loops.py`,
including:

- `_compute_curve_losses`
- `bwcrps_energy_score`
- `compute_bwcrps_condition_targets`
- `compute_target_bias_curve_core`

This makes the current WNM runtime depend on the legacy surface implementation
for code that is not intrinsically surface-specific.

**Target:** extract neutral objective and empirical-target operations into small
modules, for example:

```text
model_fit_to_data/
    objectives.py
    empirical_targets.py
    fitting_targets.py
    wnm_scoring.py
    surface_optimizer.py       # historical
```

Both WNM and surface replay should depend on the neutral layer. WNM should not
depend on the surface optimizer.

## 2.2 `shared/utils.py` is a cross-era dependency sink

`shared/utils.py` is about 56 KB and combines:

- result/path helpers;
- model checkpoint save/load;
- historical surface classes and unpickling;
- surface bundle extraction;
- plotting helpers;
- circular curve estimators;
- KDE/bandwidth estimation;
- behavioral data filtering;
- training-log utilities.

Current WNM code imports this module for lightweight operations such as
`filter_data_for_fitting`, `resolve_input_path`,
`gaussian_curve_smoother`, or bandwidth functions, but importing it also pulls
surface/training dependencies.

**Target:** split it by responsibility, for example:

```text
shared/
    paths.py
    hashing.py
    circular.py
    bandwidths.py
    behavioral_data.py
    surface_io.py              # historical/raw-surface support
```

The exact filenames are less important than breaking the dependency from WNM
runtime to historical surface/training infrastructure.

## 2.3 `model_fit_to_data` is not a proper package

The repository currently relies on several incompatible import styles:

- `from continuous_fit import ...`
- `from model_fit_to_data.continuous_fit import ...`
- `try/except ModuleNotFoundError` import fallbacks
- repeated `sys.path.insert(...)` in production scripts and tests.

This makes "run this exact file from the repository root" a hidden part of the
API and weakens installation/library reuse.

**Target:**

- add a package boundary for `model_fit_to_data`;
- use one absolute import style;
- support `python -m ...` or installed console scripts;
- remove test-specific path surgery once imports are stable.

The same principle should eventually apply to the other top-level functional
areas.

## 2.4 The predictor abstraction is only half an abstraction

`shared/prediction.py` correctly centralizes WNM calculations, but
`BiasPredictor` declares only `identity()` and `with_motor_noise()`, while
the actual consumers rely on many undeclared operations
(`log_density`, `cell_probabilities`, `circular_sd`,
`signed_arc_asymmetry`, etc.).

At the same time, `predictor_from_surrogate()` only constructs WNM predictors;
surface prediction still requires a separate optimizer path.

That is a reasonable transition state but not a clean long-term interface.

**Target:** now that WNM is the default and surface is historical, define the
maintained prediction API around WNM operations and isolate surface replay in a
legacy adapter rather than preserving a nominal two-family abstraction that
cannot instantiate both families uniformly.

## 2.5 WNM loading is duplicated

The canonical artifact format is defined by
`continuous_density/wrapped_mixture_model.py:save_model/load_model`, but
`shared/surrogate.py` independently opens the pickle and reconstructs
`ConditionalWrappedMixture` from `model_config`.

A schema change can therefore update one loader without updating the other.

**Target:** after moving the runtime model out of `continuous_density/`, keep one
artifact parser/loader. `shared.surrogate` should delegate to it and add only
family/sample-count identity logic.

---

# 3. Overlapping and duplicated functionality

## 3.1 Result-key identity code appears three times

Sanitization/canonicalization logic exists independently in:

- `fit_model_to_data.py`
- `create_unified_subject_plots.py`
- `repair_stale_reproduction.py`

These functions are required to agree exactly because they map stored results to
subjects/experiments/conditions.

**Target:** one result-identity module used by fitting, plotting, repair, and
legacy migration.

## 3.2 Surface circular-SD calculation is duplicated

The same grid-based circular-SD calculation exists in:

- `surface_simulator_for_predictions/surface_simulator.py`
- `create_unified_subject_plots.py`

while `shared.prediction.SurfacePredictor` already owns the corresponding
estimator.

**Target:** route historical surface consumers through the shared surface
predictor/helper rather than maintaining consumer-local implementations.

## 3.3 File hashing has multiple implementations

Streaming SHA-256 exists independently in:

- `shared/surrogate.py:file_digest`
- `model_fit_to_data/run_fingerprint.py:file_sha256`
- `continuous_density/package_wnm_artifact.py:file_digest`

**Target:** one shared hashing helper. Artifact identity, run fingerprints, and
packaging should call the same function.

## 3.4 Workload bucketing is duplicated across repositories

`model_fit_to_data/workload_bucketing.py` is intentionally a local copy of the
generic observed-range bucketing policy also present in CDB. This removed the
runtime dependency on the sibling repository, which is correct for users, but
it leaves two implementations of the same versioned contract.

**Target options:**

1. keep both small implementations but add explicit cross-repo parity/version
   tests; or
2. later move this genuinely generic helper to a tiny dependency-neutral shared
   package.

Do not reintroduce CDB as a dependency of ordinary DM fitting merely to remove
this duplication.

## 3.5 WNM likelihood export/rescoring is split

`postprocess_fitted_likelihoods.py` and `export_wnm_fit_curves.py` both
construct WNM likelihood output/reproduction checks. The former additionally
owns exact float64 integrated reporting-cell probability.

**Target:** one WNM likelihood evaluator and one row schema. The generic
postprocessor and WNM exporter should be orchestration layers over it.

## 3.6 The tabular exporter still plots

`export_wnm_fit_curves.py` writes the authoritative WNM CSV/likelihood products
but also emits one three-panel PNG per analysis cell. The normal pipeline then
runs the dedicated subject/group/PDF plotting code separately.

**Target:** exporter produces tables/manifest only. If the compact diagnostic is
still useful, move it to a plotting module or make it an explicit optional
diagnostic command.

## 3.7 CCC tests overlap

`test_density_ccc_objective.py` and
`test_density_objective_parity.py` both test CCC decomposition/factorization.
The first also contains the stale plotter import noted above.

**Target:** keep primitive CCC mathematics in one test module and parity/routing
tests in another.

## 3.8 Wrapped-mixture mathematics is tested at multiple layers

Examples include analytic motor convolution and asymmetry checked independently
in both:

- `tests/test_wrapped_mixture.py`
- `tests/test_prediction_contract.py`

This is not inherently wrong, but several tests repeat the same numerical claim
instead of testing that the higher layer delegates to the primitive.

**Target:** mathematical identities in `test_wrapped_mixture.py`; predictor
tests should test routing, validation, identity, and API semantics.

## 3.9 Plot/SD behavior is spread over many overlapping suites

Related behavior is distributed across:

- `test_mixture_plot_curves.py`
- `test_plot_estimators.py`
- `test_sd_estimators.py`
- `test_pooled_bwcrps.py`
- `test_pooled_bwcrps_export.py`

**Target:** distinguish estimator-unit tests from plot-pipeline integration tests
and remove repeated mathematical assertions from the latter.

---

# 4. Clearly inefficient code paths

These are architecture/performance improvements, not correctness blockers.

## 4.1 Repeated WNM network evaluation for the same coordinates

`shared.prediction.mixture_plot_curves()` asks the predictor separately for
mean/resultant, density asymmetry, circular SD, pooled SD, and sometimes a
second observed-design coordinate set. Each public predictor call rebuilds the
mixture distribution with `model.apply`.

The mixture distribution is the expensive neural forward pass; all of these
statistics are cheap analytic functions of that one distribution.

**Target:** evaluate the distribution once per coordinate batch and derive all
required statistics from it.

## 4.2 Public WNM prediction loops row-by-row in Python

The current public prediction interface loops over every parameter combination
and performs multiple predictor calls. WNM naturally supports a flattened
`parameter combination × feature difference` batch.

**Target:** batch combinations and feature coordinates, evaluate once/few times,
then reshape to output curves.

## 4.3 WNM cross-evaluation can do unnecessary work for ordinary users

After fitting, ordinary CSV runs can evaluate each fitted method under the full
eight-objective set even when a user requested a small subset. Full
cross-objective matrices are valuable for model-comparison work, but they are
not necessarily the correct default cost for an end-user single-model fit.

**Target:** make the evaluation set explicit. BMC/compiled workflows can request
the full comparison matrix; ordinary fits can default to requested/maintained
objectives.

## 4.4 `ContinuousEngine._trials()` repeatedly converts datasets

Trial arrays are converted to NumPy/JAX each time fitting/evaluation requests
them.

**Target:** cache canonical JAX trial arrays when the dataset/compiled group is
installed.

## 4.5 R prediction wrapper grows data frames and sleeps unnecessarily

`surface_simulator.R` builds the long result through repeated `rbind`, which
scales poorly, and calls `Sys.sleep(2)` after a synchronous Python process has
already exited.

**Target:** build a list and `data.table::rbindlist()`; remove the sleep.

## 4.6 Plotting module is too large to reason about efficiently

`create_unified_subject_plots.py` is about 95 KB and currently owns:

- legacy result migration;
- empirical estimators;
- surface prediction preparation;
- WNM prediction preparation;
- subject figures;
- group summary figures;
- PDF slices;
- CLI orchestration.

**Target:** split data/estimator preparation from each plot family. This will
also make test boundaries substantially clearer.

---

# 5. Transition/recovery leftovers

The repository now has a maintained standardized WNM recovery path:

- `standardized_recovery.py`
- `standardized_recovery_plots.py`
- shared recovery primitives in `recovery.py`.

It also retains transition-era runners and comparison utilities such as:

- `run_recovery_panel.py`
- `single_condition_recovery.py`
- `likelihood_search.py`
- `compare_wnm_surface_fits.py`
- surface curve-cache/exhaustive-search tools.

These files contain useful historical evidence and some reusable primitives, but
they should not all remain peers of the maintained runtime indefinitely.

**Target:** classify each as either:

1. reusable primitive and move the primitive into a neutral maintained module;
2. historical/research workflow and move it under an explicit
   `research/...` tree;
3. obsolete after its findings are captured and remove it from the active tree.

Corresponding tests should follow the code rather than remain in the default
product suite.

---

# 6. `continuous_density/` must be dissolved

`continuous_density/` was a temporary research/transition workspace. It now
contains the production WNM model implementation, training scripts, simulation
design, validation experiments, plotting/reporting scripts, exploratory local
representation families, archived findings, generated outputs, and a second
test suite.

That directory no longer represents one architectural responsibility and should
not survive the transition.

## 6.1 Target functional structure

A reasonable target is:

```text
shared/
    wnm.py                     # WNM distribution/model math and artifact format
    prediction.py              # maintained prediction operations
    surrogate.py               # artifact/family resolution and identity

surface_computation/           # underlying Demixing Model simulator + training-data generation
    ...
    wnm_simulation.py
    wnm_design.py
    generate_wnm_training_data.py
    legacy_sample_import.py

surrogate_training/
    wnm/
        data.py
        train.py
        package_artifact.py
    surface_nn/                # eventual home of current neural_network_optimization
                               # if/when historical surface code is reorganized

model_fit_to_data/
    ...
    objectives.py
    empirical_targets.py
    fitting_targets.py
    wnm_scoring.py
    continuous_optimizer.py
    standardized_recovery.py

prediction_tools/              # eventual rename of surface_simulator_for_predictions
    python / R user interfaces

research/
    wnm_transition/
    wnm_representation/
    historical_surface/

docs/
    history/
        wnm_transition/

tests/
    ...                        # maintained standalone product tests
    integration/               # CDB/BMC-dependent tests
```

This exact naming can be adjusted during implementation. The important
properties are the dependency direction and the disappearance of
`continuous_density/` as a runtime namespace.

## 6.2 Production runtime model

Move:

- `continuous_density/wrapped_mixture_model.py`

to a maintained shared runtime location, for example `shared/wnm.py`.

It should own:

- `ConditionalWrappedMixture`;
- wrapped-normal density/integrals;
- circular moments/asymmetry;
- motor-noise convolution;
- the single WNM artifact save/load schema.

Then update:

- `shared.prediction`
- `shared.surrogate`
- training/packaging code
- tests

to import it from that maintained location.

Production code should then contain **zero imports from `continuous_density`**.

## 6.3 Simulation and training-data generation

Move the simulator-facing pieces to the simulation layer:

- `sim_interface.py`
- `design.py`
- `generate_training_data.py`
- `existing_samples.py`

Likely destination: a WNM sublayer of `surface_computation/`, since these
scripts use the same underlying Demixing Model simulator and produce surrogate
training/reference data rather than fitted behavioral results.

The name `surface_computation` is historically surface-specific. It can be
renamed to a broader simulator name later, but that rename is not required to
remove the temporary WNM namespace.

## 6.4 WNM training and packaging

Move:

- `data.py`
- `train.py`
- `package_wnm_artifact.py`

to a generalized surrogate-training layer.

The existing `neural_network_optimization/` directory is specifically named
for the historical surface NN, so putting WNM there directly would repeat the
same naming problem. Prefer a new functional `surrogate_training/` parent. The
surface-NN training code can later move under `surrogate_training/surface_nn/`
when convenient.

## 6.5 Production fitting/scoring/recovery code

Any functionality that has become part of the maintained behavioral fitting
contract belongs under `model_fit_to_data/`, not the research directory.

Most of this migration is already complete. New moves should be rare and should
primarily consist of extracting reusable primitives from transition scripts, not
moving entire experiments.

`standardized_recovery.py` should remain the maintained recovery entry point.

## 6.6 Research validation and representation experiments

The following classes of files should move under an explicit research tree,
because they answer historical/experimental questions rather than implement the
end-user model:

### WNM versus historical representation / validation

- `benchmark.py`
- `benchmark_density_evaluation.py`
- `benchmark_manifest.py`
- `compare_existing_model.py`
- `evaluate.py`
- `evaluate_flagship_points.py`
- `fit_demo.py`
- `inspect_checkpoint_spectrum.py`
- `plot_uev.py`
- `report.py`
- `spectrum_diagnostics.py`
- `stratified_forward_truth.py`
- `validate_subsupport_extrapolation.py`

### Local representation experiments

- `bootstrap_local_metrics.py`
- `candidate_mode_diagnostics.py`
- `classify_local_trajectories.py`
- `combine_repeated_fit_gates.py`
- `compare_local_representations.py`
- `diagnose_local_capacity.py`
- `equivalence.py`
- `fit_local_fourier.py`
- `fit_local_spline.py`
- `fit_local_wrapped_mixture.py`
- `fourier_moments.py`
- `local_distribution_gate.py`
- `local_fit_common.py`
- `local_representation_benchmark.py`
- `maxent_fourier.py`
- `mode_diagnostics.py`
- `periodic_spline.py`
- `power_calculation.py`
- `reference_noise_floor.py`
- `refresh_local_gates.py`
- `select_flagship_points.py`
- `triage_trajectories.py`
- `summarize_fit_variation.py`
- `summarize_in_sample_metrics.py`
- `summarize_local_benchmark.py`
- `summarize_optimizer_variation.py`
- `summarize_representation_table.py`
- `summarize_uncertainty_power.py`
- `plot_amplitude_slopes.py`
- `plot_flagship_densities.py`
- `plot_harmonic_spectrum.py`
- `plot_local_benchmark.py`
- `plot_uncertainty_equivalence.py`

These should not be imported by production runtime modules.

## 6.7 Findings and transition documentation

Move the `*FINDINGS.md`, `RESULTS.md`, and the historical portions of the
current `continuous_density/README.md` to something like:

```text
docs/history/wnm_transition/
```

The transition evidence should remain available, but it should not be presented
as current user/runtime documentation.

## 6.8 Tests under `continuous_density/tests/`

Redistribute by what they test:

- WNM mathematical primitive tests should merge into the maintained
  `tests/test_wrapped_mixture.py` or successor.
- WNM training/design tests should move with the training/simulation code and be
  marked as training/research tests as appropriate.
- local representation tests should move with the research experiments and stay
  outside the default product suite.
- duplicated assertions already covered by product tests should be removed rather
  than mechanically moved.

The completion condition is that `continuous_density/tests/` is gone.

## 6.9 Generated validation outputs

`continuous_density/validation_outputs/` currently tracks about 196 generated
files totaling roughly 4.85 MB. This conflicts with the repository's own policy
that experiment bundles belong under `$DEMIXING_ARTIFACT_ROOT`.

**Target:** keep only compact findings/provenance needed for the historical
record. Move generated CSV/log/result bundles to the external artifact store and
remove them from the active source tree.

---

# 7. Test architecture

The default suite should answer: "Does a standalone installed DM checkout
correctly implement the maintained WNM model and explicit historical contracts?"

Recommended layers:

## Unit

- wrapped-mixture mathematics;
- circular utilities;
- bandwidth/target construction;
- objective functions;
- surrogate identity/artifact validation;
- continuous optimizer primitives.

## Contract

- WNM predictor contract;
- fit result/fingerprint schema;
- WNM export schema;
- plotting estimators.

## Integration, standalone DM

- ordinary CSV WNM fitting;
- public prediction API;
- export;
- subject/group/PDF plots;
- standardized recovery smoke.

## Cross-repository integration

Marked and non-default unless sibling dependencies are available:

- compiled CDB bundle readers;
- production bundle catalog;
- CDB observed-design parity.

## Historical surface

Marked `legacy_surface`; run when historical reproduction changes, not as
implicit WNM product coverage.

## Research

Local representation, optimizer-comparison, and transition experiments should
be explicitly invoked and should not define the default product suite.

---

# 8. Dependency and packaging issues

`pyproject.toml` still describes the project as "simulation, NN training,
fitting, browser" and installs cloud/browser/training dependencies into the core
environment.

It also has a `dev` comment referencing symbolic tests from another repository,
while DM itself directly imports SciPy in maintained code without explicitly
declaring SciPy.

**Target extras:**

- core runtime/fitting;
- `training`;
- `browser`;
- `cloud`;
- `legacy-surface`;
- `dev`.

Declare every direct core dependency explicitly. Do not rely on transitive
installation.

---

# 9. Naming leftovers

The default prediction implementation is WNM, but the public layer is still
named:

- `surface_simulator_for_predictions/`
- `simulate_surfaces_from_file()`
- R `simulate_surfaces()`

This is now misleading.

**Target:** introduce model-neutral names such as `prediction_tools/` and
`generate_predictions()`, with temporary compatibility wrappers if external
code depends on the old names.

Likewise, the current `surface_browser` contains a direct WNM view and is no
longer purely a surface browser. This rename is lower priority than the runtime
module cleanup.

---

# 10. Documentation audit

## 10.1 `continuous_density/README.md`

This is the most contradictory current document.

It says WNM is the deployed default and then calls WNM an opt-in fitting/scoring
backend. Near the bottom it still says the bare surrogate default resolves to
the surface NN and that generic prediction has not been migrated. That is false
after the current cutover.

**Action:** stop treating this as current documentation. Move the historical
experiment/reproduction material to `docs/history/wnm_transition/`; put current
WNM training documentation beside the training code.

## 10.2 `continuous_density/__init__.py`

It says the package is an experimental prototype intentionally kept out of
production fitting. Production WNM currently imports its model from this
package.

**Action:** eliminate this contradiction by moving production WNM code out and
then retiring the package.

## 10.3 Surface-NN training documentation

`neural_network_optimization/Neural_Network_Optimization_Pipeline_Documentation.md`
still calls the epoch-1425 surface checkpoint the production checkpoint and
describes the NN as the production surrogate.

**Action:** explicitly label this document as the **historical surface-NN
training/reproduction pipeline**.

## 10.4 README mean-bias objective guidance

The root README recommends `expectation` when the scientific target is the mean
bias curve, while the root TODO says the hard-binned `expectation` objective is
retired and will not be revived.

**Action:** document the maintained mean-bias route (`smoothed_exp`, if that is
the intended maintained objective) and label `expectation` historical.

## 10.5 `sd_ident` versus `sd_spat`

The root README uses the scientific term identifiability noise
(`sd_ident`), while code, input tables, and exports use `sd_spat`.

**Action:** explicitly state that `sd_spat` is the code/storage name for the
identifiability-noise parameter and explain the historical spatial interpretation.

## 10.6 README figure documentation

`docs/README.md` says fitted curves for the Fischer README figure are exported
by `create_unified_subject_plots.py`. The current file is written by
`export_wnm_fit_curves.py`.

The default path itself is still valid; the documented producer is stale.

## 10.7 Pipeline figure

`docs/generate_pipeline_figure.py` still depicts "Neural-network training" as
the fast-model stage and defaults to an old NN prediction export. The figure is
currently experimental/not embedded, so this is low priority but should not be
promoted without updating it to WNM.

## 10.8 Installation wording

`INSTALL.md` still speaks generally about "surface tools" for artifact-root
behavior. As the functional split is cleaned up, distinguish current WNM user
workflows from historical raw-surface tooling.

---

# 11. Repository hygiene

The tracked tree is roughly 44.7 MB at the audited commit. Major non-source
categories include approximately:

- 34.8 MB under `pretrained/`, dominated by historical surface checkpoints;
- 4.85 MB / 196 files under `continuous_density/validation_outputs/`;
- 0.96 MB of test golden data;
- 0.79 MB of documentation images.

The packaged WNM artifacts themselves are under 1 MB total.

Historical surface checkpoints may still be worth shipping for reproducibility,
but this should be a deliberate release/reproduction decision rather than an
accident of the transition. Generated transition validation bundles are a clearer
candidate to move out of git.

---

# 12. Recommended implementation order

Do not perform a large refactor before establishing a meaningful baseline. The
recommended order is:

## Phase A: pre-test hygiene

Implemented on the transition branch before establishing the green baseline:

1. the stale CCC test now imports the neutral density-objective implementation;
2. default pytest collection is restricted to `tests/`;
3. CDB-dependent checks are explicit `integration` tests and skip cleanly
   without the sibling checkout;
4. transition-comparison and clearly surface-only suites are marked
   `research` / `legacy_surface` and excluded from the default product run;
5. the most misleading current documentation has been corrected;
6. obvious dead imports found during the audit were removed.

**Still required:** execute the maintained pytest baseline and
`run_smoke_wnm.sh`. Do not call the baseline green until those commands pass.

## Phase B: cut production dependencies on historical surface modules

1. Extract neutral objective/target helpers from
   `grid_based_multi_condition_optimizer_jax_loops.py`.
2. Split lightweight maintained helpers out of `shared/utils.py`.
3. Normalize `model_fit_to_data` imports/package structure.
4. Centralize result identity, hashing, and WNM likelihood export/rescoring.

Run tests after each move.

## Phase C: dissolve `continuous_density/`

1. Move WNM runtime math/artifact format to `shared`.
2. Move simulation/design/training-data code to the simulation layer.
3. Move training/packaging to a surrogate-training layer.
4. Move research/validation experiments under `research/`.
5. Move transition findings under `docs/history/`.
6. Consolidate or move its tests.
7. remove tracked validation outputs.
8. delete the empty `continuous_density/` directory.

Acceptance criterion: no production import, documentation link, or default test
collection relies on `continuous_density`.

## Phase D: performance and API cleanup

1. Batch WNM prediction and derive multiple statistics from one distribution.
2. Cache JAX trial arrays.
3. make cross-objective evaluation explicit/configurable.
4. remove plotting from the exporter.
5. split the plotting monolith.
6. introduce model-neutral public prediction names.

## Phase E: historical surface isolation

After the WNM path is green and the project comparison has passed its cutover
gates:

1. move surface-only search/cache/recovery tools under explicit historical
   namespaces;
2. mark their tests `legacy_surface`;
3. decide whether historical checkpoints remain in the normal repository or move
   to tagged/release artifacts.

---

# 13. Completion criteria for the architecture transition

The repository architecture transition is complete when all of the following are
true:

- WNM is the default runtime and no production module imports
  `continuous_density`.
- `continuous_density/` no longer exists.
- ordinary CSV fitting/prediction requires no sibling repository.
- CDB/BMC-dependent checks are explicit integration tests.
- WNM fitting does not import neutral objective code from the surface optimizer.
- result identity, hashing, WNM likelihood evaluation, and major estimators each
  have one maintained implementation.
- the default test suite contains current product contracts, not transition
  experiments by accident.
- historical surface functionality is explicit and isolated.
- current documentation describes WNM; historical documentation clearly labels
  the surface/transition context.
- generated validation bundles live outside the source tree except for deliberate
  small golden fixtures/provenance summaries.
- the documented installed-package/import path works without repository-root
  `sys.path` manipulation.
