# Demixing Model — planning history

This log records completed decisions and superseded work. `TODO.md` is the only
live work plan. Detailed numerical evidence remains in
`docs/history/wnm_transition/` and git history rather than being repeated as
future work here.

## 2026-09 — WNM production transition

- Established and executed the pre-refactor test baseline: fixed the stale CCC test import, restricted default pytest collection to maintained tests, isolated CDB checks as integration coverage, kept transition experiments off the production suite, marked historical surface coverage as opt-in legacy coverage, removed obvious dead imports, and corrected misleading post-cutover documentation.
- Selected the trajectory-trained K=12 conditional wrapped-normal mixture as
  the replacement for the surface neural-network surrogate. Packaged 20- and
  100-observation artifacts are addressed by observer sample count.
- Promoted WNM to the shared production default for fresh predictions and
  ordinary CSV fitting. The public Python/R prediction interfaces, Fischer-
  Whitney demo, WNM tabular exporter, and a dedicated end-to-end smoke now use
  that default. Compiled CDB bundles remain the stricter path for cross-model
  comparisons rather than the prerequisite for ordinary users.
- Selected one production optimizer policy for the four retained objectives:
  64 deterministic starts in two sequential batches of 32 using the pinned JAX
  L-BFGS-B port, float32 arrays, and `highest` matmul precision. The precision
  setting avoids the GPU TF32 basin changes found in likelihood diagnostics.
- Fixed the WNM search domain at feature SD `[2.5, 200]` degrees and spatial SD
  `[5, 200]` degrees. Small corpus-hull overhangs are accepted and recorded.
- Completed focused closed-loop and actual-observer recovery, objective-specific
  search comparisons, representative real-data fits, direct analytic WNM curve
  prediction, and explicit physical/model-unit exports.
- Added bundle-native fitting, strict compiled-input validation, exact observed
  dissimilarities, fit fingerprints, per-trial likelihood replay, parameter and
  curve exports, and production-bundle coverage tests.
- Retired the duplicate CSV-export path from `create_unified_subject_plots.py`. The plotter now produces figures only; `export_wnm_fit_curves.py` is the single provenance-bound WNM tabular export path.
- Completed the maintained result/plot consumer transition: unified plots now
  recover bundle-native subject/experiment/condition labels from stored
  `analysis_cell_values`, preserve report-order pairing, infer and validate the
  saved physical circular period, and the standalone PDF-slice command evaluates
  WNM fits directly through the run-fingerprinted surrogate instead of requiring
  a surface optimizer.
- Aligned PDF-slice diagnostics with the fitted WNM estimator: the feature
  kernel stays in model degrees, the stored empirical KDE bandwidth is reused
  when available, and the standalone command restores the run's recorded JAX
  matmul precision.
- Added checksum-keyed resumable recovery and simulation stages after the audit
  found that earlier shard reuse was not tied to the producing code.
- The earlier promotion gate compared WNM against the surface NN. The decision
  to replace the surface NN makes further comparator, retraining, and promotion
  work obsolete; it is not carried into `TODO.md`.
- The original continuous-density prototype progressed from a global Sobol
  wrapped-normal-mixture experiment to the trajectory-trained K=12 model with
  independent training, checkpoint-selection, and final-test data. That work is
  complete; the prototype task no longer acts as a separate plan.

## 2026-09 — architecture consolidation after WNM cutover

- Completed the repository redistribution identified by the architecture audit.
  Maintained WNM runtime now lives in `shared/wnm.py`; fitting/scoring and
  likelihood evaluation live under `model_fit_to_data/`; training and packaging
  live under `surrogate_training/wnm/`; and simulation, design, training-data
  generation, and legacy raw-sample import live under `surface_computation/`.
- Removed the temporary `continuous_density/` namespace after moving maintained
  functionality to its permanent functional homes. No maintained production code
  imports from that namespace.
- Centralized neutral fitting objectives and empirical target construction,
  result-key identity, streaming file hashing, and WNM likelihood evaluation so
  each maintained scientific/identity quantity has one production implementation.
- Split WNM-facing path, behavioral-data, circular-smoothing, and empirical
  KDE/bandwidth helpers out of the legacy `shared/utils.py` dependency sink and
  normalized maintained `model_fit_to_data.*` package imports.
- Consolidated WNM artifact loading so `shared.surrogate` delegates reconstruction
  to the maintained WNM loader instead of maintaining an independent parser.
- Moved transition findings to `docs/history/wnm_transition/`. Development-only
  transition validation, recovery, optimizer-comparison, and representation
  experiments were kept off the production branch rather than becoming part of
  the maintained runtime contract.
- Removed generated transition validation outputs from the source tree and
  separated maintained pytest coverage from integration and legacy-surface
  coverage.
- Verified the final production snapshot with both the maintained pytest suite and
  the WNM public-path end-to-end smoke before merging PR #2 into `main`.
- The detailed `ARCHITECTURE_AUDIT.md` served as the migration plan and diagnosis.
  Its implemented conclusions are summarized here; the original file remains
  recoverable from git history.

## 2026-08 — circular-axis and density-search work

- Migrated stored surfaces and the production circular representation from a
  duplicated 181-point endpoint grid to a native periodic 180-cell axis, with
  executable row-count, rotation, seam, decoder-periodicity, and loader guards.
- Added periodic decoder operations and corrected the surface training loss.
  The 20-observation surface checkpoint was retrained. The planned
  100-observation retrain and coordinated surface refit are canceled because the
  surface NN is now legacy.
- Replaced the production density curve objective with `1 - CCC`, added explicit
  objective/run fingerprints, and implemented the exact one-degree curve-cache
  search and shared scorer. Planned sub-degree surface-cache refinement and
  surface refits are canceled with the backend.

## Retired backlog

- The hard-binned `expectation` objective was superseded by compiled
  support-weighted `smoothed_exp` and is not a production objective.
- The motor-noise likelihood floor issue belongs to the gridded surface backend;
  WNM applies symmetric motor noise analytically. It remains relevant only when
  reproducing historical surface artifacts.
- The `(mu1, mu2)` joint-density idea requires new paired simulation output and
  was never scheduled as transition work.

## 2026-06 to 2026-09 — audit closure

- Fixed period-aware circular bias construction, padded-trial leakage in density
  targets, likelihood-unit inconsistencies, combined-observer report filtering,
  identifier collisions, non-finite likelihood validation, and atomic
  split-Parquet replacement.
- Replaced the mismatched legacy curve estimators with compiled shared targets
  and explicit objective contracts. The old `expectation`, gridded-density, and
  surface motor-noise findings no longer define production WNM work.
- Added run/source fingerprints, code-digest checkpointing, stable recovery
  protocols, strict compiled-bundle inputs, and production-entry-point tests.
- The independent K12 review found a confounded surface-versus-WNM promotion
  panel and several resume/manifest gates. The persistence and gate defects were
  fixed; the comparative promotion question was superseded by the decision to
  replace the surface NN.
- Distributed surface-generation locking, bundle durability, training-input,
  and retraining findings apply to the historical surface pipeline and are not
  carried forward. The repository-local `.env` was restricted from mode `0777`
  to `0600` without inspecting its contents; relocating and rotating its secrets
  remains in `TODO.md`.
- The transition handoff's predictive-contract gate is implemented. Its refit
  and report-rebuild work is represented in the repository TODOs; its proposed
  surface-versus-WNM promotion check was canceled when the surface NN became
  legacy.

## Consolidated source documents

The following former live plans were consolidated here on 2026-09-15 and
removed so they cannot compete with `TODO.md`:

- `continuous_density/TRANSITION_PLAN.md`
- `continuous_density/TRANSITION_AUDIT.md`
- `continuous_density/OPEN_DECISIONS.md`
- `CIRCULARITY_FIX_PLAN_temp.md`
- `EXHAUSTIVE_CCC_PLAN_temp.md`
- `PLAN_REVIEW_CODEX_temp.md`
- `continuous_density/AUDIT_SUMMARY.md`
- the workspace `DEMIXING_MODEL_CONSOLIDATED_AUDIT.md`
- the workspace `K12_TRANSITION_INDEPENDENT_AUDIT.md`
- `continuous_density/HANDOFF.md`
- `OPTIMUM_SEARCH_TESTS_temp.md`
- the workspace continuous-density prototype task

The tracked transition documents remain available in git history. The untracked
`*_temp.md` drafts and unversioned workspace audits were removed after their
implemented decisions, remaining defect, and canceled surface-NN work were
summarized above.

Audit and plan snapshots stored beside generated result bundles under
`$DEMIXING_ARTIFACT_ROOT` are immutable experiment provenance, not live work
plans. They remain with those artifacts and do not compete with `TODO.md`.
