**K12 WNM transition plan — 2026-09-06**

**Current execution status (updated 2026-09-07).** This document preserves the
full transition design prepared against `12f5206`; it is not a live checklist.
`TRANSITION_AUDIT.md` records the revised execution order. Work is currently
limited to recovery and the search, prediction, and fitting support required to
attribute its failures (audit R1--R6). Plotting/browser integration, downstream
rollout and regeneration, default promotion, and surface-NN retirement are
deferred until those recovery gates are met. A recovery runner and initial n=20
density panels now exist; statements below that none were found describe the
2026-09-06 baseline.

Prepared from the current working tree of the inner `demixing_model` repository
(HEAD `12f5206`, with existing uncommitted changes), the 4.1q artifacts, and the
external batch pipeline. This is a proposed implementation plan, not a record of
completed integration or a production clearance. No runtime code was changed and
no GPU jobs were launched to prepare it.

**Decision and scope**

Select the 4,400-trajectory K12 conditional wrapped-normal mixture as the
production surrogate, replacing the surface NN outright. Freeze the architecture
and training recipe while doing the migration. Introduce K12 behind the existing
fitting and prediction commands, and keep the surface NN selectable *during* the
transition so acceptance can compare the two on identical problems. Promote the
20- and 100-observation artifacts separately after each passes acceptance. Once
every production artifact has been regenerated on K12, remove the surface NN
backend, its training pipeline and its checkpoints from the runtime (step 7).
Reproduction of historical results is served by a tagged commit plus archived
checkpoints, not by a second maintained backend: two live surrogates is exactly
the parallel-copy situation AGENTS.md forbids, and the transitional selector is
therefore scaffolding with a scheduled removal, not a supported feature.

Only the NN retires. The averaged simulated surfaces stay, with their existing
generation path (S3/S4), their storage and release, and the mu2 predictions and
`surface_browser/` views they feed. They cease to be the surrogate's training input
and become a standing empirical reference artifact, which is a change of role, not a
retirement.

Two artifacts are therefore in play, and this plan is careful to name them apart.
The **training corpora** are the 4,400-trajectory, 148,076-cell histogram sets that
trained the current K12 models: n=20 is recorded under
`results/continuous_density_4.1p/artifacts/histogram_shards`, while the packaged
n=100 artifact records `continuous_density_4.1o`. They are not the 4.1q benchmark,
which is a separate, deliberately held-out 197-trajectory set
(`design_benchmark.py` raises if the two overlap). Both are 720-bin histograms
written by scripts of the same name. The corpora are new standing artifacts and
need the storage, provenance and regeneration story the averaged surfaces already
have.

`surface_browser/` gains on-demand K12 evaluation alongside the stored surfaces
rather than instead of them: K12 is continuous in feature difference and analytic in
density, so it can be evaluated at any parameter triple rather than only where a
surface was simulated, and the stored surfaces remain as the empirical reference to
plot it against. It must evaluate K12 directly and not read corpus-derived surfaces:
`build_nn_surfaces_full.py` reaches a 180x90 lattice by regrouping 720 half-degree
bins into 2-degree cells and interpolating each mu1 row across the feature
differences that trajectory happened to visit. That script exists to give the NN
comparator a like-for-like training set; routing the browser through it would bake
that interpolation into what the browser shows.

mu2 keeps working throughout, from the averaged surfaces exactly as today. K12 models
mu1 only, so the mu2 branch of the prediction API stays on its current path and there
is no gap to schedule and no interface change to announce.

**Deferred: mu1 x mu2 covariation**

Not part of this transition and not scheduled. Recorded here only because the facts
below cost a day to establish and would otherwise be rediscovered -- and because one
of them is an active trap.

*Both marginals exist; no joint exists anywhere.* The averaged surfaces carry real
mu2 (`mu2_comp1_surface` / `mu2_comp2_surface`, (167, 90), spatial grid [-498, 498]
degrees, produced by `kde_spatial` in `shared/averaging.py`) and real mu1, but as two
separate 1-D KDEs (`kde_circular`, `kde_spatial`) sharing only the feature-difference
weighting. Two marginals per component, no covariation. The training corpus is
narrower still -- mu1 histograms only, which is why `build_nn_surfaces_full.py`
zero-fills the mu2 fields of its corpus-derived surfaces.

*The trap.* `samples_*.pkl.gz` holds `mu1_samples` and `mu2_samples` as
`(n_feat_diff, n_simulations, 2)` arrays in the same file, which looks like a
preserved per-trial pairing. It is not: across the whole 10k production archive every
column of both arrays is sorted ascending (verified over
`/gmm2/sim_samples_10k_20samples`, 8,432 files, and `/gmm2/sim_samples_10k_100samples`,
21,034 files; no sampled file retains the paired `full_results` array). Independent
per-axis sorting destroys the trial correspondence. Correlating row i against row i
yields a circular-linear r of 0.97-1.00, which is a Q-Q plot of two independently
sorted samples and not a covariation. Do not report that number. The small unsorted
`sim_samples_10k_20samples_new` archives (27 cells, 1,000 simulations) are the only
extant files where the pairing survives -- enough for a pilot look, nothing more.

*So a joint needs re-simulation.* It cannot be reconstructed from any artifact on
disk. The per-trial data is cheap to keep, though:
`simulate_dual_component_bias_distribution` already appends both biases as the last
two columns of its full-results array (`jax_fit_main.py`, indices 21 and 22), and
`generate_expanded_corpus.simulate_block` simply drops column 22. Retaining it is an
extraction and accumulation change; the generative model and the EM stay
bit-identical. The 4.1p corpus took 18,789 s (5.2 h) on one GPU for 148,076 cells at
20,000 outcomes per cell, so about 10.5 GPU-hours covers both sample counts. If a
corpus re-run ever happens for another reason, keeping column 22 then is nearly free
and saves the re-run later; whatever writes it must not re-introduce the per-axis
sorting.

*Three things that would need deciding first.* Joint resolution: at 148,076 cells x 2
items, float32 uncompressed, 720x720 is ~614 GB and out, 180x180 ~38 GB, 120x120
~17 GB, 90x90 ~9.6 GB, against 140 MB today for the mu1 marginals alone -- so
full-resolution marginals plus a coarser joint, with the joint grid chosen from what
a model needs rather than from what fits. Edge policy: mu1 wraps, so the existing
"lost outcomes to the histogram edges" check can raise and be right, but mu2 is
linear and unbounded and a fixed range will genuinely be exceeded -- widen, clip with
a recorded count, or raise, and record how many fell outside either way (CLAUDE.md
section 6). And the model family: K12 is a mixture of wrapped normals over one
circular variable, while a joint over (mu1 circular, mu2 linear) is a cylindrical
density -- wrapped normal times normal per component with a correlation term, or an
explicit p(mu1) p(mu2 | mu1) -- which touches the family, the analytic moments, the
arc probabilities, and the motor convolution (motor noise convolves the mu1 axis
only). That is a research stage with its own acceptance, not a retrain, and whether
K12's current mu1 marginal survives unchanged under it is an open empirical question.

**End state per pipeline stage**

Stage numbering follows `MODEL_PIPELINE_FOR_AGENTS.md`, which is itself an update
target: its S5 currently describes the surface NN as *the* surrogate, and every
future audit reads it first.

| Stage | End state after step 7 |
|---|---|
| S1 trial simulation (generative model + EM), S2 bias readout | Unchanged and still required. It produces both the averaged surfaces and the K12 training corpus, and the recovery panel in step 5 generates from it. |
| S3 KDE surface aggregation, S4 surface parameter grid | Retained, with a changed role: they no longer feed a surrogate, and instead produce the averaged surfaces as a standing empirical reference and as the source for mu2. `shared/averaging.py`, the averaging and pull scripts, the `averaged_surfaces_dir` machinery, the `cloud/` bundling and release, the `DEMIXING_ARTIFACT_ROOT` surface release and the `averaged_surfaces_smoke_pipeline/` fixtures all stay. Say in their documentation that they are no longer a step in the live fitting path, so the KDE aggregation stops being read as one. |
| Training corpus | New standing artifacts alongside the averaged surfaces: the n=20 `4.1p` and n=100 `4.1o` histogram corpora that trained the packaged K12 models. They need the storage, provenance and regeneration story the averaged surfaces already have -- where they live, how they are rebuilt, and what identity ties each K12 artifact to its corpus. |
| `surface_browser/` | Gains on-demand K12 evaluation alongside the stored surfaces, not instead of them, so a K12 prediction can be plotted against the simulated surface at the same parameters. Parameter entry becomes continuous for the K12 view while the stored-file lattice selection in `core/data_manager.py`, `core/surface_selector.py`, `components/sidebar.py` and `main_app.py` keeps working for the surface view. |
| mu2 curves (`surface_simulator.py`, the `use_nn_surfaces=False` branch, the R wrapper's mu2 columns) | Unchanged. K12 models mu1 only; mu2 keeps its averaged-surface path, so there is no gap and no interface change. `bias_model_comparison/plots/plot_CSH2026_predictions.R` uses that branch and needs no migration; only its `_nn` twin, which loads a checkpoint, does. |
| S5 surface NN: `neural_network_optimization/`, `pretrained/model_epoch*.pkl`, `shared/utils.py:load_checkpoint` and its legacy mu1-axis handling, `tests/test_nn_circular_loss.py`, the surface-specific parts of `tests/test_mu1_circular_axis.py` | Removed at step 7. The mu1-axis guards protect the NN's 180-row surface grid; check each against the averaged-surface path before deleting it, since that path keeps the same grid and may rely on the same guard. |
| S6 data ingestion, S7 objectives, S8 hierarchical search + motor noise | Preserved exactly as specified in the target inventory below; only the surrogate call is rerouted. |
| S9 rescoring, S10 plots/exports/prediction APIs | Rerouted to the shared provider; the NN branches are deleted at step 7 rather than left as dead selectable paths. The averaged-surface and mu2 branches are untouched. |

**What the current code requires**

Paths in the table are relative to the inner DM repository unless stated otherwise.

| Area | Current implementation | Transition consequence |
|---|---|---|
| Checkpoints | `shared/utils.py:load_checkpoint` reconstructs a surface `TrainState`, including legacy mu1-axis handling. The optimizer expects epoch/loss metadata. | A WNM artifact is not load-compatible. Introduce family-aware loading without modifying the historical surface interpretation. |
| K12 implementation | `continuous_density/wrapped_mixture_model.py` already implements the network, point/grid log density, circular moments, arc probabilities, and analytic motor convolution. | Reuse it. Do not duplicate mixture mathematics in the fitter or plotting scripts. |
| Candidate artifact | 4.1q `train_wnm_full.py` saves research fit dictionaries with `variables`, steps, and validation loss; it does not use the module's self-contained `save_model` format. | Package the selected weights with architecture and scientific metadata before loading them in production. |
| Hierarchical fitting | `grid_based_multi_condition_optimizer_jax_loops.py` loads the NN, batches 3-parameter inputs, materializes surfaces, then derives every objective. `predict_nn` wraps `_predict_batch_fixed_size`. | Extract reusable target construction and loss evaluation; connect WNM predictions to the existing hierarchical strategy for a fair comparison with direct fitting. |
| Main fitting command | `fit_model_to_data.py` defaults to `density`, dispatches exhaustive search only for that method, and writes cross-objective scores as well as the requested fit. | Every written score must understand WNM, even if only density fitting is requested. |
| Exhaustive search | `exhaustive_density.py:CurveSource` already separates curve storage from loss evaluation. `curve_cache.py:build_curve_lattice` still builds surfaces to obtain asymmetry. | Use `CurveSource` with WNM curves for exhaustive search, both as a lattice reference and as a production candidate. Measure cache amortization across datasets. |
| Density target | `generate_nn_density_asymmetry_batch` derives smoothing SD from `weights_sd / feat_diff_step`; `shared/utils.py:compute_single_density_asymmetry` performs edge-padded Gaussian smoothing. | Apply the same smoothing to analytic WNM asymmetry. Scoring raw WNM asymmetry against the current target would change the estimator. |
| Trial likelihood | Hierarchical fitting indexes grid log density; `postprocess_fitted_likelihoods.py:score_fit_row` independently repeats that indexing and converts density to approximate cell mass. | Introduce explicit backend scoring semantics and share them between fitting, cross-evaluation, and rescoring. |
| Motor noise | Fitting, rescoring, and plots independently FFT-convolve surfaces, with a density floor. | WNM should widen component variances analytically, once, before requesting any output quantity. |
| Plots/exports | `create_unified_subject_plots.py` recomputes surfaces, motor noise, moments, asymmetry, pooled SD, and pooled report-order scores. `plot_pdf_slices.py` constructs the optimizer too. | Route these consumers through the same surrogate operations and fitted-run identity. Preserve pooling definitions. |
| Prediction API | `surface_simulator_for_predictions/surface_simulator.py` selects checkpoints by sample count and also supports actual averaged surfaces and mu2 predictions; its R wrapper forwards checkpoint overrides. | Update model selection and mu1 predictions while retaining the separate simulation-surface/mu2 functionality. K12 models mu1 only. |
| Identity | `run_fingerprint.py` hashes checkpoint/data and records objective/grid settings. `curve_cache.py` hashes checkpoint and curve settings, but not an explicit surrogate/evaluator contract. | Extend the existing identity mechanisms to distinguish family, sample count, and evaluation semantics. |
| Other in-repo surrogate consumers | `scripts/compare_search_backends.py` and `scripts/validate_exhaustive_reference.py` already construct the surrogate to compare search backends. `scripts/plot_CSH2026_uev_figures_nn.py`, `scripts/plot_CSH2026_predictions_nn.R`, `scripts/plot_Chetverikov2023_uev_figures_nn.R` and `demo_fischer_whitney.py` build figures and the public demo from a checkpoint; `build_curve_cache.py` hardcodes one in its usage text. | Extend the two search harnesses for step 3's benchmark rather than writing a third comparison script. Migrate the figure scripts and the demo, then drop the `_nn` suffix that only made sense while two surrogates existed. |
| Full-chain fixtures | `tests/run_smoke_pipeline.sh` trains a small NN and fits with the resulting checkpoint end to end; `run_smoke_standard.sh` and `run_smoke_compare_seeds.sh` are its siblings. | These are the only fixtures that exercise the whole chain. Give them a K12 arm before step 7, or step 7 deletes the only end-to-end coverage the repo has. |
| Test collection | The 21 wrapped-mixture test files live in `continuous_density/tests/`, outside the `tests/` tree the project actually runs (`JAX_PLATFORMS=cpu pytest tests/ -n auto`). | Production code needs production-collected tests. Move or collect the K12 contracts into `tests/` as part of step 1, not after promotion. |
| External orchestration | The epoch-1500 filename is constructed at eleven workspace sites, not one: `bias_model_comparison/pipeline/regenerate_all_fits.sh` (preflight, ordinary fits, postprocessing, motor runs), `regenerate_motor_noise_phase.sh:522`, all eight `bias_model_comparison/fits/fit_*.py` (each carrying its own `(label, checkpoint)` tuple list), and the plots/docs under `bias_model_comparison/plots/` and `analysis/POSTPROCESSING.md`. | Update every site in the same change. A default change inside DM cannot reach explicit external overrides, and fixing only the site that was noticed is the failure pattern CLAUDE.md section 3 records. Re-grep for `model_epoch` across the workspace after the change and require zero remaining hits outside archived results. |

**Existing fitting targets and validation to carry forward**

This inventory was checked against executing source and existing test definitions,
not inferred from historical audit status markers. Those tests were inspected, not
rerun for this documentation-only change. Existing parity tests establish specific
numerical contracts; they do not certify every objective as scientifically optimal
or every new WNM gradient as correct.

| Method | Definition to preserve | Existing implementation and evidence |
|---|---|---|
| `density` (production default) | Empirical wrapped-KDE signed-mass curve on 2:2:180; Gaussian feature weights SD 20 model degrees; pooled subject-condition SJ bias bandwidth. Model analytic sign mass followed by the existing 20-degree edge-padded feature smoother. Minimize 1−CCC over the curve, then sum condition losses. | `fit_model_to_data.DENSITY_CURVE_SPEC`; optimizer `_precompute_target_curves`; `shared.utils._compute_empirical_density_asymmetry_core`; `density_objective.ccc_loss`. Tests: `test_density_ccc_objective`, `test_density_objective_parity`, `test_density_degenerate_target_refusal`, `test_bandwidth_rules`, `test_density_kde_wrap`, `test_density_kde_resolution`. |
| `density_legacy` | Same configured empirical target; historical weighted MSE/range plus 1−correlation. Preserve for replay, not as the new default. | `_compute_curve_losses(loss_type="combined")`; CCC regression tests demonstrate the legacy amplitude weakness. |
| `expectation` | Circular mean within nearest-center 4-degree feature bins; trial-count weights within condition; angular MSE against predictions at the existing returned feature indices. | `compute_target_bias_curve_core` masks padding and returns indices/means/counts; `_compute_curve_losses` supplies angular loss. Preserve the terminal center-to-grid mapping, rather than reconstructing bins from prose. Target-specific validation is less extensive than density's; add extraction-parity fixtures. |
| `smoothed_exp` | Trial-weighted rolling circular mean (Gaussian SD 20) versus pointwise model mean on the feature grid; unweighted angular MSE across grid locations. | `shared.utils.compute_target_bias_rolling_curve_core`, `_precompute_target_curves`, and the live scoring branch. There is currently no added 20-degree model smoother here. Preserve and document this asymmetry; do not silently replace it with a new matched-smoothing target. |
| `likelihood` | Same cleaned trial set and model-space scaling. Historical NN score indexes grid log density; WNM point-density evaluation is a separately versioned score convention. | Hierarchical likelihood branch and `postprocess_fitted_likelihoods.score_fit_row` currently implement the reproduction contract. New continuous fit/export parity is required. |
| `crps` | Per-trial circular-distance energy score, E[d(X,y)]−0.5 E[d(X,X′)], summed over trials. Preserve current observed-bias/feature grid indexing initially; changing that would be another score version. | Hierarchical `crps` branch and its circular-distance matrix. WNM supplies probabilities for the same cells; add direct-formula parity tests for the new call site. |
| `balanced_crps` | Feature-local Gaussian-weighted empirical bias histograms. Weight feature locations uniformly where support exceeds 1% of median support; normalized score is 2 E[d(X,Y)]−E[d(X,X′)]. | `compute_bwcrps_condition_targets` and `bwcrps_energy_score`. Preserve the factor of two relative to the ordinary CRPS convention and the support-mask normalization. |
| `bias_weighted_crps` | Same empirical distributions and score as balanced CRPS, weighted by squared Gaussian-smoothed **linear signed mean bias** times the support mask; denominator floor 1e-10. | Same shared helpers. Workspace `bayesian_biases_zoo/tests/test_bwcrps_parity.py` checks live DM helpers against the reference and cross-family scores at 180/360 periods; DM `test_pooled_bwcrps_export.py` checks report-order pooling. |
| Circular SD | Reporting/validation quantity, not an existing independently selectable fitting objective. Preserve the small-sample empirical correction and data-weighted pooling of model distributions/moments. | `create_unified_subject_plots.compute_empirical_sd_curve`, `compute_feat_bin_weights`, pooled SD helpers; `tests/test_sd_estimators.py` contains numerical checks, some executed at module scope. |

The density target has important validated implementation details: it uses real
unpadded data, wraps the bias KDE, excludes zero and the antipode from the discrete
sign masks, and uses `rescale_bias_for_grid` plus its existing fallback bandwidth
floor for sub-cell KDEs. Replacing it with empirical mean(sign(bias)), a plain
histogram sign split, or a new analytic KDE integral would change the target.
Preserve pooled/per-condition/average bandwidth choices and the SJ/Silverman
selection, with the production default explicitly pooled SJ.

Workspace `bayesian_biases_zoo/tests/test_density_target_parity.py` imports the real
DM target and tests bandwidth, wrapping, feature weights, sign conventions, and
end-to-end curves. Keep this cross-family contract running after extraction, with
both repositories present so an import skip cannot masquerade as validation.
Use `test_bwcrps_parity.py` similarly. No BBZ runtime refactor is required.

Do not transfer the 4.1q resultant >=0.5 evaluation gate into production empirical
fitting. The production density refusal is instead target-curve variance <1e-10,
checked before optimization and scoped to the density objectives. Mean-only motor
unidentifiability and non-finite/undefined moment handling remain explicit.

The current empirical and predicted smoothing operations are not mathematically
identical: empirical weights follow actual trial locations, while the model uses
finite-grid edge-padded smoothing; the NN also inherited KDE smoothing during
training. Preserve the established empirical target and downstream smoother, but
require the acceptance panel to check K12 against that actual target. The raw 4.1q
curve advantage alone does not validate this fitting-estimator interface.

Before replacing target preparation, record arrays and per-condition scores from
the existing entry point on fixed unequal-length, sparse, narrow-error and reversal
fixtures at both data periods. Require unchanged targets, masks, bandwidths, and
condition aggregation from the extracted path. Separately verify model-side analytic
approximations and gradients; agreement of empirical targets does not establish them.

**Implementation sequence**

1. **Package and resolve the candidates.**

   Use the exact `alldata` checkpoint paths recorded in
   `artifacts/production_n20_manifest.json` and `production_n100_manifest.json`.
   These are full-corpus K12 fits selected at steps 90000 and 89000 respectively.
   Package without retraining: K=12, hidden widths (128, 256, 256), minimum scale
   0.25 degrees, and the existing feature transformation. Record the source
   checkpoint hash, simulator/corpus provenance, observer sample count, period
   360, per-model-degree density units, attraction-positive bias, parameter order,
   42-degree spatial-separation convention, supported domain, and artifact schema.
   Record also that these `alldata` checkpoints selected their step on an in-sample
   10% slice (`4.1q/scripts/train_wnm_full.py --all-data`), deliberately mirroring
   the production network, which has no validation split at all. Their recorded
   validation NLLs (3.978 at n=20, 3.166 at n=100) are therefore in-sample numbers
   and must never be reported as held-out scores. The 4.1q accuracy comparison is
   unaffected: `design_benchmark.py` raises if any benchmark trajectory is a
   training trajectory.

   Extend/reuse `wrapped_mixture_model.save_model/load_model`. Add a small shared
   surrogate module, proposed `shared/surrogate.py`, with one checkpoint resolver
   and family-aware loader. Expose `--surrogate surface_nn|wnm` and an explicit
   observer-sample selection consistently at public entry points; preserve
   `--checkpoint-path` overrides. Explicit checkpoint metadata determines family;
   contradictory explicit family/sample-count requests raise. Historical surface
   files continue through `load_checkpoint`, with sample identity supplied by the
   known registry or an explicit selection when metadata is absent. Do not guess
   arbitrary checkpoint identities from substrings.

   Keep the old defaults through the acceptance stage. Verify packaged predictions
   against the original research weights before integrating the fitter. Production
   inference must not import sibling experiment scripts or require the corpus.

2. **Provide a small shared prediction contract.**

   Support batched parameters and requested feature differences, with operations
   for log density at observations, first circular moment, signed-arc asymmetry,
   and grid probabilities for consumers that actually require them. Apply motor
   noise within this layer. Backend metadata travels with the loaded object.
   A pair of concrete implementations is sufficient; a plugin system is unnecessary.

   For WNM, compute means/resultants/SD and asymmetry analytically. Component 2 is
   obtained by swapping feature SD inputs, following the existing training
   convention; do not add a bias-sign flip. Map spatial noise to d-prime only at
   interfaces that need it, using 42/sd_spat consistently. Validate bounds outside
   compiled kernels; replace the optimizer's out-of-domain dummy inputs of ones
   with legal warm-up inputs, and use valid values for padded batches.

   Extract the existing curve-smoothing operation from
   `compute_single_density_asymmetry` into one reusable helper. Its surface path
   must remain numerically unchanged. Apply it to WNM's analytic signed-arc curve
   using the same feature grid, kernel width, edge padding, and configuration.
   Keep raw and fitting-smoothed curves distinguishable.

   Do not route all WNM predictions through a 180-row surface. A narrow component
   can be much smaller than the 2-degree reporting cells; renormalizing its sampled
   peak values cannot guarantee correct moments or sign mass. Use grid densities
   for display, and integrated cell probabilities or convergence-checked quadrature
   for distributional scores. Reuse the existing arc-probability implementation
   where appropriate, checking its wrap truncation over the supported scales.

3. **Implement and compare search candidates on shared targets.**

   Add a compact optimizer module, proposed
   `model_fit_to_data/continuous_optimizer.py`. Use bounded L-BFGS-B in log SDs
   with JAX value/gradients, using the already available SciPy optimizer rather
   than adding an optimization dependency. This is one candidate alongside the existing searches; neither differentiability
   nor a lower loss in one case establishes that it is the better production search. Use deterministic,
   dispersed multistart initialization and record all final losses, convergence
   statuses, iterations, and boundary hits. Test directional derivatives against
   finite differences, including wrapped-normal branch transitions and motor noise.

   For C conditions, optimize 2C feature SDs and one shared spatial SD, plus one
   shared motor SD when enabled. Preserve the existing sum of per-condition losses;
   do not introduce trial-count weighting across conditions for curve objectives.
   Preserve the empirical motor cap and the policy of skipping mean-only objectives
   when motor noise is free. A fixed zero motor SD is a separate no-motor case, not
   a log parameter. Retain legal explicit bounds; do not hide boundary solutions
   behind a saturating sigmoid.

   Extract the current one-time target preparation out of the surface optimizer,
   proposed `model_fit_to_data/fitting_targets.py`, and have both old and new
   engines call it. Reuse the existing target helpers and `density_objective.ccc_loss`.
   If public helpers move, preserve imports used by BBZ parity tests. Compute pooled
   bandwidths and empirical weights once on actual, unpadded condition observations,
   outside the differentiated objective. No surface checkpoint should be required
   merely to construct empirical targets.

   Score analytic WNM moments for `expectation`/`smoothed_exp`, fitting-smoothed
   analytic asymmetry for `density`/`density_legacy`, and cell probabilities for
   the CRPS variants. Preserve each method's target, weights, normalization, and
   feature locations as specified in the target inventory above. Reuse
   `compute_bwcrps_condition_targets` and `bwcrps_energy_score`; do not recode them
   in the new optimizer. Route cross-objective evaluation through the same scorers.

   For WNM likelihood, evaluate continuous density at each retained trial's actual
   bias and feature difference. Retain historical grid likelihood for the surface
   backend. This is an intentional, versioned evaluation change; changing search
   algorithms does not by itself authorize changing a fitting target. Batch trial
   evaluation as needed. WNM motor noise uses analytic variance addition.

   Expose `--search continuous` for WNM while preserving historical search options.
   Complete all scores written by the public command before enabling ordinary WNM
   runs. Adapt the hierarchical strategy to consume the same WNM quantities,
   reusing its enumeration, refinement, and condition-sharing logic rather than
   forcing analytic WNM curves through a coarse bias surface. At minimum compare
   density end to end; other supported objectives must pass the same scorer parity
   before comparing their searches. A search fallback must be explicit and recorded,
   and must never substitute a different surrogate.

   Connect WNM fitting-smoothed curves to `CurveSource` and the current cache
   builder. A small in-memory lattice first establishes parity; use a representative
   disk-backed cache to benchmark the cached production workflow. The exhaustive
   optimum is exact on its specified lattice, not over continuous parameters. Compare
   hierarchical, direct gradient, and exhaustive searches on identical WNM scorers,
   targets, bounds, and datasets. Include a grid/cached-seed plus gradient-polish arm;
   distinguish improvement from better basin selection versus sub-lattice refinement.

   The cached density path currently excludes motor noise. Report that limitation
   rather than applying its no-motor result to a motor-enabled problem. Compare
   hierarchical and direct searches there, using an affordable explicit joint grid
   as a diagnostic reference. Do not generalize the cache to every objective as a
   prerequisite for this benchmark.

   Historical evidence in `OPTIMUM_SEARCH_TESTS_temp.md` motivates this comparison:
   multistart gradients won a single-condition case but missed the hierarchical
   multi-condition solution, and grid-seeded polish helped. Those exploratory numbers
   used a surface checkpoint and the older combined density objective; current CCC
   and K12 differ. They establish neither the present winner nor a permanent failure
   of gradients. Re-run the comparison on current code and targets.

   Use the recovery protocol below to choose search/restart settings on development
   cases, then freeze them for held-out benchmark cases. Retain a simple existing
   method if gradients do not improve the relevant accuracy/cost tradeoff. A hybrid
   or objective-specific dispatch is acceptable if the benchmark supports it; avoid
   an unrestricted optimizer competition. Full WNM cache rollout is conditional on
   the measured winner, not prohibited or mandatory in advance.

4. **Integrate rescoring, plots, prediction APIs, and provenance.**

   Two identity defects found by audit are fixed *here*, as a consequence of
   routing consumers through the recorded surrogate identity rather than as extra
   work. Neither is retired by the transition, because both are about choosing
   between the two observer models, and n=20 and n=100 both survive it:
   `postprocess_fitted_likelihoods.infer_checkpoint_path` selects a checkpoint by
   substring-matching the results path, so a path containing `20samples` anywhere
   rescores an n=100 fit under the n=20 model (its reproduction gate should catch
   this, but `repair_stale_reproduction.py` is exactly the tool someone reaches
   for to clear the resulting "failures"); and `surface_simulator` stamps output
   rows with the requested `n_samples` regardless of the checkpoint handed to it,
   which is ungated and silent, and which keeps a live averaged-surface branch.
   A third, the plotting default of epoch 1500 against the fitter's 1425, does
   dissolve with the surface backend and needs nothing here.

   Make `postprocess_fitted_likelihoods` use the fitter's likelihood evaluator.
   Keep trial inclusion, model-space conversion, and reductions consistent.
   Continuous density in data degrees requires adding log(360/circ_space) to the
   model-degree log density. Give exact interval probability and the historical
   density-times-bin-width approximation distinct names/metadata; never silently
   change the meaning of `loglik_mass`. Preserve legacy export columns where their
   meaning can be preserved, adding explicit estimator identity for new scores.
   Any head-to-head information-criterion comparison needs a common observation
   scoring convention, independent of which objective selected the parameters.

   Update unified plots, PDF slices, and Python/R predictions to use the shared
   provider. Pool complex moments before deriving bin-pooled SD; do not average SDs
   or angles. Preserve the report-order distribution pooling before nonlinear
   CRPS scoring. Plot sampling must not become the source of fitted summary curves.
   Keep averaged-surface inspection and mu2 simulation outputs on their existing path.

   Extend `run_fingerprint` and the curve-cache key using their existing structured
   payloads: surrogate family/artifact schema, sample count, predictor/evaluator
   version, and numerical probability convention where applicable. Record the new search
   method, log parameterization, starts/seed, bounds, tolerances, and iteration cap;
   do not fingerprint a continuous run as though it used the old grid schedule. Checkpoint
   content hashes already exist; reuse them. Version objective semantics for the
   analytic/grid differences. Record equivalent identity alongside predictions and
   exported fits so historical files cannot be plotted using today's default model.
   Keep the flat results dictionary structure; use the established sidecar pattern.

   Write WNM fit outputs and any optional reference caches in new directories. A WNM run must not resume
   into an NN directory. Rescoring and plotting must validate the recorded identity
   against the supplied checkpoint, including in repair workflows (`repair_stale_reproduction.py`
   included). Historical runs retain an explicit replay path -- transitional, like the
   retained grid likelihood in step 3: both are removed at step 7, and neither may
   accumulate features in the meantime.

5. **Run parameter recovery and search comparison, then freeze acceptance.**

   Parameter recovery is a required benchmark, not just a qualitative sanity check.
   The earlier K12 CCC/MAE results evaluate the forward map at known parameters.
   Recovery tests the inverse map: generate responses at known parameters, rebuild
   empirical fitting targets from those responses, fit without revealing the truth,
   and compare recovered parameters and independent predictions.

   Existing follow-ups are in `continuous_density/AUDIT_SUMMARY.md` (next steps,
   item 5) and `continuous_density/RESULTS.md` (remaining recovery work). Workspace
   `DEMIXING_MODEL_CONSOLIDATED_AUDIT.md` identifies missing recovery oracles for
   estimator asymmetries; `BWCRPS_VALIDATION.md`, section 4, specifies recovery for
   the alternative models and is useful design context, not completed DM evidence.
   At plan preparation, no DM recovery runner or result set was found in the
   source/results search. That baseline is now superseded: the runner lives in
   `model_fit_to_data/run_recovery_panel.py`, findings in
   `continuous_density/RECOVERY_FINDINGS.md`, and generated rows/summaries under
   `results/continuous_density_4.1q/recovery/`. These historical notes are
   pointers, not evidence that their other old bug/status claims remain current.
   DM `TODO.md`, item 3, tracks this work.

   **Separate three questions in the benchmark.**

   - Search: given exactly the same surrogate, target, and simulated observations,
     which method finds a good solution reliably, and at what cost?
   - Surrogate: with a common sufficiently effective search, does K12 improve
     parameter and predictive recovery over the deployed surface NN?
   - Target/identifiability: does minimizing a chosen empirical objective recover
     the generating parameters at all, even with a good surrogate and search?

   First use a small WNM-generated closed-loop panel to check simulation/fit wiring
   and gradients. This is necessary diagnostic evidence, not final recovery evidence:
   a model fitting its own samples cannot expose its approximation to the observer.
   For the main panel generate independent behavioral outcomes with the actual DM
   observer simulator, at known feature/spatial noise and observer sample count;
   add known independent wrapped motor noise where specified. Do not generate the
   main panel from K12 or the surface NN. Preserve both item/report-order and
   data-to-model angle conventions. Reuse the actual simulator and result-handling
   helpers; do not copy BBZ's different generative model or create a second EM code.

   Proposed fixed starting design: 12 generating groups (4 single-condition,
   4 multi-condition without motor noise, 4 multi-condition with shared motor
   noise), both observer sample counts, 2 behavioral trial-count levels (100 and
   500 per condition), and 5 independent response seeds: 240 synthetic group
   datasets. Freeze actual parameter vectors and condition counts before running.
   Cover interior and boundary values, asymmetric feature noise in both orders,
   high/low d-prime, concentrated/broad responses, and realistic discrete as well
   as continuous dissimilarity designs. Respect trial-count imbalance in selected
   multi-condition cases. Observer n=20/100 and behavioral trials=100/500 are
   different experimental axes. If production datasets need other trial counts,
   settle that before simulation rather than changing the panel after results.

   Partition generating groups into development and held-out groups, stratified
   by single/multi-condition and motor status. Use development groups for starts,
   step sizes, stopping tolerances and numerical score settings; keep response
   replicates and optimizer-start seeds separate. Freeze the selected settings
   before scoring held-out groups. Pair every optimizer on identical synthetic
   datasets and give it the same bounds and target construction. Do not initialize
   main fits at the truth; truth-start runs are separately labeled diagnostics.

   Cover the four objectives retained by `bias_model_comparison`: `likelihood`,
   `bias_weighted_crps`, `density`, and `smoothed_exp`. Check optimization quality
   independently for each because their landscapes differ; the winning search need
   not be universal. Treat likelihood as the primary parameter-recovery objective
   and bias-weighted CRPS as the secondary distributional objective. Test the two
   curve objectives too, but judge them primarily by curve and held-out predictive
   recovery rather than expecting strong parameter identification. Raw `expectation`
   is superseded in the comparison analysis, and legacy density needs replay parity,
   not a full recovery campaign. Keep all methods' loss scales distinct.

   In comparisons of search, reevaluate every candidate through the same K12 scorer.
   In comparisons of deployed pipelines, fit the paired data with K12 plus its
   objective-selected search and with the surface NN plus its existing production
   search, then compare common downstream scores. The surface arm is a legacy
   baseline, not a second development target.

   Keep float64 out of the fixed matrix unless a key likelihood or BWCRPS search
   comparison shows a material precision-sensitive solution. In that case run a
   targeted matched x64 diagnostic and report it separately. JAX's x64 flag is
   global, so adopting it for the surface backend would require re-verifying that
   backend's arithmetic; otherwise defer global adoption until the surface backend
   is retired.

   Record per replicate: generating and recovered `sd_feat1`, `sd_feat2`, shared
   `sd_spat` (plus derived d-prime), shared `sd_motor`, condition contrasts, signed
   error, absolute/log-ratio error, boundary hits, objective value at truth and at
   the fit, all start outcomes, convergence status, runtime and peak memory.
   Summarize parameter bias/RMSE and failure rates across response replicates,
   separately by generating regime, sample count and trial count. Correlation of
   fitted and true parameters alone is insufficient. Report uncertainty intervals
   for these summaries; assess interval coverage only if the fitter actually
   produces parameter intervals.

   Evaluate recovered predictions against an independent simulator reference on
   dissimilarity bands, including regions absent from sparse training designs.
   Report CCC and absolute errors for bias/asymmetry/spread plus distributional
   scores and resultant-aware moment diagnostics. Better training loss need not
   imply better recovery or better held-out predictions. Dense reference curves
   can be shared across response replicates, but never used as their fit targets.

   Diagnose failures with a bounded escalation: (a) evaluate all candidate solutions
   through the same scorer and inspect optimization gaps; (b) profile weak parameters
   or competing basins; (c) fit expected/high-precision targets for a few failing
   generating cases to distinguish finite-trial variation from systematic target
   smoothing/weighting bias or surrogate error. Density asymmetry and mean-only
   objectives may not identify all noise parameters. Good curve recovery with poor
   parameter recovery must be reported as such, not declared a passed parameter
   test or automatically blamed on search. Do not demand exact recovery of an
   unidentifiable parameter; characterize the equivalence region and restrict the
   corresponding scientific interpretation.

   Compare optimization success versus wall time and evaluator calls, both at
   comparable compute budgets and at the proposed production settings. Include
   cache construction, loading, warm scoring, storage, and amortization over one
   versus many subjects. A reused cache may beat repeated continuous fits even if
   its first build is costly. Report lattice spacing/discretization separately from
   optimization failures. The choice is empirical: retain hierarchical/cached
   search, use gradients, or choose a simple hybrid based on the held-out tradeoff.
   Predeclare acceptable recovery/predictive errors and runtime budgets before the
   final panel; do not force a gradient win into the acceptance criteria.

   Synthetic datasets, references and fits should be resumable, with identities
   covering simulator, checkpoint, target definition, search settings and seeds.
   Preserve responses, parameter truth, optimizer traces and per-condition scores
   together so subsequent diagnosis does not require rerunning expensive simulation.



   Select and record the acceptance panel and thresholds before inspecting its
   results. Use both observer sample counts; 180- and 360-period data; asymmetric
   feature noise in both orders; low/high d-prime; narrow peaks; broad/low-resultant
   densities; and feature differences near 2 and 180. Include a small multi-condition
   real-data panel, report-order pooling, and motor-off/on cases. This is a fixed
   release panel, not an expanding benchmark campaign.

   | Check | Required outcome |
   |---|---|
   | Packaging | Production-loaded weights reproduce the selected research checkpoint to numerical tolerance on fixed inputs. |
   | Mathematical contract | Normalization, circular periodicity, mirrored item mapping, and analytic motor convolution agree with independent numerical checks, including narrow and broad densities. |
   | Shared scoring | Live, cache, fitted cross-objective scores, exported curves, and likelihood rescoring agree at identical parameters within declared numerical tolerances. |
   | Search quality | Compare hierarchical, cached exhaustive, multistart gradient, and grid-seeded polish on identical WNM problems; select a stable accuracy/cost tradeoff on held-out recovery cases. Record loss gaps to the lattice reference and start-to-start variability; historical surface-search fixtures remain unchanged. |
   | Estimator integration | Smoothed WNM curves reproduce direct analytic prediction followed by the existing smoothing operation; pooled SD and report-order scores preserve their estimator definitions. |
   | Accuracy retained | Re-evaluate the final packaged checkpoints on 4.1q, separately by band and random/hard group. Recover the recorded raw-curve advantage and check fitting-smoothed curves too. Name the surface baseline per sample count before running: three different NN checkpoints are currently in play -- 4.1q compared against `pretrained/model_epoch1425_10ktrain_20samples.pkl` (also `fit_model_to_data.py`'s default) at n=20 and `results/neural_net_checkpoints_100samples_circular_4.1p/model_epoch_1500.pkl` at n=100, while the external pipeline deploys `model_epoch1500_10ktrain_{20,100}samples.pkl`. Retirement is judged against what production actually runs. |
   | Full density | Score these exact checkpoints against the surface comparators on independent references, including ungated broad cases. Start with existing benchmark histograms and a consistent bin-probability comparison; use selected fresh raw/high-precision references only if a material discrepancy remains unresolved. |
   | Actual fitting | Run the public fit → export/plot → likelihood-rescore chain. Require completion, reproducible losses, and no unexplained material failures. Real-data fit quality need not always favor the more faithful simulator surrogate. |
   | Recovery | Complete the replicated simulator-based recovery protocol below. Report each noise parameter, condition contrasts, objective-specific identifiability, and held-out banded predictions; distinguish search failure, surrogate error, and target-induced non-recovery. |
   | Performance | Measure cold and warm fitting, cache build/load, amortized multi-subject throughput, and plotting time plus peak memory on the actual panel. Predeclare an acceptable runtime budget; microbenchmarks alone do not establish production throughput. |
   | Identity | Wrong sample count/backend, stale cache, and mismatched replay checkpoint are rejected; interrupted runs resume only into matching outputs. |

   Reuse/extend the existing tests for wrapped mixtures, curve-cache/live-model
   parity, exhaustive density, search dispatch, objective parity, mu1 conventions,
   pooled BWCRPS, SD estimators, and run fingerprints. Exercise public call sites,
   not only adapter helpers. Run project tests with pytest-xdist where appropriate;
   use one GPU-dependent job at a time and put temporary/compiler caches under /tmp.
   Long acceptance stages should checkpoint whole stages and write progress logs.

   Stop when this panel passes. Do not require new architecture seeds, a K48
   convergence study, or the unfinished n=100 corpus-NN run to release a replacement
   for the existing surface NN. Reopen model research only for a material failure
   attributable to the surrogate, rather than implementation or scoring semantics.

6. **Promote, regenerate, and preserve replay.**

   After acceptance, install the packaged models in `pretrained/` under explicit
   WNM/sample-count names and switch the shared resolver defaults. Update
   `pretrained/README.md`, the root README, fitting/prediction documentation, and
   continuous-density status together. Document production-selected settings rather
   than confusing them with the mixture class's generic K8/smaller-network defaults.
   Include a minimal reproducible histogram trainer/packager in DM, refactoring the
   working research trainer rather than duplicating it; detailed corpus generation
   history and validation artifacts stay outside the tracked runtime documentation.

   Update the external pipeline's preflight, search dispatch, fit, plot, likelihood
   repair, and motor-run model selection together. Select the benchmark-supported
   search and require caches only for the selected cached path. Prefer calling the shared
   resolver instead of recreating filename conventions. Preserve the downstream
   fitted-parameter/curve schema and propagate surrogate identity. Rebuild affected
   DM outputs and downstream comparison artifacts in versioned directories; BBZ
   model fits do not need retraining just because the DM surrogate changes.

   Verify a normal checkout can fit and predict using shipped WNM artifacts without
   access to the 4.1q folder or any training surfaces. Through this step the surface
   backend remains selectable and its outputs are kept: rollback selects that backend
   and its matching result/cache namespace, and does not relabel WNM outputs or
   overwrite either checkpoint. This is the last step at which rollback exists.

7. **Retire the surface NN.**

   Only after every production artifact that a conclusion rests on has been
   regenerated on K12 -- DM fits, exports, likelihood rescoring, unified and PDF-slice
   plots, the R/Python prediction outputs, and the downstream `bias_model_comparison`
   fits and reports. Until that regeneration is complete the NN is still load-bearing,
   because the artifacts on disk were produced by it.

   Entry conditions, all required: both sample counts promoted and accepted; no
   remaining `model_epoch` reference outside archived results (grep the whole
   workspace, both repositories); the smoke pipelines running on K12; K12 contracts
   collected by `tests/`; the rollback window declared closed in writing. The smoke
   pipelines currently train a small NN as one stage of a longer chain -- replace that
   stage, keeping the surface-generation stages that feed the averaged-surface path.

   Then delete rather than deprecate: the NN backend and its dispatch, the surface
   `TrainState` reconstruction and legacy mu1-axis handling in `shared/utils.py`,
   `neural_network_optimization/`, the `pretrained/model_epoch*.pkl` checkpoints, the
   NN-only tests, and the transitional `--surrogate` selector once it has one member.
   Preserve `--checkpoint-path` for selecting among WNM artifacts.

   **Retire the migration scaffolding here too.** Several tests and their recorded
   references exist to prove a routing did not change the surface backend's
   numbers, not to describe behaviour anyone will rely on afterwards. While that
   backend is production they are load-bearing -- they guard deployed numbers --
   but once it is gone they test history:

   - `tests/data/fitting_targets_golden.npz` and the pre-extraction comparisons in
     `tests/test_fitting_targets.py`
   - `tests/data/plot_estimators_golden.npz` and `tests/test_plot_estimators.py`'s
     golden and `_pre_routing_pooled_sd` comparisons
   - `tests/data/pooled_bwcrps_golden.npz` and
     `tests/test_pooled_bwcrps.py::test_the_plotting_entry_point_is_unchanged_by_the_routing`
   - `tests/record_*_golden.py`, the recorders that wrote them

   Keep whatever states a *contract* rather than an equivalence: the clamp values,
   the wrap-not-clip binning, the pooling-before-scoring order, the empty-bin NaN.
   Those survive the backend. Delete the equivalences and the .npz files with it,
   rather than carrying a reference to a model that no longer exists.

   Scope this to the NN only. The averaged surfaces, their generation and release,
   the browser's stored-surface views and the mu2 branch of the prediction API all
   stay, so no interface loses columns and no artifact loses its source. Check each
   thing being deleted against that path before deleting it: `load_checkpoint`'s
   legacy mu1-axis handling and the mu1-axis guards in particular exist for the NN
   but concern a grid the averaged surfaces share. What must not survive is a dead
   NN branch kept "just in case" -- that is a second copy of the surrogate contract
   that nothing exercises, and CLAUDE.md section 11 is explicit that a guard no
   fixture satisfies has never run.

   Reproduction of pre-transition results is served by a tag on the last commit where
   the NN ran, plus the archived checkpoints and their result namespaces stored
   outside the runtime tree. Record the tag, the checkpoint digests and the archive
   location in `pretrained/README.md`'s successor and in the root README. Do not
   claim historical figures can be regenerated from the current tree once this step
   lands -- state that they require the tag.

   Update `MODEL_PIPELINE_FOR_AGENTS.md` (S5 in particular), `MODEL_PIPELINE_FOR_REVIEWERS.md`,
   `AGENT_AUDIT_GUIDE.md`, `Batch_Fit_Analysis_Pipeline_Documentation.md`,
   `neural_network_optimization/Neural_Network_Optimization_Pipeline_Documentation.md`,
   `surface_simulator_for_predictions/README.md`, `INSTALL.md` and `docs/` in the same
   change. Two things to state rather than leave implied: that the surrogate is now
   K12, and that S3/S4 still run but no longer feed it. An audit reference that still
   describes the NN as the surrogate will mislead every subsequent review, and one
   that leaves the KDE aggregation sitting in the fitting path will mislead it the
   other way.

**Practical implementation boundaries**

Use five reviewable changes: (1) artifact packaging, shared provider, and test
collection; (2) reusable targets, competing searches, and identity; (3) rescoring,
exports, plots, and prediction APIs; (4) acceptance results, default promotion, and
batch-pipeline rollout; (5) surface-NN removal and documentation, after downstream
regeneration. Keep WNM opt-in until all runtime consumers are connected, and keep
the NN removable-but-present until (5). The first three can be developed without
retraining either family. Existing unrelated working-tree changes must be preserved.

The largest risk is inconsistent estimator semantics between fitting and its
consumers, especially smoothed asymmetry, motor noise, and likelihood units. A
surface-shaped compatibility wrapper alone would leave that risk unresolved and
could discard the narrow-density accuracy that motivated this transition.
