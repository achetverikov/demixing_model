# JAX L-BFGS-B port: WNM objective comparison

Date: 2026-09-10

## Frozen cross-objective decision

All four retained objectives use 64-start batched JAX L-BFGS-B with float32
arrays and `highest` matmul precision. Default GPU TF32, mixed precision,
objective-specific caches, SciPy unions, and conditional hierarchies are not in
the selected path. This is an executive consistency decision rather than a
claim that one search dominates every empirical objective. It improves the
rare likelihood and matched-density misses, is neutral for BWCRPS recovery,
and knowingly accepts the matched-smoothed port's 91/120 development coverage
and 44.40 worst gap. Production routing has not yet been changed.

## Scope

The faithful implementation in `JAX_L-BFGS-B_port` was connected to the
existing single-condition recovery benchmark through a thin adapter. The
benchmark reuses its frozen datasets, objective evaluators, bounds, seed-0
log-space Latin-hypercube starts, result schema, and canonical scoring. No
objective or curve implementation was copied into the port.

The primary historical panel contains all 120 n=100 development datasets and
the four WNM objective branches implemented by the original benchmark. It uses
float32 search on GPU, 32 starts in
one device batch, and common float32 CPU rescoring of the returned endpoints.
The comparators are the already selected searches:

- likelihood: 32-start serial SciPy L-BFGS-B;
- bias-weighted CRPS: 32-start serial SciPy L-BFGS-B;
- density: the 80-by-64 log curve cache plus one SciPy polish;
- smoothed expectation: the selected cache/32-start-SciPy/conditional-hierarchy
  rule.

The density and smoothed-expectation rows in that panel use the legacy,
unmatched operators. Subsequent alignment work selected matched-KDE density and
observed-design matched feature smoothing as the current WNM operators. They
are evaluated separately below against their frozen held-out SciPy fits; the
legacy rows are retained only as historical optimizer evidence.

## Historical 32-start result

The port is a major throughput improvement, but 32-start batched L-BFGS-B is
not a universal replacement for the objective-specific searches.

| Objective | Port within 0.001 of two-arm union | Selected search within 0.001 | Worst port gap | Port median seconds | Selected median seconds, amortized | Median paired speedup |
|---|---:|---:|---:|---:|---:|---:|
| likelihood | 72.5% | 99.2% | 0.0366 | 0.287 | 3.085 | 10.1x |
| bias-weighted CRPS | 100.0% | 100.0% | 0.000954 | 1.083 | 20.400 | 19.0x |
| density | 99.2% | 100.0% | 0.0246 | 0.192 | 0.808 | 4.3x |
| smoothed expectation | 78.3% | 99.2% | 34.236 | 0.191 | 7.812 | 42.0x |

The table uses the port API's lowest finite device-rescored endpoint, with its
termination status retained. A stricter adapter policy that allowed only normal
convergence reduced coverage to 66.7%, 99.2%, 96.7%, and 77.5%, respectively.
That policy is not the port's `winner_index()` contract and is especially
misleading in float32: the selected finite winner had a non-converged status in
26.7%, 30.8%, 24.2%, and 16.7% of the four panels, but was finite and was
independently rescored on CPU.

### Interpretation by objective

- **Bias-weighted CRPS:** this is the clear candidate for replacement. The
  32-start port meets the existing 0.001 development gate on all 120 datasets
  and is about 19 times faster than serial SciPy. A separate success-only
  64-start run still missed one dataset by 0.00134; combining the 32- and
  64-start designs passed all 120 at a 6.3x speedup, but this extra budget is
  unnecessary under the port's finite-endpoint policy.
- **Density:** the direct port search is close and faster than the amortized
  cached winner, but one broad/weak dataset remains 0.0246 worse. This is a
  basin/start-design failure, not general local-solver failure. A complete
  64-start follow-up also covered 119/120: `broad_1_seed1_n900` remained 0.02477
  worse, while median solve time rose from 0.192 to 0.383 seconds and paired
  speedup fell from 4.3x to 2.17x. On a different earlier outlier, 8- and
  64-start Latin hypercubes found the cached solution while the distinct
  32-start design did not. Merely doubling a non-nested start design therefore
  does not provide the missing global-search guarantee, and the cache cannot
  yet be removed.

  Promoting the 32-start port search to float64 also covered 119/120. The same
  `broad_1_seed1_n900` basin remained 0.02462 worse. Float64 eliminated
  non-converged selected endpoints (0% versus 24.2% in float32) and reduced
  ordinary numerical gaps -- its p90 gap was zero after the common float32 CPU
  rescore -- but did not solve the global-search failure. Median solve time rose
  amortized cached winner rather than faster.
- **Smoothed expectation:** the port does not replace the selected global
  search. Misses concentrate in broad/weak cases. On the worst case, 8, 16, 32,
  and 128 starts remained about 34.24 behind the selected search; 64 starts
  improved the gap only to 32.08. The useful role for the port is to batch the
  eight cache-seeded local polishes and replace the expensive 32-start SciPy
  arm inside the existing union.
- **Likelihood:** parameter endpoints are usually very close (median endpoint
  log-RMSE 0.00133 under the success-only comparison), but GPU float32 search
  does not satisfy the existing 0.001-NLL gate. Increasing one worst case from
  32 to 128 non-nested starts reduced its CPU-rescored gap from 0.0366 to 0.0112
  but did not remove it.

  A complete mixed-precision follow-up kept the checkpoint and per-trial log
  densities in float32, but accumulated the NLL and ran the optimizer state and
  line search in float64. Runtime stayed essentially unchanged, and normal
  termination improved sharply, but solution quality did not:

  | Port precision | Within 0.001 of port/SciPy union | Misses | Worst gap | Non-converged winner | Median seconds | Paired speedup |
  |---|---:|---:|---:|---:|---:|---:|
  | float32 | 72.5% | 33 | 0.03662 | 26.7% | 0.287 | 10.1x |
  | float32 model + float64 sum/optimizer | 69.2% | 37 | 0.03760 | 1.7% | 0.278 | 10.9x |

  Against the three-way union of SciPy, float32 port, and mixed-precision port,
  the mixed run covered 68.3% and had a worst gap of 0.06586. The hard
  `broad_3_seed1_n900` fit still went to the `sd_feat2` upper-bound basin
  (`[123.07, 199.89, 8.32]`), with CPU-rescored NLL 4994.53418 versus
  4994.49658 for SciPy. A direct dtype probe confirmed float32 per-trial log
  densities, a float64 sum, and float64 gradients at the optimizer interface.
  Thus summation and optimizer-state precision are not the missing ingredient:
  the float32 model values and derivatives already change the local-search
  path.

  A device/matmul diagnostic isolated the source more closely. On
  `broad_3_seed1_n900`, the CPU float32 port and CPU float32 SciPy chose the
  same start and interior basin, both with NLL 4994.49609. With default GPU
  float32 arithmetic, both the batched port and scalar SciPy chose start 8 and
  the `sd_feat2` upper-bound basin. The disagreement therefore predates any
  difference between the two L-BFGS-B implementations. Across the 32 initial
  points, default GPU versus CPU NLL differed by a median 0.830 and as much as
  46.48.

  Setting `JAX_DEFAULT_MATMUL_PRECISION=highest` reduced those initial-point
  differences to median 0.00073 and maximum 0.02344. The GPU port and GPU
  SciPy then both returned the interior basin at NLL 4994.49707, within 0.00098
  of the CPU result. Port solve time increased from 0.430 to 0.548 seconds on
  this 900-trial diagnostic. The main discrepancy is therefore the reduced-
  precision GPU matrix-multiply path used by the float32 surrogate (TF32 on
  this NVIDIA device), not a failure of the port to implement SciPy L-BFGS-B.
  Batch size 1 still selected the same bad GPU basin under default matmul
  precision, so batching is secondary. This motivated a full likelihood panel
  with float32 arrays and `highest` matmul precision.

  The complete 120-dataset development panel confirmed the single-case result.
  With 32 starts and `highest` float32 matmuls, the port was within 0.001 of the
  port/SciPy union on 120/120 datasets; the maximum CPU-rescored gap was
  0.000977. Median solve time was 0.306 seconds versus 3.085 seconds for SciPy,
  a 9.65x paired speedup, and 8.3% of selected endpoints retained a non-normal
  termination status. Median joint parameter log-RMSE was 0.1493 versus 0.1495
  for SciPy.

  Against the union of every tested likelihood strategy, the `highest` run
  covered 119/120. The exception was `broad_3_seed3_n180`, where the earlier
  default-TF32 run happened to discover a different basin whose canonical CPU
  NLL was 0.06586 lower than both SciPy and the `highest` run. That is a global-
  search opportunity revealed by altered arithmetic, not evidence that TF32 is
  a faithful objective. For replacing the selected SciPy search, `highest`
  float32 is now the development winner; held-out confirmation remains.

### Highest matmul precision on the other objectives

The same 32-start, float32-array, `highest`-matmul configuration was run on all
120 development datasets for BWCRPS, density, and smoothed expectation. Every
endpoint was rescored with the same CPU float32 evaluator used for its selected
comparator.

| Objective | Matmul | Within 0.001 of selected/run union | Misses | Worst gap | Non-converged winner | Median seconds | Paired speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| BWCRPS | default | 100.0% | 0 | 0.000938 | 30.8% | 1.083 | 19.0x |
| BWCRPS | highest | 100.0% | 0 | 0.000824 | 5.8% | 1.067 | 19.4x |
| density | default | 99.2% | 1 | 0.02463 | 24.2% | 0.192 | 4.28x |
| density | highest | 98.3% | 2 | 0.11591 | 15.0% | 0.228 | 3.63x |
| smoothed expectation | default | 78.3% | 26 | 34.2362 | 16.7% | 0.191 | 42.0x |
| smoothed expectation | highest | 82.5% | 21 | 34.2361 | 35.0% | 0.239 | 33.1x |

For BWCRPS, `highest` preserves complete coverage, slightly improves the worst
gap and parameter recovery, and sharply reduces non-normal termination without
a timing penalty. It was therefore frozen before the held-out confirmation
reported below.

For density, `highest` does not repair the cache-beating failure and adds a new
one: `broad_1_seed1_n450` is 0.11591 worse than the selected cache search even
though the default-matmul port is marginally better than the cache there. The
default/`highest` union remains 119/120 and doubles solve time, so neither the
precision change nor their union can replace the density cache.

For smoothed expectation, `highest` improves coverage from 94/120 to 99/120,
but leaves the two largest broad-case failures essentially unchanged, including
the 34.236 worst gap. This confirms that its main problem is global basin
coverage, not TF32 fidelity. The selected cache/eight-polish/conditional-
hierarchy rule remains necessary.

### Held-out 32-start component checks

The corrected comparison imported the existing alignment evaluators directly
and used the 60 frozen held-out datasets. Density and smoothed expectation use
their current matched operators; BWCRPS has no matched/unmatched variant. Every
reference arm uses 32 SciPy starts, seed 0, and no truth start. The port used
the same starts with float32 arrays and `highest` matmul precision. All stored
endpoints were rescored with the respective current evaluator on CPU.

| Objective | Port within 0.001 of union | Reference within 0.001 | Port misses | Worst port gap | Port median seconds | Reference median seconds | Paired speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| BWCRPS | 100.0% | 95.0% | 0 | 0.000721 | 1.037 | 5.490 | 5.26x |
| matched-KDE density | 100.0% | 100.0% | 0 | 1.79e-7 | 0.214 | 5.323 | 25.75x |
| matched smoothed expectation | 98.3% | 91.7% | 1 | 2.682 | 0.261 | 6.084 | 22.10x |

BWCRPS passes the frozen held-out confirmation. Its median/global joint
log-RMSE is 0.2690/0.7319 for the port versus 0.2693/0.7763 for SciPy. The port
is within 0.001 of the union on all datasets while SciPy misses that gate on
three, so the batched port replaces serial SciPy for this objective.

Matched-KDE density is effectively endpoint-equivalent: median/global joint
log-RMSE is 1.1338/1.3047 for the port and 1.1339/1.3045 for SciPy. The batched
port can replace the serial SciPy component, but this is not a production-search
decision. The cache-plus-one-polish winner was selected for the original
analytic-sign-mass operator, not matched KDE.

Matched smoothed expectation is better than SciPy by more than 0.001 on five
datasets, but misses `broad_2_seed0_n450`: port loss 8.29395 versus 5.61148.
Its median/global joint log-RMSE is 1.1466/1.2334 versus 1.1297/1.2386 for
SciPy, so recovery is mixed rather than diagnostic of a winner. A post-hoc
hard-case probe found loss 9.8699 with 64 non-nested starts and 4.6848 with 128;
this confirms start-design sensitivity but cannot select 128 starts from the
held-out case. More importantly, SciPy alone is not the selected smoothed-exp
search: this is only a component comparison. The composite cache/eight-polish/
32-start-SciPy/conditional-hierarchy winner was built for the original
pointwise operator and must be rebuilt with matched complex-moment curves before
the port can be judged against the actual strategy.

### Common 64-start policy follow-up

Complete float32/`highest` 64-start panels were run for likelihood and BWCRPS,
using two sequential 32-start device batches. Against the 32/64 union:

| Objective | Starts | Within 0.001 | Worst gap | Median seconds | Median/global log-RMSE | Within factor 1.5 |
|---|---:|---:|---:|---:|---:|---:|
| likelihood | 32 | 97.5% | 0.14844 | 0.306 | 0.1493/0.4102 | 72.5% |
| likelihood | 64 | 100.0% | 0.00073 | 0.466 | 0.1495/0.4015 | 72.5% |
| BWCRPS | 32 | 100.0% | 0.00033 | 1.067 | 0.2629/0.6480 | 52.5% |
| BWCRPS | 64 | 100.0% | 0.00028 | 1.582 | 0.2631/0.6528 | 52.5% |

The likelihood 64-start design found three materially better `broad_3` basins,
lowering NLL by 0.0662--0.1484. Against the wider optimizer union it retained
one 0.001221 gap to SciPy. BWCRPS changed only below the existing gate. Paired
parameter recovery and whole/four-band truth-curve CCC were effectively tied
for both objectives. The executive policy selects 64 starts for both: the gain
is likelihood reliability and one configuration across objectives, not a
BWCRPS recovery improvement.

### Matched-objective development comparisons

The production-search comparison was subsequently completed on all 120
development datasets. These results supersede the component-only conclusion
above for selecting the matched density and smoothed-expectation searches.

For matched-KDE density, the legacy 80-by-64 log lattice was rebuilt exactly for
each dataset and followed by one SciPy polish. It cannot be amortized as a shared
curve cache: all 120 datasets have distinct pooled-SJ bandwidths, and the fitted
matched-KDE curve depends on that bandwidth. After common CPU rescoring:

| Search | Within 0.001 of three-arm union | Worst gap | Median seconds | Median joint log-RMSE |
|---|---:|---:|---:|---:|
| 32-start JAX port | 97.5% | 0.03207 | 0.143 | 0.90135 |
| 64-start JAX port | **100.0%** | **4.17e-7** | **0.282** | **0.90135** |
| per-dataset 80-by-64 lattice + polish | **100.0%** | 0.000960 | 18.546 | 0.90549 |

The 64-start port was materially better than the lattice on three datasets and
the lattice was never better by more than 1e-6. Doubling the port budget fixed
all four material misses of the 32-start arm. It is 65.9 times faster than the
per-dataset lattice at the median and already matches the union, so combining it
with the lattice adds cost without useful search protection. Freeze the
64-start float32/`highest` port as the matched-KDE density search. This is a
development selection; the already-inspected 32-start held-out diagnostic is
not prospective confirmation of the newly selected 64-start configuration.

For matched smoothed expectation, the 409,600-curve complex-moment cache was
built once, followed by eight cache-seeded polishes. The full comparison also
included 32-start SciPy, 32-start port, the production surface hierarchy, and
32-start JAX-BADS. The cache remained essential: it was within 0.001 of the
full union on 116/120 datasets with a worst gap of 0.249, whereas the port alone
covered 86/120 with a worst gap of 44.40. Replacing the SciPy component inside
the adapted cache/continuous/conditional-hierarchy strategy with the port left
coverage and worst gap unchanged (116/120 and 0.249) while reducing median
strategy time from 7.20 to 1.99 seconds, a 3.62-fold speedup. Freeze that
component substitution, but do not call the whole matched-objective search
settled: the old conditional rule still misses four development unions and
needs a selection-rule refinement.

A complete 64-start float32/`highest` development follow-up, evaluated as two
32-start device batches, improved the standalone port from 86/120 to 91/120 at
the 0.001 gate. Its worst gap remained 44.40. Substituting it into the cache and
conditional-hierarchy composite still covered 116/120 with worst gap 0.249,
while median time increased from 1.986 to 2.032 seconds. Combining both port
designs also remained at 116/120 and took 2.320 seconds. The same four cases
were missed because the cache and continuous arm agreed closely, so the
disagreement trigger never called the hierarchy; three union winners came from
the hierarchy and one from JAX-BADS. Thus 64 starts do not repair the
selection-rule failure. The final executive policy nevertheless selects the
standalone 64-start port and drops the cache/composite to keep one optimizer
configuration across objectives; the empirical-loss misses are accepted and
must remain explicit.

Parameter recovery does not reverse these search conclusions. Median joint
log-RMSE for port versus selected search was 0.147/0.150 for likelihood,
0.263/0.259 for bias-weighted CRPS, 0.937/0.937 for density, and 0.726/0.725 for
smoothed expectation under finite-endpoint selection. Large differences in
global recovery for weak curve objectives are not evidence of a better search;
those objectives are poorly identifying, and lower empirical loss need not
move parameters toward the generator.

## Batching and compilation

On `ordinary_1_seed0_n450`, processing all 32 starts in one batch was fastest
for every objective. Relative to batch size 1, solve time fell from 1.55 to
0.25 seconds for likelihood, 1.70 to 0.97 for bias-weighted CRPS, 2.05 to 0.33
for density, and 1.35 to 0.17 for smoothed expectation. The full-panel medians
are in the main table.

With a fresh requested JAX compilation-cache directory, one 32-start compile
took 0.90--1.18 seconds on the four representative 450-trial cases. Compilation
is recorded separately and excluded from steady-state solve time. The current
port batches independent starts for one dataset; it does not yet batch distinct
datasets with different payloads.

## Next tests

1. Route the common 64-start float32/`highest` policy through the production
   fitting entry point for all four retained objectives.
2. Any reuse of the already-inspected held-out datasets at 64 starts is
   descriptive, not a first prospective confirmation.
3. Preserve the rejected cache/composite and alternative-optimizer results as
   audit evidence; they are not runtime dependencies of the selected path.

## Artifacts

- `run_wnm_optimizer_benchmark.py`: shared runner with the port adapter.
- `compare_jax_lbfgsb_port.py`: common CPU rescoring and comparison assembly.
- `jax_lbfgsb_port_{objective}_development_gpu_v1/`: full 32-start runs.
- `jax_lbfgsb_port_bias_weighted_crps64_development_gpu_v1/`: 64-start follow-up.
- `jax_lbfgsb_port_likelihood64_highest_development_gpu_v1/` and
  `jax_lbfgsb_port_bias_weighted_crps64_highest_development_gpu_v1/`: complete
  float32/`highest` 64-start development panels supporting the final policy.
- `jax_lbfgsb_port_density64_development_gpu_v1/`: complete density 64-start
  follow-up.
- `jax_lbfgsb_port_density32_float64_development_gpu_v1/`: complete density
  32-start float64 follow-up.
- `jax_lbfgsb_port_likelihood32_mixed64_development_gpu_v1/`: complete
  likelihood mixed-precision follow-up.
- `jax_lbfgsb_port_likelihood32_highest_development_gpu_v1/`: complete
  likelihood float32/highest-matmul development panel.
- `jax_lbfgsb_port_{bias_weighted_crps,density,smoothed_exp}32_highest_development_gpu_v1/`:
  complete float32/highest-matmul development panels for the legacy objective
  branches.
- `jax_lbfgsb_port_matched_kde_ccc32_highest_heldout_gpu_v1/` and
  `jax_lbfgsb_port_matched_smoothed_exp32_highest_heldout_gpu_v1/`: current-
  operator held-out comparisons.
- `jax_lbfgsb_port_bias_weighted_crps32_highest_heldout_gpu_v1/`: frozen
  BWCRPS-port held-out confirmation.
- `jax_lbfgsb_port_matched_smoothed_exp_multistart_highest_heldout_gpu_v1/`:
  post-hoc 64/128-start diagnostic on the single matched-smoothed miss.
- `wnm_matched_smoothed_exp_development_cpu_v1/`: proper matched-smoothed
  production-search comparison and strategy tables.
- `wnm_matched_smoothed_exp_port32_port64_development_cpu_v2/`: common-rescore
  32/64-start follow-up and conditional-strategy comparison.
- `wnm_matched_kde_cache_vs_port32_port64_development_cpu_v1/`: common-rescore
  matched-density comparison selecting the 64-start port.
- `jax_lbfgsb_port_likelihood_device_precision_diagnostic_v1/`: exact worst-case
  CPU, default-GPU, batch-size-one, and highest-matmul-precision comparisons.
- `jax_lbfgsb_port_scaling_gpu_v1/`: batch-size and fresh-cache compile checks.
- `jax_lbfgsb_port_multistart_gpu_v1/`: hard-case start-count probes.
- `jax_lbfgsb_port_comparison_development_v1/`: paired tables and manifest.
- `jax_lbfgsb_port_*development_gpu_v1.log` and
  `jax_lbfgsb_port_comparison_development_v1.log`: run logs.
