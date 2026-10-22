# Generate simulation data

Use these tools to simulate the Demixing Model and prepare data for a custom predictor. The [main README](../README.md) provides prediction and fitting examples using the included trained models.

## Generate training data

`generate_wnm_training_data.py` samples noise parameters and stimulus differences, simulates internal evidence, and records the response errors recovered by the observer's mixture fit.

```bash
python -m surface_computation.generate_wnm_training_data \
  --n-samples 20 \
  --n-points 4000 \
  --n-simulations 200 \
  --out /path/to/training_data
```

- `--n-samples` sets the total internal evidence samples in each simulated trial; the generator defaults to 100.
- `--n-points` sets the number of parameter combinations in the design.
- `--n-simulations` sets the repeated simulations per design point.

The directory corpus stores the parameter design, raw feature-bias outcomes for
both items, all EM diagnostics and generation metadata. The
[training tools](../surrogate_training/wnm/README.md) read it directly.

The observer uses likelihood EM with soft component and winding probabilities,
225 starts, SDs bounded below by 1 and component weights bounded below by 0.1.
Both likelihood and parameter changes must stabilize. All starts receive up to
5,000 updates; a capped selected winner continues from its cached state up to
25,000. Capped, nonfinite or likelihood-decreasing losing starts remain counted.
A selected fit that stays capped or decreased likelihood on its own path is
refitted once with ordinary (unaccelerated) EM from the same data and starts
with the CUDA backend; if it still fails, or with the JAX backend, its trial is
kept with NaN biases and `em_masked` set (`em_repaired` marks a replaced fit).
A call (one generator tile) refuses its output when more than
max(3, trials // 1000) stay masked; the allowance is per call, not corpus-wide.
The corpus records the totals in `em_summary` and `n_masked`. The exploratory grid
generator (`simulated_samples_grid.py`) deliberately stays strict and refuses any
failed selected fit.
The fit is a multistart local search; it does not certify a global optimum.

Each corpus includes per-trial `em_*` diagnostics and selected
`em_parameters` of shape `(rows, trials, 5, 2)`, in raw fit order, with fields
`(weight, feature mean, identifying mean, feature SD, identifying SD)`.
These parameters allow inspection of floor-bound spikes and nearly uniform
components; convergence alone does not establish that a circular mean is well
identified. No trials are silently excluded. The source
fingerprint includes the likelihood kernel and density helper; old simulation
shards cannot resume under the new estimator. Bundled NN checkpoints are
unchanged and require new simulation data and retraining to reflect this fitter.

The corpus uses standard `.npy` memory maps and a SQLite tile index. Array writes
are flushed before publishing a run-bound tile checksum; `--resume` verifies
checksums and recomputes missing or altered tiles. An atomic `complete.json`
marker is published only after exact tile coverage is complete. Training refuses
incomplete corpora. One process owns the writer lock; independent workers must
use separate outputs. Do not modify a completed corpus while training reads it.

Storage and training batches do not require whole-corpus assembly or a full
finite-value mask. `--shard-rows` and `--simulation-chunk` still define the same
logical RNG tiles, even though tiles now occupy slices in shared array files.
Existing NPZ files remain readable; older NPZ-tile checkpoint directories cannot
resume as this new storage format. No existing outcomes are silently migrated.

To request an optional consolidated export, use `--out /path/to/data.npz`; the
durable directory is then `/path/to/data.npz.shards`. Export streams in bounded
buffers but needs space for a second copy; initial generation checks capacity.
You can also export a completed directory later:

```bash
python -m surface_computation.simulation_corpus /path/to/training_data /path/to/data.npz
```

Use `python -m surface_computation.simulation_corpus /path/to/training_data
--verify` for an explicit full-payload checksum scan. Ordinary training opening
checks completion, array schemas and the coverage index; it relies on the
validated payload remaining immutable rather than rereading all diagnostics.

`--compress` applies only to NPZ export. The directory corpus retains exact raw
arrays. At 148,076 rows and 20,000 trials, payload storage is about 275 GB per
sample-count setting, plus checkpoint metadata. Memory mapping removes the
275 GB heap requirement and the mandatory second output copy; it does not
remove the need for that durable storage. Keep compute caches under `/tmp`,
and run one GPU job at a time on a shared device.

`--block-rows` batches different parameter combinations on the GPU. The total
trial batch is `block_rows × simulation_chunk`; increase it with attention to
memory and measured throughput. Optional `--em-stages 50 250 1000 5000` pools
unfinished starts across those rows at the specified cumulative iteration caps.
Data and cached Fourier bases are stored once per trial. With no
`--continuation-batch`, all active starts use one padded bucket. An explicit
limit bounds continuation memory and can improve throughput for large batches;
smaller chunks add serial calls, so measure throughput. Staging remains optional because
its packing and synchronization costs can outweigh the benefit. Row grouping
preserves random keys; float32 compilation context can change arithmetic by a
few ULPs and change selection among nearly tied optima. Simulation chunk size
and row batch size are recorded in the resume fingerprint.

For the maintained free-weight, wrapped-feature/linear-identifying model, add
`--em-backend cuda` to use persistent SQUAREM-accelerated EM at N=4,10,20,100,500.
This optional backend requires an NVIDIA GPU and `nvcc` on PATH; it builds its
packaged CUDA sources once under `/tmp/demixing_cuda_em`. The default remains
`--em-backend jax`. Unsupported CUDA configurations fail explicitly.

The CUDA backend retains all 225 starts and the same convergence limits. A GPU
work queue schedules independent starts, capped winners continue from cached
states and are reselected, and GPU reduction preserves all-start diagnostics
and the first-index tie rule. JAX owns the buffers and execution stream, so
input buffers are reused without host uploads or raw-pointer borrowing.
Initial statistics are calculated once inside the kernel. The observation loop
stays rolled to limit compiled kernel size, and the worker pool uses occupancy
sizing at every supported N.

`--em-starts structured` replaces the 225 random starts with a deterministic
data-derived bank of min(N, 20) + 34 starts (54 at N >= 20), built from hard
splits and local neighbourhoods of each dataset (`structured_starts.py`). It
requires the CUDA backend and is recorded in the corpus manifest. The CLI default
remains `random`; the bank is the selected setting for production corpora at all
supported N.
Repairs of failed winners reuse the starts of the original batch fit.

The CUDA kernel shares one EM-update loop body between settling and the two
SQUAREM maps, storing temporary parameter snapshots across warp lanes to reduce
register lifetimes. Density
evaluation uses one formula, outlined at N=4,10,20,100 and inlined at N=500,
following measured instruction-cache and function-call costs.

At N=500, training-data simulation chooses between two complete compiled
evaluators before each block. If every row has at least one generating feature
SD at or below `sqrt(180/0.09)` (about 44.72 degrees), it uses a single-image
shortcut where the current fitted SD and observation distance make omitted
winding weights at most `exp(-32.4)` of the principal image. Otherwise it uses
the full evaluator. Both retain the original spatial and Fourier fallbacks;
generating SDs never constrain fitted trajectories. Mixed blocks use the full
evaluator. Direct CUDA fits without generating parameters use the full evaluator.
Circular subtraction preserves local differences and compensates rounding near
the wrap seam. Both compiled identities are included in generation metadata.

After 10 ordinary updates, SQUAREM extrapolates in scaled-mean/log-SD
coordinates with an adaptive spectral-step cap of 256. Each proposal respects
the existing parameter bounds and receives an ordinary EM stabilization step;
it is accepted only if its likelihood reaches the second ordinary iterate.
Rejected proposals retain that iterate and its cached moments. Selected capped
winners continue with ordinary EM. Convergence thresholds, all 225 starts and
the requested simulation count are retained. Different optimization paths can
find different local optima, so regenerated outcomes can differ from ordinary
EM. The step cap is an optimizer setting, not an upper SD bound.
The iteration counter includes rejected stabilization attempts as well as
ordinary EM updates; extrapolated-point statistics add an observation pass.

CUDA uses its own persistent queue: omit `--em-stages` and
`--continuation-batch`. Use `--simulation-chunk` and `--block-rows` to control
batching as before; a chunk of 20,000 can process an entire 20,000-trial cell
in one call if memory permits. These execution settings do not change
`--n-simulations`. Backend, CUDA build identity and source hashes are included
in the resume manifest, preventing shards from different implementations from
being combined. CUDA/JAX reductions and JAX initialization across compilation
contexts can differ by a few float32 ULPs.

CUDA simulation validates groups of small blocks with at most 4,096 pending
trials, amortizing host counter downloads. Larger blocks are checked
individually. All failures are checked before returning outcomes or publishing
a shard; each fit retains its own convergence check after every EM update.

## Generate averaged surfaces

For direct inspection of simulated distributions or identifiability-dimension (`mu_idf`) predictions, use `simulated_samples_grid.py`. It generates samples on a parameter grid; averaging tools then turn them into density surfaces.

- [Grid simulation guide](Likelihood_Surface_Pipeline_Documentation.md)
- [Distributed generation on Vast.ai](../cloud/README_vast.md)

To average previously generated sample files, run from the repository root:

```bash
python -m surface_computation.create_averaged_surfaces_from_samples \
  --input-folder /path/to/simulated_samples \
  --output-folder /path/to/averaged_surfaces
```

`combine_mirrored_surfaces.py` supports old, uncombined raw-surface folders;
the current averaging command already combines mirrored samples.
A complete grid requires substantial GPU time and storage.
New grid outputs use an estimator namespace for sample/surface directories,
object-store prefixes, locks and completion registries, including explicit
destinations. Grid outputs also retain checked EM diagnostics. Existing moment
estimator files remain in their original directories.

## Notes for developers

`wnm_design.py` constructs parameter designs. `wnm_simulation.py` calls the JAX observer simulator and supports item-swap augmentation. `legacy_sample_import.py` loads raw outcomes from grid-corpus files.

Generated corpora and surfaces belong under `$DEMIXING_ARTIFACT_ROOT` or another configured results location. They are not shipped and are not expected in a normal checkout.

CUDA loop and queue performance validation, including the independent audit,
is retained under `$DEMIXING_ARTIFACT_ROOT/performance_followup_20261021/campaign/`.
These generated records are not shipped with the repository; measurements and
exact-output comparisons apply to the recorded compiler and GPU.

SQUAREM selection and integration validation are retained under
`$DEMIXING_ARTIFACT_ROOT/squarem_constrained_followup_20261021/` and
`$DEMIXING_ARTIFACT_ROOT/cuda_squarem_integration_20261021/`.
These generated artifacts are not shipped. The internal `cuda_em._run` accepts
`accelerate=False` for ordinary-EM reference checks; public CUDA simulation uses
SQUAREM and its build identity prevents mixing it with earlier CUDA corpora.

Structured-start selection evidence is retained under
`$DEMIXING_ARTIFACT_ROOT/density_tolerance_20261022/`,
`$DEMIXING_ARTIFACT_ROOT/density_trajectory_20261022/` and
`$DEMIXING_ARTIFACT_ROOT/narrow_interval_check_20261022/` and
`$DEMIXING_ARTIFACT_ROOT/structured_starts_low_n_20261022/`. These generated artifacts
are not shipped. At N=500 the bank was compared with 225 random starts on identical
observations, by optimum quality and by matched NN retraining against an
independent-data control. At N=4-100 the comparison is teacher-level only, on a
32-cell panel: the bank was 3.5-6.7x faster including its construction (8-9x for
fitting alone), gained the higher likelihood far
more often than it lost it, and every sampled cell passed a criterion chosen after
the first run, measured against the better of the two fits on each dataset. That
reference is a finite two-search ensemble, not a global optimum, and in some cells
the bank's histogram shift from it exceeded the random starts' shift.

Counter-guided kernel validation and its independent audit are retained under
`$DEMIXING_ARTIFACT_ROOT/cuda_counter_round_20261021/`. These generated artifacts
are not shipped; timing and exact-output evidence apply to the recorded GPU,
compiler, inputs and starting parameters.

Evaluator dispatch validation is retained under
`$DEMIXING_ARTIFACT_ROOT/gate_dispatch_20261021/` and integration checks under
`$DEMIXING_ARTIFACT_ROOT/pipeline_audit_20261021/integration/`.
These generated records are not shipped. The dispatch comparison uses corrected
subtraction as its baseline; tiny pointwise winding-moment differences remain
possible, so the tested exact final-state comparisons are workload evidence,
not a universal optimizer-equivalence guarantee.
