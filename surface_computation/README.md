# Generate simulation data

Use these tools to simulate the Demixing Model and prepare data for a custom predictor. The [main README](../README.md) provides prediction and fitting examples using the included trained models.

## Generate training data

`generate_wnm_training_data.py` samples noise parameters and stimulus differences, simulates internal evidence, and records the response errors recovered by the observer's mixture fit.

```bash
python -m surface_computation.generate_wnm_training_data \
  --n-samples 20 \
  --n-points 4000 \
  --n-simulations 200 \
  --out /path/to/training_data.npz
```

- `--n-samples` sets the total internal evidence samples in each simulated trial; the generator defaults to 100.
- `--n-points` sets the number of parameter combinations in the design.
- `--n-simulations` sets the repeated simulations per design point.

The NPZ stores the parameter design, raw feature-bias outcomes for both items, and generation metadata. The [training tools](../surrogate_training/wnm/README.md) use these outcomes to fit a conditional wrapped-normal mixture (WNM).

The observer uses likelihood EM with soft component and winding probabilities,
225 starts, SDs bounded below by 1 and component weights bounded below by 0.1.
Both likelihood and parameter changes must stabilize. All starts receive up to
5,000 updates; a capped selected winner continues from its cached state up to
25,000. Capped losing starts remain counted. Generation refuses unresolved
selected fits, nonfinite starts, or material likelihood decreases before saving.
The fit is a multistart local search; it does not certify a global optimum.

Each shard and final NPZ includes per-trial `em_*` diagnostics and selected
`em_parameters` of shape `(rows, trials, 5, 2)`, in raw fit order, with fields
`(weight, feature mean, identifying mean, feature SD, identifying SD)`.
These parameters allow inspection of floor-bound spikes and nearly uniform
components; convergence alone does not establish that a circular mean is well
identified. No trials are silently excluded. The source
fingerprint includes the likelihood kernel and density helper; old simulation
shards cannot resume under the new estimator. Bundled NN checkpoints are
unchanged and require new simulation data and retraining to reflect this fitter.

The generator saves simulation chunks beside the output so interrupted runs can resume. Keep simulation and Python caches under `/tmp`, and run one GPU job at a time on a shared device.

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
`--em-backend cuda` to use persistent ordinary EM at N=4,10,20,100,500.
This optional backend requires an NVIDIA GPU and `nvcc` on PATH; it builds its
packaged CUDA sources once under `/tmp/demixing_cuda_em`. The default remains
`--em-backend jax`. Unsupported CUDA configurations fail explicitly.

The CUDA backend retains all 225 starts and the same convergence limits. A GPU
work queue schedules independent starts, capped winners continue from cached
states and are reselected, and GPU reduction preserves all-start diagnostics
and the first-index tie rule. JAX owns the buffers and execution stream, so
input buffers are reused without host uploads or raw-pointer borrowing.
Initial statistics are calculated once inside the kernel. The worker pool uses
occupancy sizing through N=100 and a fixed pool at N=500. It does not use
SQUAREM, alternative optimizers, reduced starting-point banks or fewer trials.

CUDA uses its own persistent queue: omit `--em-stages` and
`--continuation-batch`. Use `--simulation-chunk` and `--block-rows` to control
batching as before; a chunk of 20,000 can process an entire 20,000-trial cell
in one call if memory permits. These execution settings do not change
`--n-simulations`. Backend, CUDA build identity and source hashes are included
in the resume manifest, preventing shards from different implementations from
being combined. CUDA and JAX reductions can differ by float32 rounding.

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
