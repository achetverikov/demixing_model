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

The generator saves simulation chunks beside the output so interrupted runs can resume. Keep simulation and Python caches under `/tmp`, and run one GPU job at a time on a shared device.

## Generate averaged surfaces

For direct inspection of simulated distributions, surface-network training, or identifiability-dimension (`mu2`) predictions, use `simulated_samples_grid.py`. It generates samples on a parameter grid; averaging tools then turn them into density surfaces.

- [Grid simulation guide](Likelihood_Surface_Pipeline_Documentation.md)
- [Surface averaging and network training](../neural_network_optimization/Neural_Network_Optimization_Pipeline_Documentation.md)
- [Distributed generation on Vast.ai](../cloud/README_vast.md)

A complete grid requires substantial GPU time and storage.

## Notes for developers

`wnm_design.py` constructs parameter designs. `wnm_simulation.py` calls the JAX observer simulator and supports item-swap augmentation. `legacy_sample_import.py` loads raw outcomes from grid-corpus files.

Generated corpora and surfaces belong under `$DEMIXING_ARTIFACT_ROOT` or another configured results location. They are not shipped and are not expected in a normal checkout.
