# Simulation and training-data generation

This package contains both the current WNM simulation/training-data path and the
historical averaged-surface pipeline retained for reproduction and raw `mu2`
analyses.

## Current WNM training-data path

`generate_wnm_training_data.py` samples continuous parameter designs and stores
raw EM bias outcomes. It delegates the design to `wnm_design.py` and simulator
calls to `wnm_simulation.py`, which in turn use the maintained JAX simulator.

Example:

```bash
PYTHONPATH=. python -m surface_computation.generate_wnm_training_data \
  --n-points 4000 \
  --n-simulations 200 \
  --out $DEMIXING_ARTIFACT_ROOT/wnm/train.npz
```

The resulting NPZ is consumed by `surrogate_training.wnm.data` and
`surrogate_training.wnm.train`. It contains raw outcomes rather than KDEs or
likelihood surfaces.

`legacy_sample_import.py` is a bridge for usable raw outcomes that still exist
inside the older grid corpus. It does not make the historical surface pipeline a
runtime dependency of WNM fitting or prediction.

## Historical averaged-surface path

`simulated_samples_grid.py` and its supporting lock/object-store utilities
generate the parameter-grid samples and averaged surfaces used by the historical
surface neural network. This path is retained for reproduction and for analyses
that require raw averaged surfaces, including separate `mu2` outputs.

See:

- [Likelihood_Surface_Pipeline_Documentation.md](Likelihood_Surface_Pipeline_Documentation.md)
- [../neural_network_optimization/Neural_Network_Optimization_Pipeline_Documentation.md](../neural_network_optimization/Neural_Network_Optimization_Pipeline_Documentation.md)
- [../cloud/README_vast.md](../cloud/README_vast.md)

Generated corpora and surfaces belong under `$DEMIXING_ARTIFACT_ROOT` (or the
configured results location), not in the source tree.
