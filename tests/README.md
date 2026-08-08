# Tests

Run commands from the repository root. Override the interpreter with `PYTHON_BIN` where needed.

## Focused pytest suite

```bash
PYTHONPATH=. python -m pytest tests
```

These tests cover configuration imports, CLI flag dispatch, lock and object-store backends, and fitted-result export behavior without regenerating the full model.

A few are slower because they exercise the surrogate on a small lattice rather
than mocking it, which is the only way they can check what they claim:

- `test_curve_cache_matches_live_model.py` builds a cache through the production
  builder and compares every curve against the model, because the cache is read
  *instead of* calling it — a transposed parameter order or an off-by-one in the
  slab index would leave every checksum valid.
- `test_search_dispatch.py` runs the real fitter and records which backend ran
  for which method. It does not infer the backend from fitted values: with a
  coarse cache the two searches can land on the same parameters by coincidence.
- `test_exhaustive_density.py` checks the factorisation the exhaustive scan rests
  on against brute force over the full joint product space.

## Smoke pipelines

Three shell workflows exercise the compute pipeline:

```bash
PYTHON_BIN=python bash tests/run_smoke_pipeline.sh
PYTHON_BIN=python bash tests/run_smoke_standard.sh
PYTHON_BIN=python bash tests/run_smoke_compare_seeds.sh
```

- `run_smoke_pipeline.sh` generates averaged surfaces directly in memory, trains a short-run NN, fits two subject groups, and creates plots/exports.
- `run_smoke_standard.sh` writes simulated samples first, then averages, trains, fits, and plots.
- `run_smoke_compare_seeds.sh` checks that the direct and stored-sample routes agree under the same seed within the documented float16 tolerance.

The two fit-producing scripts use the tracked input `example_data/data_color_comb_color2_two_subjects.csv`; the seed comparison builds its own small parameter list. These workflows are compute-heavy and should normally run on an NVIDIA GPU. `run_smoke_pipeline.sh` explicitly requires one unless `ALLOW_CPU=1` is set; CPU execution can be very slow.
