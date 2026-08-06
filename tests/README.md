# Tests

Run commands from the repository root. Override the interpreter with `PYTHON_BIN` where needed.

## Focused pytest suite

```bash
PYTHONPATH=. python -m pytest tests
```

These tests cover configuration imports, CLI flag dispatch, lock and object-store backends, and fitted-result export behavior without regenerating the full model.

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
