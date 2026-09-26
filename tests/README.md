# Tests

Run commands from the repository root. Override the interpreter with `PYTHON_BIN` where needed.

## Maintained pytest baseline

```bash
PYTHONPATH=. python -m pytest
```

Pytest is configured to collect `tests/` only and, by default, excludes
cross-repository `integration` checks. This is the standalone WNM and raw-surface
baseline for a normal checkout.

Opt-in suites remain available:

```bash
# Cross-repository checks; requires sibling contextual_biases_database where relevant.
PYTHONPATH=. python -m pytest -m integration
```

The maintained baseline covers configuration, WNM fitting/scoring/prediction,
current result/export/plot contracts, shared circular geometry, and supporting
runtime utilities without requiring the comparison workspace.

The fit integration tests exercise the packaged WNM and its continuous search.

## Smoke pipelines

Four shell workflows exercise the public/model-generation paths:

```bash
PYTHON_BIN=python bash tests/run_smoke_wnm.sh
PYTHON_BIN=python bash tests/run_smoke_pipeline.sh
PYTHON_BIN=python bash tests/run_smoke_standard.sh
PYTHON_BIN=python bash tests/run_smoke_compare_seeds.sh
```

- `run_smoke_wnm.sh` exercises the current end-user chain: ordinary CSV fit with the packaged WNM, tabular export, individual/group/PDF plots, and the public prediction API.
- `run_smoke_pipeline.sh` and `run_smoke_standard.sh` generate stored averaged surfaces by the direct and staged routes.
- `run_smoke_compare_seeds.sh` checks that the direct and stored-sample simulation routes agree under the same seed within the documented float16 tolerance.

The WNM fit script uses the tracked input `example_data/data_color_comb_color2_two_subjects.csv`; the surface checks build small parameter lists. Surface generation is compute-heavy and should normally run on an NVIDIA GPU. `run_smoke_pipeline.sh` explicitly requires one unless `ALLOW_CPU=1` is set.
