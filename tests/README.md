# Tests

Run commands from the repository root. Override the interpreter with `PYTHON_BIN` where needed.

## Maintained pytest baseline

```bash
PYTHONPATH=. python -m pytest
```

Pytest is configured to collect `tests/` only and, by default, excludes
`integration` and `legacy_surface` suites. This is the standalone WNM/product
baseline for a normal checkout.

Opt-in suites remain available:

```bash
# Cross-repository checks; requires sibling contextual_biases_database where relevant.
PYTHONPATH=. python -m pytest -m integration


# Historical surface-NN reproduction contracts.
PYTHONPATH=. python -m pytest -m legacy_surface
```

The maintained baseline covers configuration, WNM fitting/scoring/prediction,
current result/export/plot contracts, shared circular geometry, and supporting
runtime utilities without requiring the comparison workspace.

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

Four shell workflows exercise the public/model-generation paths:

```bash
PYTHON_BIN=python bash tests/run_smoke_wnm.sh
PYTHON_BIN=python bash tests/run_smoke_pipeline.sh
PYTHON_BIN=python bash tests/run_smoke_standard.sh
PYTHON_BIN=python bash tests/run_smoke_compare_seeds.sh
```

- `run_smoke_wnm.sh` exercises the current end-user chain: ordinary CSV fit with the packaged WNM, tabular export, individual/group/PDF plots, and the public prediction API.
- `run_smoke_pipeline.sh` and `run_smoke_standard.sh` are historical surface-generation/training checks retained for reproduction of that pipeline.
- `run_smoke_compare_seeds.sh` checks that the direct and stored-sample simulation routes agree under the same seed within the documented float16 tolerance.

The fit-producing scripts use the tracked input `example_data/data_color_comb_color2_two_subjects.csv`; the seed comparison builds its own small parameter list. These workflows are compute-heavy and should normally run on an NVIDIA GPU. `run_smoke_pipeline.sh` explicitly requires one unless `ALLOW_CPU=1` is set; CPU execution can be very slow.
