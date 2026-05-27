# Tests

This project ships a smoke test pipeline script that exercises the core
simulation, surface creation, training, and batch-fit steps on small data.

## Run

```bash
bash tests/run_smoke_pipeline.sh
```

## Notes

- The script must be run from the repo root (it `cd`s there automatically).
- You can override the Python executable with `PYTHON_BIN`, for example:

```bash
PYTHON_BIN=python3.11 bash tests/run_smoke_pipeline.sh
```

- The script expects the example data file at
  `example_data/data_color_comb_color2_two_subjects.csv`.
