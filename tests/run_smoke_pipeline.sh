#!/usr/bin/env bash
# Pipeline smoke test: simulate + average in-memory (no intermediate sample files).
# Uses --pipeline mode which generates averaged surfaces directly, skipping step 2
# of the standard approach.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
# Ensure checkout modules are importable when unpickling.
export PYTHONPATH="${PYTHONPATH:-}:$ROOT"

# Require a GPU unless explicitly bypassed.
ALLOW_CPU="${ALLOW_CPU:-0}"
if [[ "$ALLOW_CPU" != "1" ]]; then
  if ! "$PYTHON_BIN" - <<'EOF'
import jax
devs = jax.devices()
if not any(d.platform == 'gpu' for d in devs):
    raise SystemExit("No GPU device found. Set ALLOW_CPU=1 to run on CPU.")
EOF
  then
    exit 1
  fi
fi

# Simulate samples and average in memory (pipeline mode).
PARAM_DIR="$ROOT/tests/param_list"
rm -rf "$PARAM_DIR"
mkdir -p "$PARAM_DIR"

# Create a small set of parameter combinations (only filenames matter)
for combo in \
  "samples_sf1_10.0_sf2_10.0_sp_10.0.csv" \
  "samples_sf1_10.0_sf2_20.0_sp_10.0.csv" \
  "samples_sf1_20.0_sf2_10.0_sp_10.0.csv" \
  "samples_sf1_20.0_sf2_20.0_sp_20.0.csv" \
  "samples_sf1_30.0_sf2_30.0_sp_10.0.csv" \
  "samples_sf1_30.0_sf2_40.0_sp_20.0.csv"; do
  touch "$PARAM_DIR/$combo"
done

AVG_DIR="averaged_surfaces_smoke_pipeline"
rm -rf "$AVG_DIR"

$PYTHON_BIN surface_computation/simulated_samples_grid.py \
  --machine-id PC_TEST \
  --test-mode \
  --match-csv-params "$PARAM_DIR" \
  --lock-backend auto \
  --pipeline \
  --averaged-surfaces-dir "$AVG_DIR"

echo "Raw surface smoke (pipeline mode) completed successfully."
