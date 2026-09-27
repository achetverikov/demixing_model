#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
# Ensure checkout modules are importable when unpickling.
export PYTHONPATH="${PYTHONPATH:-}:$ROOT"

# Simulate samples (test mode) for a small custom parameter list.
PARAM_DIR="$ROOT/tests/param_list"
rm -rf "$PARAM_DIR"
mkdir -p "$PARAM_DIR"

# Create a small set of parameter combinations (only filenames matter).
# Off-diagonal pairs must include both canonical (sf1<sf2) and mirror (sf2,sf1)
# so that create_averaged_surfaces_from_samples.py can find both sides.
for combo in \
  "samples_sf1_10.0_sf2_10.0_sp_10.0.csv" \
  "samples_sf1_10.0_sf2_20.0_sp_10.0.csv" \
  "samples_sf1_20.0_sf2_10.0_sp_10.0.csv" \
  "samples_sf1_20.0_sf2_20.0_sp_20.0.csv" \
  "samples_sf1_30.0_sf2_30.0_sp_10.0.csv" \
  "samples_sf1_30.0_sf2_40.0_sp_20.0.csv" \
  "samples_sf1_40.0_sf2_30.0_sp_20.0.csv"; do
  touch "$PARAM_DIR/$combo"
done

$PYTHON_BIN surface_computation/simulated_samples_grid.py \
  --machine-id PC_TEST \
  --test-mode \
  --match-csv-params "$PARAM_DIR" \
  --lock-backend file

SAMPLES_DIR="sim_samples_100_50samples_circular_em_diagcov_free_weights"
AVG_DIR="averaged_surfaces_smoke_standard"
rm -rf "results/$AVG_DIR"

# Create averaged surfaces from samples.
$PYTHON_BIN -m surface_computation.create_averaged_surfaces_from_samples \
  --input-folder "$SAMPLES_DIR" \
  --output-folder "$AVG_DIR" \
  --workers 1 \
  --include-all-params

echo "Raw surface smoke (standard mode) completed successfully."
