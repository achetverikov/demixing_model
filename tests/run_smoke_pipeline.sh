#!/usr/bin/env bash
# Pipeline smoke test: simulate + average in-memory (no intermediate sample files).
# Uses --pipeline mode which generates averaged surfaces directly, skipping step 2
# of the standard approach.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
# Ensure repo modules and training model module are importable when unpickling.
export PYTHONPATH="${PYTHONPATH:-}:$ROOT:$ROOT/neural_network_optimization"

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

TRIMMED_CSV="$ROOT/example_data/data_color_comb_color2_two_subjects.csv"

# Step 0: Confirm trimmed example data exists
if [[ ! -f "$TRIMMED_CSV" ]]; then
  echo "Missing trimmed data file: $TRIMMED_CSV" >&2
  exit 1
fi

# Step 1: Simulate samples and average in-memory (pipeline mode)
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
rm -rf "$AVG_DIR" "results/checkpoints_smoke_pipeline" "results/model_fit_to_data_smoke_pipeline"

$PYTHON_BIN surface_computation/simulated_samples_grid.py \
  --machine-id PC_TEST \
  --test-mode \
  --match-csv-params "$PARAM_DIR" \
  --lock-backend auto \
  --pipeline \
  --averaged-surfaces-dir "$AVG_DIR"

# Step 2: Train mirror-aware model (short run)
$PYTHON_BIN neural_network_optimization/mirror_aware_training.py \
  --surfaces-folder "$AVG_DIR" \
  --epochs 25 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --save-dir "checkpoints_smoke_pipeline"

# Step 3: Fit model to trimmed human data
$PYTHON_BIN model_fit_to_data/fit_model_to_data.py \
  --checkpoint-path "checkpoints_smoke_pipeline/model_epoch_0025.pkl" \
  --output-dir "model_fit_to_data_smoke_pipeline" \
  --data-path "example_data/data_color_comb_color2_two_subjects.csv" \
  --subject-col subject_exp \
  --condition-col noise \
  --no-resume \
  --max-subjects 2

# Step 4: Generate unified plots/exports from smoke results
$PYTHON_BIN model_fit_to_data/create_unified_subject_plots.py \
  --results-path "model_fit_to_data_smoke_pipeline/extended_fit_results.pkl" \
  --checkpoint-path "checkpoints_smoke_pipeline/model_epoch_0025.pkl" \
  --output-dir "model_fit_to_data_smoke_pipeline" \
  --individual-plots \
  --summary-plots \
  --csv-exports

echo "Smoke pipeline (pipeline mode) completed successfully."
