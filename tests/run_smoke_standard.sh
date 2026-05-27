#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
# Ensure repo modules and training model module are importable when unpickling.
export PYTHONPATH="${PYTHONPATH:-}:$ROOT:$ROOT/neural_network_optimization"

TRIMMED_CSV="$ROOT/example_data/data_color_comb_color2_two_subjects.csv"

# Step 0: Confirm trimmed example data exists
if [[ ! -f "$TRIMMED_CSV" ]]; then
  echo "Missing trimmed data file: $TRIMMED_CSV" >&2
  exit 1
fi

# Step 1: Simulate samples (test mode) for a small custom parameter list
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

SAMPLES_DIR="sim_samples_100_50samples_circular_em_fullcov_free_weights"
AVG_DIR="averaged_surfaces_smoke_standard"
rm -rf "results/$AVG_DIR" "results/checkpoints_smoke_standard" "results/model_fit_to_data_smoke_standard"

# Step 2: Create averaged surfaces from samples
$PYTHON_BIN neural_network_optimization/create_averaged_surfaces_from_samples.py \
  --input-folder "$SAMPLES_DIR" \
  --output-folder "$AVG_DIR" \
  --workers 1 \
  --include-all-params

# Step 3: Train mirror-aware model (short run)
$PYTHON_BIN neural_network_optimization/mirror_aware_training.py \
  --surfaces-folder "$AVG_DIR" \
  --epochs 25 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --save-dir "checkpoints_smoke_standard"

# Step 4: Fit model to trimmed human data
$PYTHON_BIN model_fit_to_data/fit_model_to_data.py \
  --checkpoint-path "checkpoints_smoke_standard/model_epoch_0025.pkl" \
  --output-dir "model_fit_to_data_smoke_standard" \
  --data-path "example_data/data_color_comb_color2_two_subjects.csv" \
  --subject-col subject_exp \
  --condition-col noise \
  --no-resume \
  --max-subjects 2

# Step 5: Generate unified plots/exports from smoke results
$PYTHON_BIN model_fit_to_data/create_unified_subject_plots.py \
  --results-path "model_fit_to_data_smoke_standard/extended_fit_results.pkl" \
  --checkpoint-path "checkpoints_smoke_standard/model_epoch_0025.pkl" \
  --output-dir "model_fit_to_data_smoke_standard" \
  --individual-plots \
  --summary-plots \
  --csv-exports

echo "Smoke standard pipeline completed successfully."
