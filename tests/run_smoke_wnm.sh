#!/usr/bin/env bash
# End-to-end smoke for the public WNM path: CSV fit -> export -> plots -> prediction.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT:$ROOT/model_fit_to_data"

DATA="$ROOT/example_data/data_color_comb_color2_two_subjects.csv"
OUT="$ROOT/results/smoke_wnm"
PARAMS="$OUT/prediction_parameters.csv"

rm -rf "$OUT"
mkdir -p "$OUT"

"$PYTHON_BIN" model_fit_to_data/fit_model_to_data.py \
  --data-path "$DATA" \
  --subject-col subject_exp \
  --condition-col noise \
  --n-samples 20 \
  --continuous-starts 2 \
  --include-methods density likelihood \
  --max-subjects 1 \
  --no-resume \
  --output-dir "$OUT"

"$PYTHON_BIN" model_fit_to_data/export_wnm_fit_curves.py \
  --results-dir "$OUT" \
  --output-dir "$OUT/csv_exports" \
  --methods density likelihood

"$PYTHON_BIN" model_fit_to_data/create_unified_subject_plots.py \
  --results-path "$OUT/extended_fit_results.pkl" \
  --output-dir "$OUT" \
  --individual-plots \
  --summary-plots \
  --pdf-slices

cat > "$PARAMS" <<'EOF'
sd_feat1,sd_feat2,sd_spat
10,30,20
EOF

"$PYTHON_BIN" surface_simulator_for_predictions/surface_simulator.py \
  --input-path "$PARAMS" \
  --n-samples 20 \
  --output-path "$OUT/predictions.csv" \
  --skip-motor-noise

test -s "$OUT/extended_fit_results.pkl"
test -s "$OUT/extended_run_fingerprint.json"
test -s "$OUT/csv_exports/fitted_parameters.csv"
test -s "$OUT/csv_exports/fitted_curves.csv"
test -s "$OUT/predictions.csv"
find "$OUT/unified_subject_plots" -name '*_unified.png' -print -quit | grep -q .
find "$OUT/summary_plots" -name '*.png' -print -quit | grep -q .
find "$OUT/pdf_slice_plots" -name '*.png' -print -quit | grep -q .

"$PYTHON_BIN" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

out = Path(sys.argv[1])
fingerprint = json.loads((out / "extended_run_fingerprint.json").read_text())
payload = fingerprint.get("payload", fingerprint)
assert payload["surrogate_family"] == "wnm"
predictions = pd.read_csv(out / "predictions.csv")
assert predictions.loc[0, "surrogate_family"] == "wnm"
PY

echo "WNM public-path smoke completed successfully."
