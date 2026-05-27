#!/usr/bin/env bash
# Verify that the standard and pipeline approaches produce statistically equivalent
# averaged surfaces when given the same non-default random seed.
#
# Standard path: generate sample files → create_averaged_surfaces_from_samples.py
# Pipeline path: generate + average in-memory (no intermediate files)
#
# Note: results are not bit-for-bit identical because the standard path stores
# samples as float16, introducing ~1e-3 degree quantisation before KDE. The
# comparison uses atol=0.05 on log-density values to account for this.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT:$ROOT/neural_network_optimization"

SEED=123

PARAM_DIR="$ROOT/tests/param_list_compare"
rm -rf "$PARAM_DIR"
mkdir -p "$PARAM_DIR"

# Minimal param set: one diagonal and one off-diagonal pair (= 2 groups)
for combo in \
  "samples_sf1_10.0_sf2_10.0_sp_10.0.csv" \
  "samples_sf1_10.0_sf2_20.0_sp_10.0.csv" \
  "samples_sf1_20.0_sf2_10.0_sp_10.0.csv"; do
  touch "$PARAM_DIR/$combo"
done

STD_RESULTS="results/compare_seed_std"
PIPE_RESULTS="results/compare_seed_pipe"
STD_AVG="averaged_surfaces_compare_std"
PIPE_AVG="averaged_surfaces_compare_pipe"

rm -rf "$STD_RESULTS" "$PIPE_RESULTS"

echo "=== Standard approach (seed=$SEED) ==="
$PYTHON_BIN surface_computation/simulated_samples_grid.py \
  --machine-id PC_TEST \
  --test-mode \
  --match-csv-params "$PARAM_DIR" \
  --lock-backend file \
  --random-seed "$SEED" \
  --results-dir "$STD_RESULTS"

$PYTHON_BIN neural_network_optimization/create_averaged_surfaces_from_samples.py \
  --input-folder "$STD_RESULTS/sim_samples_100_50samples_circular_em_fullcov_free_weights" \
  --output-folder "$STD_RESULTS/$STD_AVG" \
  --workers 1 \
  --include-all-params

echo ""
echo "=== Pipeline approach (seed=$SEED) ==="
$PYTHON_BIN surface_computation/simulated_samples_grid.py \
  --machine-id PC_TEST \
  --test-mode \
  --match-csv-params "$PARAM_DIR" \
  --lock-backend file \
  --pipeline \
  --averaged-surfaces-dir "$PIPE_RESULTS/$PIPE_AVG" \
  --random-seed "$SEED" \
  --results-dir "$PIPE_RESULTS"

echo ""
echo "=== Comparing outputs ==="
$PYTHON_BIN - <<PYEOF
import sys, pickle
import numpy as np
from pathlib import Path

std_dir  = Path("$STD_RESULTS/$STD_AVG")
pipe_dir = Path("$PIPE_RESULTS/$PIPE_AVG")

std_files  = sorted(std_dir.glob("averaged_sf1_*.pkl"))
pipe_files = sorted(pipe_dir.glob("averaged_sf1_*.pkl"))

if not std_files:
    print("ERROR: no files in", std_dir); sys.exit(1)
if not pipe_files:
    print("ERROR: no files in", pipe_dir); sys.exit(1)

std_names  = {f.name for f in std_files}
pipe_names = {f.name for f in pipe_files}
if std_names != pipe_names:
    print("ERROR: file sets differ")
    print("  Standard:", sorted(std_names))
    print("  Pipeline:", sorted(pipe_names))
    sys.exit(1)

FIELDS = ["mu1_comp1_surface", "mu1_comp2_surface",
          "mu2_comp1_surface", "mu2_comp2_surface"]
ATOL = 0.05  # float16 quantisation of sample values propagates a small error through KDE

all_ok = True
for fname in sorted(std_names):
    with open(std_dir / fname, "rb") as f:
        std_surf  = pickle.load(f)["surface"]
    with open(pipe_dir / fname, "rb") as f:
        pipe_surf = pickle.load(f)["surface"]

    for field in FIELDS:
        a = np.asarray(getattr(std_surf,  field), dtype=np.float32)
        b = np.asarray(getattr(pipe_surf, field), dtype=np.float32)
        max_diff = float(np.nanmax(np.abs(a - b)))
        ok = np.allclose(a, b, atol=ATOL, equal_nan=True)
        status = "OK  " if ok else "FAIL"
        print(f"  {status}  {fname}/{field}: max_diff={max_diff:.5f}")
        if not ok:
            all_ok = False

print()
if all_ok:
    print("All surfaces match within atol=%s." % ATOL)
    sys.exit(0)
else:
    print("Mismatch detected — standard and pipeline paths diverge beyond tolerance.")
    sys.exit(1)
PYEOF

echo "Seed comparison completed successfully."
