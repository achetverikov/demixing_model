#!/usr/bin/env bash
# Verify that standard and pipeline approaches produce averaged surfaces that:
#   1. Have the correct shape (181 mu1_bias points × 90 feat_diff points)
#   2. Contain finite log-density values (no NaN / Inf)
#   3. For off-diagonal pairs (sf1 != sf2) where both paths combine canonical+mirror,
#      mu1 log-density surfaces agree within atol=0.5.
#
# Note: exact numerical agreement is NOT expected.  The two paths use different RNG
# sequences (standard runs parameters in a flat list; pipeline groups mirror pairs and
# runs two legs per group), so even with the same base seed the actual samples drawn
# differ.  Diagonal surfaces additionally differ in sample count (standard uses 1 run,
# pipeline uses 2).  The mu2 spatial KDE is also too sensitive at its extreme grid
# points (±498) with only 50 samples to be compared tightly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT:$ROOT/neural_network_optimization"

SEED=123

PARAM_DIR="$ROOT/tests/param_list_compare"
rm -rf "$PARAM_DIR"
mkdir -p "$PARAM_DIR"

# One off-diagonal pair (canonical + mirror) — the case where both paths are comparable.
for combo in \
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

MU1_FIELDS   = ["mu1_comp1_surface", "mu1_comp2_surface"]
OTHER_FIELDS = ["mu2_comp1_surface", "mu2_comp2_surface"]
# mu1 log-densities are well-constrained; mu2 at extreme grid points can diverge
# wildly with few samples, so we only check finite-ness for mu2.
MU1_ATOL = 0.5

all_ok = True
for fname in sorted(std_names):
    with open(std_dir / fname, "rb") as f:
        std_obj  = pickle.load(f)
    with open(pipe_dir / fname, "rb") as f:
        pipe_obj = pickle.load(f)

    std_surf  = std_obj["surface"]
    pipe_surf = pipe_obj["surface"]

    # 1. Shape check — mu1: (181, 90), mu2: (167, 90)
    expected = {f: (181, 90) for f in MU1_FIELDS}
    expected.update({f: (167, 90) for f in OTHER_FIELDS})
    for field in MU1_FIELDS + OTHER_FIELDS:
        a = np.asarray(getattr(std_surf,  field), dtype=np.float32)
        b = np.asarray(getattr(pipe_surf, field), dtype=np.float32)
        exp = expected[field]
        if a.shape != exp:
            print(f"  FAIL  {fname}/{field}: std shape {a.shape} != expected {exp}")
            all_ok = False
        elif b.shape != exp:
            print(f"  FAIL  {fname}/{field}: pipe shape {b.shape} != expected {exp}")
            all_ok = False
        else:
            print(f"  OK    {fname}/{field}: shape {a.shape}")

    # 2. Finite-ness check (both paths)
    for tag, surf in [("std", std_surf), ("pipe", pipe_surf)]:
        for field in MU1_FIELDS + OTHER_FIELDS:
            arr = np.asarray(getattr(surf, field), dtype=np.float32)
            n_bad = int(np.sum(~np.isfinite(arr)))
            if n_bad > 0:
                print(f"  FAIL  {fname}/{field} [{tag}]: {n_bad} non-finite values")
                all_ok = False

    # 3. mu1 value-agreement check
    for field in MU1_FIELDS:
        a = np.asarray(getattr(std_surf,  field), dtype=np.float32)
        b = np.asarray(getattr(pipe_surf, field), dtype=np.float32)
        max_diff = float(np.nanmax(np.abs(a - b)))
        ok = max_diff <= MU1_ATOL
        status = "OK  " if ok else "FAIL"
        print(f"  {status}  {fname}/{field}: max_diff={max_diff:.5f} (atol={MU1_ATOL})")
        if not ok:
            all_ok = False

print()
if all_ok:
    print("All checks passed.")
    sys.exit(0)
else:
    print("One or more checks failed.")
    sys.exit(1)
PYEOF

echo "Seed comparison completed successfully."
