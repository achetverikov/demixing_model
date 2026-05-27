#!/usr/bin/env bash
# Verify that standard and pipeline approaches produce statistically equivalent
# averaged surfaces when given the same random seed.
#
# Both approaches use parameter-hash-based seeding (see create_param_identifier +
# jax.random.fold_in in simulated_samples_grid.py), so the same (sf1,sf2,sp) draws
# IDENTICAL float32 samples in both modes.  The only source of numerical difference
# is that the standard path stores samples on disk as float16 (~0.18° quantisation
# for values near ±180°) before the KDE step.  This introduces log-density errors
# up to ~0.1 nats in the mu1 surfaces; the comparison below checks max_diff ≤ 0.15.
#
# Only off-diagonal pairs are tested: both paths combine two simulations
# (canonical + mirror), so sample counts are equal.  Diagonal pairs differ in
# structure (standard: 1 file, pipeline: 2 runs) and are out of scope here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT:$ROOT/neural_network_optimization"

SEED=123

PARAM_DIR="$ROOT/tests/param_list_compare"
rm -rf "$PARAM_DIR"
mkdir -p "$PARAM_DIR"

# One off-diagonal pair: canonical (10,20) + mirror (20,10).
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
# Tolerance reflects float16 quantisation of stored samples (max ~0.1 log-density error).
MU1_ATOL = 0.15

all_ok = True
for fname in sorted(std_names):
    with open(std_dir / fname, "rb") as f:
        std_surf  = pickle.load(f)["surface"]
    with open(pipe_dir / fname, "rb") as f:
        pipe_surf = pickle.load(f)["surface"]

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

    # 2. Finite-ness check
    for tag, surf in [("std", std_surf), ("pipe", pipe_surf)]:
        for field in MU1_FIELDS + OTHER_FIELDS:
            arr = np.asarray(getattr(surf, field), dtype=np.float32)
            n_bad = int(np.sum(~np.isfinite(arr)))
            if n_bad > 0:
                print(f"  FAIL  {fname}/{field} [{tag}]: {n_bad} non-finite values")
                all_ok = False

    # 3. mu1 value-agreement: should match within float16 quantisation noise
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
else:
    print("One or more checks failed.")

# ── plots ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plot_dir = Path("$STD_RESULTS").parent / "compare_seed_plots"
plot_dir.mkdir(parents=True, exist_ok=True)

for fname in sorted(std_names):
    with open(std_dir / fname, "rb") as f:
        std_surf  = pickle.load(f)["surface"]
    with open(pipe_dir / fname, "rb") as f:
        pipe_surf = pickle.load(f)["surface"]

    stem = fname.replace(".pkl", "")
    for field in MU1_FIELDS:
        a = np.asarray(getattr(std_surf,  field), dtype=np.float32)  # (181, 90)
        b = np.asarray(getattr(pipe_surf, field), dtype=np.float32)
        diff = b - a

        fd_grid   = np.asarray(std_surf.feat_diff_grid)
        bias_grid = np.asarray(std_surf.mu1_bias_grid)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"{stem} / {field}\nmax|diff|={np.nanmax(np.abs(diff)):.4f}  (float16 quantisation)")

        vmin = min(a.min(), b.min())
        vmax = max(a.max(), b.max())

        for ax, data, title in zip(axes[:2], [a, b], ["standard (float16→KDE)", "pipeline (float32→KDE)"]):
            im = ax.imshow(data, aspect="auto", origin="lower",
                           extent=[fd_grid[0], fd_grid[-1], bias_grid[0], bias_grid[-1]],
                           vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("feat_diff (°)")
            ax.set_ylabel("mu1_bias (°)")
            plt.colorbar(im, ax=ax, label="log density")

        abs_max = np.nanmax(np.abs(diff))
        im = axes[2].imshow(diff, aspect="auto", origin="lower",
                            extent=[fd_grid[0], fd_grid[-1], bias_grid[0], bias_grid[-1]],
                            vmin=-abs_max, vmax=abs_max, cmap="RdBu_r")
        axes[2].set_title("pipeline − standard")
        axes[2].set_xlabel("feat_diff (°)")
        plt.colorbar(im, ax=axes[2], label="Δ log density")

        plt.tight_layout()
        out = plot_dir / f"{stem}_{field}.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"Saved: {out}")

print(f"Plots saved to: {plot_dir}")

sys.exit(0 if all_ok else 1)
PYEOF

echo "Seed comparison completed successfully."
