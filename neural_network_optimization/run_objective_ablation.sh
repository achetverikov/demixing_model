#!/usr/bin/env bash
set -euo pipefail

# Controlled loss comparison against selected 100k-simulation mu1 surfaces.
# Run stages separately because reference generation and six full trainings are
# intentionally expensive:
#   ./neural_network_optimization/run_objective_ablation.sh 20 select
#   ./neural_network_optimization/run_objective_ablation.sh 20 truth
#   ./neural_network_optimization/run_objective_ablation.sh 20 train
#   ./neural_network_optimization/run_objective_ablation.sh 20 evaluate

N_SAMPLES="${1:-20}"
STAGE="${2:-select}"
if [[ "$N_SAMPLES" != "20" && "$N_SAMPLES" != "100" ]]; then
  echo "N_SAMPLES must be 20 or 100" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
PYTHON="${PYTHON:-/workspaces/.venv/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$WORKSPACE_ROOT/results}"
SOURCE_SURFACES="${SOURCE_SURFACES:-$ARTIFACT_ROOT/averaged_surfaces_10k_${N_SAMPLES}samples_circular}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$ARTIFACT_ROOT/mu1_objective_ablation_${N_SAMPLES}samples}"
MANIFEST="$EXPERIMENT_ROOT/scenarios.csv"
TRUTH="$EXPERIMENT_ROOT/reference_100k_seed314159"
TRUTH_REPEAT="$EXPERIMENT_ROOT/reference_100k_seed271828"
# This is a screening experiment, not the final production retraining. On the
# RTX 5080, batches 32/64/128 all take about nine seconds per full epoch, while
# 128 also approaches the memory limit. Keep the production batch size and cut
# the number of full-data passes instead.
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RUN_TAG="${RUN_TAG:-screen_e${EPOCHS}_b${BATCH_SIZE}}"
PROFILES="${PROFILES:-kl kl_energy circular circular_hellinger circular_log_smooth circular_curvature_10k}"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/neural_network_optimization:$REPO_ROOT/surface_computation${PYTHONPATH:+:$PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/demixing_jax_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/demixing_xdg_cache}"
mkdir -p "$EXPERIMENT_ROOT" "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME"

select_cases() {
  JAX_PLATFORMS=cpu "$PYTHON" "$REPO_ROOT/neural_network_optimization/objective_ablation.py" select \
    --surfaces-folder "$SOURCE_SURFACES" \
    --per-scenario "${PER_SCENARIO:-3}" \
    --candidate-stride "${CANDIDATE_STRIDE:-8}" \
    --min-parameter-distance "${MIN_PARAMETER_DISTANCE:-30}" \
    --output "$MANIFEST"
}

generate_truth() {
  local seed="$1"
  local output="$2"
  "$PYTHON" "$REPO_ROOT/surface_computation/simulated_samples_grid.py" \
    --machine-id "objective_${N_SAMPLES}_${seed}" \
    --grid-level 1 --no-auto-advance --lock-backend file \
    --pipeline --param-file "$MANIFEST" \
    --n-simulations "${REFERENCE_SIMULATIONS:-100000}" \
    --n-samples "$N_SAMPLES" --random-seed "$seed" \
    --bias-bandwidth 0.075 --feat-bandwidth 3.0 \
    --chunk-size 1 \
    --results-dir "$EXPERIMENT_ROOT/simulation_seed${seed}" \
    --averaged-surfaces-dir "$output"
}

train_profiles() {
  for profile in $PROFILES; do
    echo "Training $profile for the ${N_SAMPLES}-observation model"
    "$PYTHON" "$REPO_ROOT/neural_network_optimization/mirror_aware_training.py" \
      --surfaces-folder "$SOURCE_SURFACES" \
      --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --learning-rate 0.002 \
      --weight-decay 0.0001 --seed 42 \
      --loss-profile "$profile" \
      --save-dir "$EXPERIMENT_ROOT/checkpoints/$RUN_TAG/$profile" \
      --results-dir "$EXPERIMENT_ROOT"
  done
}

evaluate_profiles() {
  local checkpoint_args=()
  local profile
  for profile in $PROFILES; do
    checkpoint_args+=(--checkpoint "$profile=$EXPERIMENT_ROOT/checkpoints/$RUN_TAG/$profile")
  done
  local repeat_args=()
  if compgen -G "$TRUTH_REPEAT/averaged_sf1_*.pkl" >/dev/null; then
    repeat_args+=(--reference-repeat "$TRUTH_REPEAT")
  fi
  "$PYTHON" "$REPO_ROOT/neural_network_optimization/objective_ablation.py" evaluate \
    --truth-folder "$TRUTH" --manifest "$MANIFEST" \
    "${repeat_args[@]}" "${checkpoint_args[@]}" \
    --output "$EXPERIMENT_ROOT/evaluation_by_feat_diff.csv"
}

case "$STAGE" in
  select) select_cases ;;
  truth) generate_truth 314159 "$TRUTH" ;;
  truth-repeat) generate_truth 271828 "$TRUTH_REPEAT" ;;
  train) train_profiles ;;
  evaluate) evaluate_profiles ;;
  all)
    select_cases
    generate_truth 314159 "$TRUTH"
    train_profiles
    evaluate_profiles
    ;;
  *)
    echo "Stage must be select, truth, truth-repeat, train, evaluate, or all" >&2
    exit 2
    ;;
esac
