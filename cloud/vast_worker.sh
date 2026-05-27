#!/usr/bin/env bash
# Launch one pipeline worker per visible GPU on a Vast.ai instance.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT:$ROOT/neural_network_optimization:$ROOT/surface_computation"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -e ".[cuda]"
fi

if [[ -z "${REDIS_HOST:-}" ]]; then
  echo "REDIS_HOST is required for Vast workers." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
  GPU_IDS=(0)
fi

if [[ -n "${GPU_WORKERS:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$GPU_WORKERS"
fi

mkdir -p logs

MACHINE_BASE="${MACHINE_BASE:-$(hostname)}"
AVERAGED_SURFACES_DIR="${AVERAGED_SURFACES_DIR:-results/averaged_surfaces_vast}"
COMPLETION_REGISTRY="${COMPLETION_REGISTRY:-$(basename "$AVERAGED_SURFACES_DIR")}"
N_SIMULATIONS="${N_SIMULATIONS:-10000}"
N_SAMPLES="${N_SAMPLES:-100}"
LOCK_TTL="${LOCK_TTL:-1800}"
BUNDLE_OUTPUT="${BUNDLE_OUTPUT:-1}"
BUNDLE_DIR="${BUNDLE_DIR:-${AVERAGED_SURFACES_DIR}_bundles}"
GRID_LEVEL="${GRID_LEVEL:-}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
MAX_CHUNKS="${MAX_CHUNKS:-}"

COMMON_ARGS=(
  surface_computation/simulated_samples_grid.py
  --pipeline
  --lock-backend redis
  --lock-ttl "$LOCK_TTL"
  --averaged-surfaces-dir "$AVERAGED_SURFACES_DIR"
  --completion-registry "$COMPLETION_REGISTRY"
  --n-simulations "$N_SIMULATIONS"
  --n-samples "$N_SAMPLES"
)

if [[ "$BUNDLE_OUTPUT" == "1" || "$BUNDLE_OUTPUT" == "true" || "$BUNDLE_OUTPUT" == "yes" ]]; then
  COMMON_ARGS+=(--bundle-output --bundle-dir "$BUNDLE_DIR")
fi

if [[ -n "$GRID_LEVEL" ]]; then
  COMMON_ARGS+=(--grid-level "$GRID_LEVEL")
fi

if [[ -n "$CHUNK_SIZE" ]]; then
  COMMON_ARGS+=(--chunk-size "$CHUNK_SIZE")
fi

if [[ -n "$MAX_CHUNKS" ]]; then
  COMMON_ARGS+=(--max-chunks "$MAX_CHUNKS")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=($EXTRA_ARGS)
  COMMON_ARGS+=("${EXTRA_ARRAY[@]}")
fi

pids=()
for gpu in "${GPU_IDS[@]}"; do
  worker_id="${MACHINE_BASE}-gpu${gpu}"
  log_file="logs/vast_${worker_id}.log"
  echo "Starting $worker_id on CUDA_VISIBLE_DEVICES=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "${COMMON_ARGS[@]}" \
    --machine-id "$worker_id" >"$log_file" 2>&1 &
  pids+=("$!")
done

trap 'echo "Stopping workers"; kill "${pids[@]}" 2>/dev/null || true' INT TERM

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
