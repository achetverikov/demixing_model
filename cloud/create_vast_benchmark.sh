#!/usr/bin/env bash
# Launch one Vast.ai GPU benchmark job. The job uploads small CSV/JSON outputs.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  cloud/create_vast_benchmark.sh OFFER_ID GPU_NAME PRICE_PER_HOUR

Example:
  cloud/create_vast_benchmark.sh 35654913 RTX_4090 0.5333

Optional environment overrides:
  GIT_REPO, GIT_REF, VAST_IMAGE, VAST_DISK
  BENCHMARK_S3_PREFIX, BENCHMARK_SURFACE_COUNT, BENCHMARK_SEED
  BENCHMARK_N_SIMULATIONS, BENCHMARK_N_SAMPLES
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 3 ]]; then
  usage >&2
  exit 2
fi

OFFER_ID="$1"
GPU_NAME="$2"
PRICE_PER_HOUR="$3"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

load_env_file() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  local line name value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -n "$line" && "${line:0:1}" != "#" ]] || continue
    line="${line#export }"
    [[ "$line" == *"="* ]] || continue
    name="${line%%=*}"
    value="${line#*=}"
    name="${name%"${name##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    export "$name=$value"
  done < "$file"
}

load_env_file ".env.docker"
load_env_file ".env"

required=(
  S3_BUCKET
  S3_ENDPOINT_URL
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
)

missing=()
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    missing+=("$name")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "Missing required env vars: ${missing[*]}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ -x "../.venv/bin/python" ]]; then
  PYTHON_BIN="../.venv/bin/python"
fi

VASTAI_BIN="$(command -v vastai || true)"
if [[ -z "$VASTAI_BIN" ]]; then
  venv_vastai="$(dirname "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')")/vastai"
  if [[ ! -x "$venv_vastai" ]]; then
    echo "Installing vastai CLI into the active Python environment..."
    "$PYTHON_BIN" -m pip install vastai
  fi
  VASTAI_BIN="$venv_vastai"
fi

if [[ ! -x "$VASTAI_BIN" ]]; then
  echo "Could not find the vastai CLI executable." >&2
  exit 1
fi

VAST_KEY="${VASTAPIKEY:-${VAST_API_KEY:-}}"
if [[ -n "$VAST_KEY" ]]; then
  "$VASTAI_BIN" set api-key "$VAST_KEY"
fi

GIT_REPO="${GIT_REPO:-https://github.com/achetverikov/demixing_model.git}"
GIT_REF="${GIT_REF:-main}"
VAST_IMAGE="${VAST_IMAGE:-andreychetverikov/demixing-vast:latest}"
VAST_DISK="${VAST_DISK:-40}"
BENCHMARK_S3_PREFIX="${BENCHMARK_S3_PREFIX:-demixing/benchmarks/gpu_surface_timing}"
BENCHMARK_SURFACE_COUNT="${BENCHMARK_SURFACE_COUNT:-20}"
BENCHMARK_SEED="${BENCHMARK_SEED:-20260527}"
BENCHMARK_N_SIMULATIONS="${BENCHMARK_N_SIMULATIONS:-10000}"
BENCHMARK_N_SAMPLES="${BENCHMARK_N_SAMPLES:-20,100}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"

VAST_ENV=(
  -e "GIT_REPO=$GIT_REPO"
  -e "GIT_REF=$GIT_REF"
  -e "REPO_DIR=/workspace/demixing_model"
  -e "S3_BUCKET=$S3_BUCKET"
  -e "S3_ENDPOINT_URL=$S3_ENDPOINT_URL"
  -e "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID"
  -e "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"
  -e "AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION"
  -e "BENCHMARK_GPU_NAME=$GPU_NAME"
  -e "BENCHMARK_OFFER_ID=$OFFER_ID"
  -e "BENCHMARK_PRICE_PER_HOUR=$PRICE_PER_HOUR"
  -e "BENCHMARK_S3_PREFIX=$BENCHMARK_S3_PREFIX"
  -e "BENCHMARK_SURFACE_COUNT=$BENCHMARK_SURFACE_COUNT"
  -e "BENCHMARK_SEED=$BENCHMARK_SEED"
  -e "BENCHMARK_N_SIMULATIONS=$BENCHMARK_N_SIMULATIONS"
  -e "BENCHMARK_N_SAMPLES=$BENCHMARK_N_SAMPLES"
)

ONSTART='cd /workspace && git clone "$GIT_REPO" "$REPO_DIR" && cd "$REPO_DIR" && git checkout "$GIT_REF" && /opt/demixing-venv/bin/python cloud/benchmark_gpu_surfaces.py --gpu-name "$BENCHMARK_GPU_NAME" --offer-id "$BENCHMARK_OFFER_ID" --price-per-hour "$BENCHMARK_PRICE_PER_HOUR" --n-simulations "$BENCHMARK_N_SIMULATIONS" --n-samples ${BENCHMARK_N_SAMPLES//,/ } --surface-count "$BENCHMARK_SURFACE_COUNT" --seed "$BENCHMARK_SEED" --upload-prefix "$BENCHMARK_S3_PREFIX"'

echo "Creating Vast benchmark instance from offer $OFFER_ID"
echo "  GPU     : $GPU_NAME"
echo "  Price   : $PRICE_PER_HOUR/hr"
echo "  Image   : $VAST_IMAGE"
echo "  Repo    : $GIT_REPO@$GIT_REF"
echo "  Prefix  : $BENCHMARK_S3_PREFIX"
echo "  Workload: ${BENCHMARK_N_SIMULATIONS} x [${BENCHMARK_N_SAMPLES}], surfaces=$BENCHMARK_SURFACE_COUNT"

"$VASTAI_BIN" create instance "$OFFER_ID" \
  --image "$VAST_IMAGE" \
  --disk "$VAST_DISK" \
  --ssh \
  --direct \
  --env "${VAST_ENV[*]}" \
  --onstart-cmd "$ONSTART"
