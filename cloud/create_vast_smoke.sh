#!/usr/bin/env bash
# Create a tiny Vast.ai smoke instance using credentials from .env/.env.docker.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  cloud/create_vast_smoke.sh OFFER_ID

Optional environment overrides:
  GIT_REPO, GIT_REF, VAST_IMAGE, VAST_DISK
  VAST_SMOKE_S3_PREFIX, VAST_SMOKE_REGISTRY
  N_SIMULATIONS, N_SAMPLES, GPU_WORKERS
  GRID_LEVEL, CHUNK_SIZE, MAX_CHUNKS
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

OFFER_ID="$1"

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
  REDIS_HOST
  REDIS_PASSWORD
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
  if [[ -x "$venv_vastai" ]]; then
    VASTAI_BIN="$venv_vastai"
  else
    VASTAI_BIN="$(command -v vastai || true)"
  fi
fi

if [[ -z "$VASTAI_BIN" ]]; then
  echo "Could not find the vastai CLI executable after installation." >&2
  exit 1
fi

VASTAI=("$VASTAI_BIN")

VAST_KEY="${VASTAPIKEY:-${VAST_API_KEY:-}}"
if [[ -n "$VAST_KEY" ]]; then
  "${VASTAI[@]}" set api-key "$VAST_KEY"
fi

GIT_REPO="${GIT_REPO:-https://github.com/achetverikov/demixing_model.git}"
GIT_REF="${GIT_REF:-main}"
VAST_IMAGE="${VAST_IMAGE:-andreychetverikov/demixing-vast:latest}"
VAST_DISK="${VAST_DISK:-60}"  # image alone is ~4-5GB; offers with <60GB disk fail to start
S3_PREFIX="${VAST_SMOKE_S3_PREFIX:-demixing/vast_smoke}"
COMPLETION_REGISTRY="${VAST_SMOKE_REGISTRY:-averaged_surfaces_vast_smoke}"
N_SIMULATIONS="${N_SIMULATIONS:-100}"
N_SAMPLES="${N_SAMPLES:-20}"
GPU_WORKERS="${GPU_WORKERS:-0}"
GRID_LEVEL="${GRID_LEVEL:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"
MAX_CHUNKS="${MAX_CHUNKS:-1}"

REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_USERNAME="${REDIS_USERNAME:-default}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"
DOCKER_LOGIN_USER="${DOCKER_LOGIN_USER:-}"
DOCKER_LOGIN_PASS="${DOCKER_LOGIN_PASS:-}"
AVERAGED_SURFACES_DIR="${AVERAGED_SURFACES_DIR:-results/$COMPLETION_REGISTRY}"
BUNDLE_DIR="${BUNDLE_DIR:-results/${COMPLETION_REGISTRY}_bundles}"

VAST_ENV=(
  -e "GIT_REPO=$GIT_REPO"
  -e "GIT_REF=$GIT_REF"
  -e "REPO_DIR=/workspace/demixing_model"
  -e "REDIS_HOST=$REDIS_HOST"
  -e "REDIS_PORT=$REDIS_PORT"
  -e "REDIS_USERNAME=$REDIS_USERNAME"
  -e "REDIS_PASSWORD=$REDIS_PASSWORD"
  -e "S3_BUCKET=$S3_BUCKET"
  -e "S3_PREFIX=$S3_PREFIX"
  -e "S3_ENDPOINT_URL=$S3_ENDPOINT_URL"
  -e "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID"
  -e "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY"
  -e "AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION"
  -e "AVERAGED_SURFACES_DIR=$AVERAGED_SURFACES_DIR"
  -e "COMPLETION_REGISTRY=$COMPLETION_REGISTRY"
  -e "BUNDLE_OUTPUT=1"
  -e "BUNDLE_DIR=$BUNDLE_DIR"
  -e "N_SIMULATIONS=$N_SIMULATIONS"
  -e "N_SAMPLES=$N_SAMPLES"
  -e "GPU_WORKERS=$GPU_WORKERS"
  -e "GRID_LEVEL=$GRID_LEVEL"
  -e "CHUNK_SIZE=$CHUNK_SIZE"
  -e "MAX_CHUNKS=$MAX_CHUNKS"
)

ONSTART='cd /workspace && git clone "$GIT_REPO" "$REPO_DIR" && cd "$REPO_DIR" && git checkout "$GIT_REF" && bash cloud/vast_worker.sh; vastai destroy instance $CONTAINER_ID'

echo "Creating Vast smoke instance from offer $OFFER_ID"
echo "  Image   : $VAST_IMAGE"
echo "  Repo    : $GIT_REPO@$GIT_REF"
echo "  Prefix  : $S3_PREFIX"
echo "  Registry: $COMPLETION_REGISTRY"

VASTAI_CREATE_ARGS=(
  --image "$VAST_IMAGE"
  --disk "$VAST_DISK"
  --ssh
  --direct
  --env "${VAST_ENV[*]}"
  --onstart-cmd "$ONSTART"
)
if [[ -n "$DOCKER_LOGIN_USER" && -n "$DOCKER_LOGIN_PASS" ]]; then
  VASTAI_CREATE_ARGS+=(--login "-u $DOCKER_LOGIN_USER -p $DOCKER_LOGIN_PASS docker.io")
fi

"${VASTAI[@]}" create instance "$OFFER_ID" "${VASTAI_CREATE_ARGS[@]}"
