#!/usr/bin/env bash
# Bootstrap a Vast.ai container, then launch surface workers.
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKDIR/demixing_model}"
GIT_REPO="${GIT_REPO:-}"
GIT_REF="${GIT_REF:-main}"

if [[ -z "$GIT_REPO" ]]; then
  echo "GIT_REPO is required, for example https://github.com/<org>/demixing_model.git" >&2
  exit 1
fi

mkdir -p "$WORKDIR"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "Updating existing repo at $REPO_DIR"
  git -C "$REPO_DIR" fetch --all --tags --prune
else
  echo "Cloning $GIT_REPO into $REPO_DIR"
  git clone "$GIT_REPO" "$REPO_DIR"
fi

echo "Checking out $GIT_REF"
git -C "$REPO_DIR" checkout "$GIT_REF"

# If GIT_REF is a branch, pull its latest commit. For tags/SHAs this is a no-op.
if git -C "$REPO_DIR" symbolic-ref -q HEAD >/dev/null; then
  git -C "$REPO_DIR" pull --ff-only
fi

cd "$REPO_DIR"
echo "Running commit $(git rev-parse --short HEAD)"

exec bash cloud/vast_worker.sh
