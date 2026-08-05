#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"
BACKENDS="${CHITU_CP_ATTN_BACKENDS:-sage sparge}"

export CHITU_RUN_TASK_ID="${CHITU_RUN_TASK_ID:-flux-cp}"
export CHITU_FLUX_FLEXCACHE_STRATEGY="${CHITU_FLUX_FLEXCACHE_STRATEGY:-}"

for backend in $BACKENDS; do
  config="$ROOT_DIR/test/configs/flux_cp_${backend}.yaml"
  echo "=== Flux CP Ulysses backend: $backend ==="
  "$CHITU_BIN" run "$config"
done
