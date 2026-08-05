#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"
BACKENDS="${CHITU_CP_ATTN_BACKENDS:-sage sparge}"

export CHITU_RUN_TASK_ID="${CHITU_RUN_TASK_ID:-wan-cp}"
export CHITU_WAN_STEPS="${CHITU_WAN_STEPS:-3}"
export CHITU_FLEXCACHE_STRATEGIES="${CHITU_FLEXCACHE_STRATEGIES:-blockdance}"
export CHITU_FLEXCACHE_STRATEGIES="${CHITU_FLEXCACHE_STRATEGIES// /,}"

for backend in $BACKENDS; do
  config="$ROOT_DIR/test/configs/wan_cp_${backend}.yaml"
  echo "=== Wan CP Ulysses backend: $backend ==="
  "$CHITU_BIN" run "$config"
done
