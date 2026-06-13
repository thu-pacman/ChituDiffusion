#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"
BACKENDS="${CHITU_ATTN_BACKENDS:-torch_sdpa flash sage sparge}"

export CHITU_RUN_TASK_ID="${CHITU_RUN_TASK_ID:-wan-attn}"
export CHITU_WAN_STEPS="${CHITU_WAN_STEPS:-3}"
export CHITU_FLEXCACHE_STRATEGIES="${CHITU_FLEXCACHE_STRATEGIES:-blockdance}"
export CHITU_FLEXCACHE_STRATEGIES="${CHITU_FLEXCACHE_STRATEGIES// /,}"

for backend in $BACKENDS; do
  config_backend="${backend}"
  if [[ "$backend" == "flash_attn" ]]; then
    config_backend="flash"
  fi
  config="$ROOT_DIR/test/configs/wan_attention_${config_backend}.yaml"
  echo "=== Wan attention backend: $backend ==="
  "$CHITU_BIN" run "$config"
done
