#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"
BACKENDS="${CHITU_ATTN_BACKENDS:-torch_sdpa flash flashinfer sage sparge}"

export CHITU_RUN_TASK_ID="${CHITU_RUN_TASK_ID:-flux-attn}"
export CHITU_FLUX_FLEXCACHE_STRATEGY="${CHITU_FLUX_FLEXCACHE_STRATEGY:-}"

for backend in $BACKENDS; do
  config_backend="${backend}"
  if [[ "$backend" == "flash_attn" ]]; then
    config_backend="flash"
  fi
  config="$ROOT_DIR/test/configs/flux_attention_${config_backend}.yaml"
  echo "=== Flux attention backend: $backend ==="
  "$CHITU_BIN" run "$config"
done
