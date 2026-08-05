#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHITU_BIN="${CHITU_BIN:-$ROOT_DIR/.venv/bin/chitu}"

export CHITU_WAN_STEPS="${CHITU_WAN_STEPS:-3}"
export CHITU_FLEXCACHE_STRATEGIES="${CHITU_FLEXCACHE_STRATEGIES:-teacache pab blockdance taylorseer cubic}"

IFS=' ' read -r -a strategies <<< "$CHITU_FLEXCACHE_STRATEGIES"
export CHITU_FLEXCACHE_STRATEGIES="$(IFS=,; echo "${strategies[*]}")"

echo "=== Wan FlexCache strategies: $CHITU_FLEXCACHE_STRATEGIES ==="
"$CHITU_BIN" run "$ROOT_DIR/test/configs/wan_flexcache.yaml"

for strategy in ${CHITU_FLUX_FLEXCACHE_STRATEGIES:-teacache pab blockdance taylorseer cubic}; do
  echo "=== Flux FlexCache strategy: $strategy ==="
  CHITU_RUN_TASK_ID="flux-flexcache-$strategy" \
  CHITU_FLUX_FLEXCACHE_STRATEGY="$strategy" \
    "$CHITU_BIN" run "$ROOT_DIR/test/configs/flux_flexcache.yaml"
done
