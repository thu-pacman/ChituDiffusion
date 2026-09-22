#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MODEL=${1:-wan14b}
SEQUENCE=${2:-4096}
OUTPUT_ROOT=${3:-$PROJECT_ROOT/outputs/fast-cp/wan-scaling-same-node}

"$SCRIPT_DIR/run_local_wan_case.sh" \
    "$MODEL" nccl_agkv 1 "$SEQUENCE" native "$OUTPUT_ROOT"
for cp in 2 4 8; do
    "$SCRIPT_DIR/run_wan_matrix.sh" "$MODEL" "$SEQUENCE" "$cp" "$OUTPUT_ROOT"
done
