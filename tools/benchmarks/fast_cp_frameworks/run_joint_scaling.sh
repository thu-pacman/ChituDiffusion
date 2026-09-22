#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MODEL=${1:-qwenimage}
SEQUENCE=${2:-4096}
TEXT_SEQUENCE=${3:-512}
OUTPUT_ROOT=${4:-$PROJECT_ROOT/outputs/fast-cp/joint-attention-same-node}

"$SCRIPT_DIR/run_local_joint_case.sh" \
    "$MODEL" nccl_agkv 1 "$SEQUENCE" "$TEXT_SEQUENCE" "$OUTPUT_ROOT"
for cp in 2 4 8; do
    for method in nccl_agkv fast_agkv nccl_ulysses fast_ulysses; do
        "$SCRIPT_DIR/run_local_joint_case.sh" \
            "$MODEL" "$method" "$cp" "$SEQUENCE" "$TEXT_SEQUENCE" "$OUTPUT_ROOT"
    done
done
