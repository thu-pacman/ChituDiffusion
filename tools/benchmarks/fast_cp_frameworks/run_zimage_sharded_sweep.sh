#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
OUTPUT_ROOT=${1:-$PROJECT_ROOT/outputs/fast-cp/zimage-sharded-sweep}
SEQUENCE=${FAST_CP_BENCH_SEQUENCE:-4096}
TEXT_SEQUENCE=${FAST_CP_BENCH_TEXT_SEQUENCE:-512}

export CHITU_Z_SHARDED_TEXT=1
"$SCRIPT_DIR/run_local_joint_case.sh" \
    zimage nccl_agkv 1 "$SEQUENCE" "$TEXT_SEQUENCE" "$OUTPUT_ROOT"
for cp in 2 4 8; do
    for method in nccl_agkv fast_agkv nccl_ulysses fast_ulysses; do
        "$SCRIPT_DIR/run_local_joint_case.sh" \
            zimage "$method" "$cp" "$SEQUENCE" "$TEXT_SEQUENCE" "$OUTPUT_ROOT"
    done
done
