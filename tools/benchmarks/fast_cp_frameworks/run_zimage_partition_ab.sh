#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
OUTPUT_ROOT=${1:-$PROJECT_ROOT/outputs/fast-cp/zimage-partition-ab}
SEQUENCE=${FAST_CP_BENCH_SEQUENCE:-4096}
TEXT_SEQUENCE=${FAST_CP_BENCH_TEXT_SEQUENCE:-512}
read -r -a METHODS <<< "${FAST_CP_Z_PARTITION_METHODS:-nccl_agkv fast_agkv nccl_ulysses fast_ulysses}"

for partition in replicated sharded; do
    output_dir="$OUTPUT_ROOT/$partition"
    if [[ "$partition" == sharded ]]; then
        export CHITU_Z_SHARDED_TEXT=1
    else
        unset CHITU_Z_SHARDED_TEXT || true
    fi
    for cp in 2 4 8; do
        for method in "${METHODS[@]}"; do
            "$SCRIPT_DIR/run_local_joint_case.sh" \
                zimage "$method" "$cp" "$SEQUENCE" "$TEXT_SEQUENCE" "$output_dir"
        done
    done
done
