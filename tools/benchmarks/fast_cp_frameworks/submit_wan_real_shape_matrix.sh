#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARTITION_ARGS=()
if [[ -n "${SRUN_PARTITION:-}" ]]; then
    PARTITION_ARGS=(--partition "$SRUN_PARTITION")
fi
NODE_ARGS=()
if [[ -n "${FAST_CP_BENCH_NODE:-}" ]]; then
    NODE_ARGS=(--nodelist "$FAST_CP_BENCH_NODE")
fi

exec srun \
    "${PARTITION_ARGS[@]}" \
    --exclusive \
    --job-name "${USER:-user}-wan-real-shapes" \
    --nodes 1 \
    --ntasks 1 \
    --cpus-per-task 192 \
    --mem 1937152 \
    --gres gpu:8 \
    "${NODE_ARGS[@]}" \
    "$SCRIPT_DIR/run_wan_real_shape_matrix.sh" "$@"
