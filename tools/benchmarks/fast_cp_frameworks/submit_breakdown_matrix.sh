#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARTITION=${SRUN_PARTITION:-debug}
NODE_ARGS=()
if [[ -n "${FAST_CP_BENCH_NODE:-}" ]]; then
    NODE_ARGS=(--nodelist "$FAST_CP_BENCH_NODE")
fi

exec srun \
    -p "$PARTITION" \
    --exclusive \
    --job-name "${USER:-user}-fast-cp-breakdown" \
    --nodes 1 \
    --ntasks 1 \
    --cpus-per-task 192 \
    --mem 1937152 \
    --gres gpu:8 \
    "${NODE_ARGS[@]}" \
    "$SCRIPT_DIR/run_breakdown_matrix.sh" "$@"
