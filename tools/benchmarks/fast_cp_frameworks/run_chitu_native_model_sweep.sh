#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MODEL=${1:-all}
OUTPUT_ROOT=${2:-$PROJECT_ROOT/outputs/fast-cp/chitu-native-model-sweep}
SEQUENCE=${FAST_CP_BENCH_SEQUENCE:-4096}
TEXT_SEQUENCE=${FAST_CP_BENCH_TEXT_SEQUENCE:-512}

case "$MODEL" in
    all) MODELS=(wan1.3b wan14b qwenimage zimage) ;;
    wan1.3b|wan14b|qwenimage|zimage) MODELS=("$MODEL") ;;
    *) echo "model must be all, wan1.3b, wan14b, qwenimage, or zimage" >&2; exit 2 ;;
esac

for model in "${MODELS[@]}"; do
    if [[ "$model" == wan* ]]; then
        runner="$SCRIPT_DIR/run_local_wan_case.sh"
        "$runner" "$model" nccl_agkv 1 "$SEQUENCE" native "$OUTPUT_ROOT"
        for cp in 2 4 8; do
            for method in nccl_agkv fast_agkv nccl_ulysses fast_ulysses; do
                "$runner" "$model" "$method" "$cp" "$SEQUENCE" native "$OUTPUT_ROOT"
            done
        done
    else
        runner="$SCRIPT_DIR/run_local_joint_case.sh"
        "$runner" "$model" nccl_agkv 1 "$SEQUENCE" "$TEXT_SEQUENCE" "$OUTPUT_ROOT"
        for cp in 2 4 8; do
            for method in nccl_agkv fast_agkv nccl_ulysses fast_ulysses; do
                "$runner" "$model" "$method" "$cp" "$SEQUENCE" "$TEXT_SEQUENCE" "$OUTPUT_ROOT"
            done
        done
    fi
done
