#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MODEL=${1:-wan14b}
SEQUENCE=${2:-4096}
CP=${3:-8}
OUTPUT_ROOT=${4:-$PROJECT_ROOT/outputs/fast-cp/model-attention-same-node}

CASES=(
    "fast_agkv native"
    "fast_agkv serialized_ce"
    "fast_agkv sync"
    "fast_ulysses native"
    "fast_ulysses serialized_ce"
    "fast_ulysses sync"
    "nccl_agkv native"
    "nccl_ulysses native"
)

for case_spec in "${CASES[@]}"; do
    read -r method schedule <<<"$case_spec"
    "$SCRIPT_DIR/run_local_wan_case.sh" \
        "$MODEL" "$method" "$CP" "$SEQUENCE" "$schedule" "$OUTPUT_ROOT"
done
