#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
source "$SCRIPT_DIR/versions.env"

SOURCE_ROOT=${FAST_CP_BENCH_SOURCE_ROOT:-$PROJECT_ROOT/refs/fast-cp-frameworks}
ENV_ROOT=${FAST_CP_BENCH_ENV_ROOT:-$PROJECT_ROOT/.venv-fast-cp-frameworks}
readonly UV_EXECUTABLE=${UV_EXECUTABLE:-$(command -v uv)}
PYTHON_VERSION=${FAST_CP_BENCH_PYTHON:-3.12}
RECREATE=${FAST_CP_BENCH_RECREATE:-0}

checkout() {
    local repository=$1
    local revision=$2
    local destination=$3
    if [[ ! -d "$destination/.git" ]]; then
        "$UV_EXECUTABLE" --version >/dev/null
        git clone --filter=blob:none --depth 1 --no-checkout "$repository" "$destination"
    fi
    git -C "$destination" remote set-url origin "$repository"
    git -C "$destination" fetch --depth 1 origin "$revision"
    git -C "$destination" checkout --detach "$revision"
    test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
}

create_env() {
    local destination=$1
    if [[ "$RECREATE" == 1 || ! -x "$destination/bin/python" ]]; then
        "$UV_EXECUTABLE" venv --clear --seed --python "$PYTHON_VERSION" "$destination"
    fi
}

mkdir -p "$SOURCE_ROOT" "$ENV_ROOT"
checkout "$XDIT_REPOSITORY" "$XDIT_COMMIT" "$SOURCE_ROOT/xdit"
checkout "$SGLANG_REPOSITORY" "$SGLANG_COMMIT" "$SOURCE_ROOT/sglang"

create_env "$ENV_ROOT/xdit"
"$UV_EXECUTABLE" pip install --python "$ENV_ROOT/xdit/bin/python" \
    torch==2.13.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130
"$UV_EXECUTABLE" pip install --python "$ENV_ROOT/xdit/bin/python" \
    --editable "$SOURCE_ROOT/xdit"

create_env "$ENV_ROOT/sglang"
SGLANG_BUILD_RUST_EXTS=none "$UV_EXECUTABLE" pip install \
    --python "$ENV_ROOT/sglang/bin/python" \
    --editable "$SOURCE_ROOT/sglang/python[diffusion]"

"$ENV_ROOT/xdit/bin/python" -c \
    'import torch, xfuser; print("xdit", torch.__version__, xfuser.__file__)'
"$ENV_ROOT/sglang/bin/python" -c \
    'import torch, sglang; print("sglang", torch.__version__, sglang.__file__)'
