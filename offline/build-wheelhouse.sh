#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_PARENT="${BUNDLE_PARENT:-/cfs_cloud_code/royroychen/wheelhouse}"
BUNDLE_NAME="${BUNDLE_NAME:-wheelhouse-chitudiffusion-py312-cu130-sm120}"
BUNDLE_DIR="${BUNDLE_PARENT}/${BUNDLE_NAME}"
WHEEL_DIR="${BUNDLE_DIR}/wheels"
IMAGE="${CHITU_BUILD_IMAGE:-mirrors.tencent.com/chitudiffusion/chitudiffusion:py312-cu130}"

rm -rf "${BUNDLE_DIR}"
mkdir -p "${WHEEL_DIR}"
install -m 0644 \
  "${REPO_DIR}/offline/requirements-offline.txt" \
  "${BUNDLE_DIR}/requirements-offline.txt"
install -m 0644 \
  "${REPO_DIR}/offline/requirements-core.txt" \
  "${BUNDLE_DIR}/requirements-core.txt"

docker run --rm \
  -v "${BUNDLE_DIR}:/bundle" \
  "${IMAGE}" \
  bash -lc \
  '/usr/bin/python3.12 -m pip download \
    --only-binary=:all: \
    --dest /bundle/wheels \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -r /bundle/requirements-core.txt && \
   /usr/bin/python3.12 -m pip download \
    --only-binary=:all: \
    --dest /bundle/wheels \
    --index-url https://pypi.org/simple \
    "nvidia-cutlass-dsl[cu13]==4.6.0.dev0" \
    "cuda-python==13.0.3" && \
   /usr/bin/python3.12 -m pip download \
    --only-binary=:all: \
    --no-deps \
    --dest /bundle/wheels \
    --index-url https://pypi.org/simple \
    "flash-attn-4==4.0.0b25" \
    "quack-kernels==0.5.3"'

install -m 0755 "${REPO_DIR}/offline/install.sh" "${BUNDLE_DIR}/install.sh"
install -m 0755 "${REPO_DIR}/offline/verify.sh" "${BUNDLE_DIR}/verify.sh"

(
  cd "${BUNDLE_DIR}"
  export LC_ALL=C
  : > WHEELS.txt
  for wheel in wheels/*.whl; do
    printf '%s\n' "${wheel#wheels/}" >> WHEELS.txt
  done
  sha256sum wheels/*.whl | sort -k2 > SHA256SUMS
)

docker run --rm --network none \
  -e MIN_FREE_GIB=5 \
  -e PYTHON_BIN=/usr/bin/python3.12 \
  -v "${BUNDLE_DIR}:/bundle:ro" \
  -v "${BUNDLE_DIR}:/output" \
  -v "${REPO_DIR}:/repo:ro" \
  "${IMAGE}" \
  bash -lc \
  '/bundle/install.sh /tmp/chitu-verify /repo && \
   /bundle/verify.sh /tmp/chitu-verify && \
   /tmp/chitu-verify/bin/python -m pip freeze --exclude-editable | LC_ALL=C sort \
     > /output/requirements-lock.txt'

cat >"${BUNDLE_DIR}/BUILD-INFO.txt" <<EOF
platform=linux-x86_64
python=3.12
torch=2.10.0
pytorch_cuda=cu130
transformers=5.12.1
nvidia_cutlass_dsl=4.6.0.dev0
quack_kernels=0.5.3
flash_attn_4=4.0.0b25
build_image=${IMAGE}
built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

tar -I 'zstd -T0 -3' \
  -C "${BUNDLE_PARENT}" \
  -cf "${BUNDLE_PARENT}/${BUNDLE_NAME}.tar.zst" \
  "${BUNDLE_NAME}"
(
  cd "${BUNDLE_PARENT}"
  sha256sum "${BUNDLE_NAME}.tar.zst" >"${BUNDLE_NAME}.tar.zst.sha256"
)

echo "Created ${BUNDLE_PARENT}/${BUNDLE_NAME}.tar.zst"
