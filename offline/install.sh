#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_DIR="${BUNDLE_DIR}/wheels"
REQUIREMENTS="${BUNDLE_DIR}/requirements-core.txt"
VENV_DIR="${1:-${CHITU_VENV_DIR:-/dockerdata/chitudiffusion-venv}}"
REPO_DIR="${2:-${CHITU_REPO_DIR:-/cfs_cloud_code/royroychen/Chitu/ChituDiffusion}}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
MIN_FREE_GIB="${MIN_FREE_GIB:-30}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

warn() {
  echo "WARNING: $*" >&2
}

[[ "$(uname -s)" == "Linux" ]] || die "Only Linux is supported."
[[ "$(uname -m)" == "x86_64" ]] || die "Only Linux x86_64 is supported."
[[ -d "${WHEEL_DIR}" ]] || die "Wheel directory not found: ${WHEEL_DIR}"
[[ -f "${REQUIREMENTS}" ]] || die "Requirements file not found: ${REQUIREMENTS}"
[[ -f "${REPO_DIR}/pyproject.toml" ]] || \
  die "ChituDiffusion repository not found: ${REPO_DIR}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
  die "Python 3.12 is required. Set PYTHON_BIN to its executable."

if [[ -f "${BUNDLE_DIR}/SHA256SUMS" ]]; then
  echo "Verifying wheel checksums..."
  (cd "${BUNDLE_DIR}" && sha256sum -c --quiet SHA256SUMS) || \
    die "Wheel checksum verification failed."
fi

python_version="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${python_version}" == "3.12" ]] || \
  die "Expected Python 3.12, found ${python_version} at ${PYTHON_BIN}."

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Detected NVIDIA environment:"
  nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader || warn "nvidia-smi query failed."
else
  warn "nvidia-smi is unavailable; GPU compatibility cannot be checked now."
fi

venv_parent="$(dirname "${VENV_DIR}")"
mkdir -p "${venv_parent}"
free_kib="$(df -Pk "${venv_parent}" | awk 'NR==2 {print $4}')"
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
(( free_kib >= required_kib )) || \
  die "At least ${MIN_FREE_GIB} GiB free space is required under ${venv_parent}."

if [[ -e "${VENV_DIR}" && ! -x "${VENV_DIR}/bin/python" ]]; then
  die "Refusing to overwrite non-venv path: ${VENV_DIR}"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

echo "Installing ChituDiffusion dependencies from local wheels only..."
PIP_NO_INDEX=1 \
"${VENV_DIR}/bin/python" -m pip install \
  --no-index \
  --find-links "${WHEEL_DIR}" \
  --only-binary=:all: \
  -r "${REQUIREMENTS}"

echo "Installing the SM120 FA4 compatibility set..."
PIP_NO_INDEX=1 \
"${VENV_DIR}/bin/python" -m pip install \
  --no-index \
  --find-links "${WHEEL_DIR}" \
  --only-binary=:all: \
  "nvidia-cutlass-dsl[cu13]==4.6.0.dev0"

PIP_NO_INDEX=1 \
"${VENV_DIR}/bin/python" -m pip install \
  --no-index \
  --find-links "${WHEEL_DIR}" \
  --only-binary=:all: \
  --no-deps \
  "flash-attn-4==4.0.0b25" \
  "quack-kernels==0.5.3"

echo "Registering editable ChituDiffusion source: ${REPO_DIR}"
PIP_NO_INDEX=1 \
"${VENV_DIR}/bin/python" -m pip install \
  --no-index \
  --find-links "${WHEEL_DIR}" \
  --no-build-isolation \
  --no-deps \
  -e "${REPO_DIR}"

cat >"${VENV_DIR}/CHITUDIFFUSION_ENV" <<EOF
torch==2.10.0+cu130
transformers==5.12.1
flash-attn-4==4.0.0b25
nvidia-cutlass-dsl==4.6.0.dev0
quack-kernels==0.5.3
bundle=${BUNDLE_DIR}
source=${REPO_DIR}
installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "Installation complete: ${VENV_DIR}"
echo "Run: ${BUNDLE_DIR}/verify.sh ${VENV_DIR}"
