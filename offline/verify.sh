#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-${CHITU_VENV_DIR:-/dockerdata/chitudiffusion-venv}}"
PYTHON="${VENV_DIR}/bin/python"

[[ -x "${PYTHON}" ]] || {
  echo "ERROR: Python environment not found: ${VENV_DIR}" >&2
  exit 1
}

"${PYTHON}" - <<'PY'
from importlib import import_module, metadata, util

expected = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "torchaudio": "2.10.0",
    "triton": "3.6.0",
    "diffusers": "0.38.0",
    "transformers": "5.12.1",
    "accelerate": "1.13.0",
    "numpy": "2.2.6",
    "scipy": "1.17.1",
}

for name, version in expected.items():
    module = import_module(name)
    actual = getattr(module, "__version__", "unknown")
    if not actual.startswith(version):
        raise RuntimeError(f"{name}: expected {version}, found {actual}")
    print(f"{name}: {actual}")

expected_distributions = {
    "flash-attn-4": "4.0.0b25",
    "nvidia-cutlass-dsl": "4.6.0.dev0",
    "quack-kernels": "0.5.3",
    "cuda-python": "13.0.3",
    "cuda-bindings": "13.0.3",
}
for name, version in expected_distributions.items():
    actual = metadata.version(name)
    if actual != version:
        raise RuntimeError(f"{name}: expected {version}, found {actual}")
    print(f"{name}: {actual}")

import_module("chitu_diffusion")
print(f"chitu-diffusion: {metadata.version('chitu-diffusion')}")

import torch

if torch.version.cuda != "13.0":
    raise RuntimeError(f"Expected Torch CUDA 13.0, found {torch.version.cuda}")

print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    if capability not in {(9, 0), (12, 0)}:
        raise RuntimeError(f"Unsupported compute capability: {capability}")

    from flash_attn.cute.interface import flash_attn_varlen_func

    q = torch.randn(64, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu = torch.tensor([0, 48, 64], device="cuda", dtype=torch.int32)
    result = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=48,
        max_seqlen_k=48,
        causal=False,
    )
    out = result[0] if isinstance(result, tuple) else result
    if out.shape != q.shape or not torch.isfinite(out).all():
        raise RuntimeError("FlashAttention 4 varlen smoke test failed")
    print(f"FA4 varlen smoke test passed: {tuple(out.shape)}")
else:
    if util.find_spec("flash_attn") is None:
        raise RuntimeError("flash_attn: module not installed")
    print(
        f"flash-attn-4: {metadata.version('flash-attn-4')} "
        "(installed; GPU kernel test skipped)"
    )
    print(
        "GPU runtime checks skipped. Re-run verify.sh on the target SM90/SM120 machine."
    )
PY

"${PYTHON}" -m pip check
echo "Offline dependency check: passed"
