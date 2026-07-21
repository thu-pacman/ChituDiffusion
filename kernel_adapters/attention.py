"""Compatibility adapters for upstream attention-kernel providers.

FlashAttention 3 is supplied by its official interface or Hugging Face
``kernels``; this module normalizes their versioned public APIs without
vendoring either implementation.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
from functools import lru_cache
from types import ModuleType
from typing import Any

import torch


_HF_REPOSITORY = "kernels-community/flash-attn3"


def _official_available() -> bool:
    return importlib.util.find_spec("flash_attn_interface") is not None


def _hf_loader_available() -> bool:
    if importlib.util.find_spec("kernels") is None:
        return False
    try:
        module = importlib.import_module("kernels")
    except ImportError:
        return False
    return callable(getattr(module, "get_kernel", None))


def flash_attention3_available() -> bool:
    """Return whether an FA3 provider can be loaded.

    The Hugging Face provider may download a compiled kernel on first explicit use.
    Merely checking availability never performs network I/O.
    """

    provider = os.getenv("CHITU_FA3_PROVIDER", "auto").strip().lower()
    if provider == "official":
        return _official_available()
    if provider == "hf":
        return _hf_loader_available()
    return _official_available() or _hf_loader_available()


@lru_cache(maxsize=1)
def _load_flash_attention3() -> tuple[ModuleType, str]:
    provider = os.getenv("CHITU_FA3_PROVIDER", "auto").strip().lower()
    if provider not in {"auto", "official", "hf"}:
        raise ValueError("CHITU_FA3_PROVIDER must be one of: auto, official, hf")

    if provider in {"auto", "official"} and _official_available():
        return importlib.import_module("flash_attn_interface"), "official"

    if provider in {"auto", "hf"} and _hf_loader_available():
        kernels = importlib.import_module("kernels")
        return kernels.get_kernel(_HF_REPOSITORY), "hf"

    raise RuntimeError(
        "FlashAttention 3 is unavailable. Install the official Hopper interface "
        "or `pip install kernels` and select infer.attn_type=flash_attn3."
    )


def flash_attention3_provider() -> str | None:
    if not flash_attention3_available():
        return None
    try:
        return _load_flash_attention3()[1]
    except Exception:
        return None


def _supported_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in signature.parameters}


def _require_hopper(tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda":
        raise RuntimeError("FlashAttention 3 requires a CUDA tensor on Hopper (SM90).")
    major, minor = torch.cuda.get_device_capability(tensor.device)
    if (major, minor) != (9, 0):
        raise RuntimeError(f"FlashAttention 3 requires Hopper SM90; found SM{major}{minor}.")


def flash_attention3_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run FA3 while normalizing provider-specific return values."""

    _require_hopper(q)
    module, _ = _load_flash_attention3()
    use_varlen = cu_seqlens_q is not None
    function_name = "flash_attn_varlen_func" if use_varlen else "flash_attn_func"
    function = getattr(module, function_name, None)
    if function is None:
        raise NotImplementedError(f"The selected FlashAttention 3 provider has no {function_name}.")

    kwargs: dict[str, Any] = {
        "softmax_scale": softmax_scale,
        "causal": causal,
        "window_size": window_size,
    }
    if use_varlen:
        kwargs.update(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

    result = function(q, k, v, **_supported_kwargs(function, kwargs))
    if not isinstance(result, tuple):
        return result, None
    output = result[0]
    lse = result[1] if len(result) > 1 and isinstance(result[1], torch.Tensor) else None
    return output, lse if return_lse else None
