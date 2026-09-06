"""Load the reviewed CuTe attention without requiring the unrelated FA2 binding."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from collections.abc import Callable

import torch


def load_cute_flash_attn_fwd() -> Callable[..., tuple[torch.Tensor, ...]]:
    try:
        from flash_attn.cute.interface import _flash_attn_fwd

        return _flash_attn_fwd
    except ModuleNotFoundError as exc:
        if exc.name != "flash_attn_2_cuda":
            raise

    # The CuTe implementation is pure Python/DSL, but upstream package import
    # eagerly loads the separate FlashAttention-2 extension. Build a namespace
    # package over the same source directory when only that optional binding is
    # absent, then import the CuTe submodule normally.
    for name in tuple(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            sys.modules.pop(name, None)
    spec = importlib.machinery.PathFinder.find_spec("flash_attn", sys.path)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("the reviewed FlashAttention source package is unavailable")
    package = types.ModuleType("flash_attn")
    package.__path__ = list(spec.submodule_search_locations)
    package.__package__ = "flash_attn"
    package.__spec__ = spec
    sys.modules["flash_attn"] = package
    return importlib.import_module("flash_attn.cute.interface")._flash_attn_fwd
