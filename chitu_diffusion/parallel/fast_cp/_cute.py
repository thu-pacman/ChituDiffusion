"""Load the reviewed CuTe attention without requiring the unrelated FA2 binding."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CapturedCuteLaunch:
    """One already-specialized CuTe runner and its TVM-FFI arguments."""

    runner: Callable[..., object]
    args: tuple[object, ...]


class _CaptureRunner:
    def __init__(self, runner: Callable[..., object]) -> None:
        self.runner = runner
        self.args: tuple[object, ...] | None = None

    def __call__(self, *args: object) -> object:
        self.args = args
        return self.runner(*args)


def capture_cute_launch(
    function: Callable[..., Any],
    call: Callable[[], Any],
) -> tuple[Any, CapturedCuteLaunch | None]:
    """Run ``call`` and capture the compiled runner it dispatches.

    This is deliberately done only after a normal warmup has populated CuTe's
    compile cache. Restoring the cache in ``finally`` leaves upstream behavior
    untouched if capture is unavailable or the call raises.
    """

    cache = getattr(function, "compile_cache", None)
    if cache is None:
        return call(), None
    # FlashAttention wraps its dictionary in JITCache. Mutate the in-memory
    # mapping directly so a persistent cache does not try to serialize proxies.
    mapping = getattr(cache, "cache", cache)
    original = list(mapping.items())
    if not original:
        return call(), None
    proxies = [(key, _CaptureRunner(runner)) for key, runner in original]
    try:
        for key, proxy in proxies:
            mapping[key] = proxy
        result = call()
    finally:
        for key, runner in original:
            mapping[key] = runner
    called = [proxy for _, proxy in proxies if proxy.args is not None]
    if len(called) != 1:
        return result, None
    proxy = called[0]
    assert proxy.args is not None
    return result, CapturedCuteLaunch(proxy.runner, proxy.args)


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
