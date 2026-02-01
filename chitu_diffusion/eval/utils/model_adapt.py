import os
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Union
from contextlib import contextmanager

import torch
import torch.nn as nn

from vbench.utils import init_submodules, save_json
from .distributed import get_rank, print0


@contextmanager
def _vbench_fp32_guard():
    """
    Force default dtype to FP32 during VBench evaluation.
    This is crucial when some dims lazily build/load models inside compute_fn.
    """
    prev_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(prev_default)


def _force_fp32_recursive(obj, device: Optional[str] = None):
    """
    Recursively cast any nn.Module inside `obj` to fp32 (and optionally move to device).
    Supports nested dict/list/tuple structures.
    """
    if isinstance(obj, nn.Module):
        obj.eval()
        if device is None:
            obj.to(dtype=torch.float32)
        else:
            obj.to(device=device, dtype=torch.float32)
        return

    if isinstance(obj, dict):
        for v in obj.values():
            _force_fp32_recursive(v, device=device)
        return

    if isinstance(obj, (list, tuple)):
        for v in obj:
            _force_fp32_recursive(v, device=device)
        return

    return
