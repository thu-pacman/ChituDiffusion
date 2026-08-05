from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from kernel_adapters.fast_ulysses import FastUlyssesAllToAll, probe_fast_ulysses


def test_fast_backend_supports_only_eager_uniform_4d(monkeypatch):
    backend = FastUlyssesAllToAll.__new__(FastUlyssesAllToAll)
    monkeypatch.setattr("torch.cuda.is_current_stream_capturing", lambda: False)

    assert backend.supports(
        torch.empty(1, 8, 4, 64, dtype=torch.bfloat16),
        2,
        1,
    )
    assert not backend.supports(
        torch.empty(8, 4, 64, dtype=torch.bfloat16),
        1,
        0,
    )
    assert not backend.supports(
        torch.empty(1, 8, 4, 64, dtype=torch.uint8),
        2,
        1,
    )

    monkeypatch.setattr("torch.cuda.is_current_stream_capturing", lambda: True)
    assert not backend.supports(
        torch.empty(1, 8, 4, 64, dtype=torch.bfloat16),
        2,
        1,
    )


def test_fast_backend_auto_probe_requires_hopper_and_subgroup_feature(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr("torch.cuda.device_count", lambda: 8)
    monkeypatch.setattr("torch.cuda.can_device_access_peer", lambda src, dst: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setitem(
        sys.modules,
        "fast_ulysses",
        SimpleNamespace(
            UlyssesGroup=object,
            SUPPORTS_WORLD_PARTITION_SUBGROUPS=True,
        ),
    )

    ready, reason = probe_fast_ulysses(
        torch.device("cuda", 0),
        require_hopper=True,
        require_subgroups=True,
    )

    assert ready
    assert "SM90" in reason


def test_fast_backend_auto_probe_rejects_pre_hopper(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device: (8, 0))

    ready, reason = probe_fast_ulysses(
        torch.device("cuda", 0),
        require_hopper=True,
        require_subgroups=False,
    )

    assert not ready
    assert "older than Hopper" in reason


def test_fast_backend_auto_probe_rejects_upstream_package_for_subgroup(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr("torch.cuda.device_count", lambda: 8)
    monkeypatch.setattr("torch.cuda.can_device_access_peer", lambda src, dst: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setitem(
        sys.modules,
        "fast_ulysses",
        SimpleNamespace(UlyssesGroup=object),
    )

    ready, reason = probe_fast_ulysses(
        torch.device("cuda", 0),
        require_hopper=True,
        require_subgroups=True,
    )

    assert not ready
    assert "lacks CFP subgroup support" in reason
