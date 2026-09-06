from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from chitu_diffusion.parallel.cp import resolve_agkv_transport
from chitu_diffusion.parallel.cp.fast import FastAgkvTransport
from chitu_diffusion.parallel.cp.fast._runtime import FastUlyssesAllToAll
from chitu_diffusion.parallel.interconnect import InterconnectProfile


def test_fast_agkv_alignment_gate_uses_complete_bshd_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("torch.cuda.is_current_stream_capturing", lambda: False)

    def tensor(heads: int):
        return SimpleNamespace(
            ndim=4,
            shape=(1, 8, heads, 12),
            is_cuda=True,
            dtype=torch.bfloat16,
            element_size=lambda: 2,
        )

    key = tensor(2)
    assert FastUlyssesAllToAll.supports_agkv(key, tensor(2))
    assert not FastUlyssesAllToAll.supports_agkv(tensor(1), tensor(1))


def _fabric(monkeypatch: pytest.MonkeyPatch, nvlink: bool | None) -> None:
    monkeypatch.setattr(
        "chitu_diffusion.parallel.cp.fast.agkv_transport.local_interconnect",
        lambda: InterconnectProfile(nvlink=nvlink, numa_nodes=(0, 0, 1, 1)),
    )


def test_fast_agkv_use_ce_follows_the_fabric(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FastAgkvTransport.__new__(FastAgkvTransport)
    monkeypatch.delenv("CHITU_FAST_AGKV_USE_CE", raising=False)

    _fabric(monkeypatch, nvlink=False)
    assert transport.use_ce
    # NVLink fills a link per peer, and an unreadable fabric keeps the SM path.
    _fabric(monkeypatch, nvlink=True)
    assert not transport.use_ce
    _fabric(monkeypatch, nvlink=None)
    assert not transport.use_ce


def test_fast_agkv_use_ce_override(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FastAgkvTransport.__new__(FastAgkvTransport)
    _fabric(monkeypatch, nvlink=True)

    monkeypatch.setenv("CHITU_FAST_AGKV_USE_CE", "on")
    assert transport.use_ce
    monkeypatch.setenv("CHITU_FAST_AGKV_USE_CE", "off")
    assert not transport.use_ce
    monkeypatch.setenv("CHITU_FAST_AGKV_USE_CE", "invalid")
    with pytest.raises(ValueError, match="CHITU_FAST_AGKV_USE_CE"):
        _ = transport.use_ce


def test_fast_agkv_async_auto_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FastAgkvTransport.__new__(FastAgkvTransport)
    transport._process_group = object()
    monkeypatch.delenv("CHITU_FAST_AGKV_ASYNC", raising=False)
    monkeypatch.delenv("CHITU_FAST_AGKV_USE_CE", raising=False)
    _fabric(monkeypatch, nvlink=True)

    monkeypatch.setattr(dist, "get_world_size", lambda group: 4)
    assert transport.async_enabled
    monkeypatch.setattr(dist, "get_world_size", lambda group: 8)
    assert not transport.async_enabled
    # Copy-engine transfers take no SMs from the attention they overlap with, so
    # the width gate that protects the SM kernel does not apply to them.
    _fabric(monkeypatch, nvlink=False)
    assert transport.async_enabled


def test_fast_agkv_async_override(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FastAgkvTransport.__new__(FastAgkvTransport)
    transport._process_group = object()
    monkeypatch.setattr(dist, "get_world_size", lambda group: 8)

    monkeypatch.setenv("CHITU_FAST_AGKV_ASYNC", "on")
    assert transport.async_enabled
    monkeypatch.setenv("CHITU_FAST_AGKV_ASYNC", "off")
    assert not transport.async_enabled
    monkeypatch.setenv("CHITU_FAST_AGKV_ASYNC", "invalid")
    with pytest.raises(ValueError, match="CHITU_FAST_AGKV_ASYNC"):
        _ = transport.async_enabled


def test_resolve_agkv_transport_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHITU_AGKV_TRANSPORT", raising=False)
    assert resolve_agkv_transport() == "torch"
    assert resolve_agkv_transport("fast") == "fast_agkv"
    assert resolve_agkv_transport("nccl") == "torch"
