from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from chitu_diffusion.parallel.fast_cp._runtime import (
    FastUlyssesAllToAll,
    probe_fast_ulysses,
)
from chitu_diffusion.parallel.fast_cp.ulysses import (
    FastUlyssesSubgroupTransport,
    FastUlyssesTransport,
)
from chitu_diffusion.parallel.nccl.ulysses import TorchUlyssesTransport
from chitu_diffusion.parallel.ulysses_transport import (
    create_ulysses_transport,
    resolve_ulysses_transport,
)


def test_legacy_fast_package_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("chitu_diffusion.parallel.fast")


def test_fast_backend_supports_only_eager_uniform_4d(monkeypatch):
    backend = FastUlyssesAllToAll.__new__(FastUlyssesAllToAll)
    monkeypatch.setattr("torch.cuda.is_current_stream_capturing", lambda: False)
    tensor = SimpleNamespace(ndim=4, is_cuda=True, dtype=torch.bfloat16)

    assert backend.supports(
        tensor,
        2,
        1,
    )
    assert not backend.supports(
        SimpleNamespace(ndim=3, is_cuda=True, dtype=torch.bfloat16),
        1,
        0,
    )
    assert not backend.supports(
        SimpleNamespace(ndim=4, is_cuda=True, dtype=torch.uint8),
        2,
        1,
    )

    monkeypatch.setattr("torch.cuda.is_current_stream_capturing", lambda: True)
    assert not backend.supports(
        tensor,
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


def test_transport_names_are_explicit_and_environment_selectable(monkeypatch):
    assert resolve_ulysses_transport("fast") == "fast_ulysses"
    assert resolve_ulysses_transport("native") == "torch"
    monkeypatch.setenv("CHITU_ULYSSES_TRANSPORT", "auto")
    assert resolve_ulysses_transport() == "auto"

    with pytest.raises(ValueError, match="must be one of"):
        resolve_ulysses_transport("unknown")


def test_fast_transport_requires_static_full_world_topology():
    with pytest.raises(ValueError, match="static full-world"):
        create_ulysses_transport(
            "fast_ulysses",
            process_group=object(),
            device=torch.device("cpu"),
            static_full_world=False,
        )

    transport = create_ulysses_transport(
        "auto",
        process_group=object(),
        device=torch.device("cpu"),
        static_full_world=False,
    )
    assert isinstance(transport, TorchUlyssesTransport)


def test_ulysses_routes_all_to_all_through_selected_transport():
    calls = []

    class _Transport:
        def all_to_all(self, tensor, scatter_dim, gather_dim):
            calls.append((tensor, scatter_dim, gather_dim))
            return tensor + 1

    tensor = torch.zeros(1, 4, 4, 8)
    output = _Transport().all_to_all(tensor, 2, 1)

    assert calls == [(tensor, 2, 1)]
    torch.testing.assert_close(output, tensor + 1)


def test_fast_subgroup_transport_uses_shared_runtime(monkeypatch):
    calls = []

    class _Backend:
        @staticmethod
        def mode(scatter_dim, gather_dim):
            return 0 if (scatter_dim, gather_dim) == (2, 1) else 1

        def logical_barrier(self, **kwargs):
            calls.append(("barrier", kwargs))

        def logical_all_to_all(self, tensor, **kwargs):
            calls.append(("a2a", kwargs))
            return tensor + 1

    class _Runtime:
        pool_bytes = 4096

        def backend(self, tensor):
            return _Backend()

    class _Fallback:
        def all_to_all(self, tensor, scatter_dim, gather_dim):
            return tensor + 2

    monkeypatch.setattr(
        FastUlyssesAllToAll,
        "supports",
        staticmethod(
            lambda tensor, scatter_dim, gather_dim: tensor.dtype == torch.bfloat16
        ),
    )
    transport = FastUlyssesSubgroupTransport(
        _Runtime(),
        peer_ranks=(0, 2, 4, 6),
        fallback=_Fallback(),
    )
    transport.reset()
    tensor = torch.zeros(1, 2, 4, 8, dtype=torch.bfloat16)
    torch.testing.assert_close(transport.all_to_all(tensor, 2, 1), tensor + 1)
    torch.testing.assert_close(transport.all_to_all(tensor, 1, 2), tensor + 1)
    assert [call[0] for call in calls] == ["a2a", "barrier", "a2a", "barrier"]
    assert calls[0][1]["peer_ranks"] == (0, 2, 4, 6)
    assert calls[0][1]["mode"] == 0
    assert calls[2][1]["mode"] == 1
    assert transport.pool_bytes == 4096


def test_fast_transport_falls_back_per_unsupported_operation(monkeypatch):
    class _Backend:
        pool_bytes = 4096

        def supports(self, tensor, scatter_dim, gather_dim):
            return tensor.dtype == torch.bfloat16

        def all_to_all(self, tensor, scatter_dim, gather_dim):
            return tensor + 1

        def reset_sequence(self):
            return None

        def destroy(self):
            return None

    class _Fallback:
        name = "fallback"
        pool_bytes = 0

        def reset(self):
            return None

        def all_to_all(self, tensor, scatter_dim, gather_dim):
            return tensor + 2

        def close(self):
            return None

    transport = FastUlyssesTransport.__new__(FastUlyssesTransport)
    transport._backend = _Backend()
    transport._fallback = _Fallback()
    transport._closed = False

    bf16 = torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16)
    fp32 = bf16.float()
    monkeypatch.setattr(
        FastUlyssesAllToAll,
        "supports",
        staticmethod(
            lambda tensor, scatter_dim, gather_dim: tensor.dtype == torch.bfloat16
        ),
    )
    torch.testing.assert_close(transport.all_to_all(bf16, 2, 1), bf16 + 1)
    torch.testing.assert_close(transport.all_to_all(fp32, 2, 1), fp32 + 2)
    assert transport.pool_bytes == 4096


def test_fast_transport_sizes_symmetric_pool_from_first_attention_shape(
    monkeypatch,
):
    monkeypatch.delenv("CHITU_FAST_ULYSSES_POOL_BYTES", raising=False)
    flux = torch.empty(1, 1152, 24, 128, dtype=torch.bfloat16)
    wan = torch.empty(1, 8320, 40, 128, dtype=torch.bfloat16)

    assert FastUlyssesTransport._pool_bytes_for(flux) == 64 << 20
    assert FastUlyssesTransport._pool_bytes_for(wan) == 384 << 20

    monkeypatch.setenv("CHITU_FAST_ULYSSES_POOL_BYTES", str(768 << 20))
    assert FastUlyssesTransport._pool_bytes_for(flux) == 768 << 20
