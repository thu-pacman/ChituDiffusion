from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from chitu_diffusion.core.distributed import parallel_state
from chitu_diffusion.core.distributed.comm_group import CommGroup
from kernel_adapters.fast_ulysses import FastUlyssesAllToAll, probe_fast_ulysses


class _FakeGroup:
    def __init__(self, group_size: int, *, fast_ulysses_enabled: bool = False):
        self.group_size = group_size
        self.fast_ulysses_enabled = fast_ulysses_enabled
        self.reset_calls = 0

    def reset_all_to_all_sequence(self) -> None:
        self.reset_calls += 1


def test_get_ulysses_group_uses_regular_up_group_without_fast_backend(monkeypatch):
    cp_group = _FakeGroup(4)
    up_group = _FakeGroup(2)
    monkeypatch.setattr(parallel_state, "_CP_GROUP", cp_group)
    monkeypatch.setattr(parallel_state, "_UP_GROUP_DICT", {2: up_group})

    assert parallel_state.get_ulysses_group(2) is up_group
    assert cp_group.reset_calls == 0


def test_get_ulysses_group_reuses_and_resets_fast_cp_group(monkeypatch):
    cp_group = _FakeGroup(4, fast_ulysses_enabled=True)
    duplicate_up_group = _FakeGroup(4)
    monkeypatch.setattr(parallel_state, "_CP_GROUP", cp_group)
    monkeypatch.setattr(parallel_state, "_UP_GROUP_DICT", {4: duplicate_up_group})

    assert parallel_state.get_ulysses_group(4) is cp_group
    assert parallel_state.get_ulysses_group(4) is cp_group
    assert cp_group.reset_calls == 2


def test_get_ulysses_group_keeps_nccl_up_subgroup_for_hybrid_mode(monkeypatch):
    cp_group = _FakeGroup(8, fast_ulysses_enabled=True)
    up_group = _FakeGroup(4)
    monkeypatch.setattr(parallel_state, "_CP_GROUP", cp_group)
    monkeypatch.setattr(parallel_state, "_UP_GROUP_DICT", {4: up_group})

    assert parallel_state.get_ulysses_group(4) is up_group
    assert cp_group.reset_calls == 0


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


def test_comm_group_falls_back_when_fast_backend_does_not_support_call():
    group = CommGroup.__new__(CommGroup)
    group.group_size = 2
    group._fast_ulysses = type(
        "_UnsupportedFastBackend",
        (),
        {"supports": lambda self, tensor, scatter_dim, gather_dim: False},
    )()
    group._supports_all_to_all_single = lambda *args, **kwargs: True
    sentinel = torch.empty(0)
    group._all_to_all_single = lambda *args, **kwargs: sentinel

    actual = group.all_to_all(torch.empty(1, 8, 4, 64), 2, 1)

    assert actual is sentinel


def _distributed_correctness_check() -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    from chitu_diffusion.modules.attention.diffusion_attn_backend import (
        DiffusionAttention_with_CP,
        DiffusionAttnBackend,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cfg_size = int(os.environ.get("TEST_CFG_SIZE", "1"))
    cp_size = world_size // cfg_size
    parallel_state.initialize_diffusion_parallel_groups(
        cfg_size=cfg_size,
        cp_size=cp_size,
        up=cp_size,
    )

    generator = torch.Generator(device=f"cuda:{local_rank}").manual_seed(
        20260726 + rank
    )
    cp_group = parallel_state.get_cp_group()
    # Give CFP branches different allocation sizes. This catches accidental
    # WORLD-collective pool growth or shared bump-offset assumptions.
    branch = rank // cp_size
    shape = (1, 128 + 64 * branch, 8, 64)
    q = torch.randn(
        shape, device=f"cuda:{local_rank}", dtype=torch.bfloat16, generator=generator
    )
    k = torch.randn(
        shape, device=f"cuda:{local_rank}", dtype=torch.bfloat16, generator=generator
    )
    v = torch.randn(
        shape, device=f"cuda:{local_rank}", dtype=torch.bfloat16, generator=generator
    )

    def gather_sequence(tensor):
        pieces = [torch.empty_like(tensor) for _ in range(cp_size)]
        dist.all_gather(pieces, tensor, group=cp_group.gpu_group)
        return torch.cat(pieces, dim=1)

    full_q = gather_sequence(q)
    full_k = gather_sequence(k)
    full_v = gather_sequence(v)
    reference = F.scaled_dot_product_attention(
        full_q.transpose(1, 2),
        full_k.transpose(1, 2),
        full_v.transpose(1, 2),
    ).transpose(1, 2)
    local_cp_rank = cp_group.rank_in_group
    reference = reference[:, local_cp_rank * shape[1] : (local_cp_rank + 1) * shape[1]]

    attention = DiffusionAttention_with_CP(
        DiffusionAttnBackend("torch_sdpa"),
        ulysses_limit=cp_size,
        cp_backend="ucp",
    )
    actual = attention(q, k, v)[0]
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)

    parallel_state.destroy_diffusion_kernel_backends()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    _distributed_correctness_check()
