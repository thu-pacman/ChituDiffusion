# SPDX-License-Identifier: Apache-2.0
"""Minimal CP-attention adapters extracted from reviewed upstream projects.

Provenance:
- SGLang 1ebd6fab6c6cf8910bd3f8a13c69d5c6720a6703
  - python/sglang/kernels/ops/diffusion/triton/ulysses_qkv.py
  - python/sglang/multimodal_gen/runtime/layers/usp.py
  - python/sglang/multimodal_gen/runtime/distributed/device_communicators/ipc_a2a.py
- vLLM-Omni 9159cedbd7ce71e8d29899867e68524ead6c03bc
  - vllm_omni/diffusion/distributed/comm.py
  - vllm_omni/diffusion/attention/parallel/{ulysses,allgather_kv}.py
  - vllm_omni/diffusion/attention/backends/ring_pytorch_attn.py

These adapters intentionally expose only dense, non-causal BSHD attention for
benchmarking. They do not import either upstream runtime or silently fall back.
"""

from __future__ import annotations

import ctypes
import os
import socket
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl

AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

SGLANG_COMMIT = "1ebd6fab6c6cf8910bd3f8a13c69d5c6720a6703"
VLLM_OMNI_COMMIT = "9159cedbd7ce71e8d29899867e68524ead6c03bc"


@triton.jit
def _pack_qkv_destination_major_kernel(
    output_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    total_elements,
    rows,
    local_heads,
    head_size,
    stride_q_row,
    stride_q_head,
    stride_k_row,
    stride_k_head,
    stride_v_row,
    stride_v_head,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    dim = offsets % head_size
    head_slot = offsets // head_size
    local_head = head_slot % local_heads
    row_slot = head_slot // local_heads
    row = row_slot % rows
    destination = row_slot // rows
    global_head = destination * local_heads + local_head

    q = tl.load(
        q_ptr + row * stride_q_row + global_head * stride_q_head + dim,
        mask=mask,
    )
    k = tl.load(
        k_ptr + row * stride_k_row + global_head * stride_k_head + dim,
        mask=mask,
    )
    v = tl.load(
        v_ptr + row * stride_v_row + global_head * stride_v_head + dim,
        mask=mask,
    )
    output_base = head_slot * (3 * head_size) + dim
    tl.store(output_ptr + output_base, q, mask=mask)
    tl.store(output_ptr + output_base + head_size, k, mask=mask)
    tl.store(output_ptr + output_base + 2 * head_size, v, mask=mask)


@triton.jit
def _merge_usp_heads_kernel(
    output_ptr,
    input_ptr,
    total_elements,
    sequence,
    batch,
    local_heads,
    head_dim,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    rest = offsets
    dim = rest % head_dim
    rest //= head_dim
    head = rest % local_heads
    rest //= local_heads
    rank = rest % world_size
    rest //= world_size
    seq = rest % sequence
    batch_index = rest // sequence
    source = (
        (((rank * sequence + seq) * batch + batch_index) * local_heads) + head
    ) * head_dim + dim
    value = tl.load(input_ptr + source, mask=mask)
    tl.store(output_ptr + offsets, value, mask=mask)


def _pack_qkv_destination_major(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    world_size: int,
    output: torch.Tensor,
) -> torch.Tensor:
    rows, global_heads, head_dim = query.shape
    local_heads = global_heads // world_size
    total_elements = rows * global_heads * head_dim
    block_size = 1024
    _pack_qkv_destination_major_kernel[(triton.cdiv(total_elements, block_size),)](
        output,
        query,
        key,
        value,
        total_elements,
        rows,
        local_heads,
        head_dim,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output


def _merge_usp_heads(received: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    world_size, sequence, batch, local_heads, head_dim = received.shape
    total_elements = received.numel()
    block_size = 256
    _merge_usp_heads_kernel[(triton.cdiv(total_elements, block_size),)](
        output,
        received,
        total_elements,
        sequence,
        batch,
        local_heads,
        head_dim,
        world_size=world_size,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output


class SglangPackedUlyssesAttention:
    """SGLang packed-QKV NCCL Ulysses path with reusable staging buffers."""

    def __init__(self, process_group: object) -> None:
        self.group = process_group
        self.world_size = dist.get_world_size(process_group)
        self._buffers: dict[tuple, torch.Tensor] = {}

    def _buffer(
        self,
        role: str,
        shape: tuple[int, ...],
        like: torch.Tensor,
    ) -> torch.Tensor:
        key = (role, shape, like.dtype, like.device)
        output = self._buffers.get(key)
        if output is None:
            output = torch.empty(shape, dtype=like.dtype, device=like.device)
            self._buffers[key] = output
        return output

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention: AttentionFn,
    ) -> torch.Tensor:
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError("SGLang packed Ulysses requires matching Q/K/V shapes")
        batch, local_sequence, heads, head_dim = query.shape
        if heads % self.world_size:
            raise ValueError("head count must be divisible by the Ulysses degree")
        local_heads = heads // self.world_size
        rows = batch * local_sequence
        packed_shape = (self.world_size, rows, local_heads, 3 * head_dim)
        packed = _pack_qkv_destination_major(
            query.view(rows, heads, head_dim),
            key.view(rows, heads, head_dim),
            value.view(rows, heads, head_dim),
            self.world_size,
            self._buffer("packed_send", packed_shape, query),
        )
        received = self._buffer("packed_recv", packed_shape, query)
        dist.all_to_all_single(received, packed, group=self.group)
        if batch == 1:
            received_qkv = received.view(
                1,
                self.world_size * local_sequence,
                local_heads,
                3 * head_dim,
            )
        else:
            received_qkv = (
                received.view(
                    self.world_size,
                    batch,
                    local_sequence,
                    local_heads,
                    3 * head_dim,
                )
                .permute(1, 0, 2, 3, 4)
                .contiguous()
                .view(
                    batch,
                    self.world_size * local_sequence,
                    local_heads,
                    3 * head_dim,
                )
            )
        exchanged = tuple(
            part.contiguous() for part in received_qkv.split(head_dim, dim=-1)
        )
        local_output = attention(*exchanged)

        send_output = local_output.permute(1, 0, 2, 3).contiguous()
        received_output = self._buffer("output_recv", tuple(send_output.shape), query)
        dist.all_to_all_single(received_output, send_output, group=self.group)
        received_output = received_output.view(
            self.world_size,
            local_sequence,
            batch,
            local_heads,
            head_dim,
        )
        merged_shape = (
            batch,
            local_sequence,
            self.world_size,
            local_heads,
            head_dim,
        )
        merged = self._buffer("output_merged", merged_shape, query)
        return _merge_usp_heads(received_output, merged).view(
            batch,
            local_sequence,
            heads,
            head_dim,
        )


def _omni_all_to_all_4d(
    tensor: torch.Tensor,
    process_group: object,
    scatter_idx: int,
    gather_idx: int,
) -> torch.Tensor:
    world_size = dist.get_world_size(process_group)
    if scatter_idx == 2 and gather_idx == 1:
        batch, local_sequence, heads, head_dim = tensor.shape
        local_heads = heads // world_size
        send = (
            tensor.reshape(batch, local_sequence, world_size, local_heads, head_dim)
            .transpose(0, 2)
            .contiguous()
        )
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=process_group)
        return (
            received.reshape(
                local_sequence * world_size,
                batch,
                local_heads,
                head_dim,
            )
            .transpose(0, 1)
            .contiguous()
        )
    if scatter_idx == 1 and gather_idx == 2:
        batch, global_sequence, local_heads, head_dim = tensor.shape
        local_sequence = global_sequence // world_size
        send = (
            tensor.reshape(
                batch,
                world_size,
                local_sequence,
                local_heads,
                head_dim,
            )
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(
                world_size,
                local_heads,
                local_sequence,
                batch,
                head_dim,
            )
        )
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send, group=process_group)
        heads = local_heads * world_size
        return (
            received.reshape(heads, local_sequence, batch, head_dim)
            .transpose(0, 2)
            .contiguous()
            .reshape(batch, local_sequence, heads, head_dim)
        )
    raise ValueError("unsupported Omni Ulysses scatter/gather dimensions")


def omni_ulysses_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    process_group: object,
    attention: AttentionFn,
) -> torch.Tensor:
    """vLLM-Omni strict Ulysses: three input A2As and one output A2A."""
    exchanged = tuple(
        _omni_all_to_all_4d(tensor, process_group, 2, 1)
        for tensor in (query, key, value)
    )
    output = attention(*exchanged)
    return _omni_all_to_all_4d(output, process_group, 1, 2)


def _omni_all_gather_sequence(
    tensor: torch.Tensor,
    process_group: object,
) -> torch.Tensor:
    world_size = dist.get_world_size(process_group)
    batch, local_sequence, heads, head_dim = tensor.shape
    gathered = torch.empty(
        world_size * batch,
        local_sequence,
        heads,
        head_dim,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(gathered, tensor.contiguous(), group=process_group)
    return (
        gathered.view(world_size, batch, local_sequence, heads, head_dim)
        .permute(1, 0, 2, 3, 4)
        .reshape(batch, world_size * local_sequence, heads, head_dim)
        .contiguous()
    )


def omni_allgather_kv_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    process_group: object,
    attention: AttentionFn,
) -> torch.Tensor:
    """vLLM-Omni AllGather-KV strategy for dense non-causal attention."""
    full_key = _omni_all_gather_sequence(key, process_group)
    full_value = _omni_all_gather_sequence(value, process_group)
    return attention(query, full_key, full_value)


class _OmniRingComm:
    def __init__(self, process_group: object) -> None:
        self.group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.send_rank = dist.get_global_rank(
            process_group, (self.rank + 1) % self.world_size
        )
        self.recv_rank = dist.get_global_rank(
            process_group, (self.rank - 1) % self.world_size
        )
        self._ops: list[dist.P2POp] = []
        self._requests = None

    def send_recv(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.contiguous()
        received = torch.empty_like(tensor)
        self._ops.extend(
            (
                dist.P2POp(dist.isend, tensor, self.send_rank, group=self.group),
                dist.P2POp(dist.irecv, received, self.recv_rank, group=self.group),
            )
        )
        return received

    def commit(self) -> None:
        self._requests = dist.batch_isend_irecv(self._ops)

    def wait(self) -> None:
        for request in self._requests:
            request.wait()
        self._requests = None
        self._ops = []


def _omni_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, lse = torch.ops.aten._scaled_dot_product_efficient_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_bias=None,
        compute_log_sumexp=True,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )[:2]
    return output.transpose(1, 2), lse.float()


def _omni_merge_attention(
    output: torch.Tensor | None,
    output_lse: torch.Tensor | None,
    block_output: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_output = block_output.float()
    _, sequence, heads, _ = block_output.shape
    if block_lse.ndim == 3:
        if block_lse.shape[1] == heads and block_lse.shape[2] >= sequence:
            block_lse = block_lse[:, :, :sequence].transpose(1, 2).unsqueeze(-1)
        elif block_lse.shape[1] >= sequence and block_lse.shape[2] == heads:
            block_lse = block_lse[:, :sequence, :].unsqueeze(-1)
        else:
            raise RuntimeError(
                "unsupported vLLM-Omni Ring LSE shape: "
                f"output={tuple(block_output.shape)}, lse={tuple(block_lse.shape)}"
            )
    elif block_lse.ndim == 4:
        if block_lse.shape[1] == heads and block_lse.shape[2] >= sequence:
            block_lse = block_lse[:, :, :sequence, :].transpose(1, 2)
        elif block_lse.shape[1] >= sequence and block_lse.shape[2] == heads:
            block_lse = block_lse[:, :sequence, :, :]
        else:
            raise RuntimeError(
                "unsupported vLLM-Omni Ring LSE shape: "
                f"output={tuple(block_output.shape)}, lse={tuple(block_lse.shape)}"
            )
    else:
        raise RuntimeError(f"vLLM-Omni Ring expected 3D/4D LSE, got {block_lse.ndim}D")
    if output is None or output_lse is None:
        return block_output, block_lse
    output = output - F.sigmoid(block_lse - output_lse) * (output - block_output)
    output_lse = output_lse - F.logsigmoid(output_lse - block_lse)
    return output, output_lse


class _OmniRingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, process_group, query, key, value, scale):
        del ctx
        comm = _OmniRingComm(process_group)
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        output = output_lse = None
        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_key = comm.send_recv(key)
                next_value = comm.send_recv(value)
                comm.commit()
            block_output, block_lse = _omni_efficient_attention(
                query,
                key,
                value,
                scale,
            )
            output, output_lse = _omni_merge_attention(
                output,
                output_lse,
                block_output,
                block_lse,
            )
            if step + 1 != comm.world_size:
                comm.wait()
                key = next_key
                value = next_value
        return output.to(query.dtype)


def omni_ring_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    process_group: object,
) -> torch.Tensor:
    """vLLM-Omni PyTorch efficient-attention Ring path."""
    return _OmniRingAttention.apply(
        process_group,
        query,
        key,
        value,
        query.shape[-1] ** -0.5,
    )


_IPC_SYNC_DECL = (
    "void spin_wait(torch::Tensor flag, torch::Tensor target, "
    "torch::Tensor timed_out, torch::Tensor peer_timed_out, int64_t budget_ns);\n"
    "void bump_signal(torch::Tensor seq, torch::Tensor peer_flag);"
)

_IPC_SYNC_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__device__ __forceinline__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__global__ void spin_wait_kernel(volatile int* flag, const int* target,
                                 int* timed_out, int* peer_timed_out,
                                 unsigned long long budget_ns) {
    int t = *target;
    unsigned long long start = now_ns();
    while (*flag < t) {
        if (now_ns() - start > budget_ns) {
            *timed_out = 1;
            *peer_timed_out = 1;
            __threadfence_system();
            return;
        }
    }
    __threadfence_system();
}
__global__ void bump_signal_kernel(int* seq, volatile int* peer_flag) {
    int v = *seq + 1;
    *seq = v;
    __threadfence_system();
    *peer_flag = v;
}
void spin_wait(torch::Tensor flag, torch::Tensor target, torch::Tensor timed_out,
               torch::Tensor peer_timed_out, int64_t budget_ns) {
    spin_wait_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        (volatile int*)flag.data_ptr<int>(), target.data_ptr<int>(),
        timed_out.data_ptr<int>(), peer_timed_out.data_ptr<int>(),
        (unsigned long long)budget_ns);
}
void bump_signal(torch::Tensor seq, torch::Tensor peer_flag) {
    bump_signal_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        seq.data_ptr<int>(), (volatile int*)peer_flag.data_ptr<int>());
}
"""


class SglangIpcA2AState:
    """Two-rank CUDA-IPC state extracted from SGLang's diffusion USP path."""

    def __init__(self, process_group: object) -> None:
        if dist.get_world_size(process_group) != 2:
            raise ValueError("SGLang CUDA-IPC Ulysses requires exactly two ranks")
        self.group = process_group
        self.rank = dist.get_rank(process_group)
        self.calls = 0
        self.staging: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = (
            OrderedDict()
        )
        self._initialize()

    def _share(self, tensor: torch.Tensor) -> torch.Tensor:
        from torch.multiprocessing.reductions import reduce_tensor

        function, args = reduce_tensor(tensor)
        mine = [(function, args)]
        peer = [None]
        rank0 = dist.get_global_rank(self.group, 0)
        rank1 = dist.get_global_rank(self.group, 1)
        if self.rank == 0:
            dist.broadcast_object_list(mine, src=rank0, group=self.group)
            dist.broadcast_object_list(peer, src=rank1, group=self.group)
        else:
            dist.broadcast_object_list(peer, src=rank0, group=self.group)
            dist.broadcast_object_list(mine, src=rank1, group=self.group)
        peer_function, peer_args = peer[0]
        peer_args = list(peer_args)
        device = torch.cuda.current_device()
        for index, value in enumerate(peer_args):
            if isinstance(value, torch.device):
                peer_args[index] = torch.device("cuda", device)
            elif isinstance(value, int) and index == 6:
                peer_args[index] = device
        return peer_function(*peer_args)

    def _initialize(self) -> None:
        members: list[tuple[str, int] | None] = [None, None]
        device = torch.cuda.current_device()
        dist.all_gather_object(
            members,
            (socket.gethostname(), device),
            group=self.group,
        )
        peer = members[1 - self.rank]
        if peer is None or peer[0] != socket.gethostname():
            raise RuntimeError("SGLang CUDA-IPC requires both ranks on one host")
        peer_device = peer[1]
        if not torch.cuda.can_device_access_peer(device, peer_device):
            raise RuntimeError("SGLang CUDA-IPC requires CUDA peer access")
        ctypes.CDLL("libcudart.so").cudaDeviceEnablePeerAccess(peer_device, 0)

        from torch.utils.cpp_extension import load_inline

        cache_root = (
            Path(os.environ.get("CHITU_SGLANG_IPC_CACHE", "/tmp/chitu_sglang_ipc"))
            / f"rank_device_{device}"
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        self.ops = load_inline(
            name=f"chitu_sglang_ipc_sync_{device}",
            cpp_sources=_IPC_SYNC_DECL,
            cuda_sources=_IPC_SYNC_SRC,
            functions=["spin_wait", "bump_signal"],
            extra_cuda_cflags=["-O3"],
            build_directory=str(cache_root),
            verbose=False,
        )
        self.flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.sequence = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.timed_out = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.peer_flag = self._share(self.flag)
        self.peer_timed_out = self._share(self.timed_out)
        timeout_ms = float(os.environ.get("CHITU_SGLANG_IPC_TIMEOUT_MS", "30000"))
        self.budget_ns = int(timeout_ms * 1e6)

    def get_staging(
        self,
        local_elements: int,
        peer_elements: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (local_elements, peer_elements, dtype)
        pair = self.staging.get(key)
        if pair is None:
            local = torch.zeros(2, local_elements, dtype=dtype, device="cuda")
            pair = (local, self._share(local))
            self.staging[key] = pair
        return pair

    def next_slot(self) -> int:
        slot = self.calls % 2
        self.calls += 1
        return slot

    def signal(self) -> None:
        self.ops.bump_signal(self.sequence, self.peer_flag)

    def wait(self) -> None:
        self.ops.spin_wait(
            self.flag,
            self.sequence,
            self.timed_out,
            self.peer_timed_out,
            self.budget_ns,
        )

    def signal_and_wait(self) -> None:
        self.signal()
        self.wait()

    def check_timeout(self) -> None:
        if self.timed_out.item() != 0:
            raise RuntimeError("SGLang CUDA-IPC peer wait timed out")


def sglang_ipc_ulysses_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: SglangIpcA2AState,
    attention: AttentionFn,
) -> torch.Tensor:
    """SGLang's fused QKV input and direct-write output IPC exchange."""
    batch, local_sequence, heads, head_dim = query.shape
    if heads % 2:
        raise ValueError("SGLang CP2 IPC requires an even head count")
    local_heads = heads // 2
    rank = state.rank
    elements = batch * local_sequence * local_heads * head_dim
    local, peer = state.get_staging(3 * elements, 3 * elements, query.dtype)
    slot = state.next_slot()
    exchanged = []
    for index, tensor in enumerate((query, key, value)):
        send = tensor[
            :, :, (1 - rank) * local_heads : (2 - rank) * local_heads
        ].contiguous()
        peer[slot].narrow(0, index * elements, elements).copy_(
            send.view(-1), non_blocking=True
        )
        output = tensor.new_empty(batch, 2 * local_sequence, local_heads, head_dim)
        output.narrow(1, rank * local_sequence, local_sequence).copy_(
            tensor[:, :, rank * local_heads : (rank + 1) * local_heads]
        )
        exchanged.append(output)
    state.signal_and_wait()
    for index, output in enumerate(exchanged):
        received = (
            local[slot]
            .narrow(0, index * elements, elements)
            .view(batch, local_sequence, local_heads, head_dim)
        )
        output.narrow(1, (1 - rank) * local_sequence, local_sequence).copy_(received)

    local_output = attention(*exchanged)
    output_elements = batch * local_sequence * heads * head_dim
    local, peer = state.get_staging(
        output_elements,
        output_elements,
        local_output.dtype,
    )
    slot = state.next_slot()
    peer_output = (
        peer[slot]
        .narrow(0, 0, output_elements)
        .view(batch, local_sequence, heads, head_dim)
    )
    peer_output[:, :, rank * local_heads : (rank + 1) * local_heads].copy_(
        local_output.narrow(1, (1 - rank) * local_sequence, local_sequence),
        non_blocking=True,
    )
    state.signal()
    output = (
        local[slot]
        .narrow(0, 0, output_elements)
        .view(batch, local_sequence, heads, head_dim)
    )
    output[:, :, rank * local_heads : (rank + 1) * local_heads].copy_(
        local_output.narrow(1, rank * local_sequence, local_sequence)
    )
    state.wait()
    return output


__all__ = [
    "SGLANG_COMMIT",
    "VLLM_OMNI_COMMIT",
    "SglangIpcA2AState",
    "SglangPackedUlyssesAttention",
    "omni_allgather_kv_attention",
    "omni_ring_attention",
    "omni_ulysses_attention",
    "sglang_ipc_ulysses_attention",
]
