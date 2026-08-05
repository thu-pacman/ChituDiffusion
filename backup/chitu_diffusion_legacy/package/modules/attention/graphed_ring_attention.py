"""Graphed context-parallel (ring) attention.

A self-contained, near-linear sequence-parallel attention primitive: shard the
sequence across N ranks, and each rank computes attention over the *full*
sequence by streaming K/V around a ring while the **whole ring loop** (P2P comm +
per-step attention + online-softmax merge) is captured **once** into a CUDA Graph
and replayed on every call.

Why the CUDA Graph matters here
-------------------------------
A plain Python ring loop launches ``O(cp_size)`` NCCL send/recv + attention
kernels from the host on *every* call. As ranks increase, per-step launch and
scheduling jitter grows and ring attention stops scaling linearly. Capturing the
loop into a CUDA Graph removes all per-step host launch overhead and the
cross-rank launch skew: every rank replays a pre-recorded graph in lock-step, so
paired NCCL P2P kernels meet their peers almost immediately instead of
spin-waiting. The graph speedup therefore *grows* with ``cp_size`` -- restoring
linear scaling.

This is the standalone counterpart of the in-tree
``DiffusionAttention_with_CP._cp_forward_graphed`` path: it is deliberately built
on raw ``torch.distributed`` so it stays portable and readable, but it integrates
with ChituDiffusion by defaulting to the context-parallel (CP) process group.

Usage
-----
    from chitu_diffusion.modules.attention.graphed_ring_attention import (
        GraphedRingAttention,
    )

    # One instance per attention site (it owns the captured graph cache).
    ring_attn = GraphedRingAttention()          # defaults to the CP group
    # q, k, v: (B, S_local, H, D); sequence dim already sharded across ranks.
    out = ring_attn(q, k, v)                     # -> (B, S_local, H, D)

The ``group`` argument accepts a ChituDiffusion ``CommGroup``, a raw
``torch.distributed.ProcessGroup``, or ``None`` (resolve the CP group, else the
default WORLD group, else run single-rank).

Requirements / limitations
--------------------------
* Inputs must be CUDA tensors with **static shapes** across calls. One graph is
  captured per distinct ``(shape, dtype, scale)`` and reused; diffusion denoise
  steps share one shape, so a single capture serves the whole run.
* **Non-causal** attention only. Causal ring attention needs a zig-zag
  load-balanced partition with per-step masking; ``causal=True`` raises rather
  than returning silently-wrong results.
* If other (uncaptured) NCCL collectives run in the same process, export
  ``NCCL_GRAPH_MIXING_SUPPORT=1`` so NCCL tolerates graph-captured P2P.
* GQA/MQA is supported (K/V heads are broadcast to the Q head count).

Self-test (raw WORLD group, e.g. 4 GPUs):
    torchrun --nproc-per-node 4 -m chitu_diffusion.modules.attention.graphed_ring_attention
"""
from __future__ import annotations

import os
from logging import getLogger
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import flash_attn

    _HAS_FLASH = True
except ImportError:
    _HAS_FLASH = False

logger = getLogger(__name__)


# --------------------------------------------------------------------------- #
# Online-softmax (log-sum-exp) merge of ring partial attentions.
#
#   out : (B, S, H, D) float32 running output
#   lse : (B, S, H, 1) float32 running log-sum-exp
#   block_out : (B, S, H, D)  partial output for the current K/V block
#   block_lse : (B, H, S)     its log-sum-exp (flash / aten convention)
#
# Numerically-stable update that needs no running max:
#   lse_new = lse - log(sigmoid(lse - block_lse))
#   out_new = out - sigmoid(block_lse - lse) * (out - block_out)
# --------------------------------------------------------------------------- #
def _online_softmax_merge(out, lse, block_out, block_lse):
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(-1)  # (B,H,S) -> (B,S,H,1)
    if out is None:
        return block_out, block_lse
    new_lse = lse - F.logsigmoid(lse - block_lse)
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    return out, new_lse


# --------------------------------------------------------------------------- #
# Per-step attention kernels. Each takes/returns (B, S, H, D) and also returns
# the log-sum-exp (B, H, S) required to merge ring partials.
# --------------------------------------------------------------------------- #
def _attn_flash(q, k, v, scale, causal):
    out, lse, _ = flash_attn.flash_attn_func(
        q, k, v, softmax_scale=scale, causal=causal, return_attn_probs=True
    )
    return out, lse


def _attn_sdpa(q, k, v, scale, causal):
    qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention(
        qt, kt, vt, dropout_p=0.0, is_causal=causal, return_debug_mask=False, scale=scale
    )[:2]
    return out.transpose(1, 2), lse


_KERNELS: Dict[str, Callable] = {"flash": _attn_flash, "sdpa": _attn_sdpa}


def _select_kernel(name: str) -> Tuple[str, Callable]:
    if name == "auto":
        name = "flash" if _HAS_FLASH else "sdpa"
    if name == "flash" and not _HAS_FLASH:
        raise RuntimeError("kernel='flash' requested but flash_attn is not installed.")
    if name not in _KERNELS:
        raise ValueError(f"Unknown kernel '{name}'. Choose from {sorted(_KERNELS)} or 'auto'.")
    return name, _KERNELS[name]


def _resolve_group(group) -> Tuple[Optional["dist.ProcessGroup"], int, int, List[int]]:
    """Normalize ``group`` to (process_group, world_size, rank_in_group, peer_ranks).

    ``peer_ranks`` holds the *global* ranks indexed by position in the group, so
    P2P ops address peers correctly even for a sub-group. ``group`` may be a
    ChituDiffusion ``CommGroup``, a raw ``ProcessGroup``, or ``None``.
    """
    # ChituDiffusion CommGroup (duck-typed to avoid a hard import dependency).
    if hasattr(group, "gpu_group") and hasattr(group, "rank_list"):
        pg = group.gpu_group
        if pg.__class__.__name__ == "SingletonGroupPlaceholder":
            return None, 1, 0, [0]
        return pg, int(group.group_size), int(group.rank_in_group), list(group.rank_list)

    # None -> try the CP group, then fall back to the default WORLD group.
    if group is None:
        try:
            from chitu_diffusion.core.distributed.parallel_state import get_cp_group

            cg = get_cp_group()
        except Exception:
            cg = None
        if cg is not None and hasattr(cg, "gpu_group"):
            return _resolve_group(cg)

    if not (dist.is_available() and dist.is_initialized()):
        return None, 1, 0, [0]

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    peers = [dist.get_global_rank(group, i) for i in range(world)] if group is not None else list(range(world))
    return group, world, rank, peers


# --------------------------------------------------------------------------- #
# The module.
# --------------------------------------------------------------------------- #
class GraphedRingAttention:
    """Context-parallel ring attention with an optional captured CUDA Graph.

    Create one instance per attention site so the captured graph and its static
    I/O buffers are reused across denoise steps / forward calls.

    Args:
        group: ChituDiffusion ``CommGroup``, raw ``ProcessGroup``, or ``None``
            (resolve the CP group, else default WORLD, else single-rank).
        kernel: ``"auto"`` (flash if available else sdpa), ``"flash"`` or
            ``"sdpa"`` (pure ATen, no extra dependency).
        use_cuda_graph: capture+replay the ring loop. Disable to fall back to the
            eager Python ring loop (debugging / dynamic shapes).
        warmup_iters: eager iterations before capture (initializes NCCL P2P
            connections and kernel workspaces so they are not created mid-capture).
    """

    def __init__(
        self,
        group=None,
        *,
        kernel: str = "auto",
        use_cuda_graph: bool = True,
        warmup_iters: int = 3,
    ) -> None:
        self.pg, self.world_size, self.rank_in_group, self._peers = _resolve_group(group)
        self.kernel_name, self._kernel_fn = _select_kernel(kernel)
        self.use_cuda_graph = use_cuda_graph
        self.warmup_iters = max(1, int(warmup_iters))

        # Ring topology: send K/V to `next`, receive the next block from `prev`.
        self._send_peer = self._peers[(self.rank_in_group + 1) % self.world_size]
        self._recv_peer = self._peers[(self.rank_in_group - 1) % self.world_size]
        self._graph_cache: Dict[tuple, dict] = {}

    # -- public API -------------------------------------------------------- #
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Full-sequence attention for this rank's local query shard.

        Args:
            q, k, v: ``(B, S_local, H, D)`` (K/V may use fewer heads for GQA/MQA);
                the sequence dim is assumed already sharded across the group.
            softmax_scale: defaults to ``1/sqrt(D)``.
            causal: must be ``False`` (see module docstring).

        Returns:
            ``(B, S_local, H, D)`` attention output in ``q.dtype``.
        """
        if causal:
            raise NotImplementedError(
                "Causal ring attention requires a zig-zag load-balanced partition "
                "and per-step masking, which this module does not implement."
            )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        if self.world_size == 1:
            out, _ = self._dispatch(q, k, v, softmax_scale, False)
            return out.to(q.dtype)

        if self.use_cuda_graph and q.is_cuda:
            return self._ring_graphed(q, k, v, float(softmax_scale))
        return self._ring_eager(q, k, v, float(softmax_scale))

    def reset_graph_cache(self) -> None:
        """Drop all captured graphs and their static buffers."""
        self._graph_cache.clear()

    # -- internals --------------------------------------------------------- #
    def _dispatch(self, q, k, v, scale, causal):
        # Broadcast K/V heads to match Q for GQA/MQA before the kernel call.
        if k.shape[-2] != q.shape[-2]:
            rep = q.shape[-2] // k.shape[-2]
            k = k.repeat_interleave(rep, dim=-2)
            v = v.repeat_interleave(rep, dim=-2)
        return self._kernel_fn(q, k, v, scale, causal)

    def _ring_eager(self, q, k, v, scale) -> torch.Tensor:
        """Plain Python ring loop with compute/comm overlap."""
        out = lse = None
        steps = self.world_size
        for step in range(steps):
            if step + 1 != steps:
                send_k = k.contiguous()
                send_v = v.contiguous()
                recv_k = torch.empty_like(send_k)
                recv_v = torch.empty_like(send_v)
                reqs = dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.isend, send_k, self._send_peer, self.pg),
                        dist.P2POp(dist.isend, send_v, self._send_peer, self.pg),
                        dist.P2POp(dist.irecv, recv_k, self._recv_peer, self.pg),
                        dist.P2POp(dist.irecv, recv_v, self._recv_peer, self.pg),
                    ]
                )

            block_out, block_lse = self._dispatch(q, k, v, scale, False)
            out, lse = _online_softmax_merge(out, lse, block_out, block_lse)

            if step + 1 != steps:
                for req in reqs:
                    req.wait()
                k, v = recv_k, recv_v
        return out.to(q.dtype)

    def _ring_graphed(self, q, k, v, scale) -> torch.Tensor:
        """Capture the ring loop into a CUDA Graph (once per shape) and replay it."""
        key = (tuple(q.shape), tuple(k.shape), tuple(v.shape), q.dtype, scale)
        entry = self._graph_cache.get(key)
        if entry is None:
            entry = self._capture(q, k, v, scale)
            self._graph_cache[key] = entry
        entry["q"].copy_(q)
        entry["k"].copy_(k)
        entry["v"].copy_(v)
        entry["graph"].replay()
        return entry["out"].clone()

    def _capture(self, q, k, v, scale) -> dict:
        static_q, static_k, static_v = q.clone(), k.clone(), v.clone()

        # Warm up on a side stream first: required before capture, and it also
        # establishes the NCCL P2P connections so they are not (illegally)
        # created during graph capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup_iters):
                self._ring_eager(static_q, static_k, static_v, scale)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self._ring_eager(static_q, static_k, static_v, scale)
        torch.cuda.synchronize()

        logger.info(
            "[graphed-ring-attn] captured ring loop for q=%s (cp_size=%d, kernel=%s)",
            tuple(q.shape),
            self.world_size,
            self.kernel_name,
        )
        return {
            "graph": graph,
            "q": static_q,
            "k": static_k,
            "v": static_v,
            "out": static_out,
        }


# --------------------------------------------------------------------------- #
# Self-test: eager vs graph vs all-gather full-attention reference (WORLD group).
# Launch: torchrun --nproc-per-node <N> -m \
#         chitu_diffusion.modules.attention.graphed_ring_attention
# --------------------------------------------------------------------------- #
def _reference_full(q_local, k_local, v_local, scale, group=None):
    world = dist.get_world_size(group)
    kg = [torch.empty_like(k_local) for _ in range(world)]
    vg = [torch.empty_like(v_local) for _ in range(world)]
    dist.all_gather(kg, k_local.contiguous(), group=group)
    dist.all_gather(vg, v_local.contiguous(), group=group)
    k_full = torch.cat(kg, dim=1)
    v_full = torch.cat(vg, dim=1)
    out = F.scaled_dot_product_attention(
        q_local.transpose(1, 2), k_full.transpose(1, 2), v_full.transpose(1, 2), scale=scale
    )
    return out.transpose(1, 2).contiguous()


def _self_test() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    os.environ.setdefault("MASTER_PORT", "29597")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    dev = torch.device(f"cuda:{local}")
    B, H, D, s_local = 1, 24, 128, 4608 // world
    scale = D ** -0.5
    gen = torch.Generator(device=dev).manual_seed(1234 + rank)
    q = torch.randn(B, s_local, H, D, device=dev, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(B, s_local, H, D, device=dev, dtype=torch.bfloat16, generator=gen)
    v = torch.randn(B, s_local, H, D, device=dev, dtype=torch.bfloat16, generator=gen)

    attn = GraphedRingAttention()
    out_graph = attn(q, k, v, softmax_scale=scale)
    eager = GraphedRingAttention(use_cuda_graph=False)
    out_eager = eager(q, k, v, softmax_scale=scale)
    ref = _reference_full(q, k, v, scale)

    d_eg = (out_graph.float() - out_eager.float()).abs().max().item()
    d_ref = (out_graph.float() - ref.float()).abs().max().item()
    dist.barrier()
    if rank == 0:
        print(
            f"[graphed-ring-attn] world={world} kernel={attn.kernel_name} "
            f"s_local={s_local} | max|graph-eager|={d_eg:.2e} max|graph-full|={d_ref:.2e}",
            flush=True,
        )
    dist.barrier()
    os._exit(0)  # process-group teardown after graph capture can hang; exit hard.


if __name__ == "__main__":
    _self_test()
