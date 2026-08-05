"""Model-agnostic context-parallel (sequence-parallel) attention for the
"replicated text + sequence-sharded image" pattern shared by single-stream
(Z-Image) and dual-stream (Qwen-Image) DiT backbones.

The pattern
-----------
Every CP rank holds the **full, replicated text** tokens and **its own shard**
of the image tokens. A rank's local query (its image shard + the full text)
must attend to the *full* sequence (all image shards + text). The text K/V are
already present everywhere; only the image K/V must be exchanged across ranks.

This module owns that exchange + the local attention, exposing a single
model-agnostic entry point::

    cp = ImageContextParallelAttention(attn_backend, mode="agkv")
    cp.enable(group)
    txt_out, img_out = cp.attention(txt_q, txt_k, txt_v, img_q, img_k, img_v)

Inputs/outputs are ``[B, S, H, D]`` (sequence-major). ``img_*`` carry this
rank's local image shard; ``txt_*`` carry the full replicated text. The model
adapter is responsible only for projections, RoPE, packing order and the final
output all-gather of the local image shard -- none of which live here.

Strategies (``CHITU_*_CP_MODE`` parity with the previous in-processor impl):

* ``agkv``       - split-Q / all-gather image K/V (every rank sees full K/V);
* ``ulysses``    - all-to-all head parallelism (DeepSpeed-Ulysses);
* ``ring``       - ring P2P of image K/V with online-softmax merge;
* ``ring_graph`` - the ring loop captured once into a CUDA Graph and replayed;
* ``unified``    - 2D Ulysses x Ring (USP): Ulysses all-to-all within each
  ``up``-sized sub-group (``CP_UP`` / framework ``up`` config) and Ring P2P of
  image K/V across the sub-groups. With a topology-aware rank map this puts the
  chatty all-to-all on the fast intra-NUMA link and only ``cp/up`` ring hops on
  the slow cross-NUMA link. ``up==1`` degenerates to pure ring, ``up==cp`` to
  pure ulysses.

All modes return the SAME ``(txt_out, img_out)`` layout, so the strategy is an
implementation detail invisible to callers.
"""
from __future__ import annotations

import contextlib
import os
from logging import getLogger
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from chitu_diffusion.core.distributed.parallel_state import get_up_group
from chitu_diffusion.runtime.parallel_utils import update_out_and_lse

logger = getLogger(__name__)

_VALID_MODES = {"agkv", "ulysses", "ring", "ring_graph", "unified"}

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0  # max representable magnitude of e4m3fn


def _quantize_fp8_per_head(x: torch.Tensor):
    """Quantize a ``[B, S, H, D]`` tensor to fp8 (e4m3) with per-head scaling.

    Returns ``(x_u8, scale)`` where ``x_u8`` is the fp8 payload re-viewed as
    ``uint8`` (so NCCL, which has no fp8 dtype, can move it as raw bytes) and
    ``scale`` is the per-head ``[1, 1, H, 1]`` fp32 dequant multiplier. Per-head
    granularity keeps accuracy high (each head gets its own dynamic range) at the
    cost of only ``H`` extra scalars to communicate.
    """
    amax = x.abs().to(torch.float32).amax(dim=(0, 1, 3), keepdim=True).clamp_min(1e-12)
    scale = amax / _FP8_MAX
    x_q = (x.to(torch.float32) / scale).clamp_(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return x_q.view(torch.uint8), scale


def _dequantize_fp8(x_u8: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Inverse of :func:`_quantize_fp8_per_head`: uint8 fp8 payload -> ``out_dtype``."""
    return (x_u8.view(_FP8_DTYPE).to(torch.float32) * scale).to(out_dtype)


class ImageContextParallelAttention:
    """Replicated-text + sharded-image context-parallel attention core.

    One instance is shared by every attention block of a model (all blocks see
    identical shapes), which lets a single captured ring CUDA Graph serve the
    whole denoise run.

    Args:
        attn_backend: a raw per-rank attention kernel callable with the
            ``DiffusionAttnBackend`` signature ``(q, k, v, causal=..., \
            return_attn_probs=...) -> (out, lse, ...)``. If a CP wrapper
            (``DiffusionAttention_with_CP``) is passed, it is unwrapped to the
            raw kernel stored on ``.attn`` so CP is applied exactly once here.
        mode: one of ``agkv`` / ``ulysses`` / ``ring`` / ``ring_graph``.
        profile: when True, every op is timed with synchronized CUDA events and
            accrued into per-bucket stats (serializes execution; for analysis
            only). Also forces ``ring_graph`` to fall back to eager ring so the
            per-op buckets stay meaningful.
        use_cudagraph: capture+replay the ring loop in ``ring_graph`` mode.
    """

    def __init__(
        self,
        attn_backend,
        *,
        mode: str = "agkv",
        profile: bool = False,
        use_cudagraph: bool = True,
        ulysses_size: int = 1,
    ) -> None:
        # Unwrap the framework CP wrapper to the raw kernel: this module
        # implements CP itself, so a second (wrong/redundant) CP layer must be
        # bypassed. DiffusionAttention_with_CP stores the bare backend on `.attn`.
        self.attn_backend = getattr(attn_backend, "attn", attn_backend)

        mode = str(mode).strip().lower()
        if mode not in _VALID_MODES:
            raise ValueError(
                f"CP mode must be one of {sorted(_VALID_MODES)}, got {mode!r}."
            )
        self.mode = mode
        self.profile = bool(profile)
        self.use_cudagraph = bool(use_cudagraph)
        # Ulysses degree for the 2D ``unified`` mode (the framework ``up`` config).
        # 1 => pure ring; cp_size => pure ulysses; anything in between is USP.
        self.ulysses_size = max(1, int(ulysses_size))

        self._group = None
        self._enabled = False
        self._comm_stream = None

        self._mode_warned = False
        self._ring_graph_cache: dict = {}
        self._ring_graph_warned = False
        # Diagnostic: force the ring loop to wait for each K/V transfer *before*
        # computing (defeats compute/comm overlap). Lets us measure how much the
        # default overlap actually hides on a given interconnect.
        self._no_overlap = os.environ.get("CHITU_CP_RING_NO_OVERLAP", "0") == "1"
        # Emit torch.profiler ranges (per comm/compute op) so the chrome trace is
        # readable. Only meaningful under torch.profiler; near-zero cost otherwise.
        self._trace_mark = os.environ.get("CHITU_TORCH_TRACE", "").strip() != ""
        # Transfer the communicated image K/V as fp8 (e4m3, per-head scaled) to
        # halve the bytes on the (host-staged, comm-bound) PCIe link. Quantize
        # before the transfer, dequantize back to the compute dtype after; the
        # attention itself stays in the original precision. Applies to the AGKV
        # all-gather, and to the unified mode's Ulysses all-to-all (K/V) + ring
        # K/V P2P. Q and the output/text gathers stay bf16.
        self._fp8_kv = os.environ.get("CHITU_CP_FP8_KV", "0") == "1"
        # Cross-step KV cache (AGKV only): on "stale" steps reuse the previously
        # all-gathered *remote* image K/V and splice in only this rank's freshly
        # computed local shard -> the all-gather is skipped entirely (no comm).
        # K/V change slowly between adjacent denoise steps, so the staleness
        # error is typically smaller than fp8 quantization. Schedule: fresh for
        # the first `warmup` steps, the last `tail` steps, and every
        # `interval`-th step in between.
        self._kvcache_enabled = os.environ.get("CHITU_CP_KV_CACHE", "0") == "1"
        self._kvcache_warmup = int(os.environ.get("CHITU_CP_KV_CACHE_WARMUP", "8"))
        self._kvcache_interval = max(1, int(os.environ.get("CHITU_CP_KV_CACHE_INTERVAL", "4")))
        self._kvcache_tail = int(os.environ.get("CHITU_CP_KV_CACHE_TAIL", "3"))
        self._kvcache: dict = {}
        self._cur_step = None
        self._cur_total = None
        self._cur_tag = "cond"
        self._layer_idx = 0
        self.reset_stats()

    def begin_forward(self, step: int, total_steps: int, cfg_tag: str = "cond") -> None:
        """Announce the current denoise step before a transformer forward so the
        AGKV KV-cache can decide fresh-vs-stale and key entries per (cfg, layer).
        No-op unless the KV cache is enabled."""
        if not self._kvcache_enabled:
            return
        if step == 0 and cfg_tag == "cond":
            self._kvcache.clear()
        self._cur_step = int(step)
        self._cur_total = int(total_steps)
        self._cur_tag = str(cfg_tag)
        self._layer_idx = 0

    def _kvcache_is_fresh(self) -> bool:
        s, n = self._cur_step, self._cur_total
        if s is None:
            return True
        if s < self._kvcache_warmup:
            return True
        if n is not None and s >= n - self._kvcache_tail:
            return True
        return (s % self._kvcache_interval) == 0

    def _agkv_gather_cached(self, lid: int, img_k, img_v, grp):
        key = (self._cur_tag, lid)
        if self._kvcache_is_fresh() or key not in self._kvcache:
            full_k = self.all_gather_seq(img_k, grp, kind="kv")
            full_v = self.all_gather_seq(img_v, grp, kind="kv")
            self._kvcache[key] = (full_k.detach(), full_v.detach())
            return full_k, full_v
        # Stale: reuse the cached gather, overwriting only this rank's own shard
        # with the freshly computed local K/V (no communication).
        cached_k, cached_v = self._kvcache[key]
        r = grp.rank_in_group
        L = img_k.shape[1]
        lo, hi = r * L, (r + 1) * L
        full_k = torch.cat([cached_k[:, :lo], img_k, cached_k[:, hi:]], dim=1)
        full_v = torch.cat([cached_v[:, :lo], img_v, cached_v[:, hi:]], dim=1)
        return full_k, full_v

    def _mark(self, name: str):
        """Named profiler range, used only to annotate the chrome trace."""
        if self._trace_mark:
            return torch.profiler.record_function(name)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def enable(self, group) -> None:
        self._group = group
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._group is not None and self._group.group_size > 1

    @property
    def group(self):
        return self._group

    def _get_comm_stream(self):
        if self._comm_stream is None:
            self._comm_stream = torch.cuda.Stream()
        return self._comm_stream

    @property
    def supports_fused_overlap(self) -> bool:
        """Whether the experimental async QKV/comm overlap path is allowed.

        Profiling, fp8 communication and KV-cache intentionally stay on the
        serial path: those modes either synchronize for measurement or change
        the communication payload/schedule.
        """
        return (
            self.enabled
            and torch.cuda.is_available()
            and not self.profile
            and self.mode in ("agkv", "ulysses")
            and not self._fp8_kv
            and not self._kvcache_enabled
        )

    def _attention_from_packed(self, q, k, v, img_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attention(
            q[:, img_len:], k[:, img_len:], v[:, img_len:],
            q[:, :img_len], k[:, :img_len], v[:, :img_len],
        )

    # ------------------------------------------------------------------ #
    # Profiling
    # ------------------------------------------------------------------ #
    def reset_stats(self) -> None:
        self._kv_seconds = 0.0
        self._kv_calls = 0
        self._kv_bytes = 0
        self._out_seconds = 0.0
        self._out_calls = 0
        self._out_bytes = 0
        self._qkv_seconds = 0.0
        self._qkv_calls = 0
        self._attn_seconds = 0.0
        self._attn_calls = 0
        self._a2a_seconds = 0.0
        self._a2a_calls = 0
        self._a2a_bytes = 0
        self._ring_seconds = 0.0
        self._ring_calls = 0
        self._ring_bytes = 0

    def stats(self) -> dict:
        return {
            "mode": self.mode,
            "kv_seconds": self._kv_seconds,
            "kv_calls": self._kv_calls,
            "kv_bytes": self._kv_bytes,
            "out_seconds": self._out_seconds,
            "out_calls": self._out_calls,
            "out_bytes": self._out_bytes,
            "qkv_seconds": self._qkv_seconds,
            "qkv_calls": self._qkv_calls,
            "attn_seconds": self._attn_seconds,
            "attn_calls": self._attn_calls,
            "a2a_seconds": self._a2a_seconds,
            "a2a_calls": self._a2a_calls,
            "a2a_bytes": self._a2a_bytes,
            "ring_seconds": self._ring_seconds,
            "ring_calls": self._ring_calls,
            "ring_bytes": self._ring_bytes,
        }

    def time(self, fn, bucket: str):
        """Run ``fn`` and, when profiling, accrue its CUDA-timed duration to the
        named compute bucket (``"qkv"`` or ``"attn"``)."""
        if not self.profile:
            return fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) / 1000.0
        if bucket == "qkv":
            self._qkv_seconds += seconds
            self._qkv_calls += 1
        elif bucket == "attn":
            self._attn_seconds += seconds
            self._attn_calls += 1
        return result

    def _comm_time(self, fn, bucket: str, recv_bytes: int):
        """Run a collective/P2P op ``fn``; when profiling, accrue its CUDA-timed
        duration and moved bytes to the named bucket (``"a2a"`` or ``"ring"``)."""
        if not self.profile:
            return fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) / 1000.0
        if bucket == "a2a":
            self._a2a_seconds += seconds
            self._a2a_calls += 1
            self._a2a_bytes += recv_bytes
        elif bucket == "ring":
            self._ring_seconds += seconds
            self._ring_calls += 1
            self._ring_bytes += recv_bytes
        return result

    # ------------------------------------------------------------------ #
    # Communication primitives
    # ------------------------------------------------------------------ #
    def all_gather_seq(self, tensor: torch.Tensor, group=None, kind: str = "kv") -> torch.Tensor:
        """All-gather a ``[B, S_local, H, D]`` tensor along the sequence dim.

        ``kind`` selects the profiling bucket: ``"kv"`` (per-layer image K/V
        gather) or ``"out"`` (post-layers output gather of the image shard).
        """
        group = group if group is not None else self._group
        tensor = tensor.contiguous()
        pieces = [torch.empty_like(tensor) for _ in range(group.group_size)]
        if self.profile:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_gather(pieces, tensor, group=group.gpu_group)
            end.record()
            end.synchronize()
            seconds = start.elapsed_time(end) / 1000.0
            recv_bytes = tensor.numel() * tensor.element_size() * (group.group_size - 1)
            if kind == "out":
                self._out_seconds += seconds
                self._out_calls += 1
                self._out_bytes += recv_bytes
            else:
                self._kv_seconds += seconds
                self._kv_calls += 1
                self._kv_bytes += recv_bytes
        else:
            dist.all_gather(pieces, tensor, group=group.gpu_group)
        return torch.cat(pieces, dim=1)

    def all_gather_seq_fp8(self, tensor: torch.Tensor, group=None, kind: str = "kv") -> torch.Tensor:
        """fp8 variant of :meth:`all_gather_seq`: quantize the local ``[B, S, H, D]``
        shard to fp8 (per-head), all-gather the fp8 bytes + per-rank scales, then
        dequantize each shard with its own scale and concat along the sequence dim.

        Moves half the bytes of the bf16 path. The tiny scale all-gather is
        untimed; the profiled ``kv`` bucket reflects the (halved) fp8 payload.

        The local shard never leaves the device, so its own slot is filled with
        the original (bf16) ``tensor`` instead of the fp8 round-trip -> 1/cp of
        the K/V stays lossless at no extra communication.
        """
        group = group if group is not None else self._group
        out_dtype = tensor.dtype
        x_u8, scale = _quantize_fp8_per_head(tensor)
        x_u8 = x_u8.contiguous()
        scale = scale.contiguous()
        byte_pieces = [torch.empty_like(x_u8) for _ in range(group.group_size)]
        scale_pieces = [torch.empty_like(scale) for _ in range(group.group_size)]
        dist.all_gather(scale_pieces, scale, group=group.gpu_group)
        if self.profile:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_gather(byte_pieces, x_u8, group=group.gpu_group)
            end.record()
            end.synchronize()
            seconds = start.elapsed_time(end) / 1000.0
            recv_bytes = x_u8.numel() * x_u8.element_size() * (group.group_size - 1)
            if kind == "out":
                self._out_seconds += seconds
                self._out_calls += 1
                self._out_bytes += recv_bytes
            else:
                self._kv_seconds += seconds
                self._kv_calls += 1
                self._kv_bytes += recv_bytes
        else:
            dist.all_gather(byte_pieces, x_u8, group=group.gpu_group)
        deq = [_dequantize_fp8(p, s, out_dtype) for p, s in zip(byte_pieces, scale_pieces)]
        # Keep our own shard lossless: substitute the original bf16 for the local
        # slot (all_gather places rank i's data at index i).
        deq[group.rank_in_group] = tensor
        return torch.cat(deq, dim=1)

    def _all_to_all(self, tensor: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
        grp = self._group
        cp = grp.group_size
        recv_bytes = int(tensor.numel() * tensor.element_size() * (cp - 1) / cp)
        return self._comm_time(
            lambda: grp.all_to_all(tensor, scatter_dim, gather_dim), "a2a", recv_bytes
        )

    def _all_gather_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        grp = self._group
        tensor = tensor.contiguous()
        pieces = [torch.empty_like(tensor) for _ in range(grp.group_size)]
        recv_bytes = tensor.numel() * tensor.element_size() * (grp.group_size - 1)
        self._comm_time(
            lambda: dist.all_gather(pieces, tensor, group=grp.gpu_group), "a2a", recv_bytes
        )
        return torch.cat(pieces, dim=2).contiguous()

    def _ring_send_recv_commit(self, tensors, src_idx: int, dst_idx: int):
        grp = self._group
        recv = []
        for t in tensors:
            t = t.contiguous()
            grp.p2p_isend(t, dst=dst_idx)
            recv.append(grp.p2p_irecv(size=t.shape, dtype=t.dtype, src=src_idx))
        grp.p2p_commit()
        return tuple(recv)

    def _ring_wait(self, recv):
        self._group.p2p_wait()
        return recv

    # ------------------------------------------------------------------ #
    # Public attention entry point
    # ------------------------------------------------------------------ #
    @torch.compiler.disable
    def attention(
        self,
        txt_q: torch.Tensor,
        txt_k: torch.Tensor,
        txt_v: torch.Tensor,
        img_q: torch.Tensor,
        img_k: torch.Tensor,
        img_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full-sequence attention for the local ``[image_shard, text]`` query.

        Args:
            txt_*: full replicated text Q/K/V, ``[B, S_txt, H, D]``.
            img_*: this rank's local image shard Q/K/V, ``[B, S_img_local, H, D]``.

        Returns:
            ``(txt_out, img_out)`` where ``txt_out`` is the full text output and
            ``img_out`` is this rank's local image-shard output.
        """
        mode = self.mode
        cp = self._group.group_size
        if mode == "unified":
            return self._attn_unified(txt_q, txt_k, txt_v, img_q, img_k, img_v)
        if mode == "ulysses":
            heads = img_q.shape[2]
            if heads % cp != 0:
                if not self._mode_warned:
                    logger.warning(
                        "Ulysses CP needs heads (%d) divisible by cp_size (%d); falling back to agkv.",
                        heads, cp,
                    )
                    self._mode_warned = True
                return self._attn_agkv(txt_q, txt_k, txt_v, img_q, img_k, img_v)
            return self._attn_ulysses(txt_q, txt_k, txt_v, img_q, img_k, img_v)
        if mode in ("ring", "ring_graph"):
            use_graph = mode == "ring_graph" and self.use_cudagraph and img_q.is_cuda
            if use_graph and self.profile:
                if not self._ring_graph_warned:
                    logger.warning(
                        "CP mode=ring_graph with profiling on: CUDA Graph disabled "
                        "so per-op buckets stay meaningful. Run without profiling "
                        "to time the graph."
                    )
                    self._ring_graph_warned = True
                use_graph = False
            if use_graph:
                return self._attn_ring_graphed(txt_q, txt_k, txt_v, img_q, img_k, img_v)
            return self._attn_ring(txt_q, txt_k, txt_v, img_q, img_k, img_v)
        return self._attn_agkv(txt_q, txt_k, txt_v, img_q, img_k, img_v)

    @torch.compiler.disable
    def attention_fused(
        self,
        *,
        produce_img_q,
        produce_img_k,
        produce_img_v,
        produce_txt_q,
        produce_txt_k,
        produce_txt_v,
        num_heads: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projection-aware CP attention fast path.

        The model processor owns projection/norm/RoPE details and provides
        producer callables returning ``[B, S, H, D]`` tensors. This core owns the
        stream/event schedule that overlaps image communication with later
        producers. Unsupported modes/features fall back to the exact serial
        ``attention()`` path by materializing all producers first.
        """

        def serial():
            img_q = produce_img_q()
            img_k = produce_img_k()
            img_v = produce_img_v()
            txt_q = produce_txt_q()
            txt_k = produce_txt_k()
            txt_v = produce_txt_v()
            return self.attention(txt_q, txt_k, txt_v, img_q, img_k, img_v)

        if not self.supports_fused_overlap:
            return serial()

        if self.mode == "agkv":
            return self._attn_agkv_fused(
                produce_img_q=produce_img_q,
                produce_img_k=produce_img_k,
                produce_img_v=produce_img_v,
                produce_txt_q=produce_txt_q,
                produce_txt_k=produce_txt_k,
                produce_txt_v=produce_txt_v,
            )

        if num_heads is not None and num_heads % self._group.group_size != 0:
            if not self._mode_warned:
                logger.warning(
                    "Ulysses CP needs heads (%d) divisible by cp_size (%d); falling back to serial fused path.",
                    num_heads,
                    self._group.group_size,
                )
                self._mode_warned = True
            return serial()

        return self._attn_ulysses_fused(
            produce_img_q=produce_img_q,
            produce_img_k=produce_img_k,
            produce_img_v=produce_img_v,
            produce_txt_q=produce_txt_q,
            produce_txt_k=produce_txt_k,
            produce_txt_v=produce_txt_v,
        )

    @torch.compiler.disable
    def cp_attn_with_full_txt_fused(
        self,
        *,
        produce_txt_q,
        produce_txt_k,
        produce_txt_v,
        produce_img_q,
        produce_img_k,
        produce_img_v,
        num_heads: Optional[int] = None,
        **_: object,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias matching ``DiffusionAttention_with_CP``'s full-text fused API."""
        return self.attention_fused(
            produce_img_q=produce_img_q,
            produce_img_k=produce_img_k,
            produce_img_v=produce_img_v,
            produce_txt_q=produce_txt_q,
            produce_txt_k=produce_txt_k,
            produce_txt_v=produce_txt_v,
            num_heads=num_heads,
        )

    @torch.compiler.disable
    def attention_fused_packed(
        self,
        *,
        img_len: int,
        produce_q,
        produce_k,
        produce_v,
        num_heads: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projection-aware fast path for packed ``[image_shard, text]`` producers.

        ``produce_q/k/v`` each run once and return packed ``[B, S_img+S_txt, H, D]``.
        This is the right API for single-stream models such as Z-Image, where
        projecting image/text separately would duplicate GEMM work.
        """

        if not self.supports_fused_overlap:
            q = produce_q()
            k = produce_k()
            v = produce_v()
            return self._attention_from_packed(q, k, v, img_len)

        if self.mode == "agkv":
            return self._attn_agkv_packed_fused(img_len, produce_q, produce_k, produce_v)

        if num_heads is not None and num_heads % self._group.group_size != 0:
            if not self._mode_warned:
                logger.warning(
                    "Ulysses CP needs heads (%d) divisible by cp_size (%d); falling back to serial fused path.",
                    num_heads,
                    self._group.group_size,
                )
                self._mode_warned = True
            q = produce_q()
            k = produce_k()
            v = produce_v()
            return self._attention_from_packed(q, k, v, img_len)

        return self._attn_ulysses_packed_fused(img_len, produce_q, produce_k, produce_v)

    # ------------------------------------------------------------------ #
    # Strategies
    # ------------------------------------------------------------------ #
    def _attn_agkv(self, txt_q, txt_k, txt_v, img_q, img_k, img_v):
        """Split-Q / all-gather image K/V. Local queries attend the full K/V."""
        grp = self._group
        local_img_len = img_q.shape[1]
        if self._kvcache_enabled and self._cur_step is not None:
            lid = self._layer_idx
            self._layer_idx += 1
            full_img_key, full_img_value = self._agkv_gather_cached(lid, img_k, img_v, grp)
        else:
            gather = self.all_gather_seq_fp8 if self._fp8_kv else self.all_gather_seq
            full_img_key = gather(img_k, grp, kind="kv")
            full_img_value = gather(img_v, grp, kind="kv")
        q = torch.cat([img_q, txt_q], dim=1).contiguous()
        k = torch.cat([full_img_key, txt_k], dim=1).contiguous()
        v = torch.cat([full_img_value, txt_v], dim=1).contiguous()
        out = self.time(lambda: self.attn_backend(q, k, v, causal=False)[0], "attn")
        return out[:, local_img_len:], out[:, :local_img_len]

    def _attn_agkv_fused(
        self,
        *,
        produce_img_q,
        produce_img_k,
        produce_img_v,
        produce_txt_q,
        produce_txt_k,
        produce_txt_v,
    ):
        grp = self._group
        cur = torch.cuda.current_stream()
        cs = self._get_comm_stream()

        img_k = produce_img_k()
        cs.wait_stream(cur)
        img_k.record_stream(cs)
        with torch.cuda.stream(cs):
            full_img_key = self.all_gather_seq(img_k, grp, kind="kv")
        ev_k = torch.cuda.Event()
        ev_k.record(cs)

        img_v = produce_img_v()
        cs.wait_stream(cur)
        img_v.record_stream(cs)
        with torch.cuda.stream(cs):
            full_img_value = self.all_gather_seq(img_v, grp, kind="kv")
        ev_v = torch.cuda.Event()
        ev_v.record(cs)

        img_q = produce_img_q()
        txt_q = produce_txt_q()
        txt_k = produce_txt_k()
        txt_v = produce_txt_v()
        local_img_len = img_q.shape[1]

        cur.wait_event(ev_k)
        cur.wait_event(ev_v)
        full_img_key.record_stream(cur)
        full_img_value.record_stream(cur)

        q = torch.cat([img_q, txt_q], dim=1).contiguous()
        k = torch.cat([full_img_key, txt_k], dim=1).contiguous()
        v = torch.cat([full_img_value, txt_v], dim=1).contiguous()
        out = self.time(lambda: self.attn_backend(q, k, v, causal=False)[0], "attn")
        return out[:, local_img_len:], out[:, :local_img_len]

    def _attn_agkv_packed_fused(self, img_len, produce_q, produce_k, produce_v):
        grp = self._group
        cur = torch.cuda.current_stream()
        cs = self._get_comm_stream()

        k = produce_k()
        img_k, txt_k = k[:, :img_len], k[:, img_len:]
        cs.wait_stream(cur)
        img_k.record_stream(cs)
        with torch.cuda.stream(cs):
            full_img_key = self.all_gather_seq(img_k, grp, kind="kv")
        ev_k = torch.cuda.Event()
        ev_k.record(cs)

        v = produce_v()
        img_v, txt_v = v[:, :img_len], v[:, img_len:]
        cs.wait_stream(cur)
        img_v.record_stream(cs)
        with torch.cuda.stream(cs):
            full_img_value = self.all_gather_seq(img_v, grp, kind="kv")
        ev_v = torch.cuda.Event()
        ev_v.record(cs)

        q = produce_q()
        img_q, txt_q = q[:, :img_len], q[:, img_len:]
        local_img_len = img_q.shape[1]

        cur.wait_event(ev_k)
        cur.wait_event(ev_v)
        full_img_key.record_stream(cur)
        full_img_value.record_stream(cur)

        qq = torch.cat([img_q, txt_q], dim=1).contiguous()
        kk = torch.cat([full_img_key, txt_k], dim=1).contiguous()
        vv = torch.cat([full_img_value, txt_v], dim=1).contiguous()
        out = self.time(lambda: self.attn_backend(qq, kk, vv, causal=False)[0], "attn")
        return out[:, local_img_len:], out[:, :local_img_len]

    def _attn_ulysses(self, txt_q, txt_k, txt_v, img_q, img_k, img_v):
        """DeepSpeed-Ulysses: all-to-all swaps sequence-sharding for head-sharding,
        compute full-sequence attention on a head slice, then swap back."""
        grp = self._group
        cp = grp.group_size
        r = grp.rank_in_group

        # Image: [B, chunk, H, D] (seq-sharded) -> [B, full_img, H/cp, D] (head-sharded).
        img_q = self._all_to_all(img_q, scatter_dim=2, gather_dim=1)
        img_k = self._all_to_all(img_k, scatter_dim=2, gather_dim=1)
        img_v = self._all_to_all(img_v, scatter_dim=2, gather_dim=1)

        # Text replicated on every rank -> keep only this rank's head slice.
        txt_q = torch.tensor_split(txt_q, cp, dim=2)[r].contiguous()
        txt_k = torch.tensor_split(txt_k, cp, dim=2)[r].contiguous()
        txt_v = torch.tensor_split(txt_v, cp, dim=2)[r].contiguous()

        full_img_len = img_q.shape[1]
        q = torch.cat([img_q, txt_q], dim=1)
        k = torch.cat([img_k, txt_k], dim=1)
        v = torch.cat([img_v, txt_v], dim=1)
        out = self.time(lambda: self.attn_backend(q, k, v, causal=False)[0], "attn")

        img_out = out[:, :full_img_len]
        txt_out = out[:, full_img_len:]
        # Image: [B, full_img, H/cp, D] -> [B, chunk, H, D] (own chunk, full heads).
        img_out = self._all_to_all(img_out, scatter_dim=1, gather_dim=2)
        # Text: gather head slices across the CP group to rebuild all heads.
        txt_out = self._all_gather_heads(txt_out)
        return txt_out.contiguous(), img_out.contiguous()

    def _attn_ulysses_fused(
        self,
        *,
        produce_img_q,
        produce_img_k,
        produce_img_v,
        produce_txt_q,
        produce_txt_k,
        produce_txt_v,
    ):
        grp = self._group
        cp = grp.group_size
        r = grp.rank_in_group
        cur = torch.cuda.current_stream()
        cs = self._get_comm_stream()

        def a2a_on_comm(t):
            if t.shape[2] % cp != 0:
                raise ValueError(f"Ulysses CP needs heads ({t.shape[2]}) divisible by cp_size ({cp})")
            cs.wait_stream(cur)
            t.record_stream(cs)
            with torch.cuda.stream(cs):
                out = grp.all_to_all(t, scatter_dim=2, gather_dim=1)
            ev = torch.cuda.Event()
            ev.record(cs)
            return out, ev

        img_k = produce_img_k()
        a2a_k, ev_k = a2a_on_comm(img_k)

        img_v = produce_img_v()
        a2a_v, ev_v = a2a_on_comm(img_v)

        img_q = produce_img_q()
        a2a_q, ev_q = a2a_on_comm(img_q)

        txt_q = produce_txt_q()
        txt_k = produce_txt_k()
        txt_v = produce_txt_v()

        for ev in (ev_k, ev_v, ev_q):
            cur.wait_event(ev)
        for t in (a2a_k, a2a_v, a2a_q):
            t.record_stream(cur)

        txt_q = torch.tensor_split(txt_q, cp, dim=2)[r].contiguous()
        txt_k = torch.tensor_split(txt_k, cp, dim=2)[r].contiguous()
        txt_v = torch.tensor_split(txt_v, cp, dim=2)[r].contiguous()

        full_img_len = a2a_q.shape[1]
        q = torch.cat([a2a_q, txt_q], dim=1)
        k = torch.cat([a2a_k, txt_k], dim=1)
        v = torch.cat([a2a_v, txt_v], dim=1)
        out = self.time(lambda: self.attn_backend(q, k, v, causal=False)[0], "attn")

        img_out = out[:, :full_img_len]
        txt_out = out[:, full_img_len:]
        img_out = self._all_to_all(img_out, scatter_dim=1, gather_dim=2)
        txt_out = self._all_gather_heads(txt_out)
        return txt_out.contiguous(), img_out.contiguous()

    def _attn_ulysses_packed_fused(self, img_len, produce_q, produce_k, produce_v):
        grp = self._group
        cp = grp.group_size
        r = grp.rank_in_group
        cur = torch.cuda.current_stream()
        cs = self._get_comm_stream()

        def a2a_on_comm(t):
            if t.shape[2] % cp != 0:
                raise ValueError(f"Ulysses CP needs heads ({t.shape[2]}) divisible by cp_size ({cp})")
            cs.wait_stream(cur)
            t.record_stream(cs)
            with torch.cuda.stream(cs):
                out = grp.all_to_all(t, scatter_dim=2, gather_dim=1)
            ev = torch.cuda.Event()
            ev.record(cs)
            return out, ev

        k = produce_k()
        img_k, txt_k = k[:, :img_len], k[:, img_len:]
        a2a_k, ev_k = a2a_on_comm(img_k)

        v = produce_v()
        img_v, txt_v = v[:, :img_len], v[:, img_len:]
        a2a_v, ev_v = a2a_on_comm(img_v)

        q = produce_q()
        img_q, txt_q = q[:, :img_len], q[:, img_len:]
        a2a_q, ev_q = a2a_on_comm(img_q)

        for ev in (ev_k, ev_v, ev_q):
            cur.wait_event(ev)
        for t in (a2a_k, a2a_v, a2a_q):
            t.record_stream(cur)

        txt_q = torch.tensor_split(txt_q, cp, dim=2)[r].contiguous()
        txt_k = torch.tensor_split(txt_k, cp, dim=2)[r].contiguous()
        txt_v = torch.tensor_split(txt_v, cp, dim=2)[r].contiguous()

        full_img_len = a2a_q.shape[1]
        qq = torch.cat([a2a_q, txt_q], dim=1)
        kk = torch.cat([a2a_k, txt_k], dim=1)
        vv = torch.cat([a2a_v, txt_v], dim=1)
        out = self.time(lambda: self.attn_backend(qq, kk, vv, causal=False)[0], "attn")

        img_out = out[:, :full_img_len]
        txt_out = out[:, full_img_len:]
        img_out = self._all_to_all(img_out, scatter_dim=1, gather_dim=2)
        txt_out = self._all_gather_heads(txt_out)
        return txt_out.contiguous(), img_out.contiguous()

    def _attn_ring(self, txt_q, txt_k, txt_v, img_q, img_k, img_v):
        """Ring attention: stream image K/V chunks around the CP ring, merging
        partial outputs with online softmax. Replicated text K/V joins step 0."""
        grp = self._group
        cp = grp.group_size
        r = grp.rank_in_group
        txt_len = txt_q.shape[1]

        cur_k = img_k.contiguous()
        cur_v = img_v.contiguous()

        # Fixed local queries: [text, image_chunk]. Only K/V rotate.
        q = torch.cat([txt_q, img_q], dim=1).contiguous()
        dst_idx = (r - 1) % cp  # send to the previous rank
        src_idx = (r + 1) % cp  # receive from the next rank
        recv_bytes_each = (
            cur_k.numel() * cur_k.element_size() + cur_v.numel() * cur_v.element_size()
        )

        out = lse = None
        for step in range(cp):
            need_comm = step + 1 != cp
            recv = None
            received = None
            if need_comm:
                with self._mark("cp_ring_commit"):
                    recv = self._ring_send_recv_commit((cur_k, cur_v), src_idx, dst_idx)
                if self.profile:
                    # Profile mode serializes comm for a clean, comparable bucket.
                    received = self._comm_time(lambda: self._ring_wait(recv), "ring", recv_bytes_each)
                elif self._no_overlap:
                    # Diagnostic: wait before computing -> no compute/comm overlap.
                    received = self._ring_wait(recv)

            if step == 0:
                block_k = torch.cat([txt_k, cur_k], dim=1)
                block_v = torch.cat([txt_v, cur_v], dim=1)
            else:
                block_k = cur_k
                block_v = cur_v

            with self._mark("cp_ring_attn"):
                block_out, block_lse = self.time(
                    lambda bk=block_k, bv=block_v: self.attn_backend(
                        q, bk, bv, causal=False, return_attn_probs=True
                    )[:2],
                    "attn",
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            if need_comm:
                if received is not None:
                    cur_k, cur_v = received
                else:
                    with self._mark("cp_ring_wait"):
                        cur_k, cur_v = self._ring_wait(recv)

        out = out.to(q.dtype)
        txt_out = out[:, :txt_len]
        img_out = out[:, txt_len:]
        return txt_out.contiguous(), img_out.contiguous()

    def _attn_unified(self, txt_q, txt_k, txt_v, img_q, img_k, img_v):
        """2D Ulysses x Ring (USP). Ulysses all-to-all swaps seq-sharding for
        head-sharding *within* each ``up``-sized sub-group; the residual
        ``ring_steps = cp / up`` image K/V chunks are then streamed P2P *across*
        the sub-groups with online-softmax merge. Falls back to pure ring when
        ``up`` does not divide cp_size. Layout-identical to every other mode.
        """
        grp = self._group
        cp = grp.group_size
        local_rank = grp.rank_in_group

        ulysses_size = self.ulysses_size
        if ulysses_size > cp:
            ulysses_size = cp
        if cp % ulysses_size != 0:
            if not self._mode_warned:
                logger.warning(
                    "Unified CP needs up (%d) to divide cp_size (%d); falling back to pure ring.",
                    ulysses_size, cp,
                )
                self._mode_warned = True
            ulysses_size = 1
        ring_steps = cp // ulysses_size
        seq_dim, head_dim = 1, 2

        # Full image length BEFORE the Ulysses all-to-all (each rank holds 1/cp).
        img_seq_len = int(img_q.shape[1] * cp)

        ulysses_group = None
        if ulysses_size > 1:
            ulysses_group = get_up_group(ulysses_size)
            u_rank = ulysses_group.rank_in_group
            # Text replicated everywhere -> keep only this rank's head slice.
            txt_q = torch.tensor_split(txt_q, ulysses_size, head_dim)[u_rank].contiguous()
            txt_k = torch.tensor_split(txt_k, ulysses_size, head_dim)[u_rank].contiguous()
            txt_v = torch.tensor_split(txt_v, ulysses_size, head_dim)[u_rank].contiguous()
            # Image: seq-sharded -> head-sharded within the sub-group. Q stays
            # bf16; K/V optionally go fp8 to halve the all-to-all bytes (the
            # unified bottleneck) when CHITU_CP_FP8_KV is set.
            img_q = self._up_all_to_all(ulysses_group, img_q, head_dim, seq_dim)
            a2a_kv = self._up_all_to_all_fp8 if self._fp8_kv else self._up_all_to_all
            img_k = a2a_kv(ulysses_group, img_k, head_dim, seq_dim)
            img_v = a2a_kv(ulysses_group, img_v, head_dim, seq_dim)

        original_local_img_len = img_q.shape[1] // ulysses_size if ulysses_size > 1 else img_q.shape[1]
        ring_block_seq = img_k.shape[1]
        local_img_len = img_q.shape[1]
        txt_len = txt_q.shape[1]
        q = torch.cat([txt_q, img_q], dim=1).contiguous()

        ring_next = (local_rank - ulysses_size) % cp  # send to
        ring_prev = (local_rank + ulysses_size) % cp  # recv from
        ring_block = local_rank // ulysses_size

        cur_k, cur_v = img_k.contiguous(), img_v.contiguous()
        if self._fp8_kv:
            recv_bytes_each = cur_k.numel() + cur_v.numel()  # fp8: 1 byte/elem
        else:
            recv_bytes_each = cur_k.numel() * cur_k.element_size() + cur_v.numel() * cur_v.element_size()
        out = lse = None
        for ring_step in range(ring_steps):
            owner_block = (ring_block + ring_step) % ring_steps
            start = owner_block * ring_block_seq
            valid_len = max(0, min(cur_k.shape[1], img_seq_len - start))

            need_comm = ring_step + 1 != ring_steps
            recv = None
            if need_comm:
                if self._fp8_kv:
                    # Stream the K/V chunk as fp8 bytes + per-head scales; the
                    # receiver dequantizes with the sender's scale (P2P, so the
                    # scale travels with the data -> no global reduction needed).
                    k_u8, k_sc = _quantize_fp8_per_head(cur_k)
                    v_u8, v_sc = _quantize_fp8_per_head(cur_v)
                    recv = self._ring_send_recv_commit((k_u8, k_sc, v_u8, v_sc), ring_prev, ring_next)
                else:
                    recv = self._ring_send_recv_commit((cur_k, cur_v), ring_prev, ring_next)
                if self.profile:
                    recv = self._comm_time(lambda: self._ring_wait(recv), "ring", recv_bytes_each)

            if ring_step == 0:
                block_k = torch.cat([txt_k, cur_k[:, :valid_len]], dim=1)
                block_v = torch.cat([txt_v, cur_v[:, :valid_len]], dim=1)
            else:
                block_k = cur_k[:, :valid_len]
                block_v = cur_v[:, :valid_len]

            if block_k.shape[1] > 0:
                block_out, block_lse = self.time(
                    lambda bk=block_k, bv=block_v: self.attn_backend(
                        q, bk, bv, causal=False, return_attn_probs=ring_steps > 1
                    )[:2],
                    "attn",
                )
                if ring_steps == 1:
                    out, lse = block_out, None
                else:
                    out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            if need_comm:
                got = recv if self.profile else self._ring_wait(recv)
                if self._fp8_kv:
                    k_u8, k_sc, v_u8, v_sc = got
                    cur_k = _dequantize_fp8(k_u8, k_sc, q.dtype)
                    cur_v = _dequantize_fp8(v_u8, v_sc, q.dtype)
                else:
                    cur_k, cur_v = got

        if out is None:
            raise RuntimeError("Unified CP attention produced no output blocks.")
        if lse is not None:
            out = out.to(q.dtype)
        txt_out = out[:, :txt_len]
        img_out = out[:, txt_len:txt_len + local_img_len]
        if ulysses_size > 1:
            # Image: head-sharded -> seq-sharded, keep this rank's original chunk.
            img_out = self._up_all_to_all(ulysses_group, img_out, seq_dim, head_dim)
            img_out = img_out[:, :original_local_img_len]
            # Text: gather head slices across the sub-group to rebuild all heads.
            txt_out = self._up_all_gather_heads(ulysses_group, txt_out)
        return txt_out.contiguous(), img_out.contiguous()

    def _up_all_to_all(self, up_group, tensor: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
        up = up_group.group_size
        recv_bytes = int(tensor.numel() * tensor.element_size() * (up - 1) / up)
        return self._comm_time(
            lambda: up_group.all_to_all(tensor, scatter_dim, gather_dim), "a2a", recv_bytes
        )

    def _up_all_to_all_fp8(self, up_group, tensor: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
        """fp8 variant of :meth:`_up_all_to_all` for image K/V (scatter heads,
        gather seq) in the unified mode.

        Uses a **global per-head scale** (amax max-reduced across the up-group)
        so every gathered sequence position dequantizes with the same per-head
        multiplier. That makes correctness independent of how ``all_to_all``
        orders the received seq blocks (the local-scale alternative would need
        per-source-block scale bookkeeping tied to that ordering). The extra
        all-reduce moves only ``H`` fp32 scalars; the a2a payload is halved.
        """
        up = up_group.group_size
        out_dtype = tensor.dtype
        amax = tensor.abs().to(torch.float32).amax(dim=(0, 1, 3), keepdim=True).clamp_min(1e-12)  # [1,1,H,1]
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=up_group.gpu_group)
        scale = amax / _FP8_MAX
        x_u8 = (
            (tensor.to(torch.float32) / scale)
            .clamp_(-_FP8_MAX, _FP8_MAX)
            .to(_FP8_DTYPE)
            .view(torch.uint8)
            .contiguous()
        )
        recv_bytes = int(x_u8.numel() * (up - 1) / up)  # fp8: 1 byte/elem
        out_u8 = self._comm_time(
            lambda: up_group.all_to_all(x_u8, scatter_dim, gather_dim), "a2a", recv_bytes
        )
        # This rank now owns head chunk r of size H/up; slice the matching global
        # per-head scale (identical on all ranks) and dequantize uniformly.
        heads = int(tensor.shape[2])
        hsub = heads // up
        r = up_group.rank_in_group
        sc = scale[:, :, r * hsub:(r + 1) * hsub, :]
        return _dequantize_fp8(out_u8, sc, out_dtype)

    def _up_all_gather_heads(self, up_group, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.contiguous()
        pieces = [torch.empty_like(tensor) for _ in range(up_group.group_size)]
        recv_bytes = tensor.numel() * tensor.element_size() * (up_group.group_size - 1)
        self._comm_time(
            lambda: dist.all_gather(pieces, tensor, group=up_group.gpu_group), "a2a", recv_bytes
        )
        return torch.cat(pieces, dim=2).contiguous()

    def _attn_ring_graphed(self, txt_q, txt_k, txt_v, img_q, img_k, img_v):
        """Ring attention whose entire loop is captured into a CUDA Graph.

        Keyed by (txt/img q,k,v) shape/dtype; one capture serves every layer and
        denoise step (all share one instance and identical shapes). Inputs are
        copied into static buffers before replay; the static outputs are cloned
        out so the next replay can safely overwrite them. The captured region is
        exactly the eager ``_attn_ring`` (capture-safe when profiling is off), so
        the result is bit-identical to eager ring.
        """
        cache_key = (
            tuple(txt_q.shape), tuple(txt_k.shape), tuple(txt_v.shape),
            tuple(img_q.shape), tuple(img_k.shape), tuple(img_v.shape),
            img_q.dtype,
        )
        entry = self._ring_graph_cache.get(cache_key)
        if entry is None:
            entry = self._capture_ring_graph(txt_q, txt_k, txt_v, img_q, img_k, img_v)
            self._ring_graph_cache[cache_key] = entry
        entry["txt_q"].copy_(txt_q)
        entry["txt_k"].copy_(txt_k)
        entry["txt_v"].copy_(txt_v)
        entry["img_q"].copy_(img_q)
        entry["img_k"].copy_(img_k)
        entry["img_v"].copy_(img_v)
        entry["graph"].replay()
        return entry["txt_out"].clone(), entry["img_out"].clone()

    def _capture_ring_graph(self, txt_q, txt_k, txt_v, img_q, img_k, img_v) -> dict:
        s_txt_q, s_txt_k, s_txt_v = txt_q.clone(), txt_k.clone(), txt_v.clone()
        s_img_q, s_img_k, s_img_v = img_q.clone(), img_k.clone(), img_v.clone()

        # Warm up on a side stream first: required before capture, and it also
        # establishes the NCCL P2P connections so they are not (illegally)
        # created during graph capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._attn_ring(s_txt_q, s_txt_k, s_txt_v, s_img_q, s_img_k, s_img_v)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            s_txt_out, s_img_out = self._attn_ring(
                s_txt_q, s_txt_k, s_txt_v, s_img_q, s_img_k, s_img_v
            )
        torch.cuda.synchronize()

        logger.info(
            "[image-cp ring-cudagraph] captured ring loop for img_q=%s (cp_size=%d). "
            "Set NCCL_GRAPH_MIXING_SUPPORT=1 if other NCCL collectives share the process.",
            tuple(img_q.shape),
            self._group.group_size,
        )
        return {
            "graph": graph,
            "txt_q": s_txt_q, "txt_k": s_txt_k, "txt_v": s_txt_v,
            "img_q": s_img_q, "img_k": s_img_k, "img_v": s_img_v,
            "txt_out": s_txt_out, "img_out": s_img_out,
        }


def resolve_cp_mode(env_var: str = "CHITU_Z_IMAGE_CP_MODE", default: str = "agkv") -> str:
    """Read and validate a CP-mode env var (kept here so callers stay terse)."""
    mode = os.environ.get(env_var, default).strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"{env_var} must be one of {sorted(_VALID_MODES)}, got {mode!r}."
        )
    return mode
