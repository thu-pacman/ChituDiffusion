"""Metadata-only CP attention planning with calibrated kernel wave costs."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..profiles.h20_nvlink import CUDNN_COST as _CUDNN_COST
from ..profiles.h20_nvlink import KERNEL_COSTS as _KERNEL_COSTS
from .cute import load_cute_flash_attn_fwd


@dataclass(frozen=True)
class AttentionPlan:
    """Immutable launch configuration, containing no tensors or graph state."""

    strategy: str = "sdpa"
    prefix: int = 0
    main_tile: tuple[int, int] = (128, 64)
    tail_tile: tuple[int, int] = (64, 64)
    tail_splits: int = 1
    merge: str = "cute_fp32"
    main_overlap: bool = True


@dataclass(frozen=True)
class AttentionCandidate:
    plan: AttentionPlan
    estimated_us: float


# H20 BF16 D128 coefficients from full-wave primitive calibration (job 233877).
# Each row is effective intercept_us, startup_us_per_wave, us_per_KV_token_per_wave.
# These are kernel costs, not shape/CP winners. See attention-planner-calibration.json.


def _kernel_us(queries, keys, heads, batch, sms, tile, coefficient):
    m, n = tile
    waves = math.ceil(math.ceil(queries / m) * heads * batch / sms)
    launch, startup, scan = coefficient
    return launch + waves * (startup + scan * math.ceil(keys / n) * n)


def _estimate(plan, queries, keys, heads, batch, sms):
    if plan.strategy == "cudnn":
        return _kernel_us(queries, keys, heads, batch, sms, (128, 128), _CUDNN_COST)
    coefficient = _KERNEL_COSTS[(*plan.main_tile, plan.main_overlap)]
    if plan.strategy == "cute":
        return _kernel_us(queries, keys, heads, batch, sms, plan.main_tile, coefficient)
    main_q, main_h, tail_q, tail_h = queries, heads, queries, heads
    if plan.strategy == "head_tail":
        main_h, tail_h = plan.prefix, heads - plan.prefix
    else:
        main_q, tail_q = plan.prefix, queries - plan.prefix
    elapsed = _kernel_us(main_q, keys, main_h, batch, sms, plan.main_tile, coefficient)
    splits = plan.tail_splits
    elapsed += _kernel_us(
        tail_q,
        keys // splits,
        tail_h,
        splits,
        sms,
        plan.tail_tile,
        _KERNEL_COSTS[(*plan.tail_tile, True)],
    )
    if splits > 1:
        # BF16 -> FP32 conversion plus FP32 combine; include both launches and
        # traffic using conservative effective bandwidth of 1.5 TB/s.
        merge_bytes = tail_q * tail_h * 128 * (10 * splits + 2)
        elapsed += 4.0 + merge_bytes / 1_500_000
    return elapsed


@lru_cache(maxsize=512)
def rank_attention_plans(
    *,
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    dtype: torch.dtype,
    device_name: str,
    capability: tuple[int, int],
    num_sms: int,
    cp_degree: int,
    contiguous: bool = True,
) -> tuple[AttentionCandidate, ...]:
    """Generate legal plans and rank their complete estimated graph cost.

    Local shape determines work. CP degree is metadata, not a scheduling rule.
    num_sms is the usable budget, which the caller must enforce separately.
    Only H20 has calibrated costs; other architectures use ordinary SDPA.
    """
    if cp_degree < 1:
        raise ValueError("cp_degree must be positive")
    if (
        device_name not in {"H20", "NVIDIA H20"}
        or capability != (9, 0)
        or not 0 < num_sms <= 78
        or dtype != torch.bfloat16
        or not contiguous
        or len(query_shape) != 4
        or len(key_shape) != 4
        or key_shape != value_shape
        or query_shape[0] != key_shape[0]
        or query_shape[2:] != key_shape[2:]
        or query_shape[-1] != 128
        or min(*query_shape, *key_shape) <= 0
    ):
        return (AttentionCandidate(AttentionPlan(), math.inf),)
    batch, queries, heads, _ = query_shape
    keys = key_shape[1]
    plans = [AttentionPlan(strategy="cudnn")]
    for m, n, overlap in _KERNEL_COSTS:
        plans.append(
            AttentionPlan(strategy="cute", main_tile=(m, n), main_overlap=overlap)
        )
        if batch != 1:
            continue
        q_tiles = math.ceil(queries / m)
        full_wave_tasks = (math.ceil(q_tiles * heads / num_sms) - 1) * num_sms
        boundaries = {
            "head_tail": full_wave_tasks // q_tiles,
            "query_tail": (full_wave_tasks // heads) * m,
        }
        for strategy, prefix in boundaries.items():
            limit = heads if strategy == "head_tail" else queries
            if not 0 < prefix < limit:
                continue
            for splits in (1, 2, 4, 8, 16, 32):
                if keys % splits or (splits > 1 and keys // splits < 256):
                    continue
                plans.append(
                    AttentionPlan(
                        strategy=strategy,
                        prefix=prefix,
                        main_tile=(m, n),
                        main_overlap=overlap,
                        tail_splits=splits,
                    )
                )
    candidates = [
        AttentionCandidate(p, _estimate(p, queries, keys, heads, batch, num_sms))
        for p in plans
    ]
    candidates.sort(key=lambda c: c.estimated_us)
    # Prefer a single kernel when a composition's modeled gain is below 1%.
    ordinary = min(
        (c for c in candidates if c.plan.strategy in {"cudnn", "cute"}),
        key=lambda c: c.estimated_us,
    )
    if candidates[0].estimated_us >= ordinary.estimated_us * 0.99:
        candidates.remove(ordinary)
        candidates.insert(0, ordinary)
    return tuple(candidates)


def select_attention_plan(**metadata) -> AttentionPlan:
    return rank_attention_plans(**metadata)[0].plan


def _sdpa(query, key, value):
    unaligned = query.is_cuda and any(t.data_ptr() % 16 for t in (query, key, value))
    # The installed fused SDPA also fails the offset-view correctness control.
    with sdpa_kernel(SDPBackend.MATH) if unaligned else nullcontext():
        return (
            F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                is_causal=False,
                enable_gqa=query.shape[2] != key.shape[2],
            )
            .transpose(1, 2)
            .contiguous()
        )


def _execute(plan, forward, query, key, value):
    if plan.strategy == "sdpa":
        return _sdpa(query, key, value)
    if plan.strategy == "cudnn":
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return _sdpa(query, key, value)
    if plan.strategy == "cute":
        return forward(
            query,
            key,
            value,
            causal=False,
            tile_mn=plan.main_tile,
            intra_wg_overlap=plan.main_overlap,
        )[0]
    output = torch.empty_like(query)
    axis = 2 if plan.strategy == "head_tail" else 1
    prefix_slice = [slice(None)] * 4
    tail_slice = prefix_slice.copy()
    prefix_slice[axis] = slice(None, plan.prefix)
    tail_slice[axis] = slice(plan.prefix, None)
    prefix_slice, tail_slice = tuple(prefix_slice), tuple(tail_slice)
    main_key, main_value = key, value
    tail_key, tail_value = key, value
    if axis == 2:
        main_key, main_value = key[prefix_slice], value[prefix_slice]
        tail_key, tail_value = key[tail_slice], value[tail_slice]
    forward(
        query[prefix_slice],
        main_key,
        main_value,
        causal=False,
        out=output[prefix_slice],
        tile_mn=plan.main_tile,
        intra_wg_overlap=plan.main_overlap,
    )
    splits = plan.tail_splits
    if splits == 1:
        forward(
            query[tail_slice],
            tail_key,
            tail_value,
            causal=False,
            out=output[tail_slice],
            tile_mn=plan.tail_tile,
            intra_wg_overlap=True,
        )
        return output
    qs = query[tail_slice].expand(splits, -1, -1, -1)
    ks, vs = [
        t.view(splits, t.shape[1] // splits, *t.shape[2:])
        for t in (tail_key, tail_value)
    ]
    partial, lse = forward(
        qs,
        ks,
        vs,
        causal=False,
        return_lse=True,
        tile_mn=plan.tail_tile,
        intra_wg_overlap=True,
    )[:2]
    if plan.merge == "cute_fp32":
        from flash_attn.cute.interface import flash_attn_combine

        flash_attn_combine(
            partial.float().unsqueeze(1),
            lse.transpose(-1, -2).unsqueeze(1),
            out=output[tail_slice],
            return_lse=False,
        )
    else:
        from .merge import fused_merge_flash_partials

        running, running_lse = partial[:1].float(), lse[:1].clone()
        for index in range(1, splits):
            if not fused_merge_flash_partials(
                running, running_lse, partial[index : index + 1], lse[index : index + 1]
            ):
                raise RuntimeError("Selected FP32 merge rejected the tensor layout")
        output[tail_slice].copy_(running)
    return output


def _signature(inputs):
    return tuple(
        (
            t.device,
            t.dtype,
            tuple(t.shape),
            tuple(t.stride()),
            t.data_ptr(),
            t.requires_grad,
        )
        for t in inputs
    )


def _layout_signature(inputs):
    return tuple(
        (
            t.device,
            t.dtype,
            tuple(t.shape),
            tuple(t.stride()),
            t.requires_grad,
            t.data_ptr() % 16,
        )
        for t in inputs
    )


class PreparedAttention:
    """One serial execution slot; construct with :func:`prepare_attention`.

    Producer writes must precede this call on the current stream. Fast output
    is reusable scratch: consume it before the next invocation. Concurrent
    streams require separate slots. Changed addresses/layout/shapes use SDPA
    without input copies; prepare another slot to optimize the new buffers.
    """

    def __init__(self, inputs, cp_degree, available_sms=None):
        self.inputs = tuple(inputs)
        self.signature = _signature(inputs)
        self.layout_signature = _layout_signature(inputs)
        self.plan = AttentionPlan()
        self.graph = None
        self.output = None
        self.forward = None
        self.replays = 0
        self.fallbacks = 0
        query, key, value = inputs
        eligible = (
            query.is_cuda
            and all(t.device == query.device for t in inputs)
            and all(t.dtype == query.dtype for t in inputs)
            and not any(t.requires_grad for t in inputs)
            and all(t.data_ptr() % 16 == 0 for t in inputs)
        )
        if cp_degree < 1:
            raise ValueError("cp_degree must be positive")
        if available_sms is not None and available_sms < 1:
            raise ValueError("available_sms must be positive")
        if eligible:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare attention before CUDA graph capture")
            props = torch.cuda.get_device_properties(query.device)
            if (
                available_sms is not None
                and available_sms > props.multi_processor_count
            ):
                raise ValueError("available_sms exceeds the device SM count")
            self.plan = select_attention_plan(
                query_shape=tuple(query.shape),
                key_shape=tuple(key.shape),
                value_shape=tuple(value.shape),
                dtype=query.dtype,
                device_name=props.name,
                capability=(props.major, props.minor),
                num_sms=props.multi_processor_count
                if available_sms is None
                else available_sms,
                cp_degree=cp_degree,
                contiguous=all(t.is_contiguous() for t in inputs),
            )
        if self.plan.strategy == "sdpa":
            return
        if self.plan.strategy != "cudnn":
            try:
                self.forward = load_cute_flash_attn_fwd()
            except (ImportError, OSError):
                self.plan = AttentionPlan()
                return
        with torch.cuda.device(query.device):
            stream = torch.cuda.Stream(device=query.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _execute(self.plan, self.forward, *inputs)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = _execute(self.plan, self.forward, *inputs)

    def __call__(self, query, key, value):
        inputs = (query, key, value)
        if self.graph is None:
            self.fallbacks += 1
            return _sdpa(*inputs)
        if _signature(inputs) != self.signature:
            if _layout_signature(inputs) == self.layout_signature:
                return _execute(self.plan, self.forward, *inputs)
            self.fallbacks += 1
            return _sdpa(*inputs)
        if torch.cuda.current_device() == query.device.index:
            return self._replay(inputs)
        with torch.cuda.device(query.device):
            return self._replay(inputs)

    def _replay(self, inputs):
        if torch.cuda.is_current_stream_capturing():
            return _execute(self.plan, self.forward, *inputs)
        self.replays += 1
        self.graph.replay()
        return self.output


def prepare_attention(
    query, key, value, *, cp_degree: int = 1, available_sms: int | None = None
) -> PreparedAttention:
    """Prepare noncausal, dropout-free BSND attention for already-arrived KV.

    Selection uses local tensor metadata, GPU model/SM count and CP degree.
    It performs no timing or collective communication. Prepare after the
    producer allocates stable buffers, outside CUDA graph capture.

    available_sms describes an externally enforced SM budget; it does not
    reserve SMs. The default assumes the complete device is available.
    """
    return PreparedAttention((query, key, value), cp_degree, available_sms)
