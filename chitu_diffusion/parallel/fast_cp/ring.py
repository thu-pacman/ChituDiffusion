"""Low-memory, transport-agnostic Flash Ring attention compute.

The production path intentionally keeps communication out of this module.  A
transport session exposes exactly one K/V shard per phase; this compute side
runs reviewed CuTe FlashAttention for that shard and merges the returned
partial output/LSE into FP32 running state in HBM.

Q is read again by each phase, and running O/LSE state crosses phase boundaries
through HBM.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from ._cute import load_cute_flash_attn_fwd
from ._ring_merge import fused_merge_flash_partials


@dataclass(frozen=True)
class RingKVPhase:
    """One ready K/V shard leased from a ring transport session."""

    index: int
    source_rank: int
    key: torch.Tensor
    value: torch.Tensor


class RingKVSession(Protocol):
    """Per-call transport state.

    ``acquire`` may block or enqueue a stream wait.  The shard remains valid
    until ``release`` returns.  Implementations can therefore rotate one or two
    receive buffers without materializing the global K/V sequence.

    Phase zero never leaves the local rank, so ``begin`` must not submit any
    transfer.  The compute side calls ``start`` once phase-zero attention is
    queued, which is what lets the first hop overlap it instead of preceding it
    on the host.
    """

    phase_count: int

    def start(self) -> None: ...

    def acquire(self, phase_index: int) -> RingKVPhase: ...

    def release(self, phase: RingKVPhase) -> None: ...

    def close(self) -> None: ...


class RingKVTransport(Protocol):
    """Injectable communication side of Flash Ring attention."""

    rank: int
    world_size: int

    def begin(self, key: torch.Tensor, value: torch.Tensor) -> RingKVSession: ...


class _FastCERingSession:
    """Adapt the low-level CE ticket protocol to the compute lease contract."""

    def __init__(
        self,
        runtime: object,
        launch: Callable[[], object],
        *,
        rank: int,
        world_size: int,
        local_kv: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self._runtime = runtime
        self._launch = launch
        self._state: object | None = None
        self._local_kv = local_kv
        self._rank = rank
        self.phase_count = world_size
        self._next_acquire = 0
        self._next_release = 0
        self._closed = False

    def start(self) -> None:
        """Submit the ring traversal. Idempotent, and a no-op after the first call."""
        if self._closed:
            raise RuntimeError("Flash Ring session is closed")
        if self._state is None:
            self._state = self._launch()

    def acquire(self, phase_index: int) -> RingKVPhase:
        if self._closed:
            raise RuntimeError("Flash Ring session is closed")
        if phase_index != self._next_acquire or phase_index != self._next_release:
            raise RuntimeError("Flash Ring phases must be acquired and released in order")
        if phase_index == 0:
            # Phase zero is this rank's own K/V, so it is available before the
            # transport has been submitted at all.
            key, value = self._local_kv
        else:
            if self._state is None:
                raise RuntimeError(
                    "Flash Ring phases beyond zero require a started session"
                )
            key, value = self._runtime.wait_ring_ce_ready(
                self._state,
                phase=phase_index,
            )
            # The CE forwarding chain was preposted by start(). Each hop waits
            # on its arrival ticket and advances without a host call here.
        self._next_acquire += 1
        return RingKVPhase(
            index=phase_index,
            source_rank=(self._rank - phase_index) % self.phase_count,
            key=key,
            value=value,
        )

    def release(self, phase: RingKVPhase) -> None:
        if self._closed:
            raise RuntimeError("Flash Ring session is closed")
        if phase.index != self._next_release or phase.index >= self._next_acquire:
            raise RuntimeError("Flash Ring phase release is out of order")
        if phase.index > 0:
            assert self._state is not None
            self._runtime.publish_ring_ce_consumed(
                self._state,
                phase=phase.index,
            )
        self._next_release += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._next_release != self.phase_count:
            raise RuntimeError(
                "cannot cancel an in-flight Flash Ring CE traversal safely"
            )


class FastRingKVTransport:
    """Production CE ring adapter backed by ``FastUlyssesAllToAll``.

    The two symmetric buffers are persistent in the Fast Ulysses pool. Epoch
    one is also the coordinated flag-reset epoch, so int32 ticket wrap remains
    safe as long as every rank enters calls in the same order.
    """

    def __init__(
        self,
        runtime: object,
        *,
        rank: int,
        world_size: int,
        tag_prefix: str = "chitu_flash_ring",
    ) -> None:
        if world_size not in {2, 4, 8}:
            raise ValueError("Flash Ring CE supports CP2, CP4, or CP8")
        if not 0 <= rank < world_size:
            raise ValueError("Flash Ring CE rank is outside its world")
        self._runtime = runtime
        self.rank = rank
        self.world_size = world_size
        self._tag_prefix = tag_prefix
        self._epoch = 0
        self._max_epoch = ((2**31 - 1) - (world_size - 1)) // (
            world_size - 1
        ) + 1

    def begin(self, key: torch.Tensor, value: torch.Tensor) -> RingKVSession:
        if self._epoch >= self._max_epoch:
            # Drain all CE/ack streams before epoch one clears shared flags.
            # The C++ prepare then executes a rank-wide symmetric barrier.
            torch.cuda.synchronize(key.device)
            self._epoch = 1
        else:
            self._epoch += 1
        epoch = self._epoch
        # Recorded before any phase-zero work reaches the stream. The deferred
        # launch hands it to the comm stream, which would otherwise wait for
        # everything queued by the time the traversal is actually submitted.
        ready_event: torch.cuda.Event | None = None
        if key.is_cuda:
            ready_event = torch.cuda.Event()
            ready_event.record()

        def launch() -> object:
            return self._runtime.begin_ring_ce_transport(
                key,
                value,
                key_tag=f"{self._tag_prefix}_key",
                value_tag=f"{self._tag_prefix}_value",
                state_tag=f"{self._tag_prefix}_state",
                epoch=epoch,
                ready_event=ready_event,
            )

        return _FastCERingSession(
            self._runtime,
            launch,
            rank=self.rank,
            world_size=self.world_size,
            local_kv=(key, value),
        )


def _lse_output_view(lse: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Convert canonical ``[B, H, Sq]`` LSE to ``[B, Sq, H, 1]``."""

    expected = (output.shape[0], output.shape[2], output.shape[1])
    if tuple(lse.shape) != expected:
        raise ValueError(
            f"partial LSE must have shape [B, H, Sq] = {expected}, "
            f"got {tuple(lse.shape)}"
        )
    return lse.transpose(1, 2).unsqueeze(-1)


def merge_flash_partials_fp32(
    running_output: torch.Tensor | None,
    running_lse: torch.Tensor | None,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge normalized attention partials with an FP32 online softmax update.

    LSE is canonical ``[B, H, Sq]`` and output is BSHD.  Both returned running
    tensors are FP32 and are suitable for storage in HBM between phases.
    """

    if partial_output.ndim != 4:
        raise ValueError("partial output must use BSHD layout")
    if partial_lse.dtype != torch.float32:
        partial_lse = partial_lse.float()
    _lse_output_view(partial_lse, partial_output)

    if running_output is None or running_lse is None:
        # Copy rather than alias: the partials live in reusable phase scratch.
        initial_output = (
            partial_output.clone()
            if partial_output.dtype == torch.float32
            else partial_output.float()
        )
        return initial_output, partial_lse.clone()
    if running_output.dtype != torch.float32 or running_lse.dtype != torch.float32:
        raise ValueError("running output and LSE must be FP32")
    if running_output.shape != partial_output.shape:
        raise ValueError("all phase outputs must have the same shape")
    if running_lse.shape != partial_lse.shape:
        raise ValueError("all phase LSE tensors must have the same shape")

    if fused_merge_flash_partials(
        running_output, running_lse, partial_output, partial_lse
    ):
        return running_output, running_lse

    merged_lse = torch.logaddexp(running_lse, partial_lse)
    running_weight = torch.exp(running_lse - merged_lse)
    partial_weight = torch.exp(partial_lse - merged_lse)
    running_output.mul_(_lse_output_view(running_weight, running_output))
    running_output.add_(
        partial_output.float() * _lse_output_view(partial_weight, partial_output)
    )
    return running_output, merged_lse


def _accepted_kwarg_names(function: object) -> frozenset[str] | None:
    """Return accepted keyword names, or ``None`` when the callable takes ``**kwargs``."""

    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return None
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return None
    return frozenset(parameters)


def _supported_kwargs(function: object, kwargs: dict[str, object]) -> dict[str, object]:
    accepted = _accepted_kwarg_names(function)
    if accepted is None:
        return kwargs
    return {name: value for name, value in kwargs.items() if name in accepted}


class FastRingAttention:
    """Non-causal FP16/BF16 Flash Ring compute for CP2/4/8."""

    def __init__(
        self,
        transport: RingKVTransport,
        *,
        flash_attn_fwd: object | None = None,
    ) -> None:
        if transport.world_size not in {2, 4, 8}:
            raise ValueError("Flash Ring attention supports CP2, CP4, or CP8")
        if not 0 <= transport.rank < transport.world_size:
            raise ValueError("transport rank is outside its world")
        self.transport = transport
        self._flash_attn_fwd = flash_attn_fwd
        self._accepted_kwargs: frozenset[str] | None = None
        self._accepted_kwargs_resolved = False
        self._partial_buffers: tuple[torch.Tensor, torch.Tensor] | None = None
        # A ring pays one attention dispatch per phase, so the entry cost counts
        # as much as the kernel.  cuDNN is cheaper on both and is only declined
        # when a caller injects its own forward or the backend rejects a shape.
        self._use_cudnn = flash_attn_fwd is None and hasattr(
            torch.ops.aten, "_scaled_dot_product_cudnn_attention"
        )

    @staticmethod
    def supports(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> bool:
        return (
            query.ndim == key.ndim == value.ndim == 4
            and query.shape[0] == key.shape[0] == value.shape[0]
            and query.shape[2:] == key.shape[2:] == value.shape[2:]
            and query.shape[0] > 0
            and query.shape[1] > 0
            and key.shape[1] > 0
            and query.shape[2] > 0
            and 8 <= query.shape[3] <= 256
            and query.shape[3] % 8 == 0
            and query.dtype == key.dtype == value.dtype
            and query.dtype in (torch.float16, torch.bfloat16)
            and query.device == key.device == value.device
        )

    def _phase_scratch(
        self,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scratch = self._partial_buffers
        if (
            scratch is not None
            and scratch[0].shape == query.shape
            and scratch[0].dtype == query.dtype
            and scratch[0].device == query.device
        ):
            return scratch
        scratch = (
            torch.empty_like(query),
            torch.empty(
                query.shape[0],
                query.shape[2],
                query.shape[1],
                dtype=torch.float32,
                device=query.device,
            ),
        )
        self._partial_buffers = scratch
        return scratch

    @staticmethod
    def _cudnn_partial_forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        softmax_scale: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one phase through cuDNN, returning BSHD output and [B, H, Sq] LSE."""

        transposed = (query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        output, lse = torch.ops.aten._scaled_dot_product_cudnn_attention(
            *transposed,
            None,
            True,
            0.0,
            False,
            False,
            scale=softmax_scale,
        )[:2]
        # cuDNN reports LSE as [B, H, Sq, 1] and lays the output out so that the
        # transpose back to BSHD is already contiguous.
        while lse.dim() > 3 and lse.shape[-1] == 1:
            lse = lse.squeeze(-1)
        return output.transpose(1, 2).contiguous(), lse

    def _partial_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        softmax_scale: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._use_cudnn:
            try:
                return self._cudnn_partial_forward(
                    query, key, value, softmax_scale=softmax_scale
                )
            except RuntimeError:
                # cuDNN declines some shapes outright; fall back for good.
                self._use_cudnn = False
        if self._flash_attn_fwd is None:
            self._flash_attn_fwd = load_cute_flash_attn_fwd()
        if not self._accepted_kwargs_resolved:
            self._accepted_kwargs = _accepted_kwarg_names(self._flash_attn_fwd)
            self._accepted_kwargs_resolved = True
        accepted = self._accepted_kwargs
        kwargs: dict[str, object] = {
            "causal": False,
            "return_lse": True,
            "softmax_scale": softmax_scale,
        }
        if accepted is None or {"out", "lse"} <= accepted:
            # Reuse phase scratch so CuTe never allocates on the critical path.
            scratch_output, scratch_lse = self._phase_scratch(query)
            kwargs["out"] = scratch_output
            kwargs["lse"] = scratch_lse
        if accepted is not None:
            kwargs = {name: v for name, v in kwargs.items() if name in accepted}
        result = self._flash_attn_fwd(query, key, value, **kwargs)  # type: ignore[operator]
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError(
                "Flash Ring requires a FlashAttention forward returning (output, LSE)"
            )
        output, lse = result[:2]
        if not isinstance(output, torch.Tensor) or not isinstance(lse, torch.Tensor):
            raise RuntimeError("FlashAttention returned invalid output/LSE values")
        return output, lse

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool = False,
        softmax_scale: float | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if causal:
            raise NotImplementedError("Flash Ring attention is non-causal only")
        if not self.supports(query, key, value):
            raise ValueError(
                "expected same-device BSHD FP16/BF16 Q/K/V with positive dimensions "
                "and head_dim in [8, 256] divisible by 8"
            )

        session = self.transport.begin(key.contiguous(), value.contiguous())
        running_output: torch.Tensor | None = None
        running_lse: torch.Tensor | None = None
        seen_sources: set[int] = set()
        try:
            if session.phase_count != self.transport.world_size:
                raise RuntimeError(
                    "ring session must expose exactly one phase per CP rank"
                )
            for phase_index in range(session.phase_count):
                phase = session.acquire(phase_index)
                try:
                    if phase.index != phase_index:
                        raise RuntimeError("transport returned an out-of-order phase")
                    if not 0 <= phase.source_rank < self.transport.world_size:
                        raise RuntimeError("transport returned an invalid source rank")
                    if phase.source_rank in seen_sources:
                        raise RuntimeError("transport returned a source rank twice")
                    seen_sources.add(phase.source_rank)
                    if (
                        phase.key.ndim != 4
                        or phase.value.shape != phase.key.shape
                        or phase.key.shape[0] != query.shape[0]
                        or phase.key.shape[2:] != query.shape[2:]
                        or phase.key.dtype != query.dtype
                        or phase.key.device != query.device
                        or phase.key.shape[1] == 0
                    ):
                        raise RuntimeError("transport returned an incompatible K/V shard")
                    partial_output, partial_lse = self._partial_forward(
                        query,
                        phase.key,
                        phase.value,
                        softmax_scale=softmax_scale,
                    )
                    if phase_index == 0:
                        # Phase zero needs no remote data, so the ring is kicked
                        # only now: the host spends its dispatch budget on
                        # attention first, and the first hop then rides the Copy
                        # Engine underneath it rather than in front of it.
                        session.start()
                    running_output, running_lse = merge_flash_partials_fp32(
                        running_output,
                        running_lse,
                        partial_output,
                        partial_lse,
                    )
                finally:
                    session.release(phase)
        finally:
            session.close()

        if len(seen_sources) != self.transport.world_size:
            raise RuntimeError("ring session did not visit every source rank")
        assert running_output is not None and running_lse is not None
        output = running_output.to(query.dtype)
        return (output, running_lse) if return_lse else output
