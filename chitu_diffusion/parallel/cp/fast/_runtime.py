"""Optional adapter for triple-mu/fast-ulysses.

Upstream: https://github.com/triple-mu/fast-ulysses
Reviewed commit: 6e5dcb24dc44e781ac3091d1d9b3f9fef314fb87
License: Apache-2.0
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def ring_ce_ticket(world_size: int, epoch: int, phase: int) -> int:
    """Return the monotonic ticket for one remote Flash Ring phase."""
    if world_size not in (2, 4, 8):
        raise ValueError("Flash Ring CE world_size must be 2, 4, or 8")
    if epoch <= 0:
        raise ValueError("Flash Ring CE epoch must be positive")
    if phase <= 0 or phase >= world_size:
        raise ValueError("Flash Ring CE phase must be in [1, world_size)")
    ticket = (epoch - 1) * (world_size - 1) + phase
    if ticket > 2**31 - 1:
        raise ValueError("Flash Ring CE ticket exceeds int32")
    return ticket


def ring_ce_slot(world_size: int, epoch: int, phase: int) -> int:
    """Select one of the two symmetric landing slots."""
    return ring_ce_ticket(world_size, epoch, phase) & 1


@dataclass(frozen=True)
class RingCETransportState:
    """Buffers and guards retained for one Flash Ring traversal.

    Phase zero returns ``local_key``/``local_value`` by identity. Remote phases
    alternate between the two rows of each symmetric landing tensor.
    """

    local_key: torch.Tensor
    local_value: torch.Tensor
    landing_key: torch.Tensor
    landing_value: torch.Tensor
    arrivals: torch.Tensor
    acks: torch.Tensor
    # Peer key, peer value, peer arrivals, local acks, and peer acks, resolved
    # once so no phase pays for a tag string or a symmetric-pool lookup.
    handles: tuple[int, ...]
    launch_event: torch.cuda.Event
    key_tag: str
    value_tag: str
    state_tag: str
    epoch: int
    world_size: int
    device_index: int
    # Slicing a landing row costs two dispatches, so keep both slots resolved.
    slot_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]

    def phase_kv(self, phase: int) -> tuple[torch.Tensor, torch.Tensor]:
        if phase == 0:
            return self.local_key, self.local_value
        return self.slot_kv[ring_ce_slot(self.world_size, self.epoch, phase)]


class AsyncFastAgkvHandle:
    def __init__(
        self,
        output: tuple[torch.Tensor, torch.Tensor],
        done: torch.cuda.Event,
    ) -> None:
        self._output = output
        self._done = done

    def wait(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.current_stream().wait_event(self._done)
        return self._output


def probe_fast_ulysses(
    device: torch.device,
    *,
    require_hopper: bool,
    require_subgroups: bool,
    require_fast_agkv: bool = False,
) -> tuple[bool, str]:
    """Check local prerequisites without initializing the WORLD-collective runtime."""
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    if require_hopper and major < 9:
        return (
            False,
            f"GPU compute capability {major}.{minor} is older than Hopper (SM90)",
        )

    local_device_count = torch.cuda.device_count()
    peer_count = local_device_count
    for peer in range(peer_count):
        if peer != device_index and not torch.cuda.can_device_access_peer(
            device_index, peer
        ):
            return (
                False,
                f"CUDA peer access is unavailable from device {device_index} to {peer}",
            )

    try:
        import fast_ulysses
    except (ImportError, OSError) as exc:
        return False, f"fast_ulysses extension cannot be loaded: {exc}"
    if not hasattr(fast_ulysses, "UlyssesGroup"):
        return False, "fast_ulysses.UlyssesGroup is unavailable"
    if require_subgroups and not getattr(
        fast_ulysses, "SUPPORTS_WORLD_PARTITION_SUBGROUPS", False
    ):
        return False, "installed fast_ulysses package lacks CFP subgroup support"
    if require_fast_agkv and not hasattr(torch.ops.fast_ulysses, "all_gather_kv_4d"):
        return False, "installed fast_ulysses package lacks the Fast AGKV operator"
    return True, f"SM{major}{minor}, CUDA P2P and fast_ulysses extension available"


def _use_tma_from_env() -> bool | None:
    value = os.environ.get("CHITU_FAST_ULYSSES_USE_TMA", "auto").strip().lower()
    if value in ("", "auto"):
        return None
    if value in ("1", "true", "on", "tma"):
        return True
    if value in ("0", "false", "off", "non_tma", "non-tma"):
        return False
    raise ValueError(
        f"CHITU_FAST_ULYSSES_USE_TMA must be auto, 0/non_tma, or 1/tma; got {value!r}"
    )


class FastUlyssesAllToAll:
    """Normalize fast-ulysses to ChituDiffusion's 4D all-to-all interface."""

    def __init__(
        self,
        process_group,
        device: torch.device,
        *,
        pool_bytes: int | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("fast_ulysses requires CUDA.")
        # The patched bootstrap initializes NVSHMEM over ``process_group``.
        # Validate that this runtime team is node-local; outer Ring groups may
        # still span nodes through NCCL.
        group_size = dist.get_world_size(process_group)
        hosts: list[str | None] = [None] * group_size
        dist.all_gather_object(hosts, socket.gethostname(), group=process_group)
        if len(set(hosts)) != 1:
            raise RuntimeError(
                "fast_ulysses is a node-local NVLink backend, but its runtime group "
                f"spans hosts {sorted(set(hosts))}."
            )
        try:
            from fast_ulysses import UlyssesGroup
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "CHITU_CP_ULYSSES_BACKEND=fast_ulysses was requested, but the "
                "fast_ulysses extension is unavailable. Install it with NVSHMEM "
                "3.7+ and an architecture-matched CUDA toolchain."
            ) from exc

        if pool_bytes is None:
            pool_bytes = int(
                os.environ.get("CHITU_FAST_ULYSSES_POOL_BYTES", str(2 << 30))
            )
        if pool_bytes <= 0:
            raise ValueError("fast_ulysses pool size must be positive")
        self.pool_bytes = pool_bytes
        self._group = UlyssesGroup(
            process_group=process_group,
            device=device,
            initial_pool_bytes=pool_bytes,
        )
        self._use_tma = _use_tma_from_env()
        self._world_size = dist.get_world_size(process_group)
        self._sequence_idx = 0
        self._logical_config_cache: set[tuple[object, ...]] = set()

    def reset_sequence(self) -> None:
        self._sequence_idx = 0

    @staticmethod
    def supports(
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
    ) -> bool:
        """Whether this call is safe for the NVSHMEM kernel path.

        CUDA Graph replay cannot reuse the host-incremented barrier epoch, and
        the extension intentionally implements only uniform fp16/bf16 4D
        Ulysses swaps. Other Chitu all-to-all layouts retain their NCCL path.
        """
        return (
            tensor.ndim == 4
            and tensor.is_cuda
            and tensor.dtype in (torch.float16, torch.bfloat16)
            and (scatter_dim, gather_dim) in ((2, 1), (1, 2))
            and not torch.cuda.is_current_stream_capturing()
        )

    @staticmethod
    def mode(scatter_dim: int, gather_dim: int) -> int:
        if scatter_dim == 2 and gather_dim == 1:
            return 0
        if scatter_dim == 1 and gather_dim == 2:
            return 1
        raise NotImplementedError(
            "fast_ulysses supports only Ulysses head/sequence swaps, got "
            f"scatter_dim={scatter_dim}, gather_dim={gather_dim}."
        )

    def all_to_all(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
    ) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError(
                f"fast_ulysses requires a 4D [B,S,H,D] tensor, got {tensor.ndim}D."
            )
        mode = self.mode(scatter_dim, gather_dim)
        tag = f"chitu_cp_{self._sequence_idx}"
        self._sequence_idx += 1
        return self._group.all_to_all_single_4d(
            tensor,
            mode=mode,
            tag=tag,
            use_tma=self._use_tma,
        )

    def all_to_all_tagged(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        *,
        tag: str,
    ) -> torch.Tensor:
        return self._group.all_to_all_single_4d(
            tensor,
            mode=self.mode(scatter_dim, gather_dim),
            tag=tag,
            use_tma=self._use_tma,
        )

    @staticmethod
    def supports_agkv(key: torch.Tensor, value: torch.Tensor) -> bool:
        return (
            key.ndim == 4
            and value.shape == key.shape
            and key.is_cuda
            and value.is_cuda
            and key.dtype == value.dtype
            and key.dtype in (torch.float16, torch.bfloat16)
            and key.shape[2] * key.shape[3] * key.element_size() % 16 == 0
            and not torch.cuda.is_current_stream_capturing()
        )

    def all_gather_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_tag: str,
        value_tag: str,
        use_ce: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.supports_agkv(key, value):
            raise ValueError("tensors are unsupported by Fast AGKV")
        if not hasattr(torch.ops.fast_ulysses, "all_gather_kv_4d"):
            raise RuntimeError(
                "installed fast_ulysses extension does not provide Fast AGKV"
            )
        return torch.ops.fast_ulysses.all_gather_kv_4d(
            self._group._group,
            key.contiguous(),
            value.contiguous(),
            key_tag,
            value_tag,
            use_ce,
        )

    def begin_all_gather_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_tag: str,
        value_tag: str,
        use_ce: bool,
    ) -> AsyncFastAgkvHandle:
        if not self.supports_agkv(key, value):
            raise ValueError("tensors are unsupported by Fast AGKV")
        if not hasattr(torch.ops.fast_ulysses, "all_gather_kv_4d"):
            raise RuntimeError(
                "installed fast_ulysses extension does not provide Fast AGKV"
            )
        key = key.contiguous()
        value = value.contiguous()
        key.record_stream(self._group._comm_stream)
        value.record_stream(self._group._comm_stream)
        output, done = self._group._launch_on_comm_stream(
            [],
            lambda: torch.ops.fast_ulysses.all_gather_kv_4d(
                self._group._group,
                key,
                value,
                key_tag,
                value_tag,
                use_ce,
            ),
        )
        return AsyncFastAgkvHandle(output, done)

    def begin_full_mesh_stream_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        source_lengths: Sequence[int],
        *,
        key_tag: str,
        value_tag: str,
        flag_tag: str,
        epoch: int,
        use_sm: bool = False,
        source_chunk: bool = False,
        chunk_count: int = 1,
        copy_local: bool = True,
        ready_event: torch.cuda.Event | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.cuda.Event,
    ]:
        """Publish streaming K/V while a fused attention kernel consumes it."""
        if not self.supports_agkv(key, value):
            raise ValueError("tensors are unsupported by full-mesh streaming")
        if not hasattr(torch.ops.fast_ulysses, "full_mesh_stream_kv_4d"):
            raise RuntimeError(
                "installed fast_ulysses extension lacks full_mesh_stream_kv_4d"
            )
        key = key.contiguous()
        value = value.contiguous()
        key.record_stream(self._group._comm_stream)
        value.record_stream(self._group._comm_stream)

        def launch():
            return torch.ops.fast_ulysses.full_mesh_stream_kv_4d(
                self._group._group,
                key,
                value,
                list(source_lengths),
                key_tag,
                value_tag,
                flag_tag,
                epoch,
                use_sm,
                source_chunk,
                chunk_count,
                copy_local,
            )

        if ready_event is None:
            output, done = self._group._launch_on_comm_stream([], launch)
        else:
            self._group._comm_stream.wait_event(ready_event)
            with torch.cuda.stream(self._group._comm_stream):
                output = launch()
            done = ready_event
        output_key, output_value, phase_flags, epoch_guard = output
        return output_key, output_value, phase_flags, epoch_guard, done

    @staticmethod
    def wait_full_mesh_ready(
        flags: torch.Tensor,
        *,
        index: int,
        epoch: int,
    ) -> None:
        """Suspend the current CUDA stream until a peer publishes an arrival."""
        torch.ops.fast_ulysses.full_mesh_wait_ready(flags, index, epoch)

    def publish_full_mesh_consumed(
        self,
        *,
        flag_tag: str,
        epoch: int,
        epoch_guard: torch.Tensor,
    ) -> torch.Tensor:
        """Release peers to reuse the symmetric buffer this rank just read.

        Issued on the caller's stream, which must already be ordered behind the
        consuming kernel. Peers hold their next send until this arrives, so the
        payload of one epoch cannot overwrite the payload another rank is still
        reading.
        """
        if not hasattr(torch.ops.fast_ulysses, "full_mesh_publish_consumed"):
            raise RuntimeError(
                "installed fast_ulysses extension lacks full_mesh_publish_consumed"
            )
        return torch.ops.fast_ulysses.full_mesh_publish_consumed(
            self._group._group,
            flag_tag,
            epoch,
            epoch_guard,
        )

    def begin_ring_ce_transport(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_tag: str,
        value_tag: str,
        state_tag: str,
        epoch: int,
        ready_event: torch.cuda.Event | None = None,
    ) -> RingCETransportState:
        """Start phase-1 CE delivery while retaining phase-0 K/V zero-copy.

        ``ready_event`` lets a caller submit this after it has already queued
        phase-zero attention: the comm stream then waits only for that event
        instead of for everything standing on the caller's stream, so the first
        hop rides the Copy Engine while phase zero occupies the SMs.
        """
        ring_ce_ticket(self._world_size, epoch, 1)
        if not self.supports_agkv(key, value):
            raise ValueError("tensors are unsupported by Flash Ring CE")
        if not key.is_contiguous() or not value.is_contiguous():
            raise ValueError(
                "Flash Ring CE requires contiguous K/V to preserve zero-copy local use"
            )
        if not hasattr(torch.ops.fast_ulysses, "ring_ce_prepare_kv_4d"):
            raise RuntimeError(
                "installed fast_ulysses extension lacks Flash Ring CE transport"
            )
        key.record_stream(self._group._comm_stream)
        value.record_stream(self._group._comm_stream)

        def launch():
            return torch.ops.fast_ulysses.ring_ce_prepare_kv_4d(
                self._group._group,
                key,
                value,
                key_tag,
                value_tag,
                state_tag,
                epoch,
            )

        if ready_event is None:
            output, done = self._group._launch_on_comm_stream([], launch)
        else:
            self._group._comm_stream.wait_event(ready_event)
            with torch.cuda.stream(self._group._comm_stream):
                output = launch()
            done = ready_event
        landing_key, landing_value, arrivals, acks, handles = output
        return RingCETransportState(
            local_key=key,
            local_value=value,
            landing_key=landing_key,
            landing_value=landing_value,
            arrivals=arrivals,
            acks=acks,
            handles=tuple(handles),
            launch_event=done,
            key_tag=key_tag,
            value_tag=value_tag,
            state_tag=state_tag,
            epoch=epoch,
            world_size=self._world_size,
            device_index=key.device.index,
            slot_kv=(
                (landing_key[0], landing_value[0]),
                (landing_key[1], landing_value[1]),
            ),
        )

    @staticmethod
    def wait_ring_ce_ready(
        state: RingCETransportState,
        *,
        phase: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wait for a complete remote K/V phase and return its landing views."""
        ring_ce_ticket(state.world_size, state.epoch, phase)
        torch.ops.fast_ulysses.ring_ce_wait_ready(
            state.arrivals,
            state.world_size,
            state.epoch,
            phase,
        )
        return state.phase_kv(phase)

    def forward_ring_ce(
        self,
        state: RingCETransportState,
        *,
        phase: int,
    ) -> None:
        """Forward the current landing slot into the right peer's next slot."""
        if phase <= 0 or phase >= state.world_size - 1:
            raise ValueError("Flash Ring CE can forward phases [1, world_size-1)")
        peer_key, peer_value, peer_arrivals, local_acks, _ = state.handles
        torch.ops.fast_ulysses.ring_ce_forward_kv_4d(
            self._group._group,
            state.landing_key,
            state.landing_value,
            peer_key,
            peer_value,
            peer_arrivals,
            local_acks,
            state.epoch,
            phase,
        )

    def publish_ring_ce_consumed(
        self,
        state: RingCETransportState,
        *,
        phase: int,
    ) -> None:
        """Let the left peer reuse a slot after this stream consumes it."""
        ring_ce_ticket(state.world_size, state.epoch, phase)
        torch.ops.fast_ulysses.ring_ce_publish_consumed(
            self._group._group,
            state.handles[4],
            state.device_index,
            state.epoch,
            phase,
        )

    def begin_all_to_all(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        *,
        tag: str,
        barrier: bool,
        use_ce: bool,
    ) -> Any:
        """Launch one upstream async collective on its communication stream."""
        mode = self.mode(scatter_dim, gather_dim)
        if use_ce:
            return self._group.all_to_all_single_4d_ce_async(
                tensor,
                mode=mode,
                tag=tag,
                barrier=barrier,
            )
        return self._group.all_to_all_single_4d_async(
            tensor,
            mode=mode,
            tag=tag,
            use_tma=self._use_tma,
            barrier=barrier,
        )

    @staticmethod
    def logical_cache_key(
        tensor: torch.Tensor,
        *,
        logical_world_size: int,
        mode: int,
        subgroup_type: str,
    ) -> tuple[object, ...]:
        return (
            logical_world_size,
            mode,
            tuple(tensor.shape),
            tensor.dtype,
            subgroup_type,
        )

    def logical_all_to_all(
        self,
        tensor: torch.Tensor,
        *,
        peer_ranks: Sequence[int],
        mode: int,
        tag: str,
        subgroup_type: str,
        storage_slots: int = 1,
        storage_index: int = 0,
    ) -> torch.Tensor:
        if not hasattr(
            torch.ops.fast_ulysses,
            "logical_subgroup_all_to_all_single_4d_ce",
        ):
            raise RuntimeError(
                "installed fast_ulysses extension lacks logical subgroup A2A"
            )
        key = self.logical_cache_key(
            tensor,
            logical_world_size=len(peer_ranks),
            mode=mode,
            subgroup_type=subgroup_type,
        )
        self._logical_config_cache.add(key)
        return torch.ops.fast_ulysses.logical_subgroup_all_to_all_single_4d_ce(
            self._group._group,
            tensor.contiguous(),
            list(peer_ranks),
            mode,
            tag,
            storage_slots,
            storage_index,
        )

    def logical_barrier(
        self,
        *,
        peer_ranks: Sequence[int],
        tag: str,
        epoch: int,
    ) -> torch.Tensor:
        return torch.ops.fast_ulysses.logical_subgroup_barrier(
            self._group._group,
            list(peer_ranks),
            tag,
            epoch,
        )

    def prepare_u2fm2_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_tag: str,
        key_tag: str,
        value_tag: str,
        flag_tag: str,
        ulysses_barrier_tag: str,
        ulysses_epoch: int,
        full_mesh_epoch: int,
        segment_stride: int,
        remote_blocks: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not hasattr(torch.ops.fast_ulysses, "u2fm2_prepare_qkv"):
            raise RuntimeError(
                "installed fast_ulysses extension lacks fused U2xFM2 Q/K/V preparation"
            )
        return torch.ops.fast_ulysses.u2fm2_prepare_qkv(
            self._group._group,
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            query_tag,
            key_tag,
            value_tag,
            flag_tag,
            ulysses_barrier_tag,
            ulysses_epoch,
            full_mesh_epoch,
            segment_stride,
            remote_blocks,
        )

    def destroy(self) -> None:
        logger.info("Destroying fast-ulysses resources")
        self._group.destroy()
        logger.info("Destroyed fast-ulysses resources")
