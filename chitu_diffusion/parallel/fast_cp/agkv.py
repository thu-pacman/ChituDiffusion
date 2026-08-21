"""Experimental NVSHMEM full-mesh streaming attention for Hopper.

The transport publishes local-first source chunks through Copy Engines, with
optional phase-major stripes and SM direct-write for experiments. A versioned
CuTe SM90 overlay waits on arrival flags directly in the TMA producer loop,
keeping the online-softmax state inside one attention kernel.

Shards carry per-source arrival flags rather than a fixed stride, so the global
sequence only has to be long enough to give every rank one token. It
needs no padding and no divisibility by the tile size or the CP degree.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ._runtime import FastUlyssesAllToAll
from ._cute import CapturedCuteLaunch, capture_cute_launch, load_cute_flash_attn_fwd


def _merge_cp2_partials(outputs: torch.Tensor, lses: torch.Tensor) -> torch.Tensor:
    merged_lse = torch.logaddexp(lses[0], lses[1])
    local_weight = torch.exp(lses[0] - merged_lse)
    remote_weight = torch.exp(lses[1] - merged_lse)
    return (
        outputs[0].float() * local_weight.transpose(-2, -1).unsqueeze(-1)
        + outputs[1].float() * remote_weight.transpose(-2, -1).unsqueeze(-1)
    ).to(outputs.dtype)


_merge_cp2_partials_compiled = torch.compile(_merge_cp2_partials, dynamic=False)


@dataclass(frozen=True)
class _DirectStreamingLaunch:
    captured: CapturedCuteLaunch
    dynamic_indices: tuple[int, int, int, int, int]
    output: torch.Tensor

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        flags: torch.Tensor,
        epoch: torch.Tensor,
    ) -> torch.Tensor:
        args = list(self.captured.args)
        for index, tensor in zip(
            self.dynamic_indices,
            (query, key, value, flags, epoch),
            strict=True,
        ):
            args[index] = tensor.detach()
        self.captured.runner(*args)
        return self.output


def _captured_tensor_index(
    args: tuple[object, ...], target: torch.Tensor
) -> int | None:
    matches = [
        index
        for index, value in enumerate(args)
        if isinstance(value, torch.Tensor)
        and value.device == target.device
        and value.data_ptr() == target.data_ptr()
    ]
    return matches[0] if len(matches) == 1 else None


class FastAgkvAttention:
    """Static, non-causal, single-node full-world CP attention."""

    def __init__(
        self,
        process_group: object,
        device: torch.device,
        *,
        pool_bytes: int | None = None,
        use_sm_transport: bool | None = None,
        source_chunk_streaming: bool | None = None,
        chunk_count: int = 1,
        scheduler: str = "fixed",
        buffer_depth: int | None = None,
        cp2_kernel: str = "fused",
        cp2_tile_n: int | None = None,
        cp2_num_threads: int | None = None,
        cp2_intra_wg_overlap: bool = True,
        cp2_mma_pv_is_rs: bool = True,
    ) -> None:
        if process_group is None or not dist.is_initialized():
            raise RuntimeError("full-mesh fused attention requires torch.distributed")
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        if self.world_size not in {2, 4, 8}:
            raise ValueError("full-mesh fused attention supports CP2, CP4, or CP8")
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) != (9, 0):
            raise RuntimeError(
                f"full-mesh fused attention requires Hopper SM90, got {major}.{minor}"
            )
        self.rank = dist.get_rank(process_group)
        self.device = device
        # Copy Engines leave every SM to attention, so they win once compute
        # dominates. SM direct-write is only competitive for tiny sequences,
        # where it still trails by well under a percent.
        self.source_chunk_streaming = (
            True if source_chunk_streaming is None else source_chunk_streaming
        )
        self.use_sm_transport = False if use_sm_transport is None else use_sm_transport
        if chunk_count not in {1, 2, 3, 4}:
            raise ValueError("chunk_count must be one of 1, 2, 3, or 4")
        if scheduler not in {"fixed", "ready_mask"}:
            raise ValueError("scheduler must be 'fixed' or 'ready_mask'")
        if not self.source_chunk_streaming and chunk_count != 1:
            raise ValueError("chunk_count requires source_chunk_streaming")
        if self.use_sm_transport and chunk_count != 1:
            raise ValueError("chunked Full-Mesh currently requires Copy Engines")
        if scheduler == "ready_mask" and chunk_count == 1:
            raise ValueError("ready_mask scheduling requires chunk_count > 1")
        if scheduler == "ready_mask" and chunk_count * self.world_size > 31:
            raise ValueError("ready_mask scheduling supports at most 31 K/V segments")
        if buffer_depth is not None and buffer_depth not in {1, 2}:
            raise ValueError("buffer_depth must be 1 or 2")
        if cp2_kernel not in {"fused", "split_merge"}:
            raise ValueError("cp2_kernel must be 'fused' or 'split_merge'")
        if cp2_tile_n not in {None, 64, 96, 128}:
            raise ValueError("cp2_tile_n must be None, 64, 96, or 128")
        if cp2_num_threads not in {None, 256, 384}:
            raise ValueError("cp2_num_threads must be None, 256, or 384")
        self.chunk_count = chunk_count
        self.scheduler = scheduler
        # One symmetric buffer makes every send wait for all peers to consume the
        # previous epoch. Alternating two buffers moves that gate two epochs
        # back so it is already satisfied, but measurement says the gate is not
        # what delays arrivals: on H20 CP4/CP8 it changed nothing from S=4096 to
        # S=75600 and cost 11% at CP8 S=4096, for twice the symmetric memory.
        # It stays available for experiments; the default is one buffer.
        self.buffer_depth = 1 if buffer_depth is None else buffer_depth
        self.cp2_kernel = cp2_kernel
        self.cp2_tile_n = cp2_tile_n
        self.cp2_num_threads = cp2_num_threads
        self.cp2_intra_wg_overlap = cp2_intra_wg_overlap
        self.cp2_mma_pv_is_rs = cp2_mma_pv_is_rs
        self._buffer_depth_is_automatic = buffer_depth is None
        self._transport = FastUlyssesAllToAll(
            process_group,
            device,
            pool_bytes=pool_bytes,
        )
        self._calls = 0
        self._slot_epochs = [0, 0]
        self._flags: torch.Tensor | None = None
        self._acks: torch.Tensor | None = None
        self._live_state: dict[int, torch.Tensor] = {}
        self._live_acks: dict[int, torch.Tensor] = {}
        self._cp2_partial_shape: tuple[int, ...] | None = None
        self._cp2_partial_out: torch.Tensor | None = None
        self._cp2_partial_lse: torch.Tensor | None = None
        self._cp2_initialized: set[tuple[object, ...]] = set()
        self._cp2_ready_events = [torch.cuda.Event() for _ in range(4)]
        self._cp2_ready_slot = 0
        self._closed = False
        # Building the metadata with torch.tensor(..., device=) costs a blocking
        # cudaStreamSynchronize per call, which pins the host to the previous
        # kernel and leaves the transport nothing to overlap with. Stage it
        # through pinned memory instead, and rotate over enough buffers that the
        # reuse wait is always for a copy several calls old.
        self._metadata_slot = 0
        self._metadata_host: list[torch.Tensor] = []
        self._metadata_device: list[torch.Tensor] = []
        self._metadata_events: list[torch.cuda.Event] = []
        self._direct_streaming: dict[tuple[object, ...], _DirectStreamingLaunch] = {}
        self._direct_seen: set[tuple[object, ...]] = set()

        try:
            _flash_attn_fwd = load_cute_flash_attn_fwd()
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                "the reviewed FlashAttention CuTe checkout is unavailable"
            ) from exc
        required = {
            "_streaming_kv",
            "_kv_ready_flags",
            "_kv_slots",
            "_kv_epoch_tensor",
            "_kv_world_size",
            "_secondary_key",
            "_secondary_value",
        }
        if not required.issubset(inspect.signature(_flash_attn_fwd).parameters):
            raise RuntimeError(
                "FlashAttention CuTe SM90 is missing the Chitu full-mesh overlay"
            )
        self._flash_attn_fwd = _flash_attn_fwd

    def _cp2_fused_config(self, global_sequence: int) -> tuple[int | None, int]:
        if self.cp2_tile_n is not None or self.cp2_num_threads is not None:
            return self.cp2_tile_n, self.cp2_num_threads or 384
        if "h20" in torch.cuda.get_device_name(self.device).lower():
            return (64, 256) if global_sequence <= 8192 else (128, 384)
        return None, 384

    @classmethod
    def shard_lengths(cls, global_sequence: int, world_size: int) -> tuple[int, ...]:
        """Canonical CP split balanced to one token, with no empty shards."""
        if global_sequence < world_size:
            raise ValueError(
                f"full-mesh streaming needs at least one token per rank, "
                f"got {global_sequence} tokens for CP{world_size}"
            )
        base, remainder = divmod(global_sequence, world_size)
        return tuple(
            base + (1 if rank < remainder else 0) for rank in range(world_size)
        )

    def _slot_starts(self, lengths: tuple[int, ...]) -> list[int]:
        """Token offset of each K/V source slot, in ascending address order.

        The CuTe producer converts these offsets with its compile-time ``tile_n``.
        This keeps the metadata valid for every upstream SM90 head-dimension
        configuration without duplicating FlashAttention's tile selection here.
        """
        if self.source_chunk_streaming:
            starts = []
            offset = 0
            for chunk in range(self.chunk_count):
                for slot in range(self.world_size):
                    source = (
                        self.rank - (self.world_size - 1 - slot)
                    ) % self.world_size
                    begin = lengths[source] * chunk // self.chunk_count
                    end = lengths[source] * (chunk + 1) // self.chunk_count
                    starts.append(offset)
                    offset += end - begin
            return starts
        else:
            # Phase-major stripes make each slot a phase spanning one full shard
            # worth of tokens, contributed by every source at once.
            slot_order = list(range(self.world_size))
        starts = []
        offset = 0
        for slot in slot_order:
            starts.append(offset)
            offset += lengths[slot]
        return starts

    def _buffer_slot_and_epoch(self) -> tuple[int, int]:
        """Pick this call's buffer and the next epoch of that buffer.

        Each buffer keeps its own densely increasing epoch, because the
        transport gates a send on peers having acknowledged exactly the previous
        epoch of the same buffer. The counter only advances once the transport
        actually accepted the call, so a rejected attempt leaves no gap.
        """
        slot = self._calls % self.buffer_depth
        return slot, self._slot_epochs[slot] + 1

    def _downgrade_buffer_depth(self, error: RuntimeError, slot: int) -> bool:
        """Fall back to one buffer when the symmetric pool cannot hold two.

        Only the very first use of the second buffer can fail this way, since
        the pool never releases what it handed out.
        """
        if (
            not self._buffer_depth_is_automatic
            or self.buffer_depth == 1
            or slot == 0
            or "SymmetricHeapPool OOM" not in str(error)
        ):
            return False
        self.buffer_depth = 1
        return True

    _METADATA_RING = 4

    def _publish_metadata(
        self, values: list[int], device: torch.device
    ) -> torch.Tensor:
        """Upload the epoch and slot offsets without stalling the host."""
        if not self._metadata_host or self._metadata_host[0].numel() != len(values):
            self._metadata_host = [
                torch.empty(len(values), dtype=torch.int32, pin_memory=True)
                for _ in range(self._METADATA_RING)
            ]
            self._metadata_device = [
                torch.empty(len(values), dtype=torch.int32, device=device)
                for _ in range(self._METADATA_RING)
            ]
            self._metadata_events = [
                torch.cuda.Event() for _ in range(self._METADATA_RING)
            ]
            self._metadata_slot = 0
        slot = self._metadata_slot
        self._metadata_slot = (slot + 1) % self._METADATA_RING
        self._metadata_events[slot].synchronize()
        host = self._metadata_host[slot]
        for index, value in enumerate(values):
            host[index] = value
        device_view = self._metadata_device[slot]
        device_view.copy_(host, non_blocking=True)
        self._metadata_events[slot].record()
        return device_view

    def _reset_epoch(self) -> None:
        """Restart the epoch counter without stranding a consumer.

        The transport never clears arrival flags or consumption acknowledgements
        after the first call, because clearing races against peers that publish
        before this rank gets there. Correctness then rests on epochs
        increasing, so restarting the counter needs both to be zeroed while
        nobody is publishing.
        """
        torch.cuda.synchronize()
        dist.barrier(self.process_group)
        for state in (*self._live_state.values(), *self._live_acks.values()):
            state.zero_()
        torch.cuda.synchronize()
        dist.barrier(self.process_group)
        self._slot_epochs = [0, 0]
        self._calls = 0

    def _cp2_buffers(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = tuple(query.shape)
        if self._cp2_partial_shape != shape or (
            self._cp2_partial_out is not None
            and self._cp2_partial_out.dtype != query.dtype
        ):
            self._cp2_partial_shape = shape
            self._cp2_partial_out = torch.empty(
                (2, *shape), dtype=query.dtype, device=query.device
            )
            self._cp2_partial_lse = torch.empty(
                (2, shape[0], shape[2], shape[1]),
                dtype=torch.float32,
                device=query.device,
            )
        assert self._cp2_partial_out is not None
        assert self._cp2_partial_lse is not None
        return self._cp2_partial_out, self._cp2_partial_lse

    @staticmethod
    def supports(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
        return (
            query.ndim == 4
            and key.ndim == 4
            and value.shape == key.shape
            and query.shape[0] == key.shape[0]
            and query.shape[0] > 0
            and query.shape[2:] == key.shape[2:]
            and query.shape[2] > 0
            and 8 <= query.shape[-1] <= 256
            and query.shape[-1] % 8 == 0
            and query.dtype == key.dtype == value.dtype
            and query.dtype in (torch.float16, torch.bfloat16)
            and query.is_cuda
            and key.is_cuda
            and value.is_cuda
        )

    def _streaming_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        flags: torch.Tensor,
        epoch: torch.Tensor,
        *,
        kv_slots: int,
        kv_world_size: int,
    ) -> torch.Tensor:
        layout = (
            tuple(query.shape),
            tuple(query.stride()),
            tuple(key.shape),
            query.dtype,
            query.device,
            flags.numel(),
            epoch.numel(),
            kv_slots,
            kv_world_size,
            *(tensor.data_ptr() % 16 for tensor in (query, key, value, flags, epoch)),
        )
        direct = self._direct_streaming.get(layout)
        if direct is not None:
            return direct(query, key, value, flags, epoch)

        def launch():
            return self._flash_attn_fwd(
                query,
                key,
                value,
                causal=False,
                _streaming_kv=True,
                _kv_ready_flags=flags,
                _kv_slots=kv_slots,
                _kv_epoch_tensor=epoch,
                _kv_world_size=kv_world_size,
            )

        if layout in self._direct_seen:
            result, captured = capture_cute_launch(self._flash_attn_fwd, launch)
            if captured is not None:
                indices = tuple(
                    _captured_tensor_index(captured.args, tensor)
                    for tensor in (query, key, value, flags, epoch)
                )
                if all(index is not None for index in indices):
                    output = result[0]
                    assert isinstance(output, torch.Tensor)
                    self._direct_streaming[layout] = _DirectStreamingLaunch(
                        captured,
                        tuple(int(index) for index in indices),
                        output,
                    )
            return result[0]

        result = launch()
        self._direct_seen.add(layout)
        return result[0]

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        causal: bool = False,
        global_sequence: int | None = None,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("full-mesh fused attention is closed")
        if causal:
            raise NotImplementedError("full-mesh fused attention is non-causal only")
        if not self.supports(query, key, value):
            raise ValueError(
                "expected contiguous-compatible BSHD FP16/BF16 tensors with "
                "matching positive batch and head_dim in [8, 256] divisible by 8"
            )
        if query.shape[1] != key.shape[1]:
            raise ValueError("each CP rank must provide equally sized local Q/K/V")
        if global_sequence is None:
            global_sequence = key.shape[1] * self.world_size
        lengths = self.shard_lengths(global_sequence, self.world_size)
        if lengths[self.rank] != key.shape[1]:
            raise ValueError(
                f"rank {self.rank} holds {key.shape[1]} tokens but the canonical "
                f"CP{self.world_size} split of {global_sequence} expects "
                f"{lengths[self.rank]}"
            )

        if max(self._slot_epochs) >= 2**31 - 2:
            self._reset_epoch()
        slot, epoch = self._buffer_slot_and_epoch()
        cp2_remote_only = (
            self.world_size == 2
            and self.source_chunk_streaming
            and not self.use_sm_transport
            and self.chunk_count == 1
            and self.scheduler == "fixed"
        )
        cp2_split = cp2_remote_only and self.cp2_kernel == "split_merge"
        cp2_fused = cp2_remote_only and self.cp2_kernel == "fused"
        cp2_partial_out: torch.Tensor | None = None
        cp2_partial_lse: torch.Tensor | None = None
        transport_ready: torch.cuda.Event | None = None
        cp2_local_queued = False
        cp2_layout = (slot, tuple(key.shape), key.dtype)
        if cp2_split:
            cp2_partial_out, cp2_partial_lse = self._cp2_buffers(query)
            if cp2_layout in self._cp2_initialized:
                transport_ready = self._cp2_ready_events[self._cp2_ready_slot]
                self._cp2_ready_slot = (self._cp2_ready_slot + 1) % len(
                    self._cp2_ready_events
                )
                transport_ready.record()
                # Queue useful work before the CPU submits the CE descriptors.
                # The comm stream waits only for transport_ready, so host issue
                # and the remote transfer proceed while this kernel is running.
                self._flash_attn_fwd(
                    query,
                    key,
                    value,
                    causal=False,
                    return_lse=True,
                    out=cp2_partial_out[0],
                    lse=cp2_partial_lse[0],
                )
                cp2_local_queued = True
        # One host-visible tensor carries the epoch plus token-space slot starts.
        # CP2 split attention waits directly on its one arrival flag and does
        # not need this metadata upload.
        slot_starts = self._slot_starts(lengths)
        kv_slots = (
            self.world_size * self.chunk_count
            if self.source_chunk_streaming
            else self.world_size
        )
        kv_world_size = (
            -1
            if self.scheduler == "ready_mask"
            and global_sequence % 256 == 0
            and all(start % 256 == 0 for start in slot_starts)
            else (1 if self.source_chunk_streaming else self.world_size)
        )
        epoch_tensor = (
            None
            if cp2_remote_only
            else self._publish_metadata([epoch, *slot_starts], query.device)
        )
        while True:
            try:
                (
                    streamed_key,
                    streamed_value,
                    phase_flags,
                    epoch_guard,
                    done,
                ) = self._transport.begin_full_mesh_stream_kv(
                    key,
                    value,
                    lengths,
                    key_tag=f"chitu_full_mesh_key_{slot}",
                    value_tag=f"chitu_full_mesh_value_{slot}",
                    flag_tag=f"chitu_full_mesh_flags_{slot}",
                    epoch=epoch,
                    use_sm=self.use_sm_transport,
                    source_chunk=self.source_chunk_streaming,
                    chunk_count=self.chunk_count,
                    copy_local=not cp2_remote_only,
                    ready_event=transport_ready,
                )
                break
            except RuntimeError as exc:
                if not self._downgrade_buffer_depth(exc, slot):
                    raise
                slot, epoch = self._buffer_slot_and_epoch()
                epoch_tensor = (
                    None
                    if cp2_remote_only
                    else self._publish_metadata([epoch, *slot_starts], query.device)
                )
        self._slot_epochs[slot] = epoch
        self._flags = phase_flags
        self._live_state[slot] = phase_flags
        if cp2_fused:
            remote_length = lengths[1 - self.rank]
            cp2_tile_n, cp2_num_threads = self._cp2_fused_config(global_sequence)
            output = self._flash_attn_fwd(
                query,
                key,
                value,
                causal=False,
                tile_mn=(128, cp2_tile_n) if cp2_tile_n else None,
                num_threads=cp2_num_threads,
                intra_wg_overlap=self.cp2_intra_wg_overlap,
                mma_pv_is_rs=self.cp2_mma_pv_is_rs,
                _kv_ready_flags=phase_flags.view(-1),
                _kv_epoch_tensor=epoch_guard,
                _secondary_key=streamed_key[:, :remote_length],
                _secondary_value=streamed_value[:, :remote_length],
            )[0]
        elif cp2_split:
            # CP2 has exactly one remote shard. Compute the local partial while
            # the peer writes the remote shard, then let the second kernel wait
            # on that shard's arrival flag and merge both online-softmax states.
            # This avoids materializing the local shard in the symmetric buffer.
            assert cp2_partial_out is not None
            assert cp2_partial_lse is not None
            if not cp2_local_queued:
                # First use of a layout initializes symmetric slices and runs
                # one SM barrier. Finish that setup before launching a
                # persistent attention kernel; later calls take the overlap path.
                torch.cuda.current_stream().wait_event(done)
                self._flash_attn_fwd(
                    query,
                    key,
                    value,
                    causal=False,
                    return_lse=True,
                    out=cp2_partial_out[0],
                    lse=cp2_partial_lse[0],
                )
                self._cp2_initialized.add(cp2_layout)
            remote_length = lengths[1 - self.rank]
            self._transport.wait_full_mesh_ready(
                phase_flags,
                index=0,
                epoch=epoch,
            )
            self._flash_attn_fwd(
                query,
                streamed_key[:, :remote_length],
                streamed_value[:, :remote_length],
                causal=False,
                return_lse=True,
                out=cp2_partial_out[1],
                lse=cp2_partial_lse[1],
            )
            output = _merge_cp2_partials_compiled(cp2_partial_out, cp2_partial_lse)
        else:
            assert epoch_tensor is not None
            # The event covers this rank's own shard copy and its arrival flag;
            # remote copies ride separate streams and are not part of it. The
            # same-device D2D copy must finish before the persistent kernel can
            # occupy every SM while waiting on that local flag.
            torch.cuda.current_stream().wait_event(done)
            output = self._streaming_forward(
                query,
                streamed_key,
                streamed_value,
                phase_flags.view(-1),
                epoch_tensor,
                kv_slots=kv_slots,
                # Ready-mask currently schedules complete 256-token-aligned
                # segments. Irregular boundaries retain the proven fixed traversal
                # because one TMA tile may otherwise straddle two arrival flags.
                kv_world_size=kv_world_size,
            )
        # Only now may peers overwrite the symmetric buffer. There is one buffer
        # per tag, so without this acknowledgement a rank that reaches its next
        # call first would replace K/V a slower peer is still reading, and that
        # peer's arrival flag would jump past the epoch it is waiting for.
        self._acks = self._transport.publish_full_mesh_consumed(
            flag_tag=f"chitu_full_mesh_flags_{slot}",
            epoch=epoch,
            epoch_guard=epoch_guard,
        )
        self._live_acks[slot] = self._acks
        self._calls += 1
        epoch_guard.record_stream(torch.cuda.current_stream())
        return output

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._transport.destroy()
