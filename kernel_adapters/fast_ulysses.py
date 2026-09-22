"""Optional adapter for triple-mu/fast-ulysses.

Upstream: https://github.com/triple-mu/fast-ulysses
Reviewed commit: 6e5dcb24dc44e781ac3091d1d9b3f9fef314fb87
License: Apache-2.0
"""

from __future__ import annotations

import logging
import os
import socket

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def probe_fast_ulysses(
    device: torch.device,
    *,
    require_hopper: bool,
    require_subgroups: bool,
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
    if dist.is_initialized() and dist.get_world_size() > local_device_count:
        return (
            False,
            "distributed WORLD spans more ranks than this node has CUDA devices",
        )
    peer_count = dist.get_world_size() if dist.is_initialized() else local_device_count
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

    def __init__(self, process_group, device: torch.device) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("fast_ulysses requires CUDA.")
        # NVSHMEM is initialized over WORLD even when the transport team is one
        # CFP branch, so validate the complete bootstrap world is single-node.
        hosts: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(hosts, socket.gethostname(), group=dist.group.WORLD)
        if len(set(hosts)) != 1:
            raise RuntimeError(
                "fast_ulysses is a single-node NVLink backend, but WORLD "
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

        pool_bytes = int(os.environ.get("CHITU_FAST_ULYSSES_POOL_BYTES", str(2 << 30)))
        self._group = UlyssesGroup(
            process_group=process_group,
            device=device,
            initial_pool_bytes=pool_bytes,
        )
        self._use_tma = _use_tma_from_env()
        self._sequence_idx = 0

    def reset_sequence(self) -> None:
        self._sequence_idx = 0

    def supports(
        self,
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
            and tensor.dtype in (torch.float16, torch.bfloat16)
            and (scatter_dim, gather_dim) in ((2, 1), (1, 2))
            and not torch.cuda.is_current_stream_capturing()
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
        if scatter_dim == 2 and gather_dim == 1:
            mode = 0
        elif scatter_dim == 1 and gather_dim == 2:
            mode = 1
        else:
            raise NotImplementedError(
                "fast_ulysses supports only Ulysses head/sequence swaps, got "
                f"scatter_dim={scatter_dim}, gather_dim={gather_dim}."
            )

        tag = f"chitu_cp_{self._sequence_idx}"
        self._sequence_idx += 1
        return self._group.all_to_all_single_4d(
            tensor,
            mode=mode,
            tag=tag,
            use_tma=self._use_tma,
        )

    def destroy(self) -> None:
        logger.info("Destroying fast-ulysses resources")
        self._group.destroy()
        logger.info("Destroyed fast-ulysses resources")
