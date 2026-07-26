from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Iterator

import torch
import torch.distributed as dist

from .topology import UspTopology


@dataclass(frozen=True)
class ActiveLaneTopology:
    ranks: tuple[int, ...]
    rank: int
    rank_in_lane: int
    width: int
    process_group: object | None

    @property
    def is_leader(self) -> bool:
        return self.rank_in_lane == 0


@dataclass(frozen=True)
class CfgParallelTopology:
    """CFP-2 placement for one lane: cond half, uncond half, paired shards."""

    lane_ranks: tuple[int, ...]
    cp_ranks: tuple[int, ...]
    cfg_ranks: tuple[int, int]
    branch_index: int
    cp_rank: int
    cp_width: int
    cp_process_group: object | None
    cfg_process_group: object

    @property
    def branch_name(self) -> str:
        return "cond" if self.branch_index == 0 else "uncond"


def cfg_parallel_rank_groups(
    lane_ranks: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, int], ...]]:
    """Return the two CP halves and shard-aligned CFG pairs for a CFP-2 lane."""
    lane = tuple(int(rank) for rank in lane_ranks)
    if len(lane) < 2 or len(lane) % 2:
        raise ValueError("CFP-2 requires an even lane width of at least 2")
    stride = len(lane) // 2
    cp_groups = (lane[:stride], lane[stride:])
    cfg_pairs = tuple(zip(cp_groups[0], cp_groups[1], strict=True))
    return cp_groups, cfg_pairs


class EpeParallelContext:
    """Instance-local EPAC process-group registry and active lane view.

    All configured groups are created in a deterministic order during startup.
    Activating a lane only changes an instance field; it never calls new_group().
    The registry is shared by context parallelism and CFG parallelism.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        local_rank: int,
        allowed_widths: tuple[int, ...],
        groups: dict[tuple[int, ...], object | None],
        usp_topologies: dict[tuple[int, ...], UspTopology],
        cfg_topologies: dict[tuple[int, ...], CfgParallelTopology],
        owned_groups: tuple[object, ...],
        owns_world: bool,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_rank = int(local_rank)
        self.allowed_widths = allowed_widths
        self._groups = groups
        self._usp_topologies = usp_topologies
        self._cfg_topologies = cfg_topologies
        self._owned_groups = owned_groups
        self._owns_world = owns_world
        self._worker_control_plane: object | None = None
        self._owned_control_groups: list[object] = []
        self._lock = RLock()
        self._closed = False
        self._active = self._topology_for(tuple(range(world_size)))

    @classmethod
    def from_torchrun(
        cls,
        *,
        allowed_widths: tuple[int, ...] | None = None,
        init_process_group: bool = True,
        backend: str | None = None,
        owns_process_group: bool | None = None,
        ulysses_degree: int = 1,
    ) -> "EpeParallelContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        owns_world = False

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if world_size > 1 and not dist.is_initialized():
            if not init_process_group:
                raise RuntimeError("torch.distributed is not initialized")
            selected_backend = backend or (
                "nccl" if torch.cuda.is_available() else "gloo"
            )
            init_kwargs = {}
            if selected_backend == "nccl":
                init_kwargs["device_id"] = torch.device("cuda", local_rank)
            dist.init_process_group(selected_backend, **init_kwargs)
            owns_world = True
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            if owns_process_group is not None:
                owns_world = bool(owns_process_group)

        widths = allowed_widths or tuple(
            width for width in range(1, world_size + 1) if world_size % width == 0
        )
        widths = tuple(int(width) for width in widths)
        cls._validate_widths(widths, world_size)
        if ulysses_degree < 1:
            raise ValueError("ulysses_degree must be positive")

        groups: dict[tuple[int, ...], object | None] = {}
        owned: list[object] = []
        world_ranks = tuple(range(world_size))
        groups[world_ranks] = dist.group.WORLD if dist.is_initialized() else None

        if dist.is_initialized():
            for width in widths:
                if width in {1, world_size}:
                    continue
                for offset in range(0, world_size, width):
                    ranks = tuple(range(offset, offset + width))
                    group = dist.new_group(ranks=list(ranks))
                    groups[ranks] = group
                    if rank in ranks:
                        owned.append(group)
        for width in widths:
            if width != 1:
                continue
            for current_rank in range(world_size):
                groups[(current_rank,)] = None

        auxiliary_groups: dict[tuple[int, ...], object | None] = {}

        def process_group_for(ranks: tuple[int, ...]) -> object | None:
            if len(ranks) == 1:
                return None
            if ranks in groups:
                return groups[ranks]
            if ranks in auxiliary_groups:
                return auxiliary_groups[ranks]
            group = dist.new_group(ranks=list(ranks))
            auxiliary_groups[ranks] = group
            if rank in ranks:
                owned.append(group)
            return group

        usp_topologies: dict[tuple[int, ...], UspTopology] = {}
        topology_widths = tuple(
            sorted({*widths, *(width // 2 for width in widths if width % 2 == 0)})
        )
        for width in topology_widths:
            effective_ulysses = min(int(ulysses_degree), width)
            while width % effective_ulysses:
                effective_ulysses -= 1
            ring_degree = width // effective_ulysses
            for offset in range(0, world_size, width):
                lane = tuple(range(offset, offset + width))
                ulysses_groups = tuple(
                    tuple(
                        lane[ring_index * effective_ulysses + ulysses_index]
                        for ulysses_index in range(effective_ulysses)
                    )
                    for ring_index in range(ring_degree)
                )
                ring_groups = tuple(
                    tuple(
                        lane[ring_index * effective_ulysses + ulysses_index]
                        for ring_index in range(ring_degree)
                    )
                    for ulysses_index in range(effective_ulysses)
                )
                for subgroup in (*ulysses_groups, *ring_groups):
                    process_group_for(subgroup)
                if rank not in lane:
                    continue
                ulysses_ranks = next(group for group in ulysses_groups if rank in group)
                ring_ranks = next(group for group in ring_groups if rank in group)
                usp_topologies[lane] = UspTopology(
                    lane_ranks=lane,
                    ulysses_ranks=ulysses_ranks,
                    ring_ranks=ring_ranks,
                    ulysses_rank=ulysses_ranks.index(rank),
                    ring_rank=ring_ranks.index(rank),
                    ulysses_degree=effective_ulysses,
                    ring_degree=ring_degree,
                    ulysses_process_group=process_group_for(ulysses_ranks),
                    ring_process_group=process_group_for(ring_ranks),
                )

        cfg_topologies: dict[tuple[int, ...], CfgParallelTopology] = {}
        for width in widths:
            if width < 2 or width % 2:
                continue
            for offset in range(0, world_size, width):
                lane = tuple(range(offset, offset + width))
                cp_groups, cfg_pairs = cfg_parallel_rank_groups(lane)
                for subgroup in (*cp_groups, *cfg_pairs):
                    process_group_for(subgroup)
                if rank not in lane:
                    continue
                branch_index = 0 if rank in cp_groups[0] else 1
                cp_ranks = cp_groups[branch_index]
                cfg_ranks = next(pair for pair in cfg_pairs if rank in pair)
                cfg_topologies[lane] = CfgParallelTopology(
                    lane_ranks=lane,
                    cp_ranks=cp_ranks,
                    cfg_ranks=cfg_ranks,
                    branch_index=branch_index,
                    cp_rank=cp_ranks.index(rank),
                    cp_width=len(cp_ranks),
                    cp_process_group=process_group_for(cp_ranks),
                    cfg_process_group=process_group_for(cfg_ranks),
                )

        groups.update(auxiliary_groups)

        return cls(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            allowed_widths=widths,
            groups=groups,
            usp_topologies=usp_topologies,
            cfg_topologies=cfg_topologies,
            owned_groups=tuple(owned),
            owns_world=owns_world,
        )

    @staticmethod
    def _validate_widths(widths: tuple[int, ...], world_size: int) -> None:
        if not widths or tuple(sorted(set(widths))) != widths:
            raise ValueError("allowed_widths must be sorted, unique, and non-empty")
        if 1 not in widths or world_size not in widths:
            raise ValueError("allowed_widths must include 1 and world_size")
        invalid = [width for width in widths if width < 1 or world_size % width]
        if invalid:
            raise ValueError(
                f"lane widths {invalid} must be positive divisors of world size {world_size}"
            )

    def _topology_for(self, ranks: tuple[int, ...]) -> ActiveLaneTopology:
        if self.rank not in ranks:
            raise ValueError(f"rank {self.rank} is not a member of lane {ranks}")
        if ranks not in self._groups:
            raise ValueError(f"lane {ranks} was not pre-created")
        return ActiveLaneTopology(
            ranks=ranks,
            rank=self.rank,
            rank_in_lane=ranks.index(self.rank),
            width=len(ranks),
            process_group=self._groups[ranks],
        )

    @property
    def active(self) -> ActiveLaneTopology:
        with self._lock:
            if self._closed:
                raise RuntimeError("parallel context is closed")
            return self._active

    @property
    def active_usp(self) -> UspTopology:
        topology = self.active
        try:
            return self._usp_topologies[topology.ranks]
        except KeyError as exc:
            raise RuntimeError(
                f"USP topology was not initialized for lane {topology.ranks}"
            ) from exc

    def cfg_parallel_topology(
        self, lane_ranks: tuple[int, ...]
    ) -> CfgParallelTopology | None:
        """Return this rank's CFP-2 topology, or None for an ineligible lane."""
        lane = tuple(int(rank) for rank in lane_ranks)
        return self._cfg_topologies.get(lane)

    @contextmanager
    def activate(self, ranks: tuple[int, ...]) -> Iterator[ActiveLaneTopology]:
        topology = self._topology_for(tuple(int(rank) for rank in ranks))
        with self._lock:
            previous = self._active
            self._active = topology
        try:
            yield topology
        finally:
            with self._lock:
                self._active = previous

    def all_gather_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        topology = self.active
        if topology.width == 1:
            return tensor
        pieces = [torch.empty_like(tensor) for _ in range(topology.width)]
        dist.all_gather(pieces, tensor.contiguous(), group=topology.process_group)
        return torch.cat(pieces, dim=1).contiguous()

    def all_gather_cfg(
        self,
        tensor: torch.Tensor,
        topology: CfgParallelTopology,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather cond and uncond predictions in stable branch order."""
        pieces = [torch.empty_like(tensor) for _ in range(2)]
        dist.all_gather(
            pieces,
            tensor.contiguous(),
            group=topology.cfg_process_group,
        )
        return pieces[0], pieces[1]

    def barrier(self) -> None:
        topology = self.active
        if topology.width > 1:
            kwargs = {}
            if torch.cuda.is_available():
                kwargs["device_ids"] = [self.local_rank]
            dist.barrier(group=topology.process_group, **kwargs)

    def initialize_worker_control_group(self) -> None:
        """Collectively create pairwise CPU groups for asynchronous rank control."""
        if self.world_size == 1 or self._worker_control_plane is not None:
            return
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        group = dist.new_group(
            ranks=list(range(self.world_size)),
            backend="gloo",
            group_desc="epac-singleton-control",
        )
        self._worker_control_plane = group
        self._owned_control_groups.append(group)

    def worker_control_group(self, peer: int) -> object:
        if self.world_size == 1:
            return None
        if not 0 < int(peer) < self.world_size:
            raise ValueError(f"invalid singleton lane peer: {peer}")
        if self._worker_control_plane is None:
            raise RuntimeError("worker control group has not been initialized")
        return self._worker_control_plane

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if dist.is_initialized():
            for group in reversed(self._owned_control_groups):
                dist.destroy_process_group(group)
            for group in reversed(self._owned_groups):
                dist.destroy_process_group(group)
            if self._owns_world:
                dist.destroy_process_group()
