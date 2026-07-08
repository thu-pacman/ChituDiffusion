# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional, Any, List, Dict

import torch
from logging import getLogger


from chitu_diffusion.core.distributed.comm_group import CommGroup
# from chitu.device_type import is_ascend
from chitu_diffusion.core.global_vars import get_global_args

def is_ascend():
    return False

logger = getLogger(__name__)

_PARALLEL_GROUPS_INITIALIZED = False

_WORLD_GROUP: Optional[CommGroup] = None
_TP_GROUP: Optional[CommGroup] = None
_DP_GROUP: Optional[CommGroup] = None
_ETP_GROUP: Optional[CommGroup] = None
_EP_GROUP: Optional[CommGroup] = None
_PP_GROUP: Optional[CommGroup] = None

_PP_PAIR_GROUP_DICT: dict[tuple[int, int], Any] = {}  # Compatible with NPU platforms


def get_global_var(name):
    var = globals().get(name)
    assert var is not None, f"global var {name} not initialized."
    return var


def get_world_group() -> CommGroup:
    return get_global_var("_WORLD_GROUP")


def get_tp_group() -> CommGroup:
    return get_global_var("_TP_GROUP")


def get_dp_group() -> CommGroup:
    return get_global_var("_DP_GROUP")


def get_etp_group() -> CommGroup:
    return get_global_var("_ETP_GROUP")


def get_ep_group() -> CommGroup:
    return get_global_var("_EP_GROUP")


def get_pp_group() -> CommGroup:
    return get_global_var("_PP_GROUP")


def get_tp_size() -> int:
    """return 1 if TP not initialized"""
    global _TP_GROUP
    if _TP_GROUP is None:
        return 1
    return _TP_GROUP.group_size


def get_dp_size() -> int:
    """return 1 if DP not initialized"""
    global _DP_GROUP
    if _DP_GROUP is None:
        return 1
    return _DP_GROUP.group_size


def get_etp_size() -> int:
    """return 1 if ETP not initialized"""
    global _ETP_GROUP
    if _ETP_GROUP is None:
        return 1
    return _ETP_GROUP.group_size


def get_ep_size() -> int:
    """return 1 if EP not initialized"""
    global _EP_GROUP
    if _EP_GROUP is None:
        return 1
    return _EP_GROUP.group_size


def get_pp_size() -> int:
    """return 1 if PP not initialized"""
    global _PP_GROUP
    if _PP_GROUP is None:
        return 1
    return _PP_GROUP.group_size


# Order of parallelism (from near to far):
# - Dense: TP -> DP -> PP
# - MoE: ETP -> EP -> PP
#
# Please note that DP communicates nearer ranks than PP, this is for converting
# DP attention to EP MoE. If want to parallelize the whole model with DP without
# conversion to EP, please launch multiple instances, following the instructions
# in `chitu/distributed/pd_disaggregation/README.md`.


def _get_first_level_rank_lists(first_level_size: int, world_size: int):
    assert world_size % first_level_size == 0
    return [
        list(range(i * first_level_size, (i + 1) * first_level_size))
        for i in range(world_size // first_level_size)
    ]


def _get_second_level_rank_lists(
    first_level_size: int, second_level_size: int, world_size: int
):
    assert world_size % (first_level_size * second_level_size) == 0
    rank_lists = []
    for i in range(world_size // (first_level_size * second_level_size)):
        for j in range(first_level_size):
            rank_lists.append(
                list(
                    range(
                        i * first_level_size * second_level_size + j,
                        (i + 1) * first_level_size * second_level_size + j,
                        first_level_size,
                    )
                )
            )
    return rank_lists


def _get_last_level_rank_lists(last_level_size: int, world_size: int):
    assert world_size % last_level_size == 0
    return [
        list(range(i, i + world_size, world_size // last_level_size))
        for i in range(world_size // last_level_size)
    ]


def get_tp_rank_lists(*, tp_size: int, world_size: int):
    return _get_first_level_rank_lists(first_level_size=tp_size, world_size=world_size)


def get_dp_rank_lists(*, tp_size: int, dp_size: int, world_size: int):
    return _get_second_level_rank_lists(
        first_level_size=tp_size, second_level_size=dp_size, world_size=world_size
    )


def get_etp_rank_lists(*, etp_size: int, world_size: int):
    return _get_first_level_rank_lists(first_level_size=etp_size, world_size=world_size)


def get_ep_rank_lists(*, etp_size: int, ep_size: int, world_size: int):
    return _get_second_level_rank_lists(
        first_level_size=etp_size, second_level_size=ep_size, world_size=world_size
    )


def get_pp_rank_lists(*, pp_size: int, world_size: int):
    return _get_last_level_rank_lists(last_level_size=pp_size, world_size=world_size)


def get_pp_pair_group(
    rank0: int, rank1: int
) -> Optional[torch.distributed.ProcessGroup]:
    return _PP_PAIR_GROUP_DICT.get((rank0, rank1), None)


def get_cpu_tp_group() -> Optional[torch.distributed.ProcessGroup]:
    return get_global_var("_TP_GROUP").cpu_group


def initialize_world_group(rank: int, local_rank: int, world_size: int):
    global _WORLD_GROUP
    assert _WORLD_GROUP is None

    _WORLD_GROUP = CommGroup([list(range(world_size))], rank, local_rank)


def initialize_tp_group(
    rank: int,
    local_rank: int,
    *,
    tp_size: int,
    world_size: int,
):
    global _TP_GROUP
    assert _TP_GROUP is None
    _TP_GROUP = CommGroup(
        get_tp_rank_lists(tp_size=tp_size, world_size=world_size), rank, local_rank
    )


def initialize_pp_group(
    rank: int,
    local_rank: int,
    *,
    pp_size: int,
    world_size: int,
):
    global _PP_GROUP
    assert _PP_GROUP is None

    pp_rank_lists = get_pp_rank_lists(pp_size=pp_size, world_size=world_size)
    _PP_GROUP = CommGroup(pp_rank_lists, rank, local_rank)

    if is_ascend():
        assert len(_PP_PAIR_GROUP_DICT) == 0
        if pp_size < 2:
            return
        ranks = pp_rank_lists[0]
        for i in range(pp_size):
            next_i = (i + 1) % pp_size
            rank_pair = [ranks[i], ranks[next_i]]
            pg = torch.distributed.new_group(rank_pair)
            _PP_PAIR_GROUP_DICT[(ranks[i], ranks[next_i])] = pg
            _PP_PAIR_GROUP_DICT[(ranks[next_i], ranks[i])] = pg


def initialize_dp_group(
    rank: int,
    local_rank: int,
    *,
    tp_size: int,
    dp_size: int,
    world_size: int,
):
    global _DP_GROUP
    assert _DP_GROUP is None
    _DP_GROUP = CommGroup(
        get_dp_rank_lists(tp_size=tp_size, dp_size=dp_size, world_size=world_size),
        rank,
        local_rank,
    )


def initialize_etp_group(
    rank: int,
    local_rank: int,
    *,
    etp_size: int,
    world_size: int,
):
    global _ETP_GROUP
    assert _ETP_GROUP is None
    _ETP_GROUP = CommGroup(
        get_etp_rank_lists(etp_size=etp_size, world_size=world_size), rank, local_rank
    )


def initialize_ep_group(
    rank: int, local_rank: int, *, etp_size: int, ep_size: int, world_size: int
):
    global _EP_GROUP
    assert _EP_GROUP is None
    _EP_GROUP = CommGroup(
        get_ep_rank_lists(etp_size=etp_size, ep_size=ep_size, world_size=world_size),
        rank,
        local_rank,
        force_no_dedup=is_ascend(),
    )


def initialize_parallel_groups(
    *, tp_size: int, dp_size: int = 1, etp_size: int = 1, ep_size: int = 1, pp_size: int
):
    global _PARALLEL_GROUPS_INITIALIZED
    assert not _PARALLEL_GROUPS_INITIALIZED

    logger.info(
        f"initialize_parallel_groups: {tp_size=}, {pp_size=}, {dp_size=} {ep_size=}"
    )
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.distributed.get_world_size()
    initialize_world_group(rank, local_rank, world_size)
    initialize_tp_group(rank, local_rank, tp_size=tp_size, world_size=world_size)
    initialize_dp_group(
        rank, local_rank, tp_size=tp_size, dp_size=dp_size, world_size=world_size
    )
    initialize_etp_group(rank, local_rank, etp_size=etp_size, world_size=world_size)
    initialize_ep_group(
        rank, local_rank, etp_size=etp_size, ep_size=ep_size, world_size=world_size
    )
    initialize_pp_group(rank, local_rank, pp_size=pp_size, world_size=world_size)

    _PARALLEL_GROUPS_INITIALIZED = True


def parallel_groups_initialized():
    return _PARALLEL_GROUPS_INITIALIZED


def destroy_parallel_groups():
    get_tp_group().destroy()
    get_pp_group().destroy()
    get_world_group().destroy()
    get_dp_group().destroy()
    # Currently we don't destroy ep_group as it is a copy of tp/dp
    # get_ep_group().destroy()


# CFG and Context Parallelism support for diffusion models
_CP_GROUP: Optional[CommGroup] = None
_CFG_GROUP: Optional[CommGroup] = None
_UP_GROUP_DICT: Optional[Dict[int, CommGroup]] = None
# Data parallel support for diffusion: each replica is a contiguous block of
# ``cfg_size * cp_size`` ranks that runs a full model independently. The replica
# group spans one such block (used for replica-local task broadcast/barrier); the
# DiT DP group connects the same in-block offset across replicas.
_DIT_REPLICA_GROUP: Optional[CommGroup] = None
_DIT_DP_GROUP: Optional[CommGroup] = None

def get_cp_group() -> CommGroup:
    return get_global_var("_CP_GROUP")

def get_cfg_group() -> CommGroup:
    return get_global_var("_CFG_GROUP")

def get_up_group(size: int) -> CommGroup:
    global _UP_GROUP_DICT
    if _UP_GROUP_DICT is None or size not in _UP_GROUP_DICT:
        raise ValueError(f"UP group of size {size} not initialized.")
    return _UP_GROUP_DICT[size]

def get_dit_replica_group() -> CommGroup:
    """The rank block (size cfg_size*cp_size) that cooperatively runs one task.

    When dp_size==1 this equals the world group, so single-replica behavior is
    unchanged. Task metadata broadcast and the per-step barrier are scoped to
    this group so different replicas can process different requests.
    """
    return get_global_var("_DIT_REPLICA_GROUP")

def get_dit_dp_group() -> CommGroup:
    """DP group connecting corresponding ranks across replicas."""
    return get_global_var("_DIT_DP_GROUP")

def get_dit_dp_size() -> int:
    global _DIT_DP_GROUP
    if _DIT_DP_GROUP is None:
        return 1
    return _DIT_DP_GROUP.group_size

def get_dit_replica_index() -> int:
    """0-based index of the replica this rank belongs to (0 when dp_size==1)."""
    global _DIT_REPLICA_GROUP
    if _DIT_REPLICA_GROUP is None:
        return 0
    replica_size = _DIT_REPLICA_GROUP.group_size
    return _DIT_REPLICA_GROUP.global_rank // replica_size


def _replica_blocks(replica_size: int, world_size: int) -> List[List[int]]:
    assert world_size % replica_size == 0, (
        f"world_size {world_size} must be divisible by replica_size {replica_size}"
    )
    return [
        list(range(base, base + replica_size))
        for base in range(0, world_size, replica_size)
    ]

def initialize_cfg_group(cfg_size: int, rank: int, local_rank: int, world_size: int, replica_size: int):
    global _CFG_GROUP
    assert _CFG_GROUP is None

    blocks = _replica_blocks(replica_size, world_size)
    if cfg_size == 1:
        # No CFG parallelism: every rank is its own singleton group.
        _CFG_GROUP = CommGroup([[idx] for idx in range(world_size)], rank, local_rank)
    elif cfg_size == 2:
        # CFG parallelism with pairs, formed within each replica block. The two
        # cp-halves of a block are paired offset-wise: (base+i, base+cp_size+i).
        assert replica_size % 2 == 0, "replica_size must be even for CFG parallelism"
        half_local = replica_size // 2
        rank_list = []
        for block in blocks:
            base = block[0]
            for i in range(half_local):
                rank_list.append([base + i, base + half_local + i])
        _CFG_GROUP = CommGroup(rank_list, rank, local_rank)
    else:
        raise ValueError("CFG size can only be 1 or 2")

def _cp_rank_lists(cfg_size: int, world_size: int, replica_size: int) -> List[List[int]]:
    blocks = _replica_blocks(replica_size, world_size)
    rank_list: List[List[int]] = []
    if cfg_size == 2:
        half_local = replica_size // 2
        for block in blocks:
            base = block[0]
            rank_list.append(list(range(base, base + half_local)))
            rank_list.append(list(range(base + half_local, base + replica_size)))
    else:
        for block in blocks:
            base = block[0]
            rank_list.append(list(range(base, base + replica_size)))
    return rank_list

def initialize_cp_group(cp_size: int, cfg_size: int, rank: int, local_rank: int, world_size: int, replica_size: int):
    global _CP_GROUP
    assert _CP_GROUP is None
    _CP_GROUP = CommGroup(_cp_rank_lists(cfg_size, world_size, replica_size), rank, local_rank)

def initialize_up_groups(up_sizes: List[int], up: int, cfg_size: int, rank: int, local_rank: int, world_size: int, replica_size: int):
    global _UP_GROUP_DICT
    assert _UP_GROUP_DICT is None

    _UP_GROUP_DICT = {}

    cp_group = get_cp_group()
    cp_group_size = cp_group.group_size

    cp_group_ranks = _cp_rank_lists(cfg_size, world_size, replica_size)

    for up_size in up_sizes:
        if up_size == 0 or cp_group_size % up_size != 0:
            continue

        if up_size > up:
            continue

        rank_list = []
        for cp_ranks in cp_group_ranks:
            for i in range(0, len(cp_ranks), up_size):
                group = cp_ranks[i:i+up_size]
                if group:
                    rank_list.append(group)

        if rank_list:
            _UP_GROUP_DICT[up_size] = CommGroup(rank_list, rank, local_rank)

def initialize_dit_replica_group(rank: int, local_rank: int, world_size: int, replica_size: int):
    global _DIT_REPLICA_GROUP
    assert _DIT_REPLICA_GROUP is None
    _DIT_REPLICA_GROUP = CommGroup(_replica_blocks(replica_size, world_size), rank, local_rank)

def initialize_dit_dp_group(rank: int, local_rank: int, world_size: int, replica_size: int):
    global _DIT_DP_GROUP
    assert _DIT_DP_GROUP is None
    dp_size = world_size // replica_size
    if dp_size == 1:
        # Single replica: every rank is its own singleton DP group.
        rank_list = [[idx] for idx in range(world_size)]
    else:
        # Connect the same in-block offset across replicas.
        rank_list = [
            [offset + r * replica_size for r in range(dp_size)]
            for offset in range(replica_size)
        ]
    _DIT_DP_GROUP = CommGroup(rank_list, rank, local_rank)

# ---------------------------------------------------------------------------
# M6 dynamic sequence-parallel (SP/CP) degree switching.
#
# ``_DYNAMIC_CP_GROUP_DICT`` maps an SP degree ``d`` -> a pre-warmed CommGroup
# whose rank_lists partition each replica block into contiguous shards of size
# ``d``. All groups are created ONCE at init so switching degree at runtime is a
# pure pointer swap (``set_active_cp_group``) with zero NCCL group-creation cost.
# ``_ACTIVE_CP_DEGREE`` tracks the degree the swappable ``_CP_GROUP`` currently
# points at; ``_BASE_CP_GROUP`` remembers the launch-time group so the baseline
# (dynamic_sp OFF) behaviour is untouched.
# ---------------------------------------------------------------------------
_DYNAMIC_CP_GROUP_DICT: Optional[Dict[int, CommGroup]] = None
_ACTIVE_CP_DEGREE: Optional[int] = None
_BASE_CP_GROUP: Optional[CommGroup] = None


def dynamic_sp_enabled() -> bool:
    return _DYNAMIC_CP_GROUP_DICT is not None


def dynamic_sp_degrees() -> List[int]:
    if _DYNAMIC_CP_GROUP_DICT is None:
        return []
    return sorted(_DYNAMIC_CP_GROUP_DICT.keys())


def get_dynamic_cp_group(degree: int) -> CommGroup:
    if _DYNAMIC_CP_GROUP_DICT is None or degree not in _DYNAMIC_CP_GROUP_DICT:
        raise ValueError(
            f"Dynamic CP group of degree {degree} not initialized "
            f"(available: {dynamic_sp_degrees()})."
        )
    return _DYNAMIC_CP_GROUP_DICT[degree]


def get_active_cp_degree() -> int:
    """Degree the swappable CP group currently points at (1 when not init)."""
    if _ACTIVE_CP_DEGREE is None:
        return get_cp_size()
    return _ACTIVE_CP_DEGREE


def set_active_cp_group(degree: int) -> None:
    """Point the global ``_CP_GROUP`` at the pre-warmed group for ``degree``.

    O(1) pointer swap over already-created communicators -- no NCCL group
    creation happens here (verified by the pre-warm at init). The Z-Image
    ``cp_forward`` reads ``get_cp_group()`` fresh every call, so the next
    transformer forward transparently shards at the new degree.
    """
    global _CP_GROUP, _ACTIVE_CP_DEGREE
    _CP_GROUP = get_dynamic_cp_group(degree)
    _ACTIVE_CP_DEGREE = degree


def get_cp_size() -> int:
    global _CP_GROUP
    if _CP_GROUP is None:
        return 1
    return _CP_GROUP.group_size


def initialize_dynamic_sp_groups(
    cfg_size: int,
    world_size: int,
    replica_size: int,
    rank: int,
    local_rank: int,
    degrees: Optional[List[int]] = None,
) -> None:
    """Pre-warm CP communicator groups for every feasible SP degree.

    Given a replica block of ``replica_size`` ranks (``cfg_size * cp_size``), the
    feasible SP degrees are the divisors of ``replica_size`` (e.g. replica_size=4
    -> {1, 2, 4}). For each degree ``d`` we build a CommGroup whose rank_lists
    split every replica block into contiguous shards of size ``d`` -- the DP /
    replica arrangement is the complement (``replica_size / d`` shards per block).
    The launch-time ``_CP_GROUP`` (degree == cp_size) is reused so nothing about
    the baseline layout changes; only the *extra* degrees allocate new groups.
    """
    global _DYNAMIC_CP_GROUP_DICT, _ACTIVE_CP_DEGREE, _BASE_CP_GROUP
    assert _CP_GROUP is not None, "CP group must be initialized before dynamic SP groups."

    base_degree = _CP_GROUP.group_size
    _BASE_CP_GROUP = _CP_GROUP

    if degrees is None:
        degrees = [d for d in range(1, replica_size + 1) if replica_size % d == 0]
    degrees = sorted(set(int(d) for d in degrees if replica_size % int(d) == 0))

    group_dict: Dict[int, CommGroup] = {}
    for degree in degrees:
        if degree == base_degree:
            # Reuse the launch-time CP group verbatim (same rank_lists).
            group_dict[degree] = _CP_GROUP
            continue
        # A CP layout at ``degree`` is exactly the standard CP rank-lists with
        # replica_size == degree (contiguous blocks of size ``degree``). new_group
        # deduping in comm_group means identical rank tuples are never rebuilt.
        rank_lists = _cp_rank_lists(cfg_size=1, world_size=world_size, replica_size=degree)
        group_dict[degree] = CommGroup(rank_lists, rank, local_rank)

    _DYNAMIC_CP_GROUP_DICT = group_dict
    _ACTIVE_CP_DEGREE = base_degree

    if rank == 0:
        logger.info(
            "Pre-warmed dynamic SP groups: degrees=%s base=%d (replica_size=%d, world=%d)",
            sorted(group_dict.keys()), base_degree, replica_size, world_size,
        )
        for degree, grp in sorted(group_dict.items()):
            logger.info("  dynamic CP degree %d: rank_list=%s", degree, grp.rank_list)


# ---------------------------------------------------------------------------
# slo_elastic pool lanes (request-level DP + per-request SP width).
#
# The pool engine partitions the world into lanes of width {1,2,4} using
# *canonical contiguous placement*: widths are placed left-to-right widest-first,
# so a lane of width ``w`` always occupies the aligned block
# ``[offset, offset+w)`` with ``offset % w == 0`` (partitions {4}, {2,2},
# {2,1,1}, {1,1,1,1} for N=4). Under that placement every rank's lane is exactly
# its aligned block of size ``w`` -- which is precisely one subgroup of the M6
# ``dynamic_sp`` degree-``w`` group. So lane subgroups need no new communicators:
# they are the pre-warmed dynamic CP groups, activated per lane per round.
# ---------------------------------------------------------------------------
def lane_offset_for_rank(rank: int, width: int) -> int:
    """Aligned block offset of the width-``width`` lane containing ``rank``."""
    return (int(rank) // int(width)) * int(width)


def get_lane_cp_group(offset: int, width: int) -> CommGroup:
    """The pre-warmed CP communicator for a canonical lane ``[offset, offset+width)``.

    Returns the M6 dynamic-SP group of degree ``width``; for the calling rank that
    group's active subgroup is exactly the rank's aligned block. Requires the pool
    lane groups to have been pre-warmed (``dynamic_sp`` / :func:`initialize_lane_groups`)
    and canonical placement (``offset % width == 0``).
    """
    width = int(width)
    offset = int(offset)
    if width <= 0 or offset % width != 0:
        raise ValueError(f"non-canonical lane placement: offset={offset}, width={width}")
    return get_dynamic_cp_group(width)


def set_active_lane_group(offset: int, width: int) -> None:
    """Point this rank's active CP group at its lane ``[offset, offset+width)``.

    O(1) pointer swap over the pre-warmed dynamic groups (no NCCL creation). After
    this call the next transformer forward on this rank shards over exactly the
    ``width`` ranks of its lane; width 1 means a DP replica (no CP collectives).
    """
    width = int(width)
    offset = int(offset)
    if width <= 0 or offset % width != 0:
        raise ValueError(f"non-canonical lane placement: offset={offset}, width={width}")
    set_active_cp_group(width)


def lane_widths_available() -> List[int]:
    """SP widths the pool engine may use (== pre-warmed dynamic-SP degrees)."""
    return dynamic_sp_degrees()


def initialize_lane_groups(
    world_size: Optional[int] = None,
    widths: Optional[List[int]] = None,
) -> None:
    """Ensure the canonical pool-lane CP groups (widths {1,2,4,...}) are pre-warmed.

    Thin wrapper over :func:`initialize_dynamic_sp_groups` for the slo_elastic pool
    engine: pre-warms one CP group per feasible width so switching a lane's width at a
    step boundary is a pure pointer swap. Idempotent -- a no-op if already pre-warmed.
    Must be called after the diffusion groups (so ``_CP_GROUP`` exists) and collectively
    on all ranks.
    """
    if dynamic_sp_enabled():
        return
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    replica_size = world_size  # pool runs one replica spanning the whole world (cfg=1)
    initialize_dynamic_sp_groups(
        cfg_size=1,
        world_size=world_size,
        replica_size=replica_size,
        rank=rank,
        local_rank=local_rank,
        degrees=widths,
    )


def initialize_diffusion_parallel_groups(
    cfg_size: int,
    cp_size: int,
    up: int = 8,
    dp_size: int = 1,
    dynamic_sp: bool = False,
):
    global _PARALLEL_GROUPS_INITIALIZED
    assert not _PARALLEL_GROUPS_INITIALIZED

    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.distributed.get_world_size()

    replica_size = cfg_size * cp_size
    assert world_size == dp_size * replica_size, (
        f"World size mismatch: {world_size} != dp_size({dp_size}) * "
        f"cfg_size({cfg_size}) * cp_size({cp_size})"
    )

    logger.info(
        f"initialize_diffusion_parallel_groups: {cfg_size=}, {cp_size=}, {up=}, "
        f"{dp_size=}, {replica_size=}"
    )

    # Initialize groups in order
    initialize_world_group(rank, local_rank, world_size)
    initialize_dit_replica_group(rank, local_rank, world_size, replica_size)
    initialize_dit_dp_group(rank, local_rank, world_size, replica_size)
    initialize_cfg_group(cfg_size, rank, local_rank, world_size, replica_size)
    initialize_cp_group(cp_size, cfg_size, rank, local_rank, world_size, replica_size)

    max_up_size = min(up, cp_size)
    up_sizes = [max_up_size, max_up_size // 2] # TODO: More up sizes to support DiTango Support
    initialize_up_groups(up_sizes, up, cfg_size, rank, local_rank, world_size, replica_size)

    # M6: pre-warm CP groups for every feasible SP degree so runtime switching
    # pays no NCCL group-creation cost. Gated by dynamic_sp; baseline untouched.
    if dynamic_sp:
        initialize_dynamic_sp_groups(
            cfg_size=cfg_size,
            world_size=world_size,
            replica_size=replica_size,
            rank=rank,
            local_rank=local_rank,
        )
    
    # Debug logging
    if rank == 0:
        logger.info(f"Replica groups initialized: {get_dit_replica_group().rank_list}")
        logger.info(f"DiT DP groups initialized: {get_dit_dp_group().rank_list}")
        logger.info(f"CFG groups initialized: {get_cfg_group().rank_list}")
        logger.info(f"CP groups initialized: {get_cp_group().rank_list}")
        for size, up_group in _UP_GROUP_DICT.items():
            logger.info(f"UP group size {size}: {up_group.rank_list}")
