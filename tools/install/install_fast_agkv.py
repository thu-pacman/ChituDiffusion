#!/usr/bin/env python3
"""Apply ChituDiffusion's Fast CP sources to a fast-ulysses checkout.

The sources live in ``chitu_diffusion/parallel/cp/fast/csrc`` and are copied
over the checkout: they are ChituDiffusion's, not patches against upstream. Two
files stay anchor-patched instead, because upstream owns them and we only add to
them -- ``bindings.cpp`` (operator registrations plus the tuned copy-engine
all-to-all body) and the two Python modules that have to follow the caller's
process group.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

INCLUDE_ANCHOR = '#include "a2a_config.cuh"\n'
INCLUDE_LINE = '#include "fast_agkv.cuh"\n'
REGISTER_ANCHOR = (
    '    m.impl("all_to_all_single_4d_ce", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::all_to_all_single_4d_ce);\n"
)
REGISTER_BLOCK = (
    "\n"
    '    m.def("all_gather_kv_4d('
    "__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
    "Tensor key, Tensor value, str key_tag, str value_tag, bool use_ce=False) "
    '-> (Tensor, Tensor)");\n'
    '    m.impl("all_gather_kv_4d", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::all_gather_kv_4d);\n"
)
ALL_GATHER_REGISTER_ANCHOR = (
    '    m.impl("all_gather_kv_4d", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::all_gather_kv_4d);\n"
)
RING_CE_REGISTER_BLOCK = (
    "\n"
    '    m.def("ring_ce_prepare_kv_4d('
    "__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
    "Tensor key, Tensor value, str key_tag, str value_tag, str state_tag, "
    'int epoch) -> (Tensor, Tensor, Tensor, Tensor, int[])");\n'
    '    m.impl("ring_ce_prepare_kv_4d", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::ring_ce_prepare_kv_4d);\n"
    "\n"
    '    m.def("ring_ce_forward_kv_4d('
    "__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
    "Tensor landing_key, Tensor landing_value, int peer_key_ptr, "
    "int peer_value_ptr, int peer_arrival_ptr, int local_ack_ptr, "
    'int epoch, int phase) -> ()");\n'
    '    m.impl("ring_ce_forward_kv_4d", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::ring_ce_forward_kv_4d);\n"
    "\n"
    '    m.def("ring_ce_wait_ready('
    'Tensor arrivals, int world_size, int epoch, int phase) -> Tensor");\n'
    '    m.impl("ring_ce_wait_ready", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::ring_ce_wait_ready);\n"
    "\n"
    '    m.def("ring_ce_publish_consumed('
    "__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
    'int peer_ack_ptr, int device_index, int epoch, int phase) -> ()");\n'
    '    m.impl("ring_ce_publish_consumed", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::ring_ce_publish_consumed);\n"
)
HYBRID_REGISTER_BLOCK = (
    "\n"
    '    m.def("logical_subgroup_all_to_all_single_4d_ce('
    "__torch__.torch.classes.fast_ulysses.UlyssesGroup group, Tensor input, "
    "int[] peer_ranks, int mode, str tag, int storage_slots=1, int storage_index=0) "
    '-> Tensor");\n'
    '    m.impl("logical_subgroup_all_to_all_single_4d_ce", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::logical_subgroup_all_to_all_single_4d_ce);\n"
    "\n"
    '    m.def("logical_subgroup_barrier('
    "__torch__.torch.classes.fast_ulysses.UlyssesGroup group, int[] peer_ranks, "
    'str tag, int epoch) -> Tensor");\n'
    '    m.impl("logical_subgroup_barrier", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::logical_subgroup_barrier);\n"
    "\n"
    '    m.def("u2fm2_prepare_qkv('
    "__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
    "Tensor query, Tensor key, Tensor value, str query_tag, str key_tag, "
    "str value_tag, str flag_tag, str ulysses_barrier_tag, int ulysses_epoch, "
    "int full_mesh_epoch, int segment_stride, int remote_blocks) "
    '-> (Tensor, Tensor, Tensor, Tensor, Tensor)");\n'
    '    m.impl("u2fm2_prepare_qkv", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::u2fm2_prepare_qkv);\n"
)
HYBRID_SCHEMA_OLD = (
    '    m.def("u2fm2_prepare_qkv(__torch__.torch.classes.fast_ulysses.UlyssesGroup '
    "group, Tensor query, Tensor key, Tensor value, str query_tag, str key_tag, "
    "str value_tag, str flag_tag, str ulysses_barrier_tag, int ulysses_epoch, "
    "int full_mesh_epoch, int segment_stride) -> "
    '(Tensor, Tensor, Tensor, Tensor, Tensor)");\n'
)
HYBRID_SCHEMA_NEW = HYBRID_SCHEMA_OLD.replace(
    "int segment_stride)", "int segment_stride, int remote_blocks)"
)
# The copy-engine all-to-all takes the fan-out the shared CE tuner measured for
# its transfer shape, instead of one stream per peer. Rationale and measurements:
# chitu_diffusion/parallel/cp/fast/csrc/ce_schedule.cuh.
CE_A2A_LAUNCH_OLD = """    launch_a2a_ce(input.data_ptr(),
                  buf.peer_ptrs,
                  dims,
                  static_cast<int>(mode),
                  static_cast<int>(input.element_size()),
                  group->ce_resources(),
                  stream);
"""
CE_A2A_LAUNCH_NEW = """    // Which fan-out wins depends on the host's interconnect, so it is measured once per
    // transfer shape (ce_schedule.cuh). The tuning barriers stay in lockstep because
    // every rank misses the same key on the same call under SPMD -- the same contract the
    // TMA/non-TMA tuner on the kernel path relies on. No layered candidates here: the
    // relay only works when every destination receives the same bytes, and in an
    // all-to-all each one receives its own slice.
    const int  elem          = static_cast<int>(input.element_size());
    const auto phase_barrier = [&group, stream] { group->fast_barrier(stream); };
    const auto launch        = [&](const CESchedule& schedule) {
        launch_a2a_ce(input.data_ptr(),
                      buf.peer_ptrs,
                      dims,
                      static_cast<int>(mode),
                      elem,
                      schedule,
                      group->ce_resources(),
                      stream);
    };
    const CEScheduleKey schedule_key{
        static_cast<int64_t>(mode == 0 ? CEPath::all_to_all_mode_0 : CEPath::all_to_all_mode_1),
        dims.b,
        dims.s_local,
        dims.n_local,
        dims.d,
        elem};
    launch(group->resolve_ce_schedule(schedule_key, [&] {
        return tune_ce_schedule(*group, /*layered_possible=*/false, launch, phase_barrier, stream);
    }));
"""
LOCAL_WORLD_REPLACEMENTS = (
    (
        "        self.peer_global_ranks = list(dist.get_process_group_ranks(pg))\n",
        "        self.peer_global_ranks = list(dist.get_process_group_ranks(pg))\n"
        "        self.peer_runtime_ranks = list(range(self.world_size))\n",
    ),
    (
        "        registrations: list[object] = [None] * dist.get_world_size()\n",
        "        registrations: list[object] = [None] * self.world_size\n",
    ),
    (
        "                tuple(int(r) for r in self.peer_global_ranks),\n",
        "                tuple(self.peer_runtime_ranks),\n",
    ),
    (
        "            group=dist.group.WORLD,\n",
        "            group=pg,\n",
    ),
    (
        "        if flattened != list(range(dist.get_world_size())):\n",
        "        if flattened != list(range(self.world_size)):\n",
    ),
    (
        "        if dist.get_rank() == 0:\n",
        "        if self.rank == 0:\n",
    ),
    (
        "        dist.broadcast(uid_t, src=0, group=dist.group.WORLD)\n",
        "        dist.broadcast(uid_t, src=self.peer_global_ranks[0], group=pg)\n",
    ),
    (
        "            dist.get_rank(),\n            dist.get_world_size(),\n",
        "            self.rank,\n            self.world_size,\n",
    ),
    (
        "        cls.init_world(uid_t.tolist(), dist.get_rank(), dist.get_world_size())\n",
        "        cls.init_world(uid_t.tolist(), self.rank, self.world_size)\n",
    ),
    (
        "        dist.barrier(group=dist.group.WORLD)\n",
        "        dist.barrier(group=pg)\n",
    ),
    (
        "            [int(r) for r in self.peer_global_ranks],\n",
        "            self.peer_runtime_ranks,\n",
    ),
)
OPTIONAL_LOCAL_WORLD_ANCHORS = {
    "        registrations: list[object] = [None] * dist.get_world_size()\n",
    "                tuple(int(r) for r in self.peer_global_ranks),\n",
    "            group=dist.group.WORLD,\n",
    "        if flattened != list(range(dist.get_world_size())):\n",
    "            dist.get_rank(),\n            dist.get_world_size(),\n",
    "        dist.barrier(group=dist.group.WORLD)\n",
}
LEGACY_WORLD_ONLY_GUARD = """        if self.world_size != dist.get_world_size():
            # The uid broadcast below runs on WORLD and nvshmem_team_split_strided is a
            # world-collective -- a subgroup-only construction would hang, so reject it up front.
            raise NotImplementedError(
                "process_group must span all ranks (NVSHMEM bootstrap is world-collective); "
                f"got a subgroup of {self.world_size}/{dist.get_world_size()} ranks"
            )
"""
SUBGROUP_FEATURE_MARKER = "SUPPORTS_WORLD_PARTITION_SUBGROUPS = True\n"


def _insert_once(text: str, anchor: str, insertion: str, label: str) -> str:
    if insertion.strip() in text:
        return text
    if text.count(anchor) != 1:
        raise RuntimeError(
            f"expected exactly one {label} anchor, found {text.count(anchor)}"
        )
    return text.replace(anchor, anchor + insertion, 1)


def _adapt_local_world_comm(text: str) -> str:
    """Make the upstream NVSHMEM world follow the supplied process group."""

    updated = text.replace(LEGACY_WORLD_ONLY_GUARD, "")
    for old, new in LOCAL_WORLD_REPLACEMENTS:
        if new in updated:
            continue
        if old in updated:
            updated = updated.replace(old, new)
        elif old in OPTIONAL_LOCAL_WORLD_ANCHORS:
            # The pinned upstream revision predates process-registration validation.
            # Those anchors only exist in newer, unpublished source snapshots.
            continue
        else:
            raise RuntimeError(f"fast-ulysses local-world anchor is missing: {old!r}")
    return updated


def apply_overlay(checkout: Path, *, check_only: bool) -> bool:
    project_root = Path(__file__).resolve().parents[2]
    sources = project_root / "chitu_diffusion" / "parallel" / "cp" / "fast" / "csrc"
    csrc = checkout.resolve() / "fast_ulysses" / "csrc"
    bindings = csrc / "bindings.cpp"
    comm_source = checkout.resolve() / "fast_ulysses" / "comm.py"
    init_source = checkout.resolve() / "fast_ulysses" / "__init__.py"
    if not bindings.is_file():
        raise FileNotFoundError(
            f"{checkout} is not a fast-ulysses checkout: {bindings} is missing"
        )

    original = bindings.read_text(encoding="utf-8")
    original_comm = comm_source.read_text(encoding="utf-8")
    updated_comm = _adapt_local_world_comm(original_comm)
    original_init = init_source.read_text(encoding="utf-8")
    updated_init = original_init
    if SUBGROUP_FEATURE_MARKER not in updated_init:
        updated_init = updated_init.rstrip() + "\n" + SUBGROUP_FEATURE_MARKER
    updated = _insert_once(
        original,
        INCLUDE_ANCHOR,
        INCLUDE_LINE,
        "C++ include",
    )
    if CE_A2A_LAUNCH_NEW not in updated:
        if updated.count(CE_A2A_LAUNCH_OLD) != 1:
            raise RuntimeError("fast-ulysses CE all-to-all launch anchor is missing")
        updated = updated.replace(CE_A2A_LAUNCH_OLD, CE_A2A_LAUNCH_NEW, 1)
    if 'm.def("all_gather_kv_4d(' not in updated:
        updated = _insert_once(
            updated,
            REGISTER_ANCHOR,
            REGISTER_BLOCK,
            "operator registration",
        )
    if 'm.def("ring_ce_prepare_kv_4d(' not in updated:
        updated = _insert_once(
            updated,
            ALL_GATHER_REGISTER_ANCHOR,
            RING_CE_REGISTER_BLOCK,
            "Flash Ring CE registration",
        )
    if 'm.def("u2fm2_prepare_qkv(' not in updated:
        updated = _insert_once(
            updated,
            ALL_GATHER_REGISTER_ANCHOR,
            HYBRID_REGISTER_BLOCK,
            "U2xFM2 operator registration",
        )
    elif HYBRID_SCHEMA_NEW not in updated:
        if updated.count(HYBRID_SCHEMA_OLD) != 1:
            raise RuntimeError("failed to rewrite the U2xFM2 operator schema")
        updated = updated.replace(HYBRID_SCHEMA_OLD, HYBRID_SCHEMA_NEW, 1)
    source_pairs = [
        (source, csrc / source.name)
        for source in sorted(sources.glob("*.cu")) + sorted(sources.glob("*.cuh"))
    ]
    if not source_pairs:
        raise FileNotFoundError(f"no Fast CP sources found in {sources}")
    changed = (
        updated != original
        or updated_comm != original_comm
        or updated_init != original_init
        or any(
            not destination.exists() or source.read_bytes() != destination.read_bytes()
            for source, destination in source_pairs
        )
    )
    if check_only:
        return changed
    bindings.write_text(updated, encoding="utf-8")
    comm_source.write_text(updated_comm, encoding="utf-8")
    init_source.write_text(updated_init, encoding="utf-8")
    for source, destination in source_pairs:
        shutil.copy2(source, destination)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkout",
        type=Path,
        help="path to a fast-ulysses source checkout",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when the overlay is not fully applied",
    )
    args = parser.parse_args()
    changed = apply_overlay(args.checkout, check_only=args.check)
    if args.check:
        return int(changed)
    print(
        "Fast CP sources "
        + ("updated" if changed else "already current")
        + f": {args.checkout}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
