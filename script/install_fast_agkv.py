#!/usr/bin/env python3
"""Apply ChituDiffusion's Fast AGKV extension to a fast-ulysses checkout."""

from __future__ import annotations

import argparse
import re
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
FULL_MESH_REGISTER_ANCHOR = (
    '    m.impl("all_gather_kv_4d", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::all_gather_kv_4d);\n"
)
FULL_MESH_REGISTER_BLOCK = (
    "\n"
    '    m.def("full_mesh_stream_kv_4d('
    "__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
    "Tensor key, Tensor value, int[] source_lengths, "
    "str key_tag, str value_tag, str flag_tag, "
    "int epoch, bool use_sm=False, bool source_chunk=False, int chunk_count=1, "
    "bool copy_local=True) "
    '-> (Tensor, Tensor, Tensor, Tensor)");\n'
    '    m.impl("full_mesh_stream_kv_4d", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::full_mesh_stream_kv_4d);\n"
    "\n"
    '    m.def("full_mesh_publish_consumed('
    "__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
    'str flag_tag, int epoch, Tensor epoch_guard) -> Tensor");\n'
    '    m.impl("full_mesh_publish_consumed", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::full_mesh_publish_consumed);\n"
)
NEW_FULL_MESH_SCHEMA = FULL_MESH_REGISTER_BLOCK.split("\n")[1] + "\n"
# Bindings from earlier overlay revisions declare a narrower schema; rewrite
# whichever one is present instead of tracking every historical spelling.
FULL_MESH_SCHEMA_PATTERN = re.compile(
    r'^[ \t]*m\.def\("full_mesh_stream_kv_4d\(.*?\);\n',
    re.MULTILINE | re.DOTALL,
)
PUBLISH_REGISTER_ANCHOR = (
    '    m.impl("full_mesh_stream_kv_4d", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::full_mesh_stream_kv_4d);\n"
)
WAIT_REGISTER_BLOCK = (
    "\n"
    '    m.def("full_mesh_wait_ready('
    'Tensor flags, int index, int epoch) -> Tensor");\n'
    '    m.impl("full_mesh_wait_ready", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::full_mesh_wait_ready);\n"
)
PUBLISH_REGISTER_BLOCK = (
    "\n"
    '    m.def("full_mesh_publish_consumed('
    "__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
    'str flag_tag, int epoch, Tensor epoch_guard) -> Tensor");\n'
    '    m.impl("full_mesh_publish_consumed", '
    "c10::DispatchKey::CompositeExplicitAutograd, "
    "&ulysses::full_mesh_publish_consumed);\n"
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
CE_STREAMS_DEFAULT = """    if (!ce_ready_) {
        ce_.streams.resize(world_size_);
        for (int i = 0; i < world_size_; ++i)
            ULYSSES_CUDA_CHECK(cudaStreamCreateWithFlags(&ce_.streams[i], cudaStreamNonBlocking));
        ce_ready_ = true;
    }
"""
CE_STREAMS_HIGH_PRIORITY = """    if (!ce_ready_) {
        int least_priority = 0;
        int greatest_priority = 0;
        ULYSSES_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(
            &least_priority, &greatest_priority));
        ce_.streams.resize(world_size_);
        for (int i = 0; i < world_size_; ++i)
            ULYSSES_CUDA_CHECK(cudaStreamCreateWithPriority(
                &ce_.streams[i], cudaStreamNonBlocking, greatest_priority));
        ce_ready_ = true;
    }
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
        "        dist.barrier(group=dist.group.WORLD)\n",
        "        dist.barrier(group=pg)\n",
    ),
    (
        "            [int(r) for r in self.peer_global_ranks],\n",
        "            self.peer_runtime_ranks,\n",
    ),
)
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

    updated = text
    for old, new in LOCAL_WORLD_REPLACEMENTS:
        if new in updated:
            continue
        if old in updated:
            updated = updated.replace(old, new)
        else:
            raise RuntimeError(f"fast-ulysses local-world anchor is missing: {old!r}")
    return updated


def apply_overlay(checkout: Path, *, check_only: bool) -> bool:
    project_root = Path(__file__).resolve().parents[1]
    overlay = project_root / "script" / "overlays" / "fast_agkv"
    csrc = checkout.resolve() / "fast_ulysses" / "csrc"
    bindings = csrc / "bindings.cpp"
    group_source = csrc / "ulysses_group.cu"
    comm_source = checkout.resolve() / "fast_ulysses" / "comm.py"
    init_source = checkout.resolve() / "fast_ulysses" / "__init__.py"
    if not bindings.is_file():
        raise FileNotFoundError(
            f"{checkout} is not a fast-ulysses checkout: {bindings} is missing"
        )

    original = bindings.read_text(encoding="utf-8")
    original_group = group_source.read_text(encoding="utf-8")
    original_comm = comm_source.read_text(encoding="utf-8")
    updated_comm = _adapt_local_world_comm(original_comm)
    original_init = init_source.read_text(encoding="utf-8")
    updated_init = original_init
    if SUBGROUP_FEATURE_MARKER not in updated_init:
        updated_init = updated_init.rstrip() + "\n" + SUBGROUP_FEATURE_MARKER
    if CE_STREAMS_HIGH_PRIORITY in original_group:
        updated_group = original_group
    elif original_group.count(CE_STREAMS_DEFAULT) == 1:
        updated_group = original_group.replace(
            CE_STREAMS_DEFAULT, CE_STREAMS_HIGH_PRIORITY, 1
        )
    else:
        raise RuntimeError("fast-ulysses CE stream creation anchor is missing")
    updated = _insert_once(
        original,
        INCLUDE_ANCHOR,
        INCLUDE_LINE,
        "C++ include",
    )
    if 'm.def("all_gather_kv_4d(' not in updated:
        updated = _insert_once(
            updated,
            REGISTER_ANCHOR,
            REGISTER_BLOCK,
            "operator registration",
        )
    if 'm.def("full_mesh_stream_kv_4d(' not in updated:
        updated = _insert_once(
            updated,
            FULL_MESH_REGISTER_ANCHOR,
            FULL_MESH_REGISTER_BLOCK,
            "full-mesh operator registration",
        )
    elif NEW_FULL_MESH_SCHEMA not in updated:
        updated, count = FULL_MESH_SCHEMA_PATTERN.subn(
            NEW_FULL_MESH_SCHEMA,
            updated,
            1,
        )
        if count != 1:
            raise RuntimeError("failed to rewrite the full-mesh operator schema")
    if 'm.def("full_mesh_wait_ready(' not in updated:
        updated = _insert_once(
            updated,
            PUBLISH_REGISTER_ANCHOR,
            WAIT_REGISTER_BLOCK,
            "full-mesh wait registration",
        )
    if 'm.def("full_mesh_publish_consumed(' not in updated:
        updated = _insert_once(
            updated,
            PUBLISH_REGISTER_ANCHOR,
            PUBLISH_REGISTER_BLOCK,
            "full-mesh consumed registration",
        )
    if 'm.def("ring_ce_prepare_kv_4d(' not in updated:
        updated = _insert_once(
            updated,
            PUBLISH_REGISTER_ANCHOR,
            RING_CE_REGISTER_BLOCK,
            "Flash Ring CE registration",
        )
    if 'm.def("u2fm2_prepare_qkv(' not in updated:
        updated = _insert_once(
            updated,
            PUBLISH_REGISTER_ANCHOR,
            HYBRID_REGISTER_BLOCK,
            "U2xFM2 operator registration",
        )
    elif HYBRID_SCHEMA_NEW not in updated:
        if updated.count(HYBRID_SCHEMA_OLD) != 1:
            raise RuntimeError("failed to rewrite the U2xFM2 operator schema")
        updated = updated.replace(HYBRID_SCHEMA_OLD, HYBRID_SCHEMA_NEW, 1)
    source_pairs = [
        (overlay / "fast_agkv.cu", csrc / "fast_agkv.cu"),
        (overlay / "fast_agkv.cuh", csrc / "fast_agkv.cuh"),
        (overlay / "symmetric_pool.cu", csrc / "symmetric_pool.cu"),
        (overlay / "symmetric_pool.cuh", csrc / "symmetric_pool.cuh"),
    ]
    changed = (
        updated != original
        or updated_group != original_group
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
    group_source.write_text(updated_group, encoding="utf-8")
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
        "Fast AGKV overlay "
        + ("updated" if changed else "already current")
        + f": {args.checkout}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
