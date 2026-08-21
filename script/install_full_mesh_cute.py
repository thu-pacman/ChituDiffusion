#!/usr/bin/env python3
"""Apply Chitu's streaming-KV overlay to a tracked FlashAttention checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def _replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if text.count(old) != 1:
        raise RuntimeError(f"expected one {label} anchor, found {text.count(old)}")
    return text.replace(old, new, 1), True


def _apply(
    path: Path,
    replacements: list[tuple[str, str, str]],
    check: bool,
    *,
    upgrades: list[tuple[str, str, str]] | None = None,
) -> bool:
    text = path.read_text()
    changed = False
    for old, new, _label in upgrades or []:
        if new not in text and old in text:
            text = text.replace(old, new)
            changed = True
    for old, new, label in replacements:
        text, replacement_changed = _replace_once(text, old, new, label)
        changed |= replacement_changed
    if changed and not check:
        path.write_text(text)
    return changed


def _apply_overlay_patch(checkout: Path, patch_name: str, check: bool) -> bool:
    patch = Path(__file__).parent / "overlays" / "full_mesh_cute" / patch_name
    command = ["git", "apply", "--check", str(patch)]
    reverse = ["git", "apply", "--reverse", "--check", str(patch)]
    if subprocess.run(
        command, cwd=checkout, check=False, capture_output=True
    ).returncode == 0:
        if not check:
            subprocess.run(
                ["git", "apply", str(patch)],
                cwd=checkout,
                check=True,
            )
        return True
    if subprocess.run(
        reverse, cwd=checkout, check=False, capture_output=True
    ).returncode == 0:
        return False
    raise RuntimeError(f"{patch_name} does not match this FlashAttention checkout")


SM90_REPLACEMENTS = [
    (
        "from cutlass.base_dsl.arch import Arch\n",
        "from cutlass.base_dsl.arch import Arch\n"
        "from cutlass.cutlass_dsl import T, dsl_user_op\n"
        "from cutlass._mlir.dialects import llvm\n",
        "CuTe DSL imports",
    ),
    (
        "\n\nclass FlashAttentionForwardSm90(FlashAttentionForwardBase):\n",
        '''

@dsl_user_op
def _ld_acquire_sys(lock_ptr: cute.Pointer, *, loc=None, ip=None) -> cutlass.Int32:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    state = llvm.inline_asm(
        T.i32(),
        [lock_ptr_i64],
        "ld.global.cv.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(state)


@dsl_user_op
def _wait_eq_sys(
    lock_ptr: cute.Pointer,
    thread_idx: int | Int32,
    flag_offset: int,
    val: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    # The spin has to be one asm block. A Python `while` here is traced by the
    # DSL as a plain Python loop: the condition is a dynamic Int32 compare, it
    # collapses to False at trace time, and the wait silently becomes dead code.
    # `thread_idx` is accepted for call-site compatibility but deliberately
    # unused -- every thread in the loading warp spins on the same address and
    # they leave together, which needs no extra warp sync.
    flag_ptr = (lock_ptr + flag_offset).toint(loc=loc, ip=ip).ir_value()
    # Two details keep the loop alive. The value goes to a block-local register
    # rather than the asm output `$0`, which is dead and can therefore share a
    # physical register with the `$2` input, making `setp.ne $0, $2` compare a
    # register with itself. And the load is an acquire, not `ld.global.cv`: a
    # relaxed load feeding only a dead value lets ptxas drop the whole loop as
    # an effect-free infinite loop.
    llvm.inline_asm(
        T.i32(),
        [flag_ptr, Int32(val).ir_value()],
        "{\\n\\t"
        ".reg .pred %chitu_p;\\n\\t"
        ".reg .b32 %chitu_v;\\n\\t"
        "CHITU_KV_WAIT${:uid}:\\n\\t"
        "ld.acquire.sys.global.b32 %chitu_v, [$1];\\n\\t"
        "setp.ne.b32 %chitu_p, %chitu_v, $2;\\n\\t"
        "@%chitu_p bra CHITU_KV_WAIT${:uid};\\n\\t"
        "mov.b32 $0, %chitu_v;\\n\\t"
        "}",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class FlashAttentionForwardSm90(FlashAttentionForwardBase):
''',
        "system-visible flag wait",
    ),
    (
        "        paged_kv_non_tma: bool = False,\n        **kwargs,\n",
        "        paged_kv_non_tma: bool = False,\n"
        "        streaming_kv: bool = False,\n"
        "        streaming_slots: int = 0,\n"
        "        streaming_world_size: int = 0,\n"
        "        **kwargs,\n",
        "streaming constructor arguments",
    ),
    (
        "        self.use_tma_KV = not paged_kv_non_tma\n",
        "        self.use_tma_KV = not paged_kv_non_tma\n"
        "        self.streaming_kv = streaming_kv\n"
        "        self.streaming_slots = streaming_slots\n"
        "        self.streaming_world_size = streaming_world_size\n",
        "streaming constructor state",
    ),
    (
        "            blocksparse_tensors,\n            self.sQ_layout,\n",
        "            blocksparse_tensors,\n"
        "            mCuTotalMBlocks if const_expr(self.streaming_kv) else None,\n"
        "            mCuTotalSplitsMBlocks if const_expr(self.streaming_kv) else None,\n"
        "            self.sQ_layout,\n",
        "kernel flag arguments",
    ),
    (
        "        blocksparse_tensors: Optional[BlockSparseTensors],\n"
        "        sQ_layout: cute.ComposedLayout,\n",
        "        blocksparse_tensors: Optional[BlockSparseTensors],\n"
        "        mKvReadyFlags: Optional[cute.Tensor],\n"
        "        mKvEpoch: Optional[cute.Tensor],\n"
        "        sQ_layout: cute.ComposedLayout,\n",
        "kernel flag signature",
    ),
    (
        "        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)\n\n"
        "        if warp_idx < 4:  # Producer\n",
        "        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)\n"
        "        kv_ready_flags = mKvReadyFlags\n"
        "        kv_epoch = mKvEpoch\n\n"
        "        if warp_idx < 4:  # Producer\n",
        "kernel flag aliases",
    ),
    (
        "                TileSchedulerCls,\n            )\n",
        "                TileSchedulerCls,\n"
        "                kv_ready_flags,\n"
        "                kv_epoch,\n"
        "            )\n",
        "load flag arguments",
    ),
    (
        "        TileSchedulerCls: Callable,\n    ):\n"
        "        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4\n"
        "        tidx, _, _ = cute.arch.thread_idx()\n",
        "        TileSchedulerCls: Callable,\n"
        "        mKvReadyFlags: Optional[cute.Tensor],\n"
        "        mKvEpoch: Optional[cute.Tensor],\n"
        "    ):\n"
        "        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4\n"
        "        tidx, _, _ = cute.arch.thread_idx()\n"
        "        phase_flags = None\n"
        "        slot_thresh = None\n"
        "        epoch = Int32(0)\n"
        "        if const_expr(self.streaming_kv):\n"
        "            phase_flags = mKvReadyFlags\n"
        "            epoch = _ld_acquire_sys(mKvEpoch.iterator)\n"
        "            # mKvEpoch is [epoch, start_token_0, ..., start_token_n].\n"
        "            # Convert token offsets once with the kernel's compile-time\n"
        "            # tile_n, preserving the upstream head-dim configuration\n"
        "            # without adding work to the per-KV-block hot loop.\n"
        "            slot_thresh = [\n"
        "                (Int32(mKvEpoch[1 + i]) + self.tile_n - 1) // self.tile_n\n"
        "                for i in range(self.streaming_slots)\n"
        "            ]\n",
        "load flag signature",
    ),
    (
        "            work_tile = tile_scheduler.initial_work_tile_info()\n"
        "            while work_tile.is_valid_tile:\n"
        "                # if work_tile.is_valid_tile:\n",
        "            work_tile = tile_scheduler.initial_work_tile_info()\n"
        "            while work_tile.is_valid_tile:\n"
        "                # if work_tile.is_valid_tile:\n"
        # Per work tile, not per kernel: the scheduler is persistent, so one CTA
        # walks several tiles and each restarts the KV descent at the top slot.
        # Carrying the cache across tiles let every tile after the first read
        # K/V without waiting on a single arrival flag, which stayed invisible
        # while the symmetric buffer happened to hold the right bytes already.
        "                last_ready_slot = Int32(self.streaming_slots)\n",
        "slot polling cache",
    ),
]

# KV blocks are consumed from high to low addresses. Wait for every newly
# touched slot between the previous and current block. The common case still
# waits for one slot, while short shards may safely share a kernel tile.
_MULTI_SLOT_WAIT_BLOCK = '''                        if const_expr(self.streaming_kv):
                            slot = Int32(0)
                            for probe in cutlass.range_constexpr(
                                1, self.streaming_slots
                            ):
                                if n_block >= slot_thresh[probe]:
                                    slot = Int32(probe)
                            if slot < last_ready_slot:
                                for wait_slot in cutlass.range_constexpr(
                                    self.streaming_slots
                                ):
                                    if wait_slot >= slot and wait_slot < last_ready_slot:
                                        for source in cutlass.range_constexpr(
                                            self.streaming_world_size
                                        ):
                                            _wait_eq_sys(
                                                phase_flags.iterator,
                                                tidx % 32,
                                                wait_slot * self.streaming_world_size + source,
                                                epoch,
                                            )
                                last_ready_slot = slot
'''
WAIT_BLOCK = '''                        if const_expr(self.streaming_kv):
                            slot = Int32(0)
                            for probe in cutlass.range_constexpr(
                                1, self.streaming_slots
                            ):
                                if n_block >= slot_thresh[probe]:
                                    slot = Int32(probe)
                            if slot < last_ready_slot:
                                if slot + 1 == last_ready_slot:
                                    for source in cutlass.range_constexpr(
                                        self.streaming_world_size
                                    ):
                                        _wait_eq_sys(
                                            phase_flags.iterator,
                                            tidx % 32,
                                            slot * self.streaming_world_size + source,
                                            epoch,
                                        )
                                else:
                                    for wait_slot in cutlass.range_constexpr(
                                        self.streaming_slots
                                    ):
                                        if wait_slot >= slot and wait_slot < last_ready_slot:
                                            for source in cutlass.range_constexpr(
                                                self.streaming_world_size
                                            ):
                                                _wait_eq_sys(
                                                    phase_flags.iterator,
                                                    tidx % 32,
                                                    wait_slot * self.streaming_world_size + source,
                                                    epoch,
                                                )
                                last_ready_slot = slot
'''

_OLD_STREAMING_METADATA = '''        phase_flags = None
        slot_thresh = None
        epoch = Int32(0)
        if const_expr(self.streaming_kv):
            phase_flags = mKvReadyFlags
            epoch = _ld_acquire_sys(mKvEpoch.iterator)
            # mKvEpoch is [epoch, thresh_0, ..., thresh_{slots-1}], where
            # thresh_j is the lowest KV block index that touches slot j.
            # Slots are ordered by ascending address, so the thresholds are
            # ascending and slot lengths need not be block-aligned. They are
            # fixed for the launch, so an ordinary cached load is enough.
            slot_thresh = [
                Int32(mKvEpoch[1 + i]) for i in range(self.streaming_slots)
            ]
'''
_TOKEN_START_STREAMING_METADATA = '''        phase_flags = None
        slot_starts = None
        epoch = Int32(0)
        if const_expr(self.streaming_kv):
            phase_flags = mKvReadyFlags
            epoch = _ld_acquire_sys(mKvEpoch.iterator)
            # mKvEpoch is [epoch, start_token_0, ..., start_token_n].
            # The producer maps token offsets with its own compile-time
            # tile_n, preserving every upstream head-dim configuration.
            slot_starts = [
                Int32(mKvEpoch[1 + i]) for i in range(self.streaming_slots)
            ]
'''
_NEW_STREAMING_METADATA = '''        phase_flags = None
        slot_thresh = None
        epoch = Int32(0)
        if const_expr(self.streaming_kv):
            phase_flags = mKvReadyFlags
            epoch = _ld_acquire_sys(mKvEpoch.iterator)
            # mKvEpoch is [epoch, start_token_0, ..., start_token_n].
            # Convert token offsets once with the kernel's compile-time
            # tile_n, preserving the upstream head-dim configuration
            # without adding work to the per-KV-block hot loop.
            slot_thresh = [
                (Int32(mKvEpoch[1 + i]) + self.tile_n - 1) // self.tile_n
                for i in range(self.streaming_slots)
            ]
'''
_OLD_WAIT_BLOCK = '''                        if const_expr(self.streaming_kv):
                            slot = Int32(0)
                            for probe in cutlass.range_constexpr(
                                1, self.streaming_slots
                            ):
                                if n_block >= slot_thresh[probe]:
                                    slot = Int32(probe)
                            if slot != last_ready_slot:
                                for source in cutlass.range_constexpr(
                                    self.streaming_world_size
                                ):
                                    _wait_eq_sys(
                                        phase_flags.iterator,
                                        tidx % 32,
                                        slot * self.streaming_world_size + source,
                                        epoch,
                                    )
                                last_ready_slot = slot
'''
_TOKEN_START_WAIT_BLOCK = '''                        if const_expr(self.streaming_kv):
                            block_start = n_block * self.tile_n
                            slot = Int32(0)
                            for probe in cutlass.range_constexpr(
                                1, self.streaming_slots
                            ):
                                if block_start >= slot_starts[probe]:
                                    slot = Int32(probe)
                            if slot < last_ready_slot:
                                for wait_slot in cutlass.range_constexpr(
                                    self.streaming_slots
                                ):
                                    if wait_slot >= slot and wait_slot < last_ready_slot:
                                        for source in cutlass.range_constexpr(
                                            self.streaming_world_size
                                        ):
                                            _wait_eq_sys(
                                                phase_flags.iterator,
                                                tidx % 32,
                                                wait_slot * self.streaming_world_size + source,
                                                epoch,
                                            )
                                last_ready_slot = slot
'''
SM90_UPGRADES = [
    (
        _OLD_STREAMING_METADATA,
        _NEW_STREAMING_METADATA,
        "token-space streaming metadata",
    ),
    (
        _TOKEN_START_STREAMING_METADATA,
        _NEW_STREAMING_METADATA,
        "precomputed streaming thresholds",
    ),
    (
        "            last_ready_slot = Int32(-1)\n",
        "            last_ready_slot = Int32(self.streaming_slots)\n",
        "streaming slot cache sentinel",
    ),
    (
        "            last_ready_slot = Int32(self.streaming_slots)\n"
        "            while work_tile.is_valid_tile:\n"
        "                # if work_tile.is_valid_tile:\n",
        "            while work_tile.is_valid_tile:\n"
        "                # if work_tile.is_valid_tile:\n"
        "                last_ready_slot = Int32(self.streaming_slots)\n",
        "per-work-tile slot cache reset",
    ),
    (
        _OLD_WAIT_BLOCK,
        WAIT_BLOCK,
        "head-dimension-aware slot wait",
    ),
    (
        _OLD_WAIT_BLOCK.replace("                        ", "                                "),
        WAIT_BLOCK.replace("                        ", "                                "),
        "nested head-dimension-aware slot wait",
    ),
    (
        _TOKEN_START_WAIT_BLOCK,
        WAIT_BLOCK,
        "precomputed head-dimension-aware slot wait",
    ),
    (
        _TOKEN_START_WAIT_BLOCK.replace(
            "                        ", "                                "
        ),
        WAIT_BLOCK.replace("                        ", "                                "),
        "nested precomputed head-dimension-aware slot wait",
    ),
    (
        _MULTI_SLOT_WAIT_BLOCK,
        WAIT_BLOCK,
        "single-slot fast-path wait",
    ),
    (
        _MULTI_SLOT_WAIT_BLOCK.replace(
            "                        ", "                                "
        ),
        WAIT_BLOCK.replace("                        ", "                                "),
        "nested single-slot fast-path wait",
    ),
]

SM90_REPLACEMENTS.extend(
    [
        (
            "                    if is_kv_load_warp:\n"
            "                        pipeline_k.producer_acquire(kv_producer_state)\n",
            "                    if is_kv_load_warp:\n"
            + WAIT_BLOCK
            + "                        pipeline_k.producer_acquire(kv_producer_state)\n",
            "initial phase wait",
        ),
        (
            "                                if const_expr(not self.use_tma_KV):\n"
            "                                    paged_kv_manager.load_page_table(n_block)\n"
            "                                pipeline_k.producer_acquire(kv_producer_state)\n",
            "                                if const_expr(not self.use_tma_KV):\n"
            "                                    paged_kv_manager.load_page_table(n_block)\n"
            + WAIT_BLOCK.replace("                        ", "                                ")
            + "                                pipeline_k.producer_acquire(kv_producer_state)\n",
            "serial phase wait",
        ),
        (
            "                                kv_producer_state_prev = kv_producer_state.clone()\n"
            "                                kv_producer_state.advance()\n"
            "                                pipeline_k.producer_acquire(kv_producer_state)\n",
            "                                kv_producer_state_prev = kv_producer_state.clone()\n"
            "                                kv_producer_state.advance()\n"
            + WAIT_BLOCK.replace("                        ", "                                ")
            + "                                pipeline_k.producer_acquire(kv_producer_state)\n",
            "overlapped phase wait",
        ),
    ]
)

INTERFACE_REPLACEMENTS = [
    (
        "    disable_scheduler_metadata: bool = False,\n"
        ") -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:\n",
        "    disable_scheduler_metadata: bool = False,\n"
        "    _streaming_kv: bool = False,\n"
        "    _kv_ready_flags: Optional[torch.Tensor] = None,\n"
        "    _kv_slots: int = 0,\n"
        "    _kv_epoch_tensor: Optional[torch.Tensor] = None,\n"
        "    _kv_world_size: int = 0,\n"
        ") -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:\n",
        "streaming API arguments",
    ),
    (
        "\n    is_static_persistent = (\n",
        "\n    if _streaming_kv:\n"
        "        assert not is_varlen_q and not use_block_sparsity\n"
        "        assert _kv_ready_flags is not None\n"
        "        assert _kv_epoch_tensor is not None\n"
        "        cu_total_m_blocks = _kv_ready_flags\n"
        "        cu_total_splits_m_blocks = _kv_epoch_tensor\n"
        "\n    is_static_persistent = (\n",
        "streaming pointer aliases",
    ),
    (
        "        fa_logging.get_fa_log_level(),\n    )\n",
        "        fa_logging.get_fa_log_level(),\n"
        "        _streaming_kv,\n"
        "        _kv_slots,\n"
        "        _kv_world_size,\n"
        "    )\n",
        "streaming compile key",
    ),
    (
        "                paged_kv_non_tma=page_size not in [None, tile_n],\n            )\n",
        "                paged_kv_non_tma=page_size not in [None, tile_n],\n"
        "                streaming_kv=_streaming_kv,\n"
        "                streaming_slots=_kv_slots,\n"
        "                streaming_world_size=_kv_world_size,\n"
        "            )\n",
        "streaming kernel configuration",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cute = args.checkout / "flash_attn/cute"
    sm90 = cute / "flash_fwd_sm90.py"
    sm90_text = sm90.read_text()
    changed = False
    if "self.streaming_ready_mask" not in sm90_text:
        changed = _apply(
            sm90,
            SM90_REPLACEMENTS,
            args.check,
            upgrades=SM90_UPGRADES,
        )
    changed |= _apply(cute / "interface.py", INTERFACE_REPLACEMENTS, args.check)
    if not args.check or not changed:
        if "cached_ready_mask" not in sm90_text:
            if "self.streaming_ready_mask" not in sm90_text:
                changed |= _apply_overlay_patch(
                    args.checkout, "ready_mask.patch", args.check
                )
            changed |= _apply_overlay_patch(
                args.checkout, "ready_mask_net_win.patch", args.check
            )
        else:
            changed |= _apply_overlay_patch(
                args.checkout, "ready_mask_net_win.patch", args.check
            )
    if args.check:
        return int(changed)
    print("CuTe full-mesh overlay " + ("updated" if changed else "already current"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
