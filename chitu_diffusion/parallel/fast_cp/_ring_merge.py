"""Single-launch online-softmax state merge for the Flash Ring phase loop.

The eager expression costs six elementwise launches per phase, which dominates
the host critical path once the ring runs eight phases.  This kernel folds them
into one launch over the running FP32 state.

The merge algebra follows vLLM's Apache-2.0 ``merge_attn_states`` at commit
``616c9bd0f4efff9761fd762e4247e18b6bd7eb23``.  The layout differs: Flash Ring
keeps BSHD output with canonical ``[B, H, Sq]`` LSE for both the running state
and the incoming partial, so one offset expression addresses both.
"""

from __future__ import annotations

import importlib.util

import torch

_TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None

if _TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def _ring_merge_kernel(
        out_ptr,
        lse_ptr,
        block_out_ptr,
        block_lse_ptr,
        total_rows,
        sequence: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_dim: tl.constexpr,
        rows_per_block: tl.constexpr,
    ):
        row = tl.program_id(0) * rows_per_block + tl.arange(0, rows_per_block)
        row_mask = row < total_rows
        token_head = sequence * heads
        batch = row // token_head
        within_batch = row % token_head
        token = within_batch // heads
        head = within_batch % heads

        # Running and partial LSE are both canonical [B, H, Sq].
        lse_offset = (batch * heads + head) * sequence + token
        running_lse = tl.load(lse_ptr + lse_offset, mask=row_mask, other=0.0)
        partial_lse = tl.load(block_lse_ptr + lse_offset, mask=row_mask, other=0.0)
        maximum = tl.maximum(running_lse, partial_lse)
        running_weight = tl.exp(running_lse - maximum)
        partial_weight = tl.exp(partial_lse - maximum)
        total = running_weight + partial_weight

        column = tl.arange(0, block_dim)
        mask = row_mask[:, None] & (column < head_dim)[None, :]
        base = row[:, None] * head_dim + column[None, :]
        running = tl.load(out_ptr + base, mask=mask, other=0.0)
        partial = tl.load(block_out_ptr + base, mask=mask, other=0.0).to(tl.float32)
        merged = (
            running * running_weight[:, None] + partial * partial_weight[:, None]
        ) / total[:, None]
        tl.store(out_ptr + base, merged, mask=mask)
        tl.store(lse_ptr + lse_offset, tl.log(total) + maximum, mask=row_mask)


def fused_merge_supported(
    running_output: torch.Tensor,
    running_lse: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
) -> bool:
    return (
        _TRITON_AVAILABLE
        and running_output.is_cuda
        and running_output.device
        == running_lse.device
        == partial_output.device
        == partial_lse.device
        and running_output.dtype == torch.float32
        and running_lse.dtype == torch.float32
        and partial_lse.dtype == torch.float32
        and partial_output.dtype
        in {torch.float16, torch.bfloat16, torch.float32}
        and running_output.ndim == 4
        and partial_output.shape == running_output.shape
        and running_lse.shape
        == (
            running_output.shape[0],
            running_output.shape[2],
            running_output.shape[1],
        )
        and partial_lse.shape == running_lse.shape
        and running_output.is_contiguous()
        and running_lse.is_contiguous()
        and partial_output.is_contiguous()
        and partial_lse.is_contiguous()
        and 1 <= running_output.shape[-1] <= 256
    )


# A ring phase re-launches this kernel with an unchanging signature, so the
# generic Triton entry re-derives a specialization key it already computed.  It
# costs ~40us of host time per launch, which the eight-phase CP8 ring cannot
# afford at short sequences.  Bind the compiled launcher once per signature.
_MERGE_RUNNERS: dict[tuple, object] = {}


def _bound_runner(compiled: object, grid: tuple[int, ...]) -> object | None:
    """Return Triton's lean launcher, or ``None`` on an unexpected build.

    ``CompiledKernel`` indexes the grid unconditionally in all three dimensions,
    unlike ``JITFunction``, which pads it first.
    """

    getter = getattr(compiled, "__getitem__", None)
    if getter is None:
        return None
    try:
        return getter((grid[0], 1, 1))
    except Exception:  # pragma: no cover - depends on the installed Triton
        return None


def fused_merge_flash_partials(
    running_output: torch.Tensor,
    running_lse: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
) -> bool:
    """Merge one phase into the running FP32 state in place.

    Returns ``False`` without touching the state when the fast path does not
    apply, so the caller can fall back to its eager merge.
    """

    if not fused_merge_supported(
        running_output, running_lse, partial_output, partial_lse
    ):
        return False
    batch, sequence, heads, head_dim = running_output.shape
    total_rows = batch * sequence * heads
    block_dim = triton.next_power_of_2(head_dim)
    # The merge only streams memory, so shape the tile for wide per-thread access
    # rather than arithmetic width.  One row per program leaves four bytes per
    # thread and reaches 1.24 TB/s on H20; this 16 KiB tile reaches 2.69 TB/s.
    rows_per_block = max(1, 4096 // block_dim)
    grid = (triton.cdiv(total_rows, rows_per_block),)
    # Triton specializes on constexprs, on the runtime row count, and on 16-byte
    # pointer alignment, so every one of those has to select the bound launcher.
    signature = (
        sequence,
        heads,
        head_dim,
        block_dim,
        rows_per_block,
        total_rows,
        partial_output.dtype,
        running_output.data_ptr() % 16 == 0,
        running_lse.data_ptr() % 16 == 0,
        partial_output.data_ptr() % 16 == 0,
        partial_lse.data_ptr() % 16 == 0,
    )
    runner = _MERGE_RUNNERS.get(signature)
    if runner is not None and runner is not False:
        try:
            # The bound launcher takes every declared parameter, constexprs
            # included, in declaration order.
            runner(
                running_output,
                running_lse,
                partial_output,
                partial_lse,
                total_rows,
                sequence,
                heads,
                head_dim,
                block_dim,
                rows_per_block,
            )
            return True
        except TypeError:  # pragma: no cover - depends on the installed Triton
            # The launcher ABI rejected the call before touching the device, so
            # the state is untouched and the generic entry can still run.
            _MERGE_RUNNERS[signature] = False

    compiled = _ring_merge_kernel[grid](
        running_output,
        running_lse,
        partial_output,
        partial_lse,
        total_rows,
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        block_dim=block_dim,
        rows_per_block=rows_per_block,
        num_warps=8,
    )
    if runner is None:
        bound = _bound_runner(compiled, grid)
        _MERGE_RUNNERS[signature] = False if bound is None else bound
    return True
