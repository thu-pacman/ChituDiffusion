# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-python (CPU) unit tests for the M6 SP-degree latent-migration index math.

Run: python -m pytest test/test_sp_migration.py -q
 or: python test/test_sp_migration.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chitu_diffusion.runtime.sp_migration import (  # noqa: E402
    gather_full_from_shards,
    rank_migration_plan,
    reshard_full_to_shard,
    shard_bounds,
    slice_intersections,
)


def test_shard_bounds_divisible_tiles_sequence():
    L, d = 4096, 4
    bounds = [shard_bounds(L, d, r) for r in range(d)]
    assert bounds == [(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096)]
    # Contiguous, gap-free, covers [0, L).
    assert bounds[0][0] == 0 and bounds[-1][1] == L
    for (a_lo, a_hi), (b_lo, b_hi) in zip(bounds, bounds[1:]):
        assert a_hi == b_lo
    # Equal sizes when divisible.
    assert len({hi - lo for lo, hi in bounds}) == 1


def test_shard_bounds_remainder_tensor_split_convention():
    L, d = 10, 4  # 10 = 3 + 3 + 2 + 2 (first `rem=2` shards get the extra)
    bounds = [shard_bounds(L, d, r) for r in range(d)]
    sizes = [hi - lo for lo, hi in bounds]
    assert sizes == [3, 3, 2, 2]
    assert bounds[0][0] == 0 and bounds[-1][1] == L
    for (a_lo, a_hi), (b_lo, b_hi) in zip(bounds, bounds[1:]):
        assert a_hi == b_lo


def test_slice_intersections_tile_exactly():
    for L, d1, d2 in [(4096, 1, 2), (4096, 2, 4), (4096, 4, 2), (4096, 4, 1), (12, 3, 4), (10, 4, 3)]:
        transfers = slice_intersections(L, d1, d2)
        # The union of all transfers tiles [0, L) with no gaps/overlaps.
        covered = sorted((t.lo, t.hi) for t in transfers)
        pos = 0
        for lo, hi in covered:
            assert lo == pos, f"gap/overlap at {pos} for L={L} {d1}->{d2}: {covered}"
            assert hi > lo
            pos = hi
        assert pos == L
        # Every transfer's dst owns [lo,hi) in the NEW layout; src owns it in OLD.
        for t in transfers:
            nlo, nhi = shard_bounds(L, d2, t.dst)
            olo, ohi = shard_bounds(L, d1, t.src)
            assert nlo <= t.lo and t.hi <= nhi
            assert olo <= t.lo and t.hi <= ohi


def test_send_recv_plans_are_mutually_consistent():
    """Every slice a rank RECEIVES must be exactly SENT by its declared peer."""
    for L, d1, d2 in [(4096, 1, 2), (4096, 2, 4), (4096, 4, 2), (4096, 4, 1), (4096, 2, 2)]:
        max_deg = max(d1, d2)
        plans = {r: rank_migration_plan(L, d1, d2, r) for r in range(max_deg)}
        # Build the set of (src, dst, global_lo, global_hi) each rank sends...
        sent = set()
        for r, plan in plans.items():
            for s in plan.sends:
                sent.add((r, s.peer, s.global_lo, s.global_hi))
        # ...and the set each rank expects to receive; they must match exactly.
        recvd = set()
        for r, plan in plans.items():
            for rv in plan.recvs:
                recvd.add((rv.peer, r, rv.global_lo, rv.global_hi))
        assert sent == recvd, f"send/recv mismatch for L={L} {d1}->{d2}"


def test_plan_reconstructs_new_shard_bitwise():
    """Applying keeps + recvs (sourced from old shards) rebuilds the exact new
    shard that reshard_full_to_shard would produce -- validated bitwise."""
    torch.manual_seed(0)
    L, D = 4096, 8
    full = torch.randn(1, L, D)
    for d1, d2 in [(1, 2), (2, 4), (4, 2), (4, 1), (2, 2), (1, 4), (4, 1)]:
        old_shards = {r: reshard_full_to_shard(full, d1, r) for r in range(d1)}
        for new_rank in range(d2):
            plan = rank_migration_plan(L, d1, d2, new_rank)
            new_lo, new_hi = shard_bounds(L, d2, new_rank)
            rebuilt = torch.empty(1, new_hi - new_lo, D)
            # keeps: slices already on this rank (old-local -> new-local).
            for k in plan.keeps:
                old_lo = shard_bounds(L, d1, new_rank)[0]
                src = old_shards[new_rank][:, k.lo - old_lo:k.hi - old_lo]
                rebuilt[:, k.lo - new_lo:k.hi - new_lo] = src
            # recvs: slices pulled from the old owner.
            for rv in plan.recvs:
                old_lo = shard_bounds(L, d1, rv.peer)[0]
                src = old_shards[rv.peer][:, rv.global_lo - old_lo:rv.global_hi - old_lo]
                rebuilt[:, rv.local_offset:rv.local_offset + rv.count] = src
            expected = reshard_full_to_shard(full, d2, new_rank)
            assert torch.equal(rebuilt, expected), f"mismatch {d1}->{d2} rank {new_rank}"


def test_gather_scatter_roundtrip_matches_direct_reshard():
    torch.manual_seed(1)
    L, D = 4096, 8
    full = torch.randn(1, L, D)
    for d1, d2 in [(1, 4), (2, 4), (4, 2), (4, 1)]:
        shards = [reshard_full_to_shard(full, d1, r) for r in range(d1)]
        regathered = gather_full_from_shards(shards)
        assert torch.equal(regathered, full)
        for new_rank in range(d2):
            got = reshard_full_to_shard(regathered, d2, new_rank)
            expected = reshard_full_to_shard(full, d2, new_rank)
            assert torch.equal(got, expected)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} SP-migration index-math tests passed.")


if __name__ == "__main__":
    _run_all()
