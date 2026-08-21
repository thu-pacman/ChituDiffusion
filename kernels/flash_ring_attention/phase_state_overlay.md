# CuTe persistent phase-state overlay contract

Status: **design overlay, not installed and not claimed as implemented**.

The working Python path launches one reviewed CuTe forward per K/V shard and
stores FP32 running O/LSE in HBM.  A true persistent implementation should be
an optional SM90 overlay on the reviewed FlashAttention CuTe forward, gated by
new private arguments so upstream behavior remains unchanged.

## Proposed private interface

The forward interface should accept:

- `_ring_phase_count: Int32` (`2`, `4`, or `8`);
- `_ring_k_slots`, `_ring_v_slots`: device pointer tables, one K/V shard per
  phase (or a base pointer plus per-phase strides);
- `_ring_k_lengths`: token count per phase;
- `_ring_ready_flags` and `_ring_epoch`: transport readiness metadata;
- `_ring_consumed_flags`: optional phase release metadata;
- `_ring_source_order`: optional source-rank diagnostics.

All arguments must be absent on the normal FlashAttention path.  Enable only
for non-causal FP16/BF16 BSHD, SM90, matching heads/head dimension, and positive
K lengths.  Reject unsupported configurations instead of silently selecting a
different numerical path.

## Kernel state machine

For each Q tile, one persistent CTA performs:

1. Load the Q tile once into shared memory.
2. Initialize row max `m = -inf`, row sum `l = 0`, and output accumulator
   `acc_o = 0` in registers.
3. For phase `p = 0 .. phase_count-1`:
   - wait for `ready_flags[p] == epoch` before issuing TMA for that slot;
   - select the phase K/V pointer and its exact token length;
   - run the existing K/V tile producer and GMMA consumer loops;
   - update `m`, `l`, and `acc_o` with the existing online-softmax recurrence;
   - publish `consumed_flags[p]` only after all reads from the slot are complete.
4. After the final phase, normalize `acc_o / l`, store output once, and store
   `m + log(l)` as FP32 LSE if requested.

The phase loop must surround the existing K/V tile loop, not the Q-tile loop.
The accumulator, row max, and row sum must be allocated outside the phase loop.
No output/LSE epilogue may run before the final phase.

## Synchronization invariants

- A wait must occur in the producer before the first TMA read from a phase slot.
- A slot cannot be released while any producer stage can still reference it.
- Every CTA processing a Q tile observes the same immutable epoch and phase
  metadata for the launch.
- Irregular shard tails use the existing K bounds/predication; a TMA tile must
  never cross from one transport slot into another.
- The producer/consumer pipeline is drained or safely rebound at each slot
  boundary.  Reusing a stage whose prior TMA is in flight is invalid.
- Empty shards are rejected by the Python gate.

## What can actually stay resident

Measured on H20 (78 SMs, one FA3-style CTA per SM, `tile_m = 128`):

| item | verdict | reason |
| --- | --- | --- |
| Q tile in shared memory | yes | 128×128 BF16 is 32 KiB, loaded once per CTA |
| row max / row sum in registers | yes | two FP32 values per row, unchanged from FA3 |
| O accumulator in registers across phases | **no** | see below |

The blocker is CTA count, not register count.  A CTA can only hold its
accumulator across the phase loop if it owns the same Q tile for the whole
launch, so every Q tile needs a resident CTA.  Q tiles number
`B × H × ceil(S_local / tile_m)`: at `B1,H40,D128,S4096,CP8` that is `4 × 40 =
160`, and at `S75600,CP8` it is `74 × 40 = 2960`.  Both far exceed the ~78
resident CTAs, and `H = 40` means no realistic shape gets under that bound.

Two K/V slots make this a correctness constraint rather than a tuning choice.
The left peer may only overwrite a slot after every local CTA has finished
reading it, so the consumed flag needs grid-wide agreement.  With a
non-persistent grid the un-launched CTAs cannot report, the launched ones block
waiting for a refill that needs those reports, and the ring deadlocks.  So two
slots force a persistent grid, and a persistent grid forces each CTA to own
several Q tiles, whose accumulators cannot all live in registers.

The practical consequence is mild: keep the running O/LSE in HBM and reload it
per Q tile per phase.  At `S4096,CP8` the state is 10.0 MiB and the seven middle
phases move about 147 MB, roughly 0.04 ms against a 0.70 ms budget.  At
`S75600,CP8` it is 193 MB of state and 2.7 GB of traffic, about 0.68 ms against
104 ms.  Both are under 1%.

So the overlay should be described as **single-launch phase-loop**, not
register-resident.  The wins that matter are paying host issue cost once instead
of once per phase, and letting the K/V tile loop run across the phase boundary
so the shard split stops costing about 15% of compute.

## Acceptance gate

Before enabling this overlay, compare output and FP32 LSE against the implemented
split/merge path for CP2/4/8, both dtypes, uneven shard lengths, head dimensions
used by ChituDiffusion, multiple batches/heads, and non-default softmax scales.
Then measure memory and latency on Hopper.  Only a compiled overlay that keeps
Q in shared memory and runs the phase loop inside a single launch may be
described as persistent; per the residency analysis above, do not claim
register-resident accumulators.
