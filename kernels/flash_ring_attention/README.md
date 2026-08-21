# Flash Ring attention compute

This directory documents the compute half of low-memory, non-causal context
parallel attention.  Transport is deliberately injected; the runtime/transport
task can implement the protocols in
`chitu_diffusion.parallel.fast_cp.ring` without coupling this work
to NVSHMEM, NCCL, or a particular ring schedule.

## Implemented correctness path

`FastRingAttention` supports FP16/BF16 BSHD tensors and CP2/4/8:

1. `RingKVTransport.begin()` creates a per-call session.
2. Every phase leases exactly one source rank's K/V shard.
3. The reviewed `load_cute_flash_attn_fwd()` is called with
   `causal=False, return_lse=True`.
4. The normalized partial output and FP32 LSE are merged using
   `logaddexp` and FP32 weights.
5. Running output and LSE live in HBM between phases.  The final output is cast
   once to the input dtype; optional returned LSE remains FP32.

The transport session owns readiness waits and receive-buffer lifetime.
Phase zero reads local K/V without transport. `start()` then preposts the full
forwarding chain while phase-zero attention runs; GPU-side arrival tickets
release each hop. `acquire(i)` waits only for phase `i`, `release()` publishes
slot reuse, and `close()` finishes the exchange. Two K/V receive buffers visit
every source rank exactly once.

This path holds only one transport K/V shard plus FP32 running O/LSE, rather
than all-gathering global K/V.  CuTe still reloads Q and launches once per phase.
It does **not** keep Q, online-softmax state, or the accumulator resident in
shared memory/registers across phases.

## Persistent CuTe phase-state overlay

`phase_state_overlay.md` specifies the safe overlay boundary for a future
single-launch CuTe kernel.  It is design-level only: no upstream patch is
installed by this change, and the Python implementation does not claim
cross-phase register residency.

The initial prepost experiment deadlocked because acknowledgements shared the
right-going CE stream: a future copy waiting for an ACK could sit ahead of the
ACK its peer needed. The retained implementation moves ACK publication to an
independent stream and waits for both readers of a landing slot: the phase's
attention-consumed event and its forwarding-done event. This removes the cycle
while keeping the forwarding chain independent of Python phase dispatch.

## H20 gate result

`benchmark_flash_ring_roofline.py` and
`benchmark_flash_ring_attention.py` use all-rank critical-path medians.  For
BF16 `B=1, S_global=4096, H=40, D=128` on H20:

The benchmark reports host wall time per iteration next to the GPU window,
because host issue rate, not NVLink bandwidth, sets the short-sequence result.
Isolated CP8 transport moves only about 0.22 ms of DMA (7 hops × 10.5 MiB at
396 GB/s) inside a 0.53 ms window; the rest is per-phase `cuStreamWaitValue32` /
`cudaMemcpyAsync` / `cuStreamWriteValue32` submission.  Host cost is a constant
of roughly 1.5 ms per CP8 call regardless of sequence length, so the result is
entirely a question of whether the GPU has enough work to hide it.

CP8 against the Full-Mesh column of `kernels/cp_attention_exploration_data_zh.md`:

| `S_global` | Flash Ring online | Full-Mesh | gap | exposed transport |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1.605 ms | 0.696 ms | +131% | 159% |
| 8,192 | 2.011 ms | 1.730 ms | +16% | 28% |
| 16,384 | 6.089 ms | 5.574 ms | +9.2% | 16% |
| 75,600 | 106.96 ms | 103.74 ms | +3.1% | 8% |

At `S=75600` CP8 the online path is within 0.3% of its own prefetched compute
(`106.96` vs `106.66 ms`), so the CE ring is effectively hidden and the 3%
overhead gate passes.  CP4 at the same length is `210.20` against Full-Mesh
`206.94 ms`.

## Where the long-sequence gap actually is

Hiding the transport does not close the gap, because the ring's compute path is
itself more expensive than one gathered attention.  `benchmark_flash_ring_roofline.py`
separates the two on a single device, CP8 `S=75600`:

| stage | median | delta |
| --- | ---: | ---: |
| one gathered attention (what Full-Mesh computes) | 103.73 ms | — |
| eight shard attentions, no merge | 104.47 ms | +0.74 ms (0.71%) |
| plus seven FP32 merges | 105.75 ms | +1.27 ms (1.23%) |

So the shard split is nearly free at this size: `9450×9450` tiles still saturate
the MMA pipeline.  Roughly two thirds of the compute gap is the merge, which is
pure HBM streaming that a fused phase loop would not perform at all.  The merge
moves 3.42 GB across the seven middle phases and currently runs at 2.69 TB/s;
the remaining distributed gap on top of the 1.94% measured here is all-rank
critical-path skew across eight devices.

That long-sequence regime is the one this design targets, because it is where
the memory difference against a materialized global K/V is real.  CP8
`S=75600` holds 387 MB of symmetric K/V (two ping-pong slots) plus 195 MB of
FP32 running O/LSE, so 582 MB against Full-Mesh's 1.55 GB of global K/V plus a
97 MB output, or 65% less.  The FP32 accumulator must be counted: it is state
Full-Mesh keeps in registers and Flash Ring cannot.

Below roughly `S=8192` the host constant dominates and Flash Ring should not be
used; Full-Mesh and Fast Ulysses both win there and memory is not the binding
constraint.

## Against Fast Ulysses

Full-Mesh is the wrong sole reference, because Fast Ulysses is also `O(T/N)` and
is therefore the method Flash Ring actually has to displace.  CP8, `H=40,D=128`:

| `S_global` | Flash Ring | Fast Ulysses | gap |
| ---: | ---: | ---: | ---: |
| 4,096 | 1.605 ms | 0.627 ms | +156% |
| 8,192 | 2.011 ms | 1.693 ms | +18.8% |
| 16,384 | 6.089 ms | 5.687 ms | +7.1% |
| 75,600 | 106.96 ms | 104.70 ms | +2.2% |

At `S=75600` the gap closes as CP degree falls, because Ulysses' `N/2` volume
advantage shrinks while Flash Ring's merge count falls with `N-1`: CP4 is
`210.20` against `209.63 ms` (+0.27%) and CP2 is `416.99` against `416.09 ms`
(+0.22%), both at parity.

Memory does not rescue the CP8 case.  Fast Ulysses reserves a symmetric pool of
four `[S_global, H/N, D]` results, 387 MB at CP8 `S=75600`, against Flash Ring's
582 MB.  Ulysses runs one unchunked attention, so it pays neither the shard
split nor the FP32 accumulator, and it is both faster and smaller.

The one structural property Flash Ring has that Ulysses does not is independence
from head count.  Ulysses requires `H % N == 0`; Wan 1.3B has 12 heads and
cannot run CP8 at all.  That, plus CP2/CP4 parity at long sequence, is the
honest scope of this path today.

Sections 29–34 of `kernels/cp_attention_exploration_data_zh.md` close the
exploration with a 15-cell three-way matrix over `S` and CP degree.  Flash Ring
wins none of them, so its only remaining indication is the intersection of two
constraints: global K/V does not fit in HBM **and** `H % N != 0`.  Everywhere
else Fast AGKV (fused Full-Mesh) or Fast Ulysses strictly dominates it.

Optimizations that produced the current numbers, and one that did not:

- caching the accepted-kwarg set instead of calling `inspect.signature` per
  phase, and passing preallocated `out`/`lse` scratch into the CuTe forward;
- a single-launch Triton merge (`ring_state_merge.py`) replacing six elementwise
  launches per phase, tiled 32 rows at a time so each thread moves 64 bytes;
  one row per program only reached 1.24 TB/s against 2.69 TB/s for the tile,
  which is worth 1.5 ms of the CP8 `S=75600` result on its own;
- reusing CUDA events in the C++ ring instead of create/destroy per phase;
- local phase-zero launch before transport submission, followed by GPU-ticket
  preposting of all forwarding hops on the CE stream; CP4 `S=4096` is
  `0.957/0.960 ms` in two profiler-free runs versus `0.962 ms` before prepost;
- rejected: a `torch.compile` merge, which moved CP8 `S=4096` from
  `1.570 / 2.524 ms` to `2.086 / 3.205 ms` because Dynamo guard time costs more
  host cycles than the fused merge saves in HBM traffic.

Together these took CP8 `S=4096` from `2.524` to `1.605 ms` and its prefetched
compute from `1.570` to `0.767 ms`, and CP8 `S=75600` from `107.32` to
`106.96 ms`.

## Tests

CPU tests inject a deterministic partial-attention function and mock transport:

```bash
pytest -q test/test_flash_ring_attention.py
```

They cover CP2/4/8, FP16/BF16, uneven shard lengths, custom softmax scale,
transport phase lifetime, FP32 state initialization/update, and rejection of
causal or unsupported CP configurations.

GPU tests need Hopper.  `test/test_ring_state_merge_gpu.py` runs on one device
and checks the Triton merge against the eager expression across dtypes, head
dimensions, and wide LSE ranges, plus the decline path for unsupported layouts.
`test/test_flash_ring_attention_gpu.py` needs CP2/4/8 and covers the CE
transport end to end, including dynamic shapes and epoch reset.
