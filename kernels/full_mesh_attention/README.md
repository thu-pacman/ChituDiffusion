# NVSHMEM Full-Mesh Fused CP Attention

This directory contains the benchmark harness and versioned adaptation layer
for a Hopper-only, non-causal context-parallel attention experiment.

## Upstream provenance

- Repository: `https://github.com/Dao-AILab/flash-attention`
- Reviewed commit: `d7e4dba3e568106b0f1b6323b07c1272f53679b3`
- Kernel source: `flash_attn/cute/flash_fwd_sm90.py`
- License: BSD-3-Clause

Despite the earlier “FA3 CuTe” working name, this source is FlashAttention-4's
CuTe DSL SM90 implementation. Official FlashAttention-3 is the separate Hopper
C++ extension. The validated environment uses `nvidia-cutlass-dsl==4.6.1` and
`quack-kernels==0.6.2`; CUTLASS DSL 4.5.2 is not compatible with this reviewed
FlashAttention revision.

The upstream checkout is managed under the ignored `refs/kernels` directory:

```bash
python script/kernel_watch.py sync flash-attention
```

The adaptation imports the reviewed CuTe SM90 kernel instead of copying its
large dependency graph into this repository. Project-owned changes are kept in
this directory as small, reviewable wrappers and overlays with the upstream
commit recorded above.

Apply both required overlays before building/running:

```bash
python script/install_full_mesh_cute.py refs/kernels/flash-attention
python script/install_fast_agkv.py refs/fast-ulysses
```

## Stage 0a: bare-kernel gate

The first gate measures local-Q/full-KV attention without communication:

```bash
CHITU_PYTHON_BIN=/path/to/python \
script/srun_direct.sh 1 1 \
  kernels/full_mesh_attention/benchmark_stage0a.py \
  --output outputs/full_mesh_attention/stage0a.json
```

Default shape: global sequence 4096, 40 heads, head dimension 128, BF16, and
CP degrees 2/4/8. The harness forces the reference path to cuDNN SDPA and
checks the CuTe output with `rtol=2e-2, atol=2e-3`.

## Scope

The experiment is intentionally limited to:

- NVIDIA Hopper SM90;
- static single-node full-world CP;
- BF16/FP16, any positive batch size, and equal Q/K/V head dimensions from
  8 through 256 in multiples of 8;
- dense non-causal self-attention;
- explicit opt-in until correctness and performance gates pass.

No pipeline or Diffusers model code should depend on this experiment before
the final unified benchmark demonstrates a win over kernel-matched cuDNN Ring.

## Streaming design and measured result

`FastAgkvAttention` defaults to local-first source chunks over Copy
Engines at every CP degree. Each rank places its complete local K/V in the
highest source slot and broadcasts one complete shard to every peer. This
reduces Copy Engine descriptors by a factor of CP and gives attention local
work while remote shards arrive. The CuTe producer polls each slot once per CTA
with cache-volatile loads immediately before issuing the corresponding TMA K/V
loads. Slot starts are passed in token space and converted once by the producer
with the `tile_n` selected for its head-dimension-specialized kernel. The common
case waits one newly touched slot; when short shards share a tile, the producer
waits every newly touched slot. Online softmax remains inside one kernel; no
partial output/LSE merge is emitted.

After the first normal CuTe specialization, the runtime captures the compiled
TVM-FFI runner and uses a direct launcher for matching shapes. Each call only
replaces Q/K/V, arrival-flag, and epoch pointers and reuses stream-ordered O/LSE
scratch. On H20 CP4 `S=4096`, the same-node wrapper/direct A/B is
`1.113/1.058 ms` (`-4.9%`) with steady-state maximum absolute error
`4.88e-4`; attention begins about `26 us` earlier in Nsight Systems.

Launching attention after only local K/V and submitting all remote copies
afterward was also retested and rejected. It delayed P2P by about `200 us`,
extended slot polling, and regressed to `1.162 ms`; that split transport is not
part of the retained implementation.

Two transports are implemented:

- Copy Engine: does not consume SM slots and is the default at every CP degree.
- SM direct-write: device kernels write peer symmetric pointers directly. It is
  an experiment-only switch because its grid competes with the high-occupancy
  attention kernel. At CP4 and a 75776-token sequence it exposes 3.29 ms of
  communication versus 0.52 ms for Copy Engines.

Transport remains in the compiled fast-ulysses CUDA extension. CuTe DSL is
used only for attention and flag polling: it has no supported NVSHMEM
device-header/device-link path. The CUDA transport resolves symmetric peer
pointers on the host and uses ordinary system-visible peer stores.

H20, BF16, `S_global=4096, H=40, D=128` median latency:

| Path | CP2 | CP4 | CP8 |
| --- | ---: | ---: | ---: |
| Fused full-mesh (default) | 1.546 ms | 0.957 ms | 0.696 ms |
| cuDNN Ring | 1.708 ms | 1.337 ms | 1.577 ms |
| first full-mesh prototype | 1.966 ms | 1.475 ms | 1.708 ms |

The default path improves over kernel-matched cuDNN Ring by 9.5%, 28.4%, and
55.9%. Communication hiding is only 28.4%/51.3%/52.4% here, which is expected:
attention is `O(S²)` while communication is `O(S)`, so a 4096-token sequence
leaves too little compute to hide behind.

Long sequences are the meaningful overlap test. All paths run Wan-14B's exact
`75600` tokens:

| Path | CP2 | CP4 | CP8 |
| --- | ---: | ---: | ---: |
| Fused full-mesh (default) | 412.09 ms | 206.94 ms | 103.74 ms |
| Fast AGKV | 416.02 ms | 210.43 ms | 107.83 ms |
| Fast Ulysses + SDPA | 416.09 ms | 209.63 ms | 104.70 ms |

Full-mesh is the fastest measured path at every CP degree, by 0.94%, 1.28%, and
0.92% over the next best. Exposed communication is at or below measurement noise
at CP2/CP4 and 0.23 ms at CP8, so communication is effectively fully hidden.

## Sequence-length flexibility

Shards carry one arrival flag each and the kernel resolves a block to its slot
through a threshold table rather than a fixed stride, so the global sequence
needs no padding and no divisibility by a kernel tile or by the CP degree. The
only requirement is that every rank gets one token, i.e. `S_global >= CP`.

`FastAgkvAttention.shard_lengths` is the canonical split: balanced to a
single token, with low ranks taking the remainder. Token-space slot starts travel
in the same device tensor as the epoch; the producer converts them once using
its compile-time `tile_n`. Batch and sequence changes therefore reuse a compiled
kernel, while each head dimension keeps FlashAttention's normal tuned compile
configuration.

Removing the padding is worth 0.35% at CP2 on the Wan-14B shape, and the
threshold lookup is not measurable against the integer division it replaced:
tile-aligned `75776` still measures 415.95/207.91/104.12 ms for CP2/4/8.

Dynamic-shape validation on H20 covers batches 1/2/3, CP2/4/8, uneven sequence
splits, multiple source slots inside one tile, and the SM90 tile buckets at head
dimensions 64/80/96/128/192/256. The largest observed BF16 absolute error is
`1.953e-3`. Rechecking the original `B=1,S=4096,H=40,D=128,CP8` shape gives
`0.694 ms` versus the documented `0.696 ms`, so the specialized fast path is
preserved within measurement noise.
