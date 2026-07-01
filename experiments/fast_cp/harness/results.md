# Harness results: async QKV-comm overlap

All runs on the H20 Slurm cluster (`debug` partition, 1 node, exclusive), via
`script/srun_direct.sh`. Driver: `cp_async_harness.py` feeding synthetic packed
Q/K/V into the real CP core; overlap schedule in `overlap_cp.py`
(`OverlapCPAttention`, a subclass — the core `image_cp_attention.py` is
untouched). bf16. Wall-clock = median ms per attention call. `delta` is
`(serial - overlap) / serial`; **positive = overlap faster**.

## 中文阶段总结：PCIe/CP overlap 方向

当前 harness 的定位是：用合成 Q/K/V 驱动真实
`ImageContextParallelAttention`，在不改 core 主路径的前提下验证 CP
attention 的通信/计算重叠、CUDA graph、Ulysses all-to-all 后端，以及
text+image 混合序列下的结构性调度。所有主要实验的数值正确性都通过：
`max_abs_diff=0`，CUDA graph 版本也 `GRAPH ALL RANKS PASS`。

### 1. AGKV overlap：机制成立，但收益取决于链路是否 comm-bound

on-node 快链路下，eager overlap 基本不赚：

| case | serial ms | overlap ms | delta |
| --- | ---: | ---: | ---: |
| AGKV cp4, img=4096, H=24 | 1.830 | 1.897 | -3.7% |
| Ulysses cp4, img=4096, H=24 | 1.606 | 1.617 | -0.7% |
| AGKV cp8, img=16384, H=24 | 6.915 | 6.858 | +0.8% |
| AGKV cp8, img=65536, H=24 | 59.664 | 59.061 | +1.0% |

component profile 显示，cp4 AGKV 下 projection 约 0.685 ms，gather
暴露通信只有约 0.32 ms；通信太小，stream/event/record_stream 和 NCCL/GEMM
SM 争用就足以吃掉收益。

在人为 comm-bound 的设置下（`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`），AGKV
overlap 立刻变成正收益：

| case | serial ms | overlap ms | delta |
| --- | ---: | ---: | ---: |
| AGKV cp4, img=16384, H=24 | 20.263 | 19.223 | +5.1% |
| AGKV cp4, img=16384, H=48 | 43.600 | 40.554 | +7.0% |
| AGKV cp4, img=8192, H=48 | 19.335 | 17.575 | +9.1% |
| Ulysses cp4, img=16384, H=24 | 12.929 | 12.966 | -0.3% |

结论：AGKV 的 overlap 机制是对的，PCIe/跨节点/慢链路上值得保留；单节点快链路上
eager overlap 不应期待明显收益。

### 2. CUDA graph：AGKV 能把 CPU launch 开销压下去

修正 graph schedule 后，AGKV CUDA graph overlap 在 H20 上变成正收益：

| case | serial ms | overlap ms | serial-graph ms | overlap-graph ms | best vs eager serial |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph overlap | 1.837 | 1.883 | 1.816 | 1.681 | +8.5% |
| graph overlap + `CHITU_CP_COMPILE_TAIL=1` | 1.833 | 1.817 | 1.884 | 1.654 | +9.8% |

trace 里 NCCL all-gather 和后续 projection GEMM 的 overlap ratio 约 0.998。
`torch.compile` 只能减少 tail 里的细碎 cat/copy 噪声，不能消除 NCCL/GEMM
资源争用；它是小加成，不是主收益来源。

### 3. Ulysses：当前最优通信 API 是 `all_to_all_single`

Ulysses 的原始 workflow 是：

```text
img_q/k/v: [B, S_img_local, H, D]
  -> input all_to_all
  -> [B, S_img_global, H/cp, D]

txt_q/k/v: [B, S_txt, H, D]
  -> 本地切 head
  -> [B, S_txt, H/cp, D]

joint SDPA over [image_global + text]

img_out: all_to_all 回 [B, S_img_local, H, D]
txt_out: all_gather heads 回 [B, S_txt, H, D]
```

把 list-based `all_to_all` 换成 packed `dist.all_to_all_single` 后，绝对时间和
kernel 数都有改善：

| Ulysses a2a backend | serial ms | overlap ms | serial-graph ms | overlap-graph ms |
| --- | ---: | ---: | ---: | ---: |
| list `all_to_all` | 1.604 | 1.615 | 1.563 | 1.500 |
| packed `all_to_all_single` | 1.526 | 1.544 | 1.511 | 1.460 |
| symmetric tensor + NCCL `all_to_all_single` | 1.524 | 1.527 | 1.490 | 1.439 |
| symmetric-memory pipelined P2P hook | 1.689 | 1.694 | 1.602 | 1.546 |

`all_to_all_single` 使短 trace 的 GPU kernel 数从 210 降到 160，elementwise
copy 从 75 降到 35，cat-copy 从 40 降到 20。`symm_single` 可用但没有本质减少
pack/unpack；`symm_pipe` 正确且能 capture，但 barrier/spin/P2P copy 太碎，反而慢。

结论：在不改 projection 输出 layout、不改 attention 输入 layout 的前提下，
`all_to_all_single` 是当前 Ulysses 的工程最优解。下一层优化不该继续换 all2all
API，而应减少 pack/unpack。

### 4. projection 直出 rank-major：下一条更可能有价值的路线

当前 Ulysses input all-to-all 前有一次 pack：

```text
projection -> [B, S, H, D]
pack copy  -> [rank, B, S, H/cp, D]
all_to_all_single
```

理想路线是 projection 直接写 all-to-all send layout：

```text
projection -> [rank, B, S, H/cp, D]
all_to_all_single
```

这个可以省掉 Q/K/V input-side rank-major pack。注意：简单 Triton pack kernel
只能替换 `permute().contiguous()`，仍然有一次全量 copy；真正想赢，需要把 rank-major
layout 融进 projection 的 store/epilogue，且不能让 projection GEMM 本身变慢。

### 5. text/image split：语义正确，但 naive split 不赚

因为 non-causal attention 沿 query 维度可分，我们尝试把 joint attention 拆成
image-query attention 和 text-query attention，让 text attention 覆盖 `img_q`
input all-to-all 或 `img_out` output all-to-all。三个 schedule 都数值正确：
`txt_q`、`txt_out`、`txt_both` 均 `max_abs_diff=0`，graph 也通过。

但性能结果不理想：

| `CHITU_ULYSSES_SPLIT` | serial ms | overlap ms | serial-graph ms | overlap-graph ms | best vs eager serial |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` | 1.526 | 1.536 | 1.489 | 1.431 | +6.3% |
| `txt_q` | 1.526 | 1.656 | 1.492 | 1.576 | -3.2% |
| `txt_out` | 1.525 | 1.640 | 1.490 | 1.534 | -0.5% |
| `txt_both` | 1.524 | 1.794 | 1.490 | 1.709 | -12.2% |

`txt_out` trace 证明 output `SendRecv` 确实和 text SDPA overlap 起来了，但代价是
joint SDPA 被拆成两个小 SDPA：5 次 replay 中 SDPA kernel 从 5 个变成 10 个，
SDPA 总时间从约 2.51 ms 增到约 3.37 ms；多出来的 SDPA 损失远大于被隐藏的
output all-to-all。

结论：text/image split 是语义正确的结构性方向，但 naive query split 当前不迁移。
除非 text 序列显著变长，或者能保持 fused/joint SDPA 的效率，否则不值得。

### 当前迁移建议

1. AGKV overlap + CUDA graph：值得保留，适合 PCIe/跨节点/慢链路，需 flag 控制。
2. Ulysses：优先迁移 packed `all_to_all_single`，这是当前 layout 下最稳的收益。
3. `symm_single`：保留实验分支；`symm_pipe`：记录为未来 TODO，不迁移。
4. Ulysses 下一步优化：研究 projection 直出 rank-major layout，减少 input pack。
5. text/image split：当前 naive 版本不迁移，只保留结论和 trace。

## Headline

- **Correctness: overlap is bit-identical to serial** for both AGKV and Ulysses
  (`max_abs_diff = 0`, PSNR ~205-208 dB), including in the comm-bound regime.
- **On-node NVLink (real Z-Image-ish sizes): overlap is neutral-to-slightly-negative.**
  The intra-node collective is so cheap there is almost nothing to hide, and the
  comm-stream machinery (events, `record_stream`, NCCL/GEMM SM contention) costs
  as much as the comm it hides. This confirms the plan's §0 prediction.
- **Comm-bound regime (P2P+SHM disabled): overlap delivers a real, growing
  speedup — up to +9.1% for AGKV**, scaling with the projection/comm ratio
  exactly as predicted. This proves the overlap *mechanism* works; on NVLink the
  *opportunity* is just near-zero.

## 1. Correctness (`--check`)

| mode | cp | img(global) | H | result |
| --- | ---: | ---: | ---: | --- |
| agkv | 4 | 4096 | 24 | max_abs_diff=0, PSNR 208 dB, ALL RANKS PASS |
| ulysses | 4 | 4096 | 24 | max_abs_diff=0, PSNR 208 dB, ALL RANKS PASS |
| agkv (P2P off, big proj) | 4 | 8192 | 48 | max_abs_diff=0, PSNR 206 dB, ALL RANKS PASS |

## 2. Wall-clock, on-node NVLink (P2P on) — neutral

| mode | cp | img | B | H×D | serial ms | overlap ms | delta |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| agkv | 4 | 4096 | 1 | 24×128 | 1.830 | 1.897 | **-3.7%** |
| ulysses | 4 | 4096 | 1 | 24×128 | 1.606 | 1.617 | -0.7% |
| agkv | 8 | 16384 | 1 | 24×128 | 6.915 | 6.858 | +0.8% |
| agkv | 8 | 16384 | 4 | 24×128 | 28.213 | 28.195 | +0.1% |
| agkv | 8 | 65536 | 1 | 24×128 | 59.664 | 59.061 | +1.0% |
| agkv | 8 | 32768 | 2 | 24×128 | 40.186 | 40.129 | +0.1% |

Component profile (`--profile-parts`), cp4 AGKV, 1024²/4096 img tokens:
`proj(q+k+v)=0.685ms  gather(k+v incl proj)=0.775ms  serial_attn_total=1.804ms`.
Exposed gather comm ≈ 0.32 ms — far smaller than the 0.685 ms of projection that
would hide it, and dwarfed by the per-call stream overhead. There is simply not
enough comm to overlap on NVLink.

## 3. Wall-clock, comm-bound regime (`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`) — overlap wins

| mode | cp | img | B | H×D | serial ms | overlap ms | delta |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| agkv | 4 | 16384 | 1 | 24×128 | 20.263 | 19.223 | **+5.1%** |
| agkv | 4 | 16384 | 1 | 48×128 | 43.600 | 40.554 | **+7.0%** |
| agkv | 4 | 8192 | 1 | 48×128 | 19.335 | 17.575 | **+9.1%** |
| ulysses | 4 | 16384 | 1 | 24×128 | 12.929 | 12.966 | -0.3% |

Component profile (`--profile-parts`), cp4 AGKV, P2P off, img=16384:
`proj(q+k+v)=1.898ms  gather(k+v incl proj)=10.411ms  serial_attn_total=18.792ms`.
Now the exposed gather (~8.5 ms) is large and the ~1.9 ms of projection can be
hidden behind it. The AGKV delta grows monotonically as the projection share
rises (more heads / smaller image), which is the signature of genuine overlap,
not noise.

## 4. Reading

- **AGKV overlap is correct and effective; its payoff is gated entirely by how
  comm-bound the link is.** NVLink-within-a-node is not comm-bound for these
  shapes, so the on-node delta is ~0. The same code wins by up to ~9% the moment
  comm dominates (cross-node, P2P-disabled, or larger CP degree with slower
  inter-node fabric).
- **Ulysses input-only overlap is neutral.** Its input all-to-all is smaller than
  AGKV's gather, and the output collectives (held serial in phase 1, per the
  plan) dominate — so there is little input-side comm to hide. Extending overlap
  to the output collectives is the lever for Ulysses, deferred per §9.2 of the
  plan.

## 5. Implications for migration

- Porting `attention_fused()` into the core is **safe** (bit-identical) and worth
  doing behind the `CHITU_CP_OVERLAP_QKV` flag, but on a single H20 node it
  should be expected to be **neutral**, not a speedup. Keep it off by default
  on-node; the value appears on slower/inter-node CP fabrics.
- The natural next experiment for a *positive on-NVLink* result is the **CUDA
  graph phase (§9a)**: the on-node loss is dominated by CPU-side per-op launch +
  stream-management overhead, which is exactly what graph replay removes. If a
  captured overlap subgraph turns the on-node delta from negative to ≥0, that is
  the win condition for NVLink.
- The Ulysses lever is **output-collective overlap**, not more input tuning.

## 6. CUDA graph + compile-tail follow-up

After fixing the graph schedule to preserve the eager AGKV ordering
(`K-proj -> K all-gather`, `V-proj` overlaps K gather, `V all-gather`,
`Q-proj` overlaps V gather), CUDA graph overlap becomes positive on H20:

| case | serial ms | overlap ms | serial-graph ms | overlap-graph ms | best vs eager serial |
| --- | ---: | ---: | ---: | ---: | ---: |
| graph overlap | 1.837 | 1.883 | 1.816 | 1.681 | +8.5% |
| graph overlap + `CHITU_CP_COMPILE_TAIL=1` | 1.833 | 1.817 | 1.884 | 1.654 | +9.8% |

The trace confirms real overlap: in
`traces/agkv_overlap_graph_short.overlap_graph.rank0.json`, each NCCL
all-gather overlaps the following projection GEMM with ~0.998 average overlap
ratio; `traces/agkv_serial_graph_short.serial_graph.rank0.json` has 0 overlap.

`torch.compile` on the AGKV tail is a small extra win, not a silver bullet. It
does not remove NCCL/GEMM resource contention, and it does not fuse the NCCL
copies. It does reduce the tail assembly noise: in the short trace,
`CatArrayBatchedCopy` kernels drop from 25 to 10 over 5 replays, replaced by 15
small Triton fused pointwise kernels; overlap-graph improves from ~1.68 ms to
~1.65 ms in this harness run.

## 7. Ulysses input-side CUDA graph overlap

Initial Ulysses graph support mirrors the phase-1 eager schedule: only input
K/V/Q image all-to-all is overlapped with V/Q projection; output image all-to-all
and text head all-gather remain serial after attention.

| case | serial ms | overlap ms | serial-graph ms | overlap-graph ms | best vs eager serial |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ulysses input overlap graph | 1.605 | 1.618 | 1.562 | 1.500 | +6.5% |

Correctness remains bit-identical (`max_abs_diff=0`, graph all ranks pass).
The short trace
`traces/ulysses_overlap_graph_short.overlap_graph.rank0.json` shows the first
input `SendRecv` all-to-all overlapping following projection GEMMs with ~0.995
overlap ratio. The later `SendRecv` / `AllGather` kernels after SDPA are the
deliberately serial output collectives.

### all_to_all_single and symmetric-memory probes

Replacing the harness' list-based `all_to_all` with a packed
`dist.all_to_all_single` backend (`CHITU_HARNESS_A2A=single`) improves absolute
Ulysses time and removes many pack/unpack kernels:

| Ulysses a2a backend | serial ms | overlap ms | serial-graph ms | overlap-graph ms |
| --- | ---: | ---: | ---: | ---: |
| list `all_to_all` | 1.604 | 1.615 | 1.563 | 1.500 |
| packed `all_to_all_single` | 1.526 | 1.544 | 1.511 | 1.460 |
| symmetric tensor + NCCL `all_to_all_single` | 1.524 | 1.527 | 1.490 | 1.439 |
| symmetric-memory pipelined P2P hook | 1.689 | 1.694 | 1.602 | 1.546 |

Short trace kernel count drops from 210 to 160; elementwise copy kernels drop
from 75 to 35 and cat-copy kernels from 40 to 20.

`probe_symm_mem_a2a.py` also shows that this environment can allocate symmetric
memory, rendezvous it across ranks, and run `all_to_all_single` on symmetric
tensors:

```text
torch=2.9.1+cu130 cuda=13.0 nccl=(2, 27, 7)
baseline all_to_all_single 0.064-0.069 ms/call
symm tensor all_to_all_single 0.052-0.055 ms/call
has _pipelined_produce_and_all2all=True
```

Inside the harness, the symmetric tensor path is correct and captureable, but
the current shape does not show a decisive win over packed `all_to_all_single`.
The short trace is nearly identical to `single`: 165 GPU events vs 160, with
the same pack/unpack copies and slightly longer NCCL `SendRecv` total in this
run (1.130 ms vs 0.954 ms over 5 replays). The one bug fixed here was slot
reuse: serial Ulysses now resets the symmetric-buffer sequence each call, so it
does not allocate/rendezvous a fresh static buffer every benchmark iteration.

The private `_pipelined_produce_and_all2all` hook is available, correct, and
CUDA-graph captureable with the default CUDA symmetric allocator, but it is not
faster for this Ulysses payload. Standalone probe time is ~0.135 ms/call vs
~0.064 ms/call for normal `all_to_all_single`; harness trace event count jumps
to 470 because the hook emits many barrier/spin/P2P-copy kernels:

```text
symm_pipe trace over 5 replays:
160 symmetric_memory::barrier_kernel
140 direct-copy elementwise kernels
20 spin kernels
15 Memcpy PtoP
```

`CHITU_HARNESS_SYMM_BACKEND=NVSHMEM` was not usable on the tested node: NVSHMEM
failed during IB transport setup and the Slurm step had to be cancelled. For
now, the migration candidate remains packed `all_to_all_single`; `symm_single`
is a useful experimental branch, and `symm_pipe` should stay probe-only unless
we can fuse the actual chunk producer into it and remove the extra copy/barrier
tax.

Future TODO: keep `symm_pipe` as a probe-only branch and revisit it only if we
can fuse the real projection/chunk producer into `_pipelined_produce_and_all2all`
or make NVSHMEM stable on the cluster. The current copy-only producer path is
correct but too fragmented to migrate.

### Projection direct-to-rank-major layout

The next promising Ulysses input-side experiment is to make Q/K/V projection
write image tokens directly in the `all_to_all_single` send layout:

```text
current:
  projection -> [B, S, H, D]
  pack copy  -> [rank, B, S, H/cp, D]
  all_to_all_single

target:
  projection -> [rank, B, S, H/cp, D]
  all_to_all_single
```

This can remove the input-side rank-major pack for Q/K/V. It does not remove
the post-a2a unpack (`[rank,B,S,H/cp,D] -> [B,rank*S,H/cp,D]`) unless attention
also accepts the rank-major sequence layout, and it does not address the output
image all-to-all pack/unpack.

### Text/Image split Ulysses attention

Another structural direction for Z-Image/Qwen-style CP attention is to exploit
the replicated text sequence. Non-causal attention is separable along the query
dimension, so the joint query attention
`attn(cat([img_q, txt_q]), cat([img_k, txt_k]), cat([img_v, txt_v]))` can be
split into image-query attention and text-query attention with shared K/V.

The harness has an experimental switch:

```bash
CHITU_HARNESS_A2A=single CHITU_ULYSSES_SPLIT=txt_q    # text attention covers img-Q input a2a
CHITU_HARNESS_A2A=single CHITU_ULYSSES_SPLIT=txt_out  # text attention covers img-output a2a
CHITU_HARNESS_A2A=single CHITU_ULYSSES_SPLIT=txt_both # split text queries across both gaps
```

This is likely a better direction than searching for another all-to-all API,
because it creates new hideable compute around both the image-Q input all-to-all
and the image-output all-to-all.

First harness result (`CHITU_HARNESS_A2A=single`, cp4, bf16, img=4096,
txt=512): all three split schedules are bit-identical to serial, including
CUDA graph capture (`max_abs_diff=0`, graph all ranks pass), but the naive split
is slower because it turns one joint SDPA into two or three smaller SDPAs.

| `CHITU_ULYSSES_SPLIT` | serial ms | overlap ms | serial-graph ms | overlap-graph ms | best vs eager serial |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` | 1.526 | 1.536 | 1.489 | 1.431 | +6.3% |
| `txt_q` | 1.526 | 1.656 | 1.492 | 1.576 | -3.2% |
| `txt_out` | 1.525 | 1.640 | 1.490 | 1.534 | -0.5% |
| `txt_both` | 1.524 | 1.794 | 1.490 | 1.709 | -12.2% |

The `txt_out` trace
`traces/ulysses_split_txt_out_graph_short.overlap_graph.rank0.json` confirms the
intended overlap: image-output `SendRecv` runs while text-query SDPA is active.
But over 5 graph replays, SDPA kernels double from 5 to 10 and SDPA time grows
from ~2.51 ms to ~3.37 ms, while the hidden output all-to-all is only tens of
microseconds per replay. So this split is not useful at 512 text tokens unless
we can keep a fused/joint SDPA shape or the text sequence becomes much longer.

## Repro

```bash
export CHITU_PROJECT_ROOT=$PWD SRUN_PARTITION=debug SRUN_EXTRA_ARGS="--exclusive"

# correctness
bash script/srun_direct.sh 1 4 experiments/fast_cp/harness/cp_async_harness.py --mode agkv --check

# on-node A/B (expect ~neutral)
bash script/srun_direct.sh 1 4 experiments/fast_cp/harness/cp_async_harness.py --mode agkv --bench

# comm-bound A/B (expect overlap to win)
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
  bash script/srun_direct.sh 1 4 experiments/fast_cp/harness/cp_async_harness.py \
    --mode agkv --bench --img-tokens 8192 --heads 48

# component ceiling
bash script/srun_direct.sh 1 4 experiments/fast_cp/harness/cp_async_harness.py --mode agkv --profile-parts
```
