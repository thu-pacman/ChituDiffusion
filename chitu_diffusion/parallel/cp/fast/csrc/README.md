# Fast CP 扩展源码

## 中文

本目录是 ChituDiffusion 在 `fast-ulysses` 之上自有的 CUDA/C++ 源码。
`tools/install/install_fast_agkv.py` 把这些文件拷入指定的 `fast-ulysses`
checkout 并重新编译扩展；它们不是针对上游的补丁，而是由本仓库维护的实现：

| 文件 | 内容 |
|---|---|
| `fast_agkv.{cu,cuh}` | K/V sequence all-gather（SM 与 CE 两条路径）、Ring CE、U2FM2 |
| `ce_schedule.{cu,cuh}` | 两条 CE 路径共用的扇出调度、NUMA 布局探测和逐 shape 调优 |
| `all_to_all_ce.cu` | uniform 4D all-to-all 的 CE 传输 |
| `ulysses_group.{cu,cuh}` | group 状态：CE stream、调度与 NUMA 布局缓存 |
| `symmetric_pool.{cu,cuh}` | NVSHMEM symmetric heap pool |

`bindings.cpp` 与 `comm.py`、`__init__.py` 仍由安装器按锚点打补丁：这三个文件
归上游所有，我们只在其上追加算子注册、CE all-to-all 的调度调用，以及让 NVSHMEM
world 跟随调用方 process group 的改动。锚点缺失时安装器直接报错，用来暴露上游
漂移。

Fast AGKV 将本地 `[B, S_local, H, D]` K/V 直接写入每个 rank 的最终
`[B, S_global, H, D]` buffer，合并 K/V launch 和完成握手，避免 NCCL
`all_gather_into_tensor` 的中间组织开销。原独立 Full-Mesh fused attention 实验
已合并后删除，只保留统一的 Fast AGKV transport 和仍被 Ring/U2FM2 共用的底层
通信能力。

适用条件：

- 单机、P2P 全互通、静态 full-world CP，world size 为 2–8；NVLink 和 PCIe 主机
  都支持，传输路径按互连选择（见下）；
- FP16/BF16 contiguous BSHD，行宽必须 16-byte 对齐；
- `generate`/静态 pipeline；动态 EPE lane 会使用 Torch/NCCL fallback；
- 传输路径由主机互连决定，`CHITU_FAST_AGKV_USE_CE=auto|on|off` 可覆盖。NVLink
  主机（如 H20）走 SM direct-write，每个 peer 有独立链路可填；PCIe 主机走 CE，
  一张 GPU 的所有 peer 共用一个 egress 端口，此时铺开只会掉带宽——8x RTX PRO
  5000 CP8 上 CE all-gather 比 SM 快 1.1–1.2 倍且不占 SM。
- Wan processor 会将 K/V all-gather 与 Q projection/norm/RoPE 重叠；走 CE 时始终
  重叠，走 SM 时仅在 CP≤4 重叠（SM kernel 会与 attention 抢 SM，CP8 上强制
  overlap 实测更慢）。`CHITU_FAST_AGKV_ASYNC=auto|on|off` 可覆盖此策略。

安装到已检出的 `fast-ulysses`：

```bash
python tools/install/install_fast_agkv.py /path/to/fast-ulysses
# 然后按照 fast-ulysses 的 NVSHMEM/CUDA 构建说明重新安装扩展。
# --check 只报告是否需要重新安装（需要时退出码为 1）。
```

运行时选择：

```bash
chitu ... --agkv-transport fast_agkv
# 或
CHITU_AGKV_TRANSPORT=fast_agkv chitu ...
```

H20、BF16、`[1,4096,24,128]` 的单元测试 microbenchmark 中，direct-write
K/V 通信相对优化后的 NCCL AGKV 分别在 CP2/CP4/CP8 达到约
`1.81x / 1.37x / 1.16x`；包含 SDPA 的完整 attention 延迟降低约
`7.1% / 7.6% / 5.5%`。这些是固定 shape microbenchmark，不应直接解释为模型
端到端收益。

Wan 1.3B attention processor（同一 shape）中，CP2/CP4 overlap 相对 NCCL
AGKV 分别降低约 `5.8% / 7.7%`；CP8 的同步 Fast AGKV 降低约 `6.9%`，而 SM 路径上
强制 overlap 比同步慢约 `19%`，所以 SM 路径的默认门禁不在 CP8 开启 overlap。

8x RTX PRO 5000（PCIe，CP8，Wan 1.3B shape）上 CE 路径的完整 attention 延迟：
NCCL `1.089 ms`、SM 同步 `0.938 ms`、CE 同步 `0.789 ms`、CE overlap
`0.725 ms`——CE 不占 SM，所以 overlap 在 CP8 仍然有收益。

all-to-all 的 CE 路径共用同一套调度。8x RTX PRO 5000、CP8、Wan 14B 720p shape
下单独耗时从 `3.438 ms`（每 peer 一条 stream）降到 `2.995 ms`（单条远端
stream），并在 GEMM 之下完全隐藏；SM kernel 单独跑仍更快（`2.274 ms`）但只能隐藏
`14%`，所以同步 Ulysses 仍走 SM，CE 只用于重叠路径。

## English

Fast AGKV extends the `fast-ulysses` NVSHMEM symmetric heap with a K/V sequence
all-gather. Each rank writes local `[B, S_local, H, D]` K/V shards directly into
the final `[B, S_global, H, D]` buffers on every peer. K and V share one launch
and one completion handshake.

This directory holds the CUDA/C++ sources ChituDiffusion owns:
`fast_agkv.{cu,cuh}` (the all-gather plus Ring CE and U2FM2),
`ce_schedule.{cu,cuh}` (the copy-engine fan-out schedule, NUMA layout probe and
per-shape tuner shared by both CE paths), `all_to_all_ce.cu`,
`ulysses_group.{cu,cuh}` and `symmetric_pool.{cu,cuh}`. The installer copies
them into a checkout rather than patching it. Only `bindings.cpp`, `comm.py` and
`__init__.py` stay anchor-patched, since upstream owns those and we only append
to them; a missing anchor is a hard error, which is how upstream drift surfaces.

The production gate requires static full-world, single-node CP with 2–8 ranks,
contiguous FP16/BF16 BSHD tensors, and 16-byte-aligned rows. Unsupported or
dynamic lanes use the Torch/NCCL fallback.

The transport follows the host fabric, and `CHITU_FAST_AGKV_USE_CE=auto|on|off`
overrides it. NVLink hosts such as H20 take the SM direct-write path, where each
peer has its own link to fill. PCIe hosts take the copy engines, where all peers
share one egress port and spreading a transfer only costs bandwidth: on 8x RTX
PRO 5000 at CP8 the CE all-gather is 1.1–1.2x faster than the SM kernel and
takes no SMs at all. Wan overlaps the K/V gather with Q projection/norm/RoPE:
always on the CE path, and only at CP≤4 on the SM path, whose kernel competes
with the attention it overlaps. `CHITU_FAST_AGKV_ASYNC=auto|on|off` controls it.

The all-to-all takes the same CE schedule. At CP8 with the Wan 14B 720p shape on
8x RTX PRO 5000 it went from 3.438 ms with a stream per peer to 2.995 ms with a
single remote stream, and it hides completely under a GEMM chain. The SM kernel
is still faster standalone (2.274 ms) but hides only 14%, so the synchronous
Ulysses transport keeps the kernel path and CE serves the overlapped one.

The former standalone Full-Mesh fused-attention experiment was removed after its
communication work was folded into the unified Fast AGKV transport. What remains
is that transport and the lower-level primitives still shared by Ring/U2FM2.

Install with `tools/install/install_fast_agkv.py`, rebuild `fast-ulysses`, and
select `--agkv-transport fast_agkv` or set `CHITU_AGKV_TRANSPORT=fast_agkv`.
