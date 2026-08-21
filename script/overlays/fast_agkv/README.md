# Fast AGKV

## 中文

Fast AGKV 是 ChituDiffusion 在 `fast-ulysses` NVSHMEM symmetric heap 之上增加的
K/V sequence all-gather。它将本地 `[B, S_local, H, D]` K/V 直接写入每个 rank 的
最终 `[B, S_global, H, D]` buffer，合并 K/V launch 和完成握手，避免 NCCL
`all_gather_into_tensor` 的中间组织开销。

同一 overlay 也承载 Full-Mesh fused attention 的 symmetric K/V buffer、
per-segment arrival flags 和 consumed-epoch handshake。安装器会同时覆盖
`symmetric_pool.{cu,cuh}`，保证 flag 只在新 slice 创建时初始化；不要只复制
`fast_agkv.cu`，否则会重新引入跨 epoch 清零竞态。

CP2 默认使用 remote-only split：只把远端 K/V 写入 symmetric buffer，本地 K/V
直接执行一个 partial attention；远端 arrival 后执行第二个 partial，并按 LSE
精确合并。这避免了本地 K/V D2D materialization。该路径只适用于 C1
source-chunk Copy Engine；CP4/CP8 和实验性 chunk/scheduler 仍使用单-kernel
streaming-KV 路径。

适用条件：

- 单机、全 NVLink、静态 full-world CP，world size 为 2–8；
- FP16/BF16 contiguous BSHD，行宽必须 16-byte 对齐；
- `generate`/静态 pipeline；动态 EPE lane 会使用 Torch/NCCL fallback；
- 当前默认使用 SM direct-write；CE 路径可通过
  `CHITU_FAST_AGKV_USE_CE=1` 实验性启用，但 H20 CP4/CP8 实测更慢。
- Wan processor 会在 CP2/CP4 将 K/V all-gather 与 Q projection/norm/RoPE
  重叠；CP8 默认使用同步 direct-write，因为强制 overlap 的实测延迟更高。
  `CHITU_FAST_AGKV_ASYNC=auto|on|off` 可覆盖此策略。

将版本化 overlay 应用到已检出的 `fast-ulysses`：

```bash
python script/install_fast_agkv.py /path/to/fast-ulysses
# 然后按照 fast-ulysses 的 NVSHMEM/CUDA 构建说明重新安装扩展。
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
AGKV 分别降低约 `5.8% / 7.7%`；CP8 的同步 Fast AGKV 降低约 `6.9%`，而强制
overlap 比同步路径慢约 `19%`，因此默认门禁不会在 CP8 开启 overlap。

## English

Fast AGKV extends the `fast-ulysses` NVSHMEM symmetric heap with a K/V sequence
all-gather. Each rank writes local `[B, S_local, H, D]` K/V shards directly into
the final `[B, S_global, H, D]` buffers on every peer. K and V share one launch
and one completion handshake.

The production gate requires static full-world, single-node NVLink CP with
2–8 ranks, contiguous FP16/BF16 BSHD tensors, and 16-byte-aligned rows.
Unsupported or dynamic lanes use the Torch/NCCL fallback. The SM direct-write
path is the default; the optional CE path is experimental and was slower on
H20 at CP4/CP8. Wan overlaps the K/V gather with Q projection/norm/RoPE at
CP2/CP4. CP8 defaults to synchronous direct-write because forced overlap was
slower; `CHITU_FAST_AGKV_ASYNC=auto|on|off` controls this gate.

At CP2, the default C1 Copy Engine path sends only the remote K/V shard. Local
K/V feeds one partial attention directly; a second partial consumes the remote
shard after its arrival flag, and an exact LSE merge produces the final output.
CP4/CP8 and experimental chunked schedulers keep the single-kernel streaming-KV
path.

Apply the versioned overlay with `script/install_fast_agkv.py`, rebuild
`fast-ulysses`, and select `--agkv-transport fast_agkv` or set
`CHITU_AGKV_TRANSPORT=fast_agkv`.
