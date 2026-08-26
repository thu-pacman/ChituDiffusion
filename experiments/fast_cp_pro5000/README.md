# RTX PRO 5000 Fast CP 微基准

本目录将 Context Parallel 通信从 H3 完整 pipeline 中独立出来，在本机
8× NVIDIA RTX PRO 5000 72GB Blackwell 上比较 Ulysses 和 AGKV transport：

- Ulysses：NCCL `all_to_all_single` 对比 Fast Ulysses；
- AGKV：NCCL K/V all-gather 对比 Fast AGKV kernel 和 Copy Engine。

计时周期与 H3 attention 的通信结构一致：

1. Q、K、V 分别执行 heads-to-sequence all-to-all；
2. attention 输出执行 sequence-to-heads all-to-all。

H3 768×768、5 秒请求对应全局 packed sequence 21,824 tokens、56 个全局
attention heads、head dimension 128。测试按照 8 卡拓扑分别构造
TP4×CP2、TP2×CP4 和 TP1×CP8 的 TP-local tensor。

## Ulysses 测试结果

H3 21,824-token shape：

- TP4×CP2：NCCL 3.092 ms，Fast 1.596 ms，**加速 1.94×**；
- TP2×CP4：NCCL 4.120 ms，Fast 2.509 ms，**加速 1.64×**；
- TP1×CP8：NCCL 5.675 ms，Fast 3.581 ms，**加速 1.58×**。

在全部测试长度 4,096–21,824 tokens 上，Fast Ulysses 都快于 NCCL，
通信周期的 median latency 降低 23%–48%。

## AGKV 测试结果

AGKV 只测 K/V sequence all-gather，不包含后续 attention。H3
21,824-token shape：

- TP4×CP2：NCCL 2.185 ms，Fast kernel 1.560 ms（**1.40×**），
  Fast CE 1.438 ms（**1.52×**）；
- TP2×CP4：NCCL 7.377 ms，Fast kernel 4.768 ms（**1.55×**），
  Fast CE 8.464 ms（**0.87×，慢于 NCCL**）；
- TP1×CP8：NCCL 18.754 ms，Fast kernel 13.236 ms（**1.42×**），
  Fast CE 35.860 ms（**0.52×，明显慢于 NCCL**）。

Fast AGKV kernel 在全部 CP degree 和 sequence length 上都快于 NCCL。
Copy Engine 只在同 NUMA 的 CP2 上获胜；CP4/CP8 的多 peer PCIe copy 无法被当前
同步微基准中的计算覆盖，且 CP8 跨 NUMA，因此 CE 路径反而更慢。当前
`FastAgkvTransport` 默认 `CHITU_FAST_AGKV_USE_CE=0`，与本机测试结论一致。

## 性能优势来自哪里

### 1. 省去 NCCL 前后的显式 layout copy

标准 Torch/NCCL Ulysses 的一次 all-to-all 包含：

1. `reshape + permute + contiguous`，把输入整理成 NCCL send layout；
2. `dist.all_to_all_single`；
3. `permute + reshape + contiguous`，恢复 `[B, S, H, D]` layout。

一次 H3 通信周期需要处理 Q、K、V 和输出，因此会反复启动 layout kernel，
并产生 send/recv 临时 tensor。Fast Ulysses 直接用源和目标 stride 描述
sequence/head 重排，将重排融合进跨 GPU 搬运，不再单独执行前后两次
`contiguous` copy。这是当前微基准最主要的收益来源。

### 2. 直接写入目标 GPU 的 symmetric buffer

Fast Ulysses 预先从 NVSHMEM symmetric heap 分配并复用输出 buffer。发送 rank
可以通过 P2P 地址直接写入目标 rank 的最终位置，不需要等待 NCCL collective
先完成到中间 buffer，再执行额外搬运。固定 tag 的 buffer 会跨迭代复用，也减少了
allocator 和 collective setup 开销。

### 3. 针对 4D Ulysses shape 的专用路径

Fast Ulysses 只处理 `[B, S, H, D]` 的固定 Ulysses 交换，能够根据 shape 在
TMA 和 non-TMA 路径之间自动选择。NCCL 是通用 collective，需要支持更多 tensor
布局、拓扑和通信模式，因此在这种规则、单机、dense shape 上有更高的固定成本。

### 4. CP2 避免跨 NUMA，收益最大

本机 GPU 之间没有 NVLink，而是 PCIe `NODE/SYS` 拓扑。CP2 使用同一 NUMA
侧的相邻 GPU，P2P 路径较短，因此 H3 shape 达到 1.94×。CP4 扩大 peer 数量，
CP8 还会跨 NUMA socket，P2P 路径和同步成本上升，所以收益下降到 1.64× 和
1.58×。这说明 Fast CP 在本机的优势主要来自减少软件和 layout 开销，而不是
NVLink 带宽。

### 5. 当前数字尚未包含通信与计算重叠

本测试调用的是 H3 当前使用的同步 `all_to_all` 路径，`TMA=auto`。Fast CP
另外提供 Copy Engine 异步接口，可将通信与 Q/K/V projection 或 attention
计算重叠，但本组数字**没有**计入这部分潜在收益。因此这里测到的是 transport
本身的差异，不应把结果解释成 Copy Engine overlap 的收益。

### 6. AGKV 同样消除了 NCCL gather 后的 layout copy

标准 AGKV 先用 NCCL 将 K/V gather 成 rank-major tensor，再执行
`unflatten + permute + reshape + contiguous`，才能得到 attention 所需的
sequence-major K/V。Fast AGKV 直接把每个 rank 的 K/V 写入 symmetric buffer
中的最终 sequence 位置，因此 kernel 路径在 CP2、CP4、CP8 都有 1.40×–1.55×
收益。

AGKV 的 Copy Engine 路径需要向多个 peer 分别发起 DMA copy。CP2 只有一个远端
peer，CE 能达到约 54 GB/s；CP4/CP8 的 peer 数增加，并跨越更复杂的 PCIe/NUMA
路径，同步执行时固定开销和最慢 peer 决定总延迟。CE 的价值应在与 attention
计算重叠后重新评估，不能根据 transport-only 同步数字直接否定异步 CE。

## MiniMax-H3 端到端验收

TP4×CP2、FA4、Fast Ulysses 已在 8×RTX PRO 5000 上通过真实权重验收：

- 768×768、5 秒、24 FPS、20 steps；
- NCCL baseline 总延迟 56.90 秒；
- 经过一次同 shape 预热后，Fast Ulysses 总延迟 55.63 秒，端到端提升 2.2%；
- DiT 每次 transition 从 2.103 秒降至 2.010 秒，提升 4.4%；
- 输出 H.264 768×768 视频和 AAC 音频，文件位于
  `outputs/h3-fast-ulysses-tp4cp2/h3-tp4cp2-fast-ulysses-fa4-768p-5s-20steps.mp4`。

H3 warmup 会先运行很小的 attention shape，因此 symmetric pool 必须按文档默认
预留 2 GiB，不能按首次 warmup tensor 缩小；否则正式 768p 请求会在 pool 固定为
64 MiB 后 OOM。

曾验证过 CE 与 QKV projection overlap；稳定版本端到端为 56.21 秒，仍慢于同步
Fast Ulysses 的 55.63 秒，且需要额外 producer fence。相关实现已回退，H3 当前
仅保留同步 Fast Ulysses 路径。

## 结果边界

- 这是 CP transport 微基准，没有同时运行 TP collective、FA4 和 projection；
- 完整 H3 中 TP 和 CP 会竞争同一 PCIe fabric，端到端收益通常低于微基准；
- 每次迭代使用所有 rank 中最慢 rank 的 latency，预热 10 次、测量 30 次；
- 计时前先与 NCCL 结果做 bit-exact 正确性校验；
- 不比较 PyTorch allocator peak memory，因为 NVSHMEM symmetric heap 不计入
  PyTorch allocator，直接比较会低估 Fast CP 的实际显存占用；
- AGKV 结果是 transport-only，不包含 fused Fast AGKV attention。

## 复现

在 ChituDiffusion 仓库根目录执行：

```bash
export NVSHMEM_HOME=$PWD/refs/nvshmem-developer
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NVSHMEM_DISABLE_NVLS=1
export NVSHMEM_REMOTE_TRANSPORT=none
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  experiments/fast_cp_pro5000/benchmark.py \
  --output outputs/fast-cp-pro5000/results-cp2.json

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  experiments/fast_cp_pro5000/benchmark.py \
  --output outputs/fast-cp-pro5000/results-cp4.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8 \
  experiments/fast_cp_pro5000/benchmark.py \
  --output outputs/fast-cp-pro5000/results-cp8.json

# AGKV：将 benchmark.py 换为 benchmark_agkv.py，并使用独立输出文件。
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  experiments/fast_cp_pro5000/benchmark_agkv.py \
  --output outputs/fast-cp-pro5000/results-agkv-cp2.json
```

原始 benchmark 数据写入已被 `.gitignore` 排除的 `outputs/`，仓库仅保留复现
脚本与汇总结论。
