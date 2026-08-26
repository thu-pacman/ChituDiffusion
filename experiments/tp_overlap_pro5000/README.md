# H3 TP GEMM / NCCL all-reduce overlap 分析

## 测量范围

- 硬件：8 × RTX PRO 5000
- 拓扑：TP4 × CP2，CP 使用同步 Fast Ulysses
- 输入：21,824 tokens，每个 CP rank 10,912 tokens
- 模型：MiniMax-H3 FL2VA Transformer，FA4
- 方法：先执行 1 次完整 warmup，再用 CUDA Event 测量每个
  `RowParallelLinear` 的本地 GEMM 和紧随其后的 NCCL all-reduce。
- 原始数据写入 `outputs/`，不纳入版本库

执行：

```bash
export NVSHMEM_HOME=$PWD/refs/nvshmem-developer
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
export NCCL_IB_DISABLE=1
export CHITU_ULYSSES_TRANSPORT=fast_ulysses
export CHITU_FAST_ULYSSES_ASYNC_CE=0

/dockerdata/chitudiffusion-venv-sm120/bin/torchrun \
  --standalone --nproc-per-node=8 \
  experiments/tp_overlap_pro5000/profile_h3.py \
  --output outputs/tp-overlap-pro5000/results.json
```

## 实测结果

8 个 rank 的稳态 DiT forward 都约为 1,991 ms。关键 rank 上：

- 101 次 RowParallel all-reduce：811.34 ms，占 forward 的 **40.74%**
- 对应的本地 RowParallel GEMM：132.89 ms，占 **6.67%**
- 50 次 attention `out_proj`：GEMM 43.48 ms，all-reduce 406.50 ms
- 50 次 MLP `fc2`：GEMM 89.37 ms，all-reduce 404.33 ms
- 单层 all-reduce 约 8.1 ms，而 attention / MLP GEMM 分别只有约
  0.87 / 1.79 ms

各 rank 的 all-reduce 占比为 40.41%–40.94%，结果一致。这里的占比是
当前同步执行路径的 GPU 关键路径时间，不包含模型加载和首次编译。

## Overlap 收益估计

沿 token 维切分 RowParallel GEMM，并让前一块的 NCCL all-reduce 与后一块
GEMM 重叠。对每层记完整 GEMM 时间为 `G`、all-reduce 时间为 `A`：

- 两块流水：理论节省 `min(G, A) / 2`
- 无限细分理想上限：理论节省 `min(G, A)`

由于本机上每层都是 `A >> G`，可隐藏的上限由 GEMM 决定：

- **2-way chunk 理论值**：节省 66.44 ms / forward，1,991.38 → 1,924.94
  ms，即 **1.0345× / 3.45%**
- **无限细分理想上限**：节省 132.89 ms / forward，1,991.38 →
  1,858.49 ms，即 **1.0715× / 7.15%**
- 粗略外推 20 个 denoise step：两块最多节省约 1.33 s，理想上限约
  2.66 s；端到端还包含 VAE 等阶段，因此整体比例会更低

这些数字假设 GEMM 和通信随分块线性缩放，且没有额外的 kernel launch、
同步、buffer 管理和资源竞争。实际 NCCL kernel 会占用部分 SM，和 GEMM
并发后可能互相减速；两块还会把 collective 数量翻倍。因此现实收益应低于
3.45%，很可能只有低个位数。

## 结论

TP all-reduce 本身确实是显著瓶颈，约占 DiT forward 的 41%；但简单
GEMM/all-reduce overlap 只能隐藏较短的本地 GEMM，不能消除 8 ms 级的
通信主体。建议若继续，只做 2-way token chunk 的隔离原型并实测；在看到
稳定收益前，不接入 H3 主链路。更大的收益需要降低 all-reduce 本身的成本，
而不只是做 overlap。
