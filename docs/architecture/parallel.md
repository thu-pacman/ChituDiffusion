# 上下文并行

并行层包含两类 Context Parallel 实现。

## NCCL CP

默认路径使用 `torch.distributed` 和 NCCL collective，适合单机或多机静态 CP。
`attention_mode=agkv` 使用 K/V gather；`attention_mode=ulysses` 使用
all-to-all，并可在剩余并行维度使用 NCCL Ring。

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --output outputs/zimage-nccl.png
```

## Fast CP

Fast AGKV 和 Fast Ulysses 使用 NVSHMEM transport，面向支持 P2P 的单机 Hopper。
扩展必须针对目标 CUDA、NVSHMEM 和 GPU 架构编译。

```bash
# Fast AGKV transport
--agkv-transport fast_agkv

# Fast Ulysses transport
--attention-mode ulysses --ulysses-transport fast_ulysses
```

Fast Ring 和 fused Fast AGKV attention 是 benchmark/实验路径，模型 pipeline 不会自动
选择它们。安装和限制见仓库内 `chitu_diffusion/parallel/cp/fast/README.md`。

![H20 单机 Fast CP scaling](../assets/fast_cp/fast-cp-h20-single-node-scaling.png)

![H20 单机 Fast CP winner matrix](../assets/fast_cp/fast-cp-h20-single-node-winner-matrix.png)

## 与其他模块的关系

并行层不管理请求、SLO 或 worker。FlexCache 可叠加静态 NCCL CP 或 Fast CP，但缓存层
不创建 process group，也不选择并行布局。
