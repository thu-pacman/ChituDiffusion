# Parallel

该包按执行域分为 `cp`、`tp` 和 `vae`。模型必须从对应子包导入，不再从
`chitu_diffusion.parallel` 根包导入符号。

## CP

`cp/` 负责 lane/process-group 生命周期、context-parallel attention 和 transport：

- `cp/nccl/`：Torch/NCCL K/V gather 与 Ulysses all-to-all，作为默认路径。
- `cp/fast/`：单机 NVSHMEM Fast AGKV/Fast Ulysses；Fast Ring 等代码仅供实验。
- `cp/context.py`：创建候选 lane group 并跟踪 active lane。
- `cp/topology.py`：Ulysses/USP topology。
- `cp/agkv_transport.py` 与 `cp/ulysses_transport.py`：transport 协议和 factory。

Fast CP 构建条件和 benchmark 见 [`cp/fast/README.md`](cp/fast/README.md)。

## TP

`tp/` 负责 tensor-parallel topology、Column/Row/MergedColumn/Replicated linear
以及按 rank 加载 checkpoint。TP 不依赖 CP 或 VAE。

## VAE

`vae/` 负责 decode lane 的最小 topology contract、空间 tile 规划、通信与 leader
重组。`VaeParallelPlacement` 统一两种归属：不指定 degree 时解码跟随当前 denoise
lane，指定 degree 时创建一个独立于 DiT TP/CP 的固定 decode group，例如 TP4×CP2
搭配 VAEP8。固定 group 的生命周期由 stage runtime 持有。具体 VAE 的 latent
归一化和单 tile decode 仍由模型包实现。

依赖方向固定为 `cp -> tp`；`vae` 持有独立 decode group；`tp` 不反向依赖
其他并行域。EPE 决定 lane，parallel 层不管理请求、SLO 或 worker。
