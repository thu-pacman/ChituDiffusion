# Parallel

该包按执行域分为 `cp`、`tp` 和 `vae`。模型必须从对应子包导入，不再从
`chitu_diffusion.parallel` 根包导入符号。根包只放跨域共用的主机事实：
`interconnect.py` 从驱动读出本机 GPU 之间的互连形态（有无 NVLink、每张卡的 NUMA
节点），供需要在"铺开传输"和"串行传输"之间取舍的调度使用——PCIe 主机一张卡的所有
peer 共用一个 egress 端口，NVLink 主机每个 peer 有独立链路，两者的最优调度相反。

## CP

`cp/` 负责 lane/process-group 生命周期、context-parallel attention 和 transport：

- `cp/nccl/`：Torch/NCCL K/V gather 与 Ulysses all-to-all，作为默认路径。
- `cp/fast/`：单机 NVSHMEM Fast AGKV/Fast Ulysses；Fast Ring 等代码仅供实验。
  AGKV 走 SM 还是 copy engine 由 `interconnect.py` 探测的互连决定。CUDA/C++
  扩展源码在 `cp/fast/csrc/`。
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
