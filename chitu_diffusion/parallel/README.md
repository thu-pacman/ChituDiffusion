# Parallel

该包只实现 Context Parallel 通信、attention 和 process-group 生命周期。

## NCCL CP

`nccl/` 提供 Torch/NCCL K/V gather、Ulysses all-to-all 和 Ring 组合。它是单机和
多机静态 CP 的默认路径。

## Fast CP

`fast_cp/` 提供单机 NVSHMEM transport：

- Fast AGKV：每个 rank 保留本地 Q 并接收全局 K/V。
- Fast Ulysses：在序列和 head 维度间重排。
- Fast Ring 与 fused Fast AGKV attention：仅供 benchmark 和实验。

构建条件、参数和 H20 图片见 [`fast_cp/README.md`](fast_cp/README.md)。

## 所有权

- `groups.py` 创建候选 lane group 并跟踪 active lane。
- `topology.py` 提供当前 lane 的 group view。
- `image_attention.py` 编排模型无关的 sharded attention。
- `agkv_transport.py` 与 `ulysses_transport.py` 定义公共 transport 协议和 factory。
- `nccl/` 与 `fast_cp/` 保存具体实现。

模型包只能依赖 `chitu_diffusion.parallel` 的公共接口，不应创建 process group 或直接
依赖具体 transport。EPE 决定 lane，parallel 层不管理请求、SLO 或 worker。
