# Parallel

该目录维护 EPAC 中与模型无关的并行拓扑、通信和计算。根目录提供稳定 facade、
topology/group 生命周期、公共 transport 协议与 factory，以及模型无关 attention
编排；`fast_cp/` 集中 Fast-CP 算法与 NVSHMEM transport，`nccl/` 集中
`torch.distributed`/NCCL 实现。它不是仅面向 context parallel 的包：当前实现 CP，
后续 CFG parallel 也应复用同一套动态 lane 和 process-group 生命周期。

- `groups.py`：启动阶段创建候选 lane group，并维护当前 active lane；
- `topology.py`：active lane 对应的 Ulysses/Ring group view；
- `usp.py`：Ulysses×Ring 兼容 facade；实现在 `fast_cp/ulysses_ring.py`；
- `image_attention.py`：统一的 AGKV/USP 配置，以及 sharded self-attention 和
  `sharded image + replicated joint/text` 两类模型无关接口。
- `ulysses_transport.py`、`agkv_transport.py`：公共协议、名称解析、能力 gate 与
  factory；
- `nccl/`：Torch Ulysses、AGKV、Ring 和 NCCL Full-Mesh prototype；
- `fast_cp/`：Fast Ulysses、Fast AGKV attention/transport、Fast Ring 和
  Ulysses×Ring；硬件、selector 和安装约束见
  [`fast_cp/README.md`](fast_cp/README.md)。

`groups.py` 是通用资源层。模型目录只能依赖根目录公共协议/facade，并保留
QKV/RoPE/布局及 overlap glue；不应 import `fast_cp/` 或 `nccl/` concrete，不应复制
通信算子或创建 process group。

USP 的 all-to-all 与 joint-ring schedule 基于 xDiT/xFuser 和 yunchang 的
Apache-2.0 实现。与 xFuser 原始全局 `PROCESS_GROUP` 不同，这里的 group 是每次调用显式
传入的，因此 EPAC 可以在 pulse 之间从 cp4 `u2r2` 切换到 cp2 `u2r1` 或 cp1 `u1r1`。
所有候选 group 必须在服务启动阶段以确定顺序创建；attention forward 不创建或销毁 group。

安装 USP 可选依赖：

```bash
uv sync --extra usp
```

CUDA 的 NCCL 路径使用 `torch.distributed.all_to_all_single`、FlashAttention 和
xDiT-style joint ring；Fast 路径按 capability gate 替换其中的 Ulysses transport。
CPU/Gloo 路径只用于多进程数值回归，不代表服务性能。

所有模型入口默认 `attention_mode="agkv"`。选择 `attention_mode="usp"` 且未指定
`ulysses_degree` 时，根据实际 attention heads、active lane 和真实节点边界选择最大
合法 U degree；剩余 degree 交给 NCCL Ring。显式 degree 是上限，也会向下选择同时
整除 heads 与 lane width 的 divisor。U row 不跨节点，Ring column 可以跨节点。

2026-07-28 在 4 x H20 上完成以下 USP smoke：Z-Image 512/2-step
（`examples/zimage_epe.py`，Slurm `201838`）、FLUX.2-klein 512/1-step
（`examples/flux2_klein_cp.py`，`201833`）、Qwen-Image 512/4-step CFP2 x CP2
（`examples/qwen_image_epac.py`，`201839`）和 Wan2.1-T2V-1.3B
832x480/17-frame/2-step CFP2 x CP2（`examples/wan_epac.py`，`201835`）。调用形式均为：

```bash
bash script/srun_direct.sh 1 4 <entry> \
  --model-path <path> --attention-mode usp --ulysses-degree 2
```

四个作业均完成生成并正常退出；Qwen PNG SHA256 为
`54c3291b13c9ddc760490eccef851c9b36feb5c7da6907da732fa313135b7291`，Wan 完成 MP4
编码。FLUX.1 的同步 facade follower parallel-VAE 收尾问题修正后，以 512/1-step
u2r2 作业 `201861` 完整验证。共享 pure self-attention 另以 4 卡 BF16 CUDA 数值
smoke `201841` 验证 u2r2，最大绝对误差不超过 `0.0078125`。
