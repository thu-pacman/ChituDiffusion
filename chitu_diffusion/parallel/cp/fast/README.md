# Fast CP

Fast CP 提供单节点 AGKV 和 Ulysses，以及覆盖 QKV projection 至 output projection
的显式推理接口。NVSHMEM transport、Attention Planner 和模型 adapter 各自独立；
runtime 统一管理通信槽位、准备流程与资源生命周期。

新环境请从[安装与启用指南](../../../../docs/usage/fast-cp-installation.md)开始：
包含依赖、固定源码版本、编译、CuTe 后端、多卡验证与两种启用方式。

## 目录与阅读入口

```text
fast/
├── __init__.py              # 公共符号，adapter 按需加载
├── api.py                   # init_fast_cp、FastCPRuntime、FastCPAttention
├── runtime/
│   ├── backend.py           # 扩展加载、对称堆、SM/TMA/CE 选择
│   ├── agkv.py              # K/V gather transport
│   └── ulysses.py           # all-to-all transport 与 logical subgroup
├── attention/
│   ├── planner.py           # 候选估算、准备与执行
│   ├── cute.py              # CuTe 后端加载
│   └── merge.py             # FP32 partial output/LSE 合并
├── projection/
│   ├── kv.py                # 对称槽位 KV producer 与 CE 发送
│   └── qk_epilogue.py       # RMSNorm/RoPE、packed QKV 融合
├── adapters/
│   ├── base.py              # 协议、能力检查、执行对象
│   ├── _execution.py        # 共用的分段 Graph/AGKV 调度
│   ├── wan.py               # Wan 契约与执行模板
│   ├── qwen_image.py        # Qwen 布局、packed/overlap 选择
│   └── zimage.py            # ZImage 分片或复制文本布局
├── profiles/
│   └── h20_nvlink.py        # 已校准的能力与 kernel 成本参数
├── csrc/                    # CUDA/C++ 扩展源码
├── experimental/
│   └── ring.py              # 独立 Fast Ring 实验
└── test/                    # 单元测试、GPU 测试与分布式验收
```

| 任务 | 入口 |
| --- | --- |
| 在模型中使用 Fast CP | 包根目录的 `init_fast_cp` 与三个 adapter，示例见下文 |
| 理解准备和资源生命周期 | [api.py](api.py) |
| 修改通信或搬运策略 | [runtime/backend.py](runtime/backend.py) |
| 修改 attention 候选与 wave 估算 | [attention/planner.py](attention/planner.py) |
| 接入新模型 | [adapters/base.py](adapters/base.py) 与对应模型文件 |
| 校准另一种设备 | [profiles/h20_nvlink.py](profiles/h20_nvlink.py) |
| 编译 CUDA 扩展 | [csrc/README.md](csrc/README.md) 与下文构建命令 |
| 运行正确性和性能回归 | [test/README.md](test/README.md) |

执行关系是 `api → adapter → projection / attention / runtime → CUDA 扩展`。
`runtime/` 不导入模型，`attention/` 不选择通信算法，`profiles/` 不持有 tensor 或
CUDA stream。模型文件声明数学和分片语义，共用 `_execution.py` 的调度，不通过
继承另一个模型复用代码。

包根目录是应用调用入口；内部目录供维护者修改实现。原未发布的平铺模块和
`experimental/` 兼容转发已删除，仓库调用方已迁移；原有顶层 transport 符号保留。
`experimental/ring.py` 不属于 wrapper 的正式执行路径。

## 使用 wrapper

当前验证范围：单节点 H20 NVLink、CP2/4/8、TP1、B1、BF16、D128、等长序列分片、
无 mask 的非 causal MHA。模型接入方显式调用以下接口；尚未自动安装到完整 pipeline。

```python
import torch
from chitu_diffusion.parallel.cp.fast import QwenImageAdapter, init_fast_cp

with torch.inference_mode(), init_fast_cp(cp_group, device=device) as cp:
    block = cp.wrap_attention(
        attention_module.eval(),
        adapter=QwenImageAdapter(),
        origin=original_region,
        algorithm="agkv",
        schedule="auto",
    )
    inputs = dict(
        hidden_states=image_shard,
        encoder_hidden_states=text_shard,
        image_rotary_emb=(image_rope_shard, text_rope_shard),
    )
    block.prepare(example_inputs=inputs)
    image_output, text_output = block(**inputs)
```

`original_region` 接收相同 keyword arguments，返回相同本地分片。准备阶段若任意
rank 能力不足，所有 rank 一致选择 origin；未提供 origin 时报错。forward 不调优，
也不会在通信已失败后切换 NCCL。

普通调用允许输入地址变化：数据复制到稳定 Graph 槽位，返回独立输出。固定地址
场景使用绑定入口，省去这两项复制：

```python
bound = block.bind(inputs=inputs)  # 所有 rank 集体准备
# 上游在 runtime 的构造 stream 上原地更新 inputs。
image_output, text_output = bound.run()
print(bound.metadata)
bound.close()
```

绑定输出是借用的 scratch，需要跨后续调用保存时由调用方 clone。绑定入口的性能
不包含普通入口的 staging 和输出复制成本。

| Adapter | 输入分布 | AGKV 路径 |
| --- | --- | --- |
| `WanAdapter()` | 序列分片，完整投影宽度 Q/K RMSNorm，可选局部 cos/sin | KV 直写，CE gather 与 Q projection 重叠 |
| `QwenImageAdapter()` | image/text 都分片，复数 RoPE，text-first | 准备时比较 packed/overlap；packed epilogue 直写对称 KV |
| `ZImageAdapter(image_tokens=..., text_distribution=...)` | image-first；文本 `sharded` 或 `replicated` | 对应文本分布的 CE overlap |

`algorithm="ulysses"` 使用串行模板，保留 head padding、Attention Planner 和输入/
输出方向独立的 SM/TMA/CE 选择。AGKV wrapper 使用 CE。每个模型文件同时包含契约、
AGKV 执行模板和 Ulysses 构建逻辑。

## 资源与失效

- 所有 rank 按相同顺序 prepare/bind 和执行。shape、stride、dtype 或输入结构改变
  时先集体重新准备。权重、LoRA、norm/RoPE 配置改变前调用 `block.invalidate()`。
- 同一 backend 的 wrapper 共用锁和 workspace，只允许在构造 stream 顺序执行。
  同形状的不同层复用通信槽位，权重、Graph 和输出各自持有。
- 默认 `pool_bytes=2 << 30`、`max_workspaces=8`。堆不自动扩容或逐槽回收；长序列
  或多个 workload 应显式预留容量。关闭 binding 不释放共享槽位。
- 已有 NVSHMEM backend 时传入 `backend=existing_backend`。借用者关闭不销毁堆；
  拥有者必须在借用 wrapper 关闭后释放资源。
- 当前使用计算分段 Graph，不支持外层 CUDA Graph capture 或并发 stream 执行。

## 独立 transport 与 attention

现有 pipeline 的 transport factory 仍支持 `--ulysses-transport fast_ulysses`
与 `--agkv-transport fast_agkv`。公共协议和 factory 位于 `cp/`，NCCL 实现位于
`cp/nccl/`。transport 的设备、dtype 和拓扑范围比 wrapper 宽；SM120 基础通信可
构建不代表模型 wrapper 已在 PRO 5000 验证。

共享 attention 的入口独立于通信。planner 从 shape、dtype、SM 数和 CP degree
生成候选；成本参数位于 `profiles/h20_nvlink.py`，不保存逐 shape 的静态选择表。

```python
from chitu_diffusion.parallel.cp.fast.attention import prepare_attention

attention = prepare_attention(query, key, value, cp_degree=cp_degree)
output = attention(query, key, value)
```

独立 transport 的常用配置如下；wrapper 的对称堆容量由 `init_fast_cp(pool_bytes=...)`
指定，AGKV 执行模板由 adapter 决定。

| 环境变量 | 作用 |
| --- | --- |
| `CHITU_ULYSSES_TRANSPORT` / `CHITU_AGKV_TRANSPORT` | 选择 `torch`、`auto` 或对应 Fast transport |
| `CHITU_FAST_ULYSSES_TRANSFER=auto/sm/tma/ce` | Ulysses 搬运方式；auto 按实际 shape 和方向测量 |
| `CHITU_FAST_ULYSSES_POOL_BYTES` / `CHITU_FAST_AGKV_POOL_BYTES` | 独立 transport 的对称堆容量 |
| `CHITU_FAST_AGKV_USE_CE=auto/on/off` | 独立 AGKV 的 CE 选择，auto 考虑本地互连 |
| `CHITU_FAST_AGKV_ASYNC=auto/on/off` | 独立 AGKV 通信与 attention 的重叠策略 |
| `FAST_ULYSSES_CE_TUNE_VERBOSE=1` | 输出扩展中的 CE 扇出调优结果 |

`auto` 在准备时能力不足可回落到 Torch/NCCL，显式 Fast 选择则报告错误。wrapper
遵循前述集体 origin 回退契约；这两套入口的调度范围不同。

## 构建

`csrc/` 保留单一源码；同一扩展可包含 SM90/SM120 机器码，由 CUDA loader 选择。
`tools/install/build_fast_cp.py` 应用源码 overlay、编译 wheel 并检查 native image；
`install_fast_agkv.py` 只负责 overlay，不编译。需要针对 Python/Torch/CUDA ABI
构建，完整命令统一维护在[安装指南](../../../../docs/usage/fast-cp-installation.md)。

## 验证与结果

在独占 8 GPU H20 节点运行：

```bash
bash chitu_diffusion/parallel/cp/fast/test/run_wrapper.sh
```

依赖统一安装在 `.deps/fast-cp/`，测试脚本自动加载 `tools/install/fast_cp_env.sh`。
可通过 `FAST_CP_DEPS_ROOT` 指定其他安装位置，详见安装指南。

脚本执行 pytest 和 24 组分布式检查，覆盖 Wan 1.3B S32760/H12、Wan 14B S75600/H40、
Qwen/ZImage image4096+text512 的 CP2/4/8，导出 JSON/CSV。所有 workload 使用
BF16/B1/D128、随机权重；计时覆盖完整 attention block，未测完整生成任务。

详细复现见 [test/README.md](test/README.md)，跨框架报告见
[Fast CP 功能文档](../../../../docs/features/fast-cp.md)，稳定 CSV 与雷达图见
[docs/assets/fast_cp](../../../../docs/assets/fast_cp/README.md)。目录重构的回归结果
与此前正式 benchmark 分开记录。

## 致谢

Fast Ulysses 后端基于 [triple-mu/fast-ulysses](https://github.com/triple-mu/fast-ulysses)，
上游许可证为 Apache-2.0。本项目增加 logical subgroup、AGKV、CE projection
overlap 和模型执行接口；CuTe FlashAttention 与 cuDNN 提供 attention 后端。
