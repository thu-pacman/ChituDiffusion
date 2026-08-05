# Image Decoder Executor 接入契约

本文面向新增 image decoder 的模型开发人员。宿主和 EPAC 已负责 stage 生命周期、
请求排队、static-DP/static-CP/elastic 调度、pulse、lane 状态交换、timeline 和异步
CPU 后处理。模型接入不应复制这些逻辑。

## 交付物

每个模型家族只需要提供两个轻量对象：

1. `DiffusionBackendFactory`：在每个 stage rank 上根据
   `ExecutorBuildContext(world, pool)` 创建模型、通信组和 executor。
2. `DiffusionBackend`：在共享基类上实现请求编解码、warmup、request-local state
   准备、condition/state-byte 计算和模型特有 VAE 参数。

不要从零实现完整 executor。模型 executor 应继承
`epac.model_executor.DiffusersBackend`，只提供 request codec、warmup、
prompt/state 准备、condition/state-byte 计算和模型特有 decode 参数；stage world 校验、
request profile、state transfer、单步调用、postprocess 和 close 生命周期由共享基类负责。
factory 使用 `build_stage_parallel_context()` 和 `scheduling_options_from_pool()`。

Z-Image 与 Flux.1 的参考实现分别位于 `models/zimage/executor.py` 和
`models/flux1/executor.py`。宿主入口只依赖
`EmbeddedDiffusionRuntime.create(..., executor_factory=...)`，不得 import 模型私有类型。

调度成本、在线校准和 planner facade 统一使用
`epac.model_scheduling.EpeSchedulingModule`。模型只保留产生 warmup 行的逻辑。离线
Diffusers 风格 API 继承 `epac.api.DiffusersEPACPipeline`，模型侧只声明 pipeline、
backend 和 typed request；不要复制 full-world generate 或 EPE service 驱动。

## 生命周期

```text
all stage ranks
  StageWorldSpec
       |
  ExecutorFactory.build()       load model + create canonical lane groups
       |
  EmbeddedDiffusionRuntime.start()
       |
  warmup(resolutions, 5 steps)  measure DiT + terminal + transfer costs
       |
leader: submit / poll / cancel
followers: worker loop
       |
  stop(graceful=True)           drain + executor.close()
```

当前首版支持两种 world ownership：executor 自己初始化 stage WORLD，或复用已经初始化且
与 stage 完全一致的 `dist.group.WORLD`。任意 subgroup 注入尚未支持，会显式失败。

## Executor 方法

| 方法 | 默认所有者 / 模型侧责任 |
| --- | --- |
| `normalize_request` / `serialize_request` | 模型实现 typed request codec；只传可序列化字段 |
| `validate_request` / `deserialize_request` | 共享基类校验 warmup shape，并复用 normalize |
| `request_profile` / `profile` | 共享基类组装 profile；模型只返回 condition 与预估 state bytes |
| `request_id` / `request_deadline_ms` | 共享基类读取 typed request 字段 |
| `prepare_request` / `warmup` | 模型运行 encoder、创建 state，并产生 DiT/terminal 实测行 |
| `denoise_step` / `synchronize_state` | 共享基类调用 pipeline 的单步和 cursor 同步接口 |
| `export_state` | 共享默认迁移 canonical latent；额外可变状态由模型覆写 |
| `finalize_gpu` | 共享基类管理 lane context；模型只提供 decode kwargs，必要时覆写 |
| `postprocess` / `abort_request` / `close` | 共享基类提供 Diffusers 默认生命周期，模型资源特殊时覆写 |

## DiT 与并行约束

- 启动时按 `allowed_lane_widths` 确定性地创建所有 canonical lane group；所有 rank 必须
  使用相同创建顺序，不能在请求中途临时 `new_group`。
- `denoise_step(..., lane_ranks=...)` 是唯一的 denoise 调度单元。LLaDA2 可以在 DiT
  内部切分序列，调用 Ulysses/AGKV attention，再 all-gather 序列。
- 不要加入 CFG parallel。condition 数量只作为 cost profile 属性参与测量。
- 每次 step 后 state cursor 只能递增一次。pulse 的 K 步由 runtime 循环调用单步接口，
  executor 不应自行执行额外 K 步。
- `export_state` 必须足以让 cp1/cp2/cp4 之间切换。若模型包含除 latent 外的可变状态，
  需要一并放入 `TransferBundle.tensors` 或可序列化 metadata。
- encoder 和 VAE module 保持原生 Diffusers 实现。宽 lane 可复用
  `parallel_tiled_vae_decode`，但所有 lane rank 必须以相同顺序进入 collective；runtime
  只在 leader 执行 D2H，CPU postprocess/PNG 继续异步。

## LLaDA2 团队需要确定的内容

由于当前仓库拿不到新增 image decoder 源码，以下内容必须由模型团队根据真实实现填写，
不能从旧版本推断：

- 对外 request schema，以及 text/VQ/SigVQ 条件如何构造；
- DiT state 中跨 step、跨 lane 必须保留的 tensor；
- image token 序列长度与 512/1024/2048 shape 的准确映射；
- 模型原生 scheduler 的 cursor/update 语义；
- decoder 输出进入 VAE 前的缩放、归一化和 dtype；
- Ulysses/AGKV attention 对 mask、padding 和 head divisibility 的限制。

## 最低验收

1. 单卡 `static_dp` 与原始模型在相同 seed/request 下完成 shape 和数值合理性检查。
2. 四卡 `static_cp` 完成 512/1024/2048、完整 denoise steps，并输出可解码图片。
3. 启动 warmup 覆盖所有声明的 `(resolution, lane_width)`，测量 DiT step、VAE/D2H
   terminal，并对所有 rank pair 测量 request state transfer。
4. elastic trace 至少覆盖 cp4 -> cp1、cp1 -> cp2 和 request completion 后 lane pull。
5. static-DP/static-CP/elastic 均无 NCCL/Gloo error，所有请求 exactly-once completion。
6. graceful stop 后无残留 worker，timeline 可绘制，HTTP 和 embedded 两个入口结果一致。

当前 Z-Image 已完成上述 runtime 路径的 4x H20 Slurm smoke 和 serve benchmark。具体命令、
指标及结果适用范围记录在顶层 `README.md`。
