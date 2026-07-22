# Image Decoder Executor 接入契约

本文面向新增 image decoder 的模型开发人员。宿主和 EPAC 已负责 stage 生命周期、
请求排队、static-DP/static-CP/elastic 调度、pulse、lane 状态交换、timeline 和异步
CPU 后处理。模型接入不应复制这些逻辑。

## 交付物

每个模型家族只需要提供两个对象：

1. `ImageDecoderExecutorFactory`：在每个 stage rank 上根据
   `ExecutorBuildContext(world, pool)` 创建模型、通信组和 executor。
2. `ImageDecoderExecutor`：定义请求编解码、request-local denoise state、单步 DiT、
   state transfer、VAE decode 和 CPU postprocess。

Z-Image 的参考实现位于 `models/zimage/executor.py`。宿主入口只依赖
`EmbeddedDiffusionRuntime.create(..., executor_factory=...)`，不得 import 模型私有类型。

## 生命周期

```text
all stage ranks
  StageWorldSpec
       |
  ExecutorFactory.build()       load model + create canonical lane groups
       |
  EmbeddedDiffusionRuntime.start()
       |
  warmup(resolutions, 5 steps)  populate measured cost model
       |
leader: submit / poll / cancel
followers: worker loop
       |
  stop(graceful=True)           drain + executor.close()
```

当前首版支持两种 world ownership：executor 自己初始化 stage WORLD，或复用已经初始化且
与 stage 完全一致的 `dist.group.WORLD`。任意 subgroup 注入尚未支持，会显式失败。

## Executor 方法

| 方法 | 模型侧责任 |
| --- | --- |
| `normalize_request` / `validate_request` | 接受模型专用 request，补默认值并拒绝未 warmup 的 shape |
| `serialize_request` / `deserialize_request` | 只传输可序列化字段，不传 CUDA tensor |
| `request_profile` | 在 prepare 前提供总步数、序列长度、batch/condition 属性 |
| `request_deadline_ms` | 返回逐请求 SLO；返回 `None` 时使用 pool 的 `default_deadline_ms` |
| `prepare_request` | 运行 encoder/tokenizer，创建独立 scheduler、timesteps、latent 和 step cursor |
| `profile` | 从 state 返回当前 `completed_steps`，结果必须与真实 cursor 一致 |
| `denoise_step` | 只推进一步；按照 `lane_ranks` 执行当前 lane 的 CP，不自行调度下一步 |
| `synchronize_state` | lane 切换后把所有 rank 的 cursor 对齐到指定 step |
| `export_state` | 返回继续推理所需的 canonical tensor；不能依赖旧 lane 的局部分片 |
| `finalize_gpu` | VAE decode 并将结果转为 CPU tensor，不做 PNG 编码 |
| `postprocess` | CPU image processor；由 runtime 线程池异步调用 |
| `abort_request` | 释放该 request 的模型私有资源，必须幂等 |
| `close` | 按 ownership 释放 lane group、model 和 process group，必须可重复调用 |

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
- encoder 和 VAE 可以先保持原生 Diffusers 实现；性能优化不应改变 executor contract。

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
3. 启动 warmup 覆盖所有声明的 `(resolution, lane_width)`，生成 measured cost table。
4. elastic trace 至少覆盖 cp4 -> cp1、cp1 -> cp2 和 request completion 后 lane pull。
5. static-DP/static-CP/elastic 均无 NCCL/Gloo error，所有请求 exactly-once completion。
6. graceful stop 后无残留 worker，timeline 可绘制，HTTP 和 embedded 两个入口结果一致。

当前 Z-Image 已完成上述 runtime 路径的 4x H20 Slurm smoke 和 serve benchmark。具体命令、
指标及结果适用范围记录在顶层 `README.md`。
