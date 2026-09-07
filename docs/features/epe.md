# EPE

EPE 是常驻服务的弹性并行执行引擎。它把一组 GPU 划分为多个 CP lane，根据请求规模、
实测代价和 deadline 决定每个请求占用哪些 rank，并在固定的 pulse 边界重新规划。

## 基本原理

一个 lane 是互不重叠的 rank 集合。例如四张 GPU 可以组成一个 CP4 lane、两个 CP2
lane，或四个单卡 lane。EPE 支持三种策略：

- `elastic` 在配置允许的 lane 宽度之间调整请求。
- `static_cp` 始终使用整个 world。
- `static_dp` 为每个 rank 分配独立请求。

服务启动时会测量不同序列长度、lane 宽度、batch size 和 CFG 条件数下的 DiT step
时间，也会测量 VAE、D2H 和 rank 间状态迁移成本。planner 使用这些数据预测请求完成
时间。

每个 pulse 包含四步：

1. planner 根据 deadline、预计剩余时间和等待时长排列请求。
2. planner 搜索不重叠的 lane 布局，优先减少 SLO 违约和饥饿请求。
3. worker 在 lease 时间内执行计划步数，空闲 lane 可以继续当前请求或拉取新请求。
4. 所有 lane 报告进度后，planner 生成下一个布局。

## 优化方案

### 基于实测代价调度

EPE 不使用固定的 CP 加速比。启动 warmup 建立 cost table，运行期间再用同一 workload
key 的观测值做 EWMA 校正。新分辨率不会直接复用其他分辨率的校正因子。

### 有界布局搜索

planner 同时考虑运行中请求和等待队列，在有限候选布局中比较 SLO 违约数、逾期程度、
饥饿请求、有效吞吐量、迁移成本和空闲 rank。候选数有上限，避免请求增加后规划时间
失控。

### 只迁移必要状态

lane 改变时，旧 lane leader 只向新加入的 rank 发送模型继续去噪所需的 canonical
state。当前通用 executor 迁移 `latents` 和 `step_index`，随后新 lane 同步 scheduler
状态。

### GPU 与 CPU 后处理解耦

VAE decode 完成后，leader 把 CPU tensor 交给线程池执行 PIL 后处理和编码。GPU lane
可以先释放并处理下一个请求。

## 验证结果

仓库的 CPU 测试覆盖 lane 约束、deadline guard、cost 校正、布局选择、pulse
rendezvous、lane pull、状态迁移和异步后处理。当前没有可发布的 EPE 端到端吞吐或 SLO
实测，因此文档不提供性能数字。

EPE 目前不保证任意 rank 故障后的恢复，也不取消已经进入去噪阶段的请求。FlexCache
不接入 EPE 服务。

lane 迁移要求模型能导出可恢复的 canonical state。带 lane 常驻 KV cache 或固定
expert 归属的模型无法满足这一点，只能配置 `static_cp`，例如 Hunyuan Image 3 的
TP×CFG×CP×EP stage。这类 backend 的 `export_state` 会直接报错，而不是静默产生错误
结果。

## 使用示例

四卡配置见 `examples/stage-zimage.yaml`。启动时，torchrun 的进程数必须与配置中的
GPU 数量一致：

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  serve --stage-config examples/stage-zimage.yaml
```

配置中的关键字段如下：

```yaml
name: image
process: image
factory: zimage
gpu: [0, 1, 2, 3]
parallelism:
  cp:
    world_size: 4
  vae:
    enabled: true
  scheduler:
    policy: elastic
    allowed_lane_widths: [1, 2, 4]
    warmup_resolutions: [512]
    warmup_steps: 3
factory_args:
  model_path: /path/to/Z-Image
  default_width: 512
  default_height: 512
service:
  host: 0.0.0.0
  port: 18200
```

## 并行配置

`parallelism` 分为四段。`cp`、`vae` 和 `scheduler` 对所有模型含义相同，`model` 只
接受当前 factory 真正实现的轴：

- `cp`：交给 CP/CFG 调度的 rank 空间，包含 `world_size`、`attention_mode`
  (`agkv` 或 `ulysses`)、`ulysses_degree` 以及 `ulysses_transport`、
  `agkv_transport`(`auto`/`nccl`/`fast`)。约束是
  `cp.world_size × model.tensor_parallel_degree = GPU 数`。
- `vae`：终端解码的并行度，包含 `enabled`、`degree` 和 `halo`。`degree` 不填时
  解码跟随产出 latent 的 lane；`degree > 1` 表示一个跨 lane 的固定解码组，只在
  `scheduler.policy: static_cp` 下允许，因为此时整个 stage 同一时刻只有一条 lane。
- `scheduler`：原 `chitu_pool` 的调度参数，包括 `policy`、`allowed_lane_widths`、
  pulse 与 deadline 控制以及 warmup 设置。
- `model`：模型专属轴。Z-Image、Qwen-Image、Wan 只有 `cfg_parallel_degree`；
  MiniMax-H3 有 `tensor_parallel_degree`；Hunyuan Image 3 有
  `tensor_parallel_degree`、`cfg_parallel_degree` 和 `expert_parallel_degree`；
  FLUX.1 没有模型专属轴。配置未实现的轴会直接报错，而不是被忽略。

旧字段 `parallelism.sp`、`parallelism.tp`、`parallelism.chitu_pool`，以及
`factory_args` 中的 `attention_mode`、`ulysses_degree`、`cfg_parallel`、
`expert_parallel_degree`、`parallel_vae`、`vae_parallel_degree`、
`vae_parallel_halo` 不再接受，加载配置时会报出对应的新路径。

已有宿主管理进程和请求入口时，可以使用 `EmbeddedDiffusionRuntime`，不启动 HTTP
server。完整代码见 `examples/epe_embedded.py`。
