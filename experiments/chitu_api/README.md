# Chitu EPE 图像推理服务实验接口

本目录提供一版可独立启动的 Z-Image EPE 图像推理服务。Pipeline 和 DiT
直接继承 Diffusers Z-Image 实现，不再调用 Chitu 的 `DiffusionBackend`、
`DiffusionTaskPool`、`Generator`、`chitu_init()` 或 stage-level runtime。

服务采用一张 GPU 对应一个 OS process 的形式。所有进程共同运行 EPE
lockstep control loop，只有 stage rank 0 对外启动 HTTP 服务并返回图片。

## 当前架构

```text
torchrun
  -> experiments.chitu_api.serve
  -> StageServiceConfig
  -> EpeZImageServiceRuntime
       |- leader-only FastAPI admission/status/image API
       |- request queue + denoise-phase lane planner
       `- EpeZImagePipeline (Diffusers-compatible)
            |- EpeParallelContext
            |- EpeZImageTransformer2DModel
            |    `- ZImageAgkvAttnProcessor
            |- native tokenizer + text encoder
            |- request-local native scheduler
            `- native VAE + image processor
```

Pipeline 对外仍保留 Diffusers 风格的 `pipeline(...)`。服务内部将一次生成拆为：

1. `prepare_request()`：编码 prompt、生成初始 latent，并创建请求独立 scheduler。
2. `denoise_step()`：读取本 phase 的 active lane，执行一个 denoise step。
3. `synchronize_state()`：在 phase 边界同步 canonical latent 和 scheduler cursor。
4. `finalize_request()`：rank 0 使用原生 VAE 生成最终图片。

rank 0 负责 HTTP admission、phase planning 和最终 PNG；所有 rank 都加载 DiT，
当前也都会加载并运行 text encoder。一次 phase 内每个 rank 最多属于一条 lane，
不同 lane 可以并行处理不同请求。

### 通信组初始化

通信组在服务启动阶段、模型权重加载之前创建：

```text
EpeZImageServiceRuntime.from_config()
  -> EpeZImagePipeline.from_pretrained()
       -> EpeParallelContext.from_torchrun()
            -> set CUDA device
            -> dist.init_process_group(NCCL)
            -> dist.new_group() for configured canonical lanes
       -> load DiT / text encoder / VAE
  -> pipeline.to(cuda)
  -> world barrier
  -> start rank-0 HTTP server
  -> CHITU_API_READY
```

以 4 卡、`allowed_lane_widths: [1,2,4]` 为例：full-world `cp4` 复用
`dist.group.WORLD`，额外创建 `[0,1]` 和 `[2,3]` 两个 `cp2` group；`cp1`
不需要实际 process group。运行时 `activate()` 只切换 instance-local topology，
不会调用 `new_group()`。

当前 ready 前只保证 process group 已创建，没有逐个 subgroup 主动执行 warmup
collective；如果部署环境要求消除 NCCL 首次 collective 的 lazy-init 抖动，还需增加
subgroup warmup。

## 目录结构

| 文件 | 作用 |
| --- | --- |
| `stage_config.example.yaml` | 面向甲方的 StageConfig 示例 |
| `serve.py` | torchrun 和 Chitu 内部 Slurm 的兼容入口 |
| `zimage/pipeline.py` | Diffusers-compatible Z-Image pipeline |
| `zimage/transformer.py` | 支持 active lane 的 Z-Image DiT |
| `zimage/attention.py` | 简化 AGKV context parallel attention |
| `zimage/parallel.py` | instance-local topology 和通信组 registry |
| `zimage/runtime.py` | instance-local 请求池和 denoise-phase EPE planner |
| `zimage/serve.py` | standalone torchrun 服务入口 |
| `service.py` | leader-only HTTP API |
| `protocol.py` | 请求、状态和健康检查数据结构 |
| `benchmark_client.py` | 回放 Chitu Poisson trace 的 HTTP 压测器 |
| `benchmark_static_cp.yaml` | 4 卡固定 `cp4`、串行请求基线 |
| `benchmark_epe.yaml` | 4 卡 `[1,2,4]` denoise-phase EPE 配置 |
| `zimage_service_slurm.yaml` | Chitu 内部 Slurm 验证配置 |
| `result.md` | 真实 GPU smoke test 和当前边界 |

## 甲方启动方式

### 1. 准备 StageConfig

从 `stage_config.example.yaml` 创建实际配置，至少修改以下字段：

```yaml
name: image_decode
gpu: [4, 5, 6, 7]

parallelism:
  sp: 4
  chitu_pool:
    policy: slo_elastic
    allowed_lane_widths: [1, 2, 4]
    switch_allowed_until_step: 20
    phase_max_steps: 1

factory_args:
  model_path: /path/to/Z-Image

service:
  host: 0.0.0.0
  port: 18080
  advertise_host: 10.0.0.8
```

`advertise_host` 必须是请求方能够访问的计算节点地址，不能在跨节点访问时配置为
`127.0.0.1`。

### 2. 使用 torchrun 启动

调度器分配物理 GPU `[4,5,6,7]` 后，启动命令为：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
../ChituDiffusion/.venv/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  --module experiments.chitu_api.serve \
  --stage-config experiments/chitu_api/stage_config.yaml
```

如果调度系统已经设置了 `CUDA_VISIBLE_DEVICES`，不要再次覆盖：

```bash
../ChituDiffusion/.venv/bin/torchrun \
  --standalone \
  --nproc-per-node="$GPU_COUNT" \
  --module experiments.chitu_api.serve \
  --stage-config /path/to/stage_config.yaml
```

启动时必须满足：

```text
--nproc-per-node
  == CUDA_VISIBLE_DEVICES 中的 GPU 数量
  == parallelism.sp
  == StageConfig.gpu 的元素数量
```

物理 GPU 会被 `CUDA_VISIBLE_DEVICES` 映射为进程内设备 `0..N-1`。Chitu
内部只使用 stage-local `RANK/LOCAL_RANK`，不会把物理 GPU ID 当作 distributed
rank。

### 3. 等待服务 ready

模型加载和通信组预创建全部完成后，rank 0 输出：

```text
CHITU_API_READY {
  "stage": "image_decode",
  "endpoint": "http://10.0.0.8:18080",
  "physical_gpu_ids": [4, 5, 6, 7],
  "world_size": 4,
  "allowed_lane_widths": [1, 2, 4],
  "runtime": "standalone_diffusers"
}
```

甲方只能在收到 `CHITU_API_READY` 后开始发送请求。follower ranks 不开放端口，
也不返回业务结果。

## 请求接口

### 提交请求

```bash
curl -sS -X POST http://10.0.0.8:18080/v1/image-decode \
  -H 'content-type: application/json' \
  -d '{
    "request_id": "demo-1",
    "prompt": "a red panda reading a book",
    "width": 1024,
    "height": 1024,
    "seed": 7,
    "num_steps": 20,
    "deadline_ms": 5000
  }'
```

服务立即返回 HTTP 202，不会等待推理完成：

```json
{
  "request_id": "demo-1",
  "status": "accepted",
  "status_url": "/v1/image-decode/demo-1",
  "image_url": "/v1/image-decode/demo-1/image"
}
```

### 查询状态并获取图片

```bash
curl -sS http://10.0.0.8:18080/v1/image-decode/demo-1
curl -o demo-1.png \
  http://10.0.0.8:18080/v1/image-decode/demo-1/image
```

请求状态包括 `pending`、`running`、`completed`、`cancelled` 和 `failed`。
只有 `completed` 状态可以下载 PNG。

### 取消与健康检查

```bash
curl -X DELETE http://10.0.0.8:18080/v1/image-decode/demo-1
curl http://10.0.0.8:18080/health
```

当前原型只保证尚未首次调度的 pending request 可以安全取消。

## EPE 行为

- `width=1` 表示一个请求使用一张 GPU，即 request-level DP lane。
- `width>1` 表示一个请求使用多张 GPU 执行 CP/SP。
- `{1,2,4}` 对应的通信组在服务 ready 前全部创建完成。
- Pipeline 将服务执行拆为 `prepare_request()`、`denoise_step()` 和
  `finalize_request()`；每个请求持有独立 scheduler、latent 和 step cursor。
- planner 在 denoise phase 边界选择布局：低并发使用宽 CP lane，高并发将 world
  划分为多个 request-level DP lanes。
- 每个 phase 最多执行 `phase_max_steps` 个 step。`switch_allowed_until_step` 之前
  可以改变 lane width，达到边界后固定使用最后一次 placement。
- phase 结束后，lane leader 将 canonical latent 同步到整个 stage world，并同步
  每个 rank 的 request-local scheduler cursor，因此下一 phase 可以换到任意预建 lane。
- DiT 将 image-token sequence 切分一次，attention 使用 local Q 和 all-gathered
  image K/V；FFN/Norm/Residual 保持 sequence-sharded，最终投影后才 all-gather
  patch output。
- tokenizer、text encoder、scheduler、VAE 和 image processor 都使用 Diffusers
  原生实现，不做并行改写。
- CFG 分支在 lane 内顺序执行，不启用 CFG parallel。
- planner 只激活启动阶段已经创建的 group，运行时不调用 `new_group()`。

2-GPU 实测布局示例：

```text
低并发：hot-first:cp2@[0, 1] step=0+1
并发到达：hot-first:cp1@[0] step=4+1 switch,
          hot-second:cp1@[1] step=0+1
并发下降：hot-second:cp2@[0, 1] step=2+1 switch
```

## 直接运行 Pipeline

不启动 HTTP 服务时，可以直接验证 Diffusers 风格 pipeline：

```bash
../ChituDiffusion/.venv/bin/torchrun \
  --standalone \
  --nproc-per-node=2 \
  --module experiments.chitu_api.zimage.smoke \
  --model-path /path/to/Z-Image \
  --output outputs/zimage.png \
  --steps 20
```

## Chitu 内部 Slurm 验证

`zimage_service_slurm.yaml` 用于 ChituDiffusion 开发环境内申请 GPU 和验证服务，
不是甲方正式启动接口：

```bash
CHITU_PYTHON_BIN=../ChituDiffusion/.venv/bin/python \
  ../ChituDiffusion/.venv/bin/python -m chitu_diffusion.cli run \
  experiments/chitu_api/zimage_service_slurm.yaml
```

Slurm 模式下物理 GPU placement 由 Slurm 所有，配置只表达 GPU 数量。不要在
Slurm 已经重映射设备后再次按物理 GPU ID 设置 `CUDA_VISIBLE_DEVICES`。

## 4 卡 Poisson 压测

### 实验口径

- 硬件：单节点 4 x H20；两臂由 Slurm 分配到同型号的不同节点。
- 模型：`/home/chenyy/WORK/models/Z-Image`。
- Trace：ChituDiffusion `gen_client_trace.py` 生成的固定 seed Poisson 到达。
- 负载：12 个 `1024 x 1024` 请求，3 req/s，trace 持续 4.41s。
- 推理：每请求 50 denoise steps、单图片。
- warmup：每臂正式计时前运行一个同 shape 的 2-step 请求。
- static CP：一次只接纳一个请求，固定 `cp4`，50 steps 为一个 phase。
- EPE：最多 4 个 inflight request，lane widths `[1,2,4]`，每 step 一个 phase。

生成完全相同的 trace：

```bash
../ChituDiffusion/.venv/bin/python \
  experiments/cp_dp_hot_switch/gen_client_trace.py \
  --rate 3.0 \
  --num-requests 12 \
  --sizes 1024x1024 \
  --steps 50 \
  --base-seed 42 \
  --rng-seed 0 \
  --out outputs/chitu-api-benchmark/poisson_1024_r3n12_50step.json
```

通过 Slurm 启动其中一臂；将配置文件替换为 `benchmark_epe.yaml` 即可运行 EPE：

```bash
srun \
  --partition=debug \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:4 \
  --cpus-per-task=96 \
  --mem=400G \
  --job-name=zimage-static-cp-bench \
  bash -lc '
    cd /home/chenyy/WORK/cyy/ChituDiffusion-hotswitch
    /home/chenyy/WORK/cyy/ChituDiffusion/.venv/bin/torchrun \
      --standalone \
      --nnodes=1 \
      --nproc-per-node=4 \
      --module experiments.chitu_api.serve \
      --stage-config experiments/chitu_api/benchmark_static_cp.yaml
  '
```

收到 `CHITU_API_READY` 后，从可访问计算节点端口的客户端回放 trace：

```bash
../ChituDiffusion/.venv/bin/python \
  -m experiments.chitu_api.benchmark_client \
  --endpoint http://COMPUTE_NODE:18200 \
  --trace outputs/chitu-api-benchmark/poisson_1024_r3n12_50step.json \
  --output outputs/chitu-api-benchmark/static_cp_metrics.json \
  --warmup-steps 2 \
  --timeout-s 3600
```

### 关键数据

两臂均完成 `12/12` 请求，日志中没有 failed、OOM、CUDA 或 NCCL error。

| 指标 | static CP | EPE | EPE 变化 |
| --- | ---: | ---: | ---: |
| makespan | 199.65s | 158.44s | -20.6% |
| throughput | 0.0601 req/s | 0.0757 req/s | +26.0% |
| mean latency | 106.23s | 102.75s | -3.3% |
| p50 latency | 106.41s | 102.93s | -3.3% |
| p95 latency | 186.15s | 153.87s | -17.3% |
| max latency | 195.24s | 154.03s | -21.1% |
| mean queue delay | 89.59s | 50.71s | -43.4% |
| p95 queue delay | 169.56s | 102.09s | -39.8% |

static CP 的首请求延迟为 `16.53s`；EPE 首请求在其他请求到达后从 `cp4`
收窄为 `cp1`，延迟为 `50.27s`。因此当前 EPE 在饱和 Poisson 负载下提高了吞吐、
queue delay 和 tail latency，但显著牺牲 isolated/early request latency。

本次 EPE 正式推理共有 600 个 request-step lane assignments：598 个使用 `cp1`，
2 个使用 `cp4`；共执行 152 个正式 phase，记录到 21 次 placement switch。当前
planner 本质上按 inflight 数量选最大可并发布局，还没有真正使用 H20 cost model、
deadline 或 SLO 做优化，因此这些数据是 runtime/并行结构基线，不是最终调度策略上限。

原始指标位于 `outputs/chitu-api-benchmark/{static_cp,epe}_metrics.json`；service
layout 日志位于同目录的 `*_service.log`。这些是可重生成产物，不纳入源码版本控制。

## 当前边界

- 当前仅支持单机 stage world 和每个请求生成一张图片。
- 当前请求仍是 Z-Image text-to-image prompt，不是 LLaDA2-Uni VQ token。
- 当前 CP backend 是 AGKV；Ulysses all-to-all backend 尚未接入。
- 当前 EPE planner 是基于 inflight 数量的简单布局规则，不是完整 cost/SLO planner。
- phase 边界目前把 canonical latent 复制到所有 rank，尚未实现只向目标 lane
  迁移的低流量方案。
- 简化 AGKV 对内部 padding 的序列是数值近似而非 bitwise 等价，正式模型仍需做
  固定 prompt 集的质量回归。
- running request 暂不支持 request-scoped cancel。
- completion 元数据和 PNG 暂存在 leader 内存中，长期服务还需要 TTL/容量淘汰。
- 原生 text encoder 和 VAE 当前在每个 rank 加载并执行，尚未优化为 leader-only。
- 后续 LLaDA2-Uni executor 需要替换 prompt encoder，接入 VQ-token conditioning。
