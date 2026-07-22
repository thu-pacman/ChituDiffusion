# Z-Image examples

先设置模型路径：

```bash
export ZIMAGE_MODEL_PATH=/path/to/Z-Image
```

## 原生 Diffusers

只导入官方 `diffusers.ZImagePipeline`，不经过 EPAC adapter 或 runtime：

```bash
../ChituDiffusion/.venv/bin/python \
  -m chitu_diffusers.examples.zimage_native \
  --local-files-only \
  --steps 50
```

## EPAC Static Generate

`EPACPipeline.generate()` 使用所有 torchrun ranks 组成的 full-world lane。该路径属于简单的序列并行，不执行
warmup，也不运行 elastic layout planner。

```bash
../ChituDiffusion/.venv/bin/torchrun \
  --standalone \
  --nproc-per-node=4 \
  --module chitu_diffusers.examples.zimage_epe \
  --local-files-only \
  --steps 50
```

单卡可以将 `torchrun` 替换为普通 Python。默认输出为
`outputs/chitu-diffusers/zimage_epe.png`。

## Embedded Runtime（无 HTTP）

该示例直接使用 `StageWorldSpec + EmbeddedDiffusionRuntime +
ZImageExecutorFactory`。rank 0 submit/poll 并保存 raw PIL completion，followers 只运行
EPAC worker：

```bash
../ChituDiffusion/.venv/bin/torchrun \
  --standalone \
  --nproc-per-node=4 \
  --module chitu_diffusers.examples.zimage_embedded \
  --model-path "$ZIMAGE_MODEL_PATH" \
  --resolution 512 \
  --steps 12
```

## EPAC Elastic Serve

所有 torchrun ranks 执行同一个 `pipeline.serve()`。启动阶段会创建 lane、对声明的
分辨率执行 random-tensor warmup，并初始化 measured cost model。rank 0 随后开放 HTTP，
其他 rank 进入 follower control loop。

```bash
../ChituDiffusion/.venv/bin/torchrun \
  --standalone \
  --nproc-per-node=4 \
  --module chitu_diffusers.examples.zimage_serve \
  --model-path "$ZIMAGE_MODEL_PATH" \
  --local-files-only \
  --attention-mode agkv \
  --warmup-resolutions 512 1024 2048 \
  --warmup-steps 5 \
  --pulse-steps 5 \
  --default-deadline-ms 30000 \
  --port 18200
```

`--default-deadline-ms` 是从到达到完成的默认端到端 SLO。请求 JSON 中的
`deadline_ms` 可以逐请求覆盖它；不传两者则按无 deadline 请求调度。

安装 `uv sync --extra usp` 后，可将 attention 参数替换为
`--attention-mode usp --ulysses-degree 2`，在四卡 lane 使用 u2r2。

发送专用请求：

```bash
curl -X POST http://NODE:18200/v1/image-decode \
  -H 'content-type: application/json' \
  -d '{
    "request_id": "req-001",
    "prompt": "a red cube on a white table",
    "width": 1024,
    "height": 1024,
    "num_steps": 50,
    "guidance_scale": 5.0,
    "seed": 7,
    "deadline_ms": 120000
  }'
```

缓存字段已经在 `CacheConfig` 中保留，但当前只有 `strategy="none"` 可执行；MeanCache 和
TeaCache 会 fail fast，不会静默降级。
