# chitu_diffusers examples

## FLUX.2-klein

FLUX.2-klein currently retains the official Diffusers single-GPU path and a
fixed full-world context-parallel baseline. The four-step distilled schedule is
not wired into EPAC elastic switching.

```bash
bash script/srun_direct.sh 1 1 chitu_diffusers/examples/flux2_klein_native.py \
  --model-path /path/to/flux2-klein \
  --output outputs/flux2_klein/native_512_4step.png

bash script/srun_direct.sh 1 4 chitu_diffusers/examples/flux2_klein_cp.py \
  --model-path /path/to/flux2-klein \
  --output outputs/flux2_klein/fixed_cp_4gpu_512_4step.png
```

固定 CP 仅支持 distilled checkpoint、AGKV 和 batch size 1。当前限制与验证
结果见 `chitu_diffusers/models/flux2_klein/README.md`。

## Qwen-Image

使用本地 Qwen-Image Diffusers 权重运行原生 50 步基线：

```bash
export CHITU_PROJECT_ROOT="$PWD"
bash script/srun_direct.sh 1 1 chitu_diffusers/examples/qwen_image_native.py \
  --model-path /path/to/Qwen-Image \
  --local-files-only --steps 50 \
  --output outputs/chitu-diffusers/qwen_image/native_512_50.png
```

四卡 EPAC 默认使用 CFP2 x CP2，并执行三次 startup warmup：

```bash
bash script/srun_direct.sh 1 4 chitu_diffusers/examples/qwen_image_embedded.py \
  --model-path /path/to/Qwen-Image \
  --resolution 512 --steps 50 --warmup-steps 3 \
  --policy elastic --record-timeline \
  --output outputs/chitu-diffusers/qwen_image/epac_4gpu_512_50.png
```

不启动 embedded worker 的同步 full-world API 可使用
`chitu_diffusers/examples/qwen_image_epac.py`。

增加 `--secondary-steps 25` 会同时提交一个短请求，形成两个 width-2 lane，
并在短请求完成后验证长请求扩容到 width 4 的 state migration。使用
`--no-cfg-parallel` 可切换到整 lane 串行 CFG 对照，`--no-parallel-vae` 可关闭
并行 VAE。

## FLUX.1-dev

以下命令通过 Slurm wrapper 使用共享 ChituDiffusion 环境：

```bash
export CHITU_PROJECT_ROOT="$PWD"
export CHITU_PYTHON_BIN=/home/chenyy/WORK/cyy/ChituDiffusion/.venv/bin/python
export FLUX1_MODEL_PATH=/home/chenyy/WORK/models/Flux-1
```

原生 Diffusers 基线：

```bash
bash script/srun_direct.sh 1 1 chitu_diffusers/examples/flux1_native.py \
  --model-path "$FLUX1_MODEL_PATH" \
  --output outputs/flux1-diffusers/native.png
```

静态 full-world AGKV lane：

```bash
bash script/srun_direct.sh 1 4 chitu_diffusers/examples/flux1_epac.py \
  --model-path "$FLUX1_MODEL_PATH" \
  --output outputs/flux1-diffusers/cp4.png
```

动态 lane 切换正确性检查：

```bash
bash script/srun_direct.sh 1 4 chitu_diffusers/examples/flux1_dynamic.py \
  --model-path "$FLUX1_MODEL_PATH" \
  --lane-widths 1,4,2 \
  --output outputs/flux1-diffusers/dynamic.png
```

embedded runtime：

```bash
bash script/srun_direct.sh 1 4 chitu_diffusers/examples/flux1_embedded.py \
  --model-path "$FLUX1_MODEL_PATH" \
  --resolution 512 \
  --steps 2 \
  --output outputs/flux1-diffusers/embedded.png
```

embedded runtime 默认按 active lane 做 parallel VAE。使用
`--no-parallel-vae` 可切回 leader-only decode；`--vae-parallel-halo N` 可调整
Flux AutoencoderKL 的 latent halo，默认值为 8。

当前 FLUX.1 路径已验证 AGKV 和 parallel VAE，尚未验收 USP、true CFG、IP-Adapter、
ControlNet、LoRA、Schnell 和 FlexCache。

## Z-Image

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

多卡 CFG 默认采用 CFP2 优先布局：二卡为 CFP2 x CP1，四卡为 CFP2 x CP2。
传入 `--no-cfg-parallel` 可运行整 lane 纯 CP 对照。

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
