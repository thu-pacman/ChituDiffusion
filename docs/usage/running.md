# 运行

`chitu generate` 执行单次生成，`chitu serve` 启动 EPE 服务。CLI 支持 `zimage`、
`flux1`、`flux2-klein`、`qwen-image` 和 `wan`。

## 单卡生成

```bash
chitu generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --prompt "a red cube on a white table" \
  --output outputs/zimage.png
```

查看某个模型的完整参数：

```bash
chitu generate --model zimage --help
```

## 静态多卡生成

所有 rank 执行同一条命令，只有 leader 写出结果。默认 transport 是 Torch/NCCL：

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model flux1 \
  --model-path /path/to/FLUX.1-dev \
  --output outputs/flux1.png
```

Fast AGKV：

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --agkv-transport fast_agkv \
  --output outputs/zimage-fast-agkv.png
```

Fast Ulysses：

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --attention-mode ulysses \
  --ulysses-transport fast_ulysses \
  --output outputs/zimage-fast-ulysses.png
```

## FlexCache

缓存参数直接附加到 `generate`：

```bash
chitu generate \
  --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --cache-strategy magcache \
  --output outputs/wan-magcache.mp4
```

策略专属参数使用对应前缀。例如 MeanCache：

```bash
--cache-strategy meancache --meancache-fresh-steps 25
```

## EPE 服务

配置中的 GPU 数必须等于 torchrun 进程数。四卡示例：

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  serve --stage-config examples/stage-zimage.yaml
```

leader 提供 HTTP 服务，其他 rank 执行 worker loop。宿主已经管理进程时，可参考
`examples/epe_embedded.py` 使用 `EmbeddedDiffusionRuntime`。

## Slurm

集群入口会为每张 GPU 启动一个 task，并设置 `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE`：

```bash
bash tools/cluster/srun_direct.sh 1 4 -m chitu_diffusion.cli \
  generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --output outputs/zimage.png
```

仓库根目录 `examples/` 还包含各模型 Python API、静态 CP、EPE embedded 和服务配置
示例。
