# 运行

`chitu generate` 执行单次生成，`chitu serve` 启动 EPE 服务。CLI 支持 `zimage`、
`flux1`、`flux2-klein`、`qwen-image`、`wan`、`llada-image` 和 `hunyuan-image3`。

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

所有 `generate` 入口都会打印 `elapsed_s=...`，并在产物旁写入同名 `.json`。
`elapsed_s` 与兼容字段 `generate_seconds` 都以秒记录本次生成耗时，包括文本编码、
去噪、VAE 解码和结果后处理，不含模型加载、图片保存或视频编码。CUDA 计时边界会
等待设备工作完成；这是单次请求耗时，首轮可能包含编译或初始化成本。
JSON 同时记录 prompt、seed、步数和并行参数；FlexCache 模型还记录缓存统计。

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

Hunyuan Image 3 的 TP/CFG/CP/EP 拓扑在启动时确定。以下使用 72 GiB 卡上验证的
TP2×CFG2×CP2×EP2，并以 VAEP8 解码：

```bash
torchrun --standalone --nproc-per-node=8 -m chitu_diffusion.cli \
  generate \
  --model hunyuan-image3 \
  --model-path /path/to/HunyuanImage-3 \
  --expert-parallel-degree 2 \
  --steps 50 \
  --output outputs/hunyuan_image3.png
```

只需满足 `world = TP × CFG × CP`、`CFG ∈ {1, 2}`、`EP` 整除 `CFG × CP`，即可改成
其他单机拓扑。未给出的 degree 由 world size 推导，例如下面等价于 TP2×CFG1×CP4×EP4：

```bash
torchrun --standalone --nproc-per-node=8 -m chitu_diffusion.cli \
  generate --model hunyuan-image3 --cfg-parallel-degree 1 ...
```

## VAE 解码

Z-Image、FLUX.1、LLaDA-Image、Wan 和 Qwen-Image 的静态并行解码使用逐层
行分片：卷积交换相邻 rank 的边界行，GroupNorm 通信全局统计量，attention
使用本地 Q 和 all-gather 后的全局 K/V（AGKV）。视频保留原有时序缓存。
该路径保留整图解码语义，浮点归约与卷积算法差异仍可能造成数值误差。

使用 `--parallel-vae` 开启，`--no-parallel-vae` 让 leader 整图解码。
Z-Image 和 LLaDA-Image 默认关闭，其余上述模型默认开启。单 rank、分片过短
或不支持的 decoder 使用 leader 路径；不支持的结构会提示原因。
开启逐层并行前需关闭 Diffusers 自带 tiling，否则会报错。

`--vae-parallel-halo N` 保留兼容（非负），但上述逐层路径按各层卷积核自动
推导 halo，不使用该值；其他模型的独立 tile 路径仍有自己的 halo 语义。
sidecar 的 `requested_parallel_vae`、`requested_vae_parallel_halo` 保留请求值，
`vae_decode_mode`（`layerwise` / `leader`）、`parallel_vae` 和
`vae_parallel_degree` 记录实际执行方式，逐层路径的 `vae_parallel_halo` 为 null。

本次实现和验证范围为静态并行，不包含动态 EPE。
支持范围、误差和性能测量见 [VAE 验证记录](../validation/vae-parallel.md)。

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

八卡 static-CP MoE 服务示例：

```bash
torchrun --standalone --nproc-per-node=8 -m chitu_diffusion.cli \
  serve --stage-config examples/stage-hunyuan-image3.yaml
```

leader 提供 HTTP 服务，其他 rank 执行 worker loop。宿主已经管理进程时，可参考
`examples/epe_embedded.py` 使用 `EmbeddedDiffusionRuntime`。

配置里的 `parallelism` 分为公共的 `cp`、`vae`、`scheduler` 和模型专属的 `model`，
字段含义见 [EPE 并行配置](../features/epe.md#并行配置)。

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
