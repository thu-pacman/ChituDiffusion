<a name="chitudiffusion"></a>

<p align="center">
  <img src="docs/assets/chitudiffusion-brand.png" alt="ChituDiffusion" width="760">
</p>

<p align="center">
  <b>中文</b> &nbsp;·&nbsp; <a href="#english-version">English</a>
</p>

<h3 align="center">支持 EPE、Fast CP、NCCL CP 与 FlexCache 的 Diffusers 推理运行时</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12--3.13-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-GPU%20required-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

ChituDiffusion 基于 Diffusers 生命周期提供模型生成、上下文并行、缓存加速和常驻服务。

> 项目处于开发者预览阶段。服务不保证任意 rank 故障后的恢复。

## 核心能力

<table>
<tr>
<td width="50%" valign="top">

### <a href="chitu_diffusion/parallel/cp/fast/README.md">Fast CP：高性能序列并行</a>

提供 Fast AGKV 与 Fast Ulysses，并保留 NCCL fallback。面向具备 GPU P2P、
NVSHMEM 和目标架构扩展的兼容单机环境优化通信路径。

</td>
<td width="50%" valign="top">

### <a href="docs/features/epe.md">EPE：弹性并行服务</a>

根据实测代价和端到端 SLO，在 pulse 边界调整 CP lane，并通过统一 executor 处理
队列、状态迁移和 worker 生命周期。

</td>
</tr>
<tr>
<td width="50%" valign="top">

### <a href="docs/features/flexcache.md">FlexCache：Cache 加速 API 和评测</a>

提供 MagCache、MeanCache、TeaCache、TaylorSeer 与 PAB 的统一 API，可叠加单卡或
静态 CP。查看 <a href="docs/features/flexcache.md#优化结果">速度与质量评测</a>。

</td>
<td width="50%" valign="top">

### <a href="docs/features/diffusers-api.md">Diffusers Plug and Play</a>

保留 tokenizer、文本编码器、scheduler、VAE 和输出处理流程，只适配模型相关的
DiT 执行与 tensor 布局。

</td>
</tr>
</table>

**支持模型**

<table>
<thead>
<tr>
<th>模型</th>
<th>生成能力</th>
<th>并行与服务能力</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="chitu_diffusion/models/flux1/README.md">FLUX.1</a></td>
<td>文生图</td>
<td>静态 CP、EPE executor、FlexCache model spec</td>
</tr>
<tr>
<td><a href="chitu_diffusion/models/flux2_klein/README.md">FLUX.2-klein</a></td>
<td>文生图</td>
<td>固定 full-world 静态 CP</td>
</tr>
<tr>
<td><a href="chitu_diffusion/models/hunyuan_image3/README.md">Hunyuan Image 3</a></td>
<td>固定尺寸文生图与图生图基础服务链路</td>
<td>可配置 TP×CFG×CP×EP decoder（72 GiB 参考配置 TP2×CFG2×CP2×EP2）、变长 AGKV attention、EPE static CP、变长 all-to-all expert dispatch、独立 VAEP</td>
</tr>
<tr>
<td><a href="chitu_diffusion/models/minimax_h3/README.md">MiniMax-H3</a></td>
<td>T2VA、first/last-frame FL2VA 基础服务链路</td>
<td>TP×CP DiT、EPE、Fast Ulysses/NCCL CP、独立 VAEP、音视频解码与 MP4 输出</td>
</tr>
<tr>
<td><a href="chitu_diffusion/models/qwen_image/README.md">Qwen-Image</a></td>
<td>文生图</td>
<td>静态 CP、EPE executor</td>
</tr>
<tr>
<td><a href="chitu_diffusion/models/wan/README.md">Wan 2.1 T2V</a></td>
<td>文生视频</td>
<td>静态 CP、EPE executor、FlexCache profiles</td>
</tr>
<tr>
<td><a href="chitu_diffusion/models/zimage/README.md">Z-Image</a></td>
<td>文生图</td>
<td>静态 CP、统一 EPE 服务入口、FlexCache integration</td>
</tr>
</tbody>
</table>

## 安装

要求 Linux、Python 3.12 或 3.13、NVIDIA GPU 和可用的 CUDA 环境。按机器 CUDA
版本选择 `pyproject.toml` 中的 PyTorch index，然后运行：

```bash
uv sync --group dev
source .venv/bin/activate
chitu --help
```

`pyproject.toml` 是唯一构建入口。

## 生成

```bash
chitu generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --prompt "a red cube on a white table" \
  --output outputs/zimage.png
```

静态多卡 NCCL CP：

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model flux1 \
  --model-path /path/to/FLUX.1-dev \
  --output outputs/flux1.png
```

单机 TP×CFG×CP×EP MoE，例如 Hunyuan Image 3 的默认 8 卡拓扑：

```bash
torchrun --standalone --nproc-per-node=8 -m chitu_diffusion.cli \
  generate \
  --model hunyuan-image3 \
  --model-path /path/to/HunyuanImage-3 \
  --steps 50 \
  --output outputs/hunyuan_image3.png
```

Fast CP 需要兼容的单机 GPU、GPU P2P、NVSHMEM 和针对目标环境编译的扩展。参数和安装
步骤见 [Fast CP README](chitu_diffusion/parallel/cp/fast/README.md)。

## FlexCache

```bash
chitu generate \
  --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --cache-strategy magcache \
  --output outputs/wan-magcache.mp4
```

缓存支持与模型、步数和 profile 绑定。不支持的组合会在运行前报错。原理、支持范围和
测试结果见 [FlexCache 文档](docs/features/flexcache.md)。

## EPE 服务

```bash
chitu serve --stage-config examples/stage-zimage.yaml
```

所有 rank 进入服务生命周期，leader 提供 HTTP，其他 rank 执行 worker loop。配置和
限制见[运行说明](docs/usage/running.md)与 [EPE 文档](docs/features/epe.md)。

## 文档与开发

- 文档首页：[`docs/index.md`](docs/index.md)
- 安装：[`docs/usage/installation.md`](docs/usage/installation.md)
- 运行：[`docs/usage/running.md`](docs/usage/running.md)
- 模型 API：[`docs/features/diffusers-api.md`](docs/features/diffusers-api.md)
- 用户示例：[`examples/README.md`](examples/README.md)

```bash
python -m pytest -q
python -m ruff check chitu_diffusion tests examples
python -m build
mkdocs build --strict
```

## 许可证

[MIT License](LICENSE)

---

<a name="english-version"></a>

<p align="center">
  <img src="docs/assets/chitudiffusion-brand.png" alt="ChituDiffusion" width="760">
</p>

<p align="center">
  <a href="#chitudiffusion">中文</a> &nbsp;·&nbsp; <b>English</b>
</p>

<h3 align="center">Diffusers inference with EPE, Fast CP, NCCL CP, and FlexCache</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12--3.13-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-GPU%20required-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

ChituDiffusion provides model generation, context parallelism, cache acceleration,
and persistent serving while preserving the Diffusers pipeline lifecycle.

> The project is a developer preview. The service does not recover from an
> arbitrary rank failure.

## Core capabilities

<table>
<tr>
<td width="50%" valign="top">

### <a href="chitu_diffusion/parallel/cp/fast/README.md">Fast CP: High-performance sequence parallelism</a>

Fast AGKV and Fast Ulysses with an NCCL fallback. The fast paths target
compatible single-node systems with GPU P2P, NVSHMEM, and extensions built for
the target architecture.

</td>
<td width="50%" valign="top">

### <a href="docs/features/epe.md">EPE: Elastic parallel serving</a>

Adjusts CP lanes at pulse boundaries using measured costs and end-to-end SLOs,
with one executor lifecycle for queues, state migration, and workers.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### <a href="docs/features/flexcache.md">FlexCache: Cache APIs and evaluation</a>

One API for MagCache, MeanCache, TeaCache, TaylorSeer, and PAB, composable with
single-GPU or static CP generation. See the
<a href="docs/features/flexcache.md#优化结果">speed and quality evaluations</a>.

</td>
<td width="50%" valign="top">

### <a href="docs/features/diffusers-api.md">Diffusers Plug and Play</a>

Preserves tokenizers, text encoders, schedulers, VAEs, and output processing.
Adapters only supply model-specific DiT execution and tensor layouts.

</td>
</tr>
</table>

**Supported models**

<table>
<thead>
<tr>
<th>Model</th>
<th>Generation</th>
<th>Parallelism and serving</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="chitu_diffusion/models/flux1/README.md">FLUX.1</a></td>
<td>Text to image</td>
<td>Static CP, EPE executor, FlexCache model spec</td>
</tr>
<tr>
<td><a href="chitu_diffusion/models/flux2_klein/README.md">FLUX.2-klein</a></td>
<td>Text to image</td>
<td>Fixed full-world static CP</td>
</tr>
<tr>
<td><a href="chitu_diffusion/models/hunyuan_image3/README.md">Hunyuan Image 3</a></td>
<td>Foundational fixed-size text-to-image and image-to-image service path</td>
<td>Configurable TP×CFG×CP×EP decoder (72 GiB reference: TP2×CFG2×CP2×EP2), variable-length AGKV attention, EPE static CP, variable-length all-to-all expert dispatch, independent VAEP</td>
</tr>
<tr>
<td><a href="chitu_diffusion/models/minimax_h3/README.md">MiniMax-H3</a></td>
<td>Foundational T2VA and first/last-frame FL2VA service path</td>
<td>TP×CP DiT, EPE, Fast Ulysses/NCCL CP, independent VAEP, audio/video decode, and MP4 output</td>
</tr>
<tr>
<td><a href="chitu_diffusion/models/qwen_image/README.md">Qwen-Image</a></td>
<td>Text to image</td>
<td>Static CP and EPE executor</td>
</tr>
<tr>
<td><a href="chitu_diffusion/models/wan/README.md">Wan 2.1 T2V</a></td>
<td>Text to video</td>
<td>Static CP, EPE executor, and FlexCache profiles</td>
</tr>
<tr>
<td><a href="chitu_diffusion/models/zimage/README.md">Z-Image</a></td>
<td>Text to image</td>
<td>Static CP, unified EPE serving, and FlexCache integration</td>
</tr>
</tbody>
</table>

## Installation

ChituDiffusion requires Linux, Python 3.12 or 3.13, an NVIDIA GPU, and a working
CUDA environment. Select the matching PyTorch index in `pyproject.toml`, then run:

```bash
uv sync --group dev
source .venv/bin/activate
chitu --help
```

`pyproject.toml` is the only build entry point.

## Generation

```bash
chitu generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --prompt "a red cube on a white table" \
  --output outputs/zimage.png
```

Static multi-GPU NCCL CP:

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model flux1 \
  --model-path /path/to/FLUX.1-dev \
  --output outputs/flux1.png
```

A single-node TP×CFG×CP×EP MoE stage, for example Hunyuan Image 3 on its default
eight-GPU topology:

```bash
torchrun --standalone --nproc-per-node=8 -m chitu_diffusion.cli \
  generate \
  --model hunyuan-image3 \
  --model-path /path/to/HunyuanImage-3 \
  --steps 50 \
  --output outputs/hunyuan_image3.png
```

Fast CP requires compatible single-node GPUs, GPU P2P, NVSHMEM, and extensions
built for the target environment. See the
[Fast CP README](chitu_diffusion/parallel/cp/fast/README.md).

## FlexCache

```bash
chitu generate \
  --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --cache-strategy magcache \
  --output outputs/wan-magcache.mp4
```

Cache support is tied to the model, step count, and profile. Unsupported
combinations fail before execution. See the
[FlexCache documentation](docs/features/flexcache.md) for principles, support,
and measured results.

## EPE serving

```bash
chitu serve --stage-config examples/stage-zimage.yaml
```

All ranks enter the service lifecycle. The leader hosts HTTP while the remaining
ranks execute worker loops. See the [running guide](docs/usage/running.md) and
[EPE documentation](docs/features/epe.md).

## Documentation and development

- Documentation: [`docs/index.md`](docs/index.md)
- Installation: [`docs/usage/installation.md`](docs/usage/installation.md)
- Running: [`docs/usage/running.md`](docs/usage/running.md)
- Model API: [`docs/features/diffusers-api.md`](docs/features/diffusers-api.md)
- Examples: [`examples/README.md`](examples/README.md)

```bash
python -m pytest -q
python -m ruff check chitu_diffusion tests examples
python -m build
mkdocs build --strict
```

## License

[MIT License](LICENSE)
