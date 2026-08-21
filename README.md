<a name="chitudiffusion"></a>

<p align="center">
  <img src="docs/assets/chitudiffusion-brand.png" alt="ChituDiffusion" width="760">
</p>

<p align="center">
  <b>🇨🇳 中文</b> &nbsp;·&nbsp; <a href="#english-version">🇺🇸 English</a>
</p>

---

<h3 align="center">ChituDiffusion：支持弹性并行、高性能上下文并行、缓存加速和统一服务的 Diffusers 推理运行时</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-GPU%20required-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

ChituDiffusion 是一个基于 Diffusers 生命周期的高性能上下文并行扩散推理运行时。它保留
上游 Diffusers 的 tokenizer、文本编码器、scheduler、VAE 和输出处理，只通过统一执行
后端并行化 DiT 去噪计算。

> **项目状态：开发者预览版。** 静态生成路径和 EPE 核心运行时已经可用并通过测试，
> 但服务端尚不具备跨 rank 容错能力。生产部署前请阅读[当前限制](#当前限制)。

## ✨ 为什么选择 ChituDiffusion

<table>
<tr>
<td width="50%">

### 🌐 Elastic Parallel Engine（EPE）
EPE 根据启动时测得的代价和端到端 SLO，在 pulse 边界调整 CP lane，并迁移请求状态。
`generate` 与常驻 `serve` 使用同一套 executor。

</td>
<td width="50%">

### ⚡ FlexCache
MagCache、MeanCache、TeaCache、TaylorSeer 和 PAB 通过 request-local hook 接入
`generate`。所有 rank 使用相同的 fresh/reuse 判定。

</td>
</tr>
<tr>
<td width="50%">

### 🚀 高性能上下文并行（Fast CP）
Fast AGKV 传输全局 K/V，Fast Ulysses 在序列和 heads 之间重排数据。两者使用
NVSHMEM 和 Copy Engine 减少通信等待。Fast Ring 用于低显存实验。

</td>
<td width="50%">

### 🧩 Diffusers-native
模型适配器保留上游 pipeline 生命周期。AGKV、USP、CFP/CP 与并行 VAE 使用共享的
request、denoise、decode 和 postprocess 协议。

</td>
</tr>
</table>

---

## 🎬 支持的模型

| 模型 | `generate` | EPE `serve` | FlexCache | 输出 |
|:---|:---|:---|:---|:---|
| **Z-Image** | 静态 CP / CFP+CP | ✅ | MeanCache、TaylorSeer、PAB | 图片 |
| **FLUX.1-dev** | 静态 CP | ✅ | MagCache、TeaCache、TaylorSeer、PAB | 图片 |
| **Qwen-Image** | 静态 CP / CFP+CP | ✅ | MagCache、MeanCache、TaylorSeer、PAB | 图片 |
| **Wan 2.1 T2V** | 静态 CP / CFP+CP | ✅ | MagCache、TeaCache、TaylorSeer、PAB | MP4 视频 |
| **FLUX.2-klein** | 静态 CP baseline | ❌ 弹性切换 | 尚未验收 | 图片 |

缓存支持范围与模型相关。未支持的缓存和模型组合会在执行前报错。

---

## ⚡ 快速开始

### 1. 安装

环境要求：Linux、NVIDIA GPU、Python 3.12+、可用的 CUDA/PyTorch 环境，以及本地或
可访问的 Diffusers 格式模型权重。

```bash
git clone <repository-url>
cd ChituDiffusion
uv sync
source .venv/bin/activate
chitu --help
```

运行 `uv sync` 前，请根据集群 CUDA 版本修改 `pyproject.toml` 中的 PyTorch index。
USP 依赖通过 `uv sync --extra usp` 安装。Fast transport 需要单独编译，步骤见
[Fast CP 文档](chitu_diffusion/parallel/fast_cp/README.md)。

### 2. 单请求生成

```bash
chitu generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --prompt "a red cube on a white table" \
  --output outputs/zimage.png
```

### 3. 多卡静态 CP

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model flux1 \
  --model-path /path/to/FLUX.1-dev \
  --prompt "a lighthouse above a stormy sea" \
  --output outputs/flux1.png
```

Slurm 环境可使用仓库 wrapper：

```bash
bash script/srun_direct.sh 1 4 chitu_diffusion/examples/zimage_epe.py \
  --model-path /path/to/Z-Image \
  --steps 12 \
  --output outputs/zimage.png
```

单机 Hopper 可以添加 `--attention-mode usp --ulysses-transport fast_ulysses`
启用 Fast Ulysses。`--agkv-transport fast_agkv` 启用 Fast AGKV K/V transport。
Fused Fast AGKV attention 和 Fast Ring 目前用于 benchmark。安装方法和 H20 scaling
图见 [Fast CP 文档](chitu_diffusion/parallel/fast_cp/README.md)。

所有分布式 rank 都必须进入 `generate`；只有 full-world leader 返回并保存最终输出。

### 4. 常驻服务

```bash
chitu serve --stage-config /path/to/stage.yaml
```

所有 rank 都进入 `serve`。stage leader 提供 HTTP，其他 rank 运行分布式 worker
loop。

---

## 🏗️ 架构

```text
Diffusers pipeline 生命周期
        |
模型相关 DiffusionBackend / executor
        |
共享 request、denoise、decode、postprocess 协议
        |
  +-----+--------------------------------+
  |                                      |
generate：full-world 静态 CP       serve：队列 + EPE pulse
                                    动态 lane + state migration
```

| 目录 | 职责 |
|:---|:---|
| [EPE runtime](chitu_diffusion/epac/) | executor、代价模型、调度策略、pulse 协议和 worker runtime |
| [`chitu_diffusion/models/`](chitu_diffusion/models/) | 模型家族相关 tensor glue 和 Diffusers adapter |
| [`chitu_diffusion/parallel/`](chitu_diffusion/parallel/) | 进程组、上下文并行 attention 和并行 VAE 通信 |
| [`chitu_diffusion/parallel/fast_cp/`](chitu_diffusion/parallel/fast_cp/) | Fast AGKV、Fast Ulysses 和 Fast Ring |
| [`chitu_diffusion/flexcache/`](chitu_diffusion/flexcache/) | request-local 缓存策略和模型 profile |
| [`chitu_diffusion/serve/`](chitu_diffusion/serve/) | 配置、HTTP 和分布式服务生命周期 |

实现细节见[运行时说明](chitu_diffusion/README.md)、[EPE 设计文档
](chitu_diffusion/epac/README.zh-CN.md)、[Fast CP 文档
](chitu_diffusion/parallel/fast_cp/README.md)和[FlexCache 文档
](chitu_diffusion/flexcache/README.zh-CN.md)。

---

## 🧠 FlexCache

FlexCache 用于单请求 `generate`。EPE 常驻服务暂时不接入缓存策略。

```bash
chitu generate \
  --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --cache-strategy magcache \
  --output outputs/wan-magcache.mp4
```

每个请求保存自己的缓存状态。串行 CFG 分支分别保存状态。fresh/reuse 判定只使用
各 rank 共有的输入和日程，保证所有 rank 以相同顺序进入 collective。

MagCache 和 MeanCache profile 只覆盖指定模型和步数。其他配置会在执行前报错。
支持范围和新策略接入方法见
[FlexCache 文档](chitu_diffusion/flexcache/README.zh-CN.md)。

可复现实验记录：

- [MagCache 对比](outputs/flexcache/magcache_compare_20260805/result.md)
- [MeanCache 对比](outputs/flexcache/meancache_compare_20260804/result.md)

---

## 🛠️ 开发与验证

```bash
python -m pytest -q
python -m ruff check chitu_diffusion test
python -m build
```

GPU 正确性必须按模型使用相同 seed 与原生 Diffusers baseline 对比。示例 launcher 和
参数见 [`chitu_diffusion/examples/README.md`](chitu_diffusion/examples/README.md)。

贡献代码时：

1. EPE 调度放在 EPE runtime，通信与 attention 后端放在 `parallel/`，缓存策略放在
   `flexcache/`。
2. checkpoint 相关 tensor 适配放在 `models/<family>/`。
3. 未支持能力必须显式报错。
4. 增加 CPU contract 测试，并记录 GPU 命令和结果。
5. rank-local 缓存状态不能改变分布式控制流。

## ⚠️ 当前限制

- 项目尚未达到 production GA，不保证任意 rank 故障后的恢复。
- EPE 暂时不接入 FlexCache。
- cache profile 只对指定模型、scheduler 和步数有效，配置变化后需要重新评估速度和质量。
- Fast transport 需要 Hopper、单机 P2P 和针对目标环境编译的 NVSHMEM 扩展。
- Fast AGKV fused attention 和 Fast Ring 仍是 benchmark 能力。
- FLUX.2-klein 目前仅提供固定静态 CP baseline。
- 历史 ChituBench、DiTango、分阶段 runtime、配置、测试和结果冻结在
  [`backup/chitu_diffusion_legacy/`](backup/chitu_diffusion_legacy/)，不会进入 wheel
  或默认测试。

## 📄 许可证

ChituDiffusion 使用 [MIT License](LICENSE)。模型权重和上游依赖仍受各自许可证与使用
政策约束。

---

<a name="english-version"></a>

<p align="center">
  <img src="docs/assets/chitudiffusion-brand.png" alt="ChituDiffusion" width="760">
</p>

<p align="center">
  <a href="#chitudiffusion">🇨🇳 中文</a> &nbsp;·&nbsp; <b>🇺🇸 English</b>
</p>

---

<h3 align="center">ChituDiffusion: Diffusers inference with elastic parallelism, high-performance context parallelism, cache acceleration, and unified serving</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-GPU%20required-76B900?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

ChituDiffusion is a Diffusers-native runtime for high-performance,
context-parallel diffusion inference. It keeps the tokenizer, text encoder,
scheduler, VAE, and output processing in the upstream Diffusers lifecycle and
parallelizes DiT denoising through one shared execution backend.

> **Project status: developer preview.** Static generation and the core EPE
> runtime are usable and tested, but the service does not yet provide
> cross-rank fault tolerance. Review [Current limitations](#current-limitations)
> before production deployment.

## ✨ Why ChituDiffusion

<table>
<tr>
<td width="50%">

### 🌐 Elastic Parallel Engine (EPE)
EPE uses startup measurements and end-to-end SLOs to adjust CP lanes and
migrate request state at pulse boundaries. `generate` and persistent `serve`
use the same executor.

</td>
<td width="50%">

### ⚡ FlexCache
MagCache, MeanCache, TeaCache, TaylorSeer, and PAB integrate with
single-request `generate` through request-local hooks. Every rank uses the same
fresh/reuse decision.

</td>
</tr>
<tr>
<td width="50%">

### 🚀 High-Performance Context Parallelism (Fast CP)
Fast AGKV transfers global K/V. Fast Ulysses redistributes data between the
sequence and head dimensions. Both use NVSHMEM and Copy Engines to reduce
communication wait time. Fast Ring supports low-memory experiments.

</td>
<td width="50%">

### 🧩 Diffusers-Native Integration
Model adapters preserve upstream pipeline lifecycles. AGKV, USP, CFP/CP, and
parallel VAE use shared request, denoise, decode, and postprocess contracts.

</td>
</tr>
</table>

---

## 🎬 Supported Models

| Model | `generate` | EPE `serve` | FlexCache | Output |
|:---|:---|:---|:---|:---|
| **Z-Image** | Static CP / CFP+CP | ✅ | MeanCache, TaylorSeer, PAB | Image |
| **FLUX.1-dev** | Static CP | ✅ | MagCache, TeaCache, TaylorSeer, PAB | Image |
| **Qwen-Image** | Static CP / CFP+CP | ✅ | MagCache, MeanCache, TaylorSeer, PAB | Image |
| **Wan 2.1 T2V** | Static CP / CFP+CP | ✅ | MagCache, TeaCache, TaylorSeer, PAB | MP4 video |
| **FLUX.2-klein** | Static-CP baseline | No elastic switching | Not validated | Image |

Cache support is model-specific. Unsupported cache and model combinations fail
before execution.

---

## ⚡ Quick Start

### 1. Install

Requirements: Linux, NVIDIA GPUs, Python 3.12+, a working CUDA/PyTorch
environment, and local or accessible Diffusers-format checkpoints.

```bash
git clone <repository-url>
cd ChituDiffusion
uv sync
source .venv/bin/activate
chitu --help
```

Select the appropriate PyTorch CUDA index in `pyproject.toml` before
`uv sync`. Install USP dependencies with `uv sync --extra usp`. Fast transports
require a separate build described in the
[Fast CP guide](chitu_diffusion/parallel/fast_cp/README.md).

### 2. Generate One Request

```bash
chitu generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --prompt "a red cube on a white table" \
  --output outputs/zimage.png
```

### 3. Run Multi-GPU Static CP

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model flux1 \
  --model-path /path/to/FLUX.1-dev \
  --prompt "a lighthouse above a stormy sea" \
  --output outputs/flux1.png
```

Use the repository wrapper on Slurm:

```bash
bash script/srun_direct.sh 1 4 chitu_diffusion/examples/zimage_epe.py \
  --model-path /path/to/Z-Image \
  --steps 12 \
  --output outputs/zimage.png
```

On a single Hopper node, add
`--attention-mode usp --ulysses-transport fast_ulysses` to enable Fast
Ulysses. `--agkv-transport fast_agkv` enables the Fast AGKV K/V transport.
Fused Fast AGKV attention and Fast Ring remain benchmark paths. See the
[Fast CP guide](chitu_diffusion/parallel/fast_cp/README.md) for build steps and
H20 scaling figures.

Every distributed rank must enter `generate`; only the full-world leader
returns and writes the final output.

### 4. Start Persistent Serving

```bash
chitu serve --stage-config /path/to/stage.yaml
```

All ranks enter `serve`. The stage leader exposes HTTP while follower ranks run
the distributed worker loop.

---

## 🏗️ Architecture

```text
Diffusers pipeline lifecycle
        |
model-specific DiffusionBackend / executor
        |
shared request, denoise, decode, and postprocess contract
        |
  +-----+--------------------------------+
  |                                      |
generate: full-world static CP     serve: queue + EPE pulses
                                   dynamic lanes + state migration
```

| Directory | Responsibility |
|:---|:---|
| [EPE runtime](chitu_diffusion/epac/) | Executor, cost model, scheduler, pulse protocol, and workers |
| [`chitu_diffusion/models/`](chitu_diffusion/models/) | Model-family tensor glue and Diffusers adapters |
| [`chitu_diffusion/parallel/`](chitu_diffusion/parallel/) | Process groups, context-parallel attention, and parallel VAE communication |
| [`chitu_diffusion/parallel/fast_cp/`](chitu_diffusion/parallel/fast_cp/) | Fast AGKV, Fast Ulysses, and Fast Ring |
| [`chitu_diffusion/flexcache/`](chitu_diffusion/flexcache/) | Request-local cache strategies and model profiles |
| [`chitu_diffusion/serve/`](chitu_diffusion/serve/) | Configuration, HTTP, and distributed service lifecycle |

See the [runtime guide](chitu_diffusion/README.md), [EPE design
guide](chitu_diffusion/epac/README.md), [Fast CP
guide](chitu_diffusion/parallel/fast_cp/README.md), and [FlexCache
guide](chitu_diffusion/flexcache/README.md).

---

## 🧠 FlexCache

FlexCache supports single-request `generate`. Persistent EPE serving does not
currently use cache strategies.

```bash
chitu generate \
  --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --cache-strategy magcache \
  --output outputs/wan-magcache.mp4
```

Each request owns its cache state. Serial CFG branches keep separate state.
Fresh/reuse decisions use inputs and schedules shared by all ranks, so every
rank enters collectives in the same order.

MagCache and MeanCache profiles cover specified models and step counts. Other
configurations fail before execution. The
[FlexCache guide](chitu_diffusion/flexcache/README.md) documents the support
matrix and strategy integration.

Reproducible reports:

- [MagCache comparison](outputs/flexcache/magcache_compare_20260805/result.md)
- [MeanCache comparison](outputs/flexcache/meancache_compare_20260804/result.md)

---

## 🛠️ Development and Validation

```bash
python -m pytest -q
python -m ruff check chitu_diffusion test
python -m build
```

GPU acceptance must compare each model against its native Diffusers baseline
with the same seed. Launchers and arguments are documented in
[`chitu_diffusion/examples/README.md`](chitu_diffusion/examples/README.md).

Contribution boundaries:

1. Keep EPE scheduling in the EPE runtime, communication or attention backends
   in `parallel/`, and cache strategies in `flexcache/`.
2. Keep checkpoint-specific tensor adaptation in `models/<family>/`.
3. Fail explicitly for unsupported capabilities.
4. Add CPU contract tests and record GPU commands and results.
5. Rank-local cache state must not change distributed control flow.

## ⚠️ Current Limitations

- The project is not production GA and does not guarantee recovery from an
  arbitrary rank failure.
- EPE does not currently integrate FlexCache.
- Cache profiles apply only to their specified models, schedulers, and step
  counts. Configuration changes require new speed and quality measurements.
- Fast transports require Hopper, single-node P2P, and NVSHMEM extensions
  compiled for the target environment.
- Fused Fast AGKV attention and Fast Ring remain benchmark capabilities.
- FLUX.2-klein currently provides a fixed static-CP baseline only.
- Historical ChituBench, DiTango, staged runtime, configurations, tests, and
  results are frozen under
  [`backup/chitu_diffusion_legacy/`](backup/chitu_diffusion_legacy/) and are
  excluded from packages and default tests.

## 📄 License

ChituDiffusion is released under the [MIT License](LICENSE). Model checkpoints
and upstream dependencies remain subject to their own licenses and acceptable
use policies.
