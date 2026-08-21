<a name="chitudiffusion"></a>

<p align="center">
  <img src="docs/assets/chitudiffusion-brand.png" alt="ChituDiffusion" width="760">
</p>

<p align="center">
  <b>🇨🇳 中文</b> &nbsp;·&nbsp; <a href="#english-version">🇺🇸 English</a>
</p>

---

<h3 align="center">ChituDiffusion：高性能 Diffusers 推理运行时 — 弹性并行 · 缓存加速 · 统一服务</h3>

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

### 🌐 Elastic Parallel Engine
基于启动实测代价和端到端 SLO，在 pulse 边界动态重组 CP lane，并支持 request state
migration。`generate` 与常驻 `serve` 共享同一 executor。

</td>
<td width="50%">

### ⚡ FlexCache
MagCache、MeanCache、TeaCache、TaylorSeer 和 PAB 以 request-local hook 解耦接入，并
保证 CFP/CP rank 控制流一致。

</td>
</tr>
<tr>
<td width="50%">

### 🚀 高性能上下文并行
AGKV 与 USP attention、CFP/CP 混合布局和并行 VAE 使用动态 lane process group，
加速图像与视频 DiT 推理；静态单节点 CP 可显式选择 capability-gated Fast Ulysses
或 Fast AGKV transport，不满足约束的动态 lane 保持 NCCL 路径。Fused Full-Mesh
与 Fast Ring 当前仅用于研究 benchmark；安装约束和 H20 性能矩阵见
[`parallel/fast_cp`](chitu_diffusion/parallel/fast_cp/README.md) 与
[`Fast CP 结果摘要`](kernels/fast_cp_results.md)。

</td>
<td width="50%">

### 🧩 Diffusers-native
模型适配器保留上游 pipeline 生命周期；AGKV、USP、CFP/CP 与并行 VAE 通过共享执行协议
组合，而不是维护第二套推理栈。

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

支持范围是显式且与模型相关的。未支持的缓存/模型组合会提前报错，不会静默退化为无缓存
执行。

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
可选 USP 后端可通过 `uv sync --extra usp` 安装。

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
bash script/srun_direct.sh 1 4 chitu_diffusion/examples/wan_epac.py \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --output outputs/wan.mp4
```

所有分布式 rank 都必须进入 `generate`；只有 full-world leader 返回并保存最终输出。

### 4. 常驻服务

```bash
chitu serve --stage-config /path/to/stage.yaml
```

所有 rank 都进入 `serve`；stage leader 提供 HTTP，其他 rank 运行分布式 worker loop。
该路径会显式拒绝非 `none` 的 FlexCache 策略。

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
| [`chitu_diffusion/epac/`](chitu_diffusion/epac/) | 模型无关 executor、代价模型、调度策略、pulse 协议和 worker runtime |
| [`chitu_diffusion/models/`](chitu_diffusion/models/) | 模型家族相关 tensor glue 和 Diffusers adapter |
| [`chitu_diffusion/parallel/`](chitu_diffusion/parallel/) | 进程组、上下文并行 attention 和并行 VAE 通信 |
| [`chitu_diffusion/flexcache/`](chitu_diffusion/flexcache/) | 不侵入模型 forward 的 request-local 缓存策略 |
| [`chitu_diffusion/serve/`](chitu_diffusion/serve/) | 配置、HTTP 和分布式服务生命周期 |

实现细节见[运行时说明](chitu_diffusion/README.md)、[EPAC 设计文档
](chitu_diffusion/epac/README.zh-CN.md)和[FlexCache 扩展指南
](chitu_diffusion/flexcache/README.zh-CN.md)。

---

## 🧠 FlexCache

FlexCache 仅用于单请求 `generate`：

```bash
chitu generate \
  --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --cache-strategy magcache \
  --output outputs/wan-magcache.mp4
```

策略状态属于单个请求，串行 CFG 分支相互隔离。fresh/reuse 判定只使用各 rank 复制的
输入或共享日程，保证 CFP/CP rank 进入相同 collective。官方 MagCache 和 MeanCache
profile 根据模型与步数选择，超出标定范围会直接报错。

完整支持约束、新策略接入流程和并行安全要求见
[FlexCache 文档](chitu_diffusion/flexcache/README.zh-CN.md)。

可复现对比记录：

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

1. 模型无关行为放在 `epac/`、`parallel/` 或 `flexcache/`。
2. checkpoint 相关 tensor 适配放在 `models/<family>/`。
3. 未支持能力必须显式报错。
4. 增加 CPU contract 测试，并记录 GPU 命令和结果。
5. 禁止用 rank-local 缓存信号改变分布式控制流。

## ⚠️ 当前限制

- 项目尚未达到 production GA，不保证任意 rank 故障后的恢复。
- FlexCache 仅支持 `generate`；常驻 EPE 服务会拒绝缓存策略。
- cache profile 针对特定模型、scheduler 和步数标定；改变条件后必须重新评估速度与质量。
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

<h3 align="center">ChituDiffusion: High-Performance Diffusers Inference — Elastic Parallelism · Cache Acceleration · Unified Serving</h3>

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

### 🌐 Elastic Parallel Engine
Startup measurements and end-to-end SLOs drive dynamic CP-lane layouts and
request-state migration at pulse boundaries. `generate` and persistent `serve`
share the same executor.

</td>
<td width="50%">

### ⚡ FlexCache
MagCache, MeanCache, TeaCache, TaylorSeer, and PAB integrate through
request-local hooks with rank-identical CFP/CP control flow.

</td>
</tr>
<tr>
<td width="50%">

### 🚀 High-Performance Context Parallelism
AGKV and USP attention, mixed CFP/CP layouts, and parallel VAE use dynamic lane
process groups to accelerate image and video DiT inference. Static single-node
CP can explicitly select capability-gated Fast Ulysses or Fast AGKV transports;
dynamic lanes retain the NCCL path when those constraints do not hold. Fused
Full-Mesh and Fast Ring remain research benchmarks; see the
[`parallel/fast_cp`](chitu_diffusion/parallel/fast_cp/README.md) constraints and the
[`Fast CP results`](kernels/fast_cp_results.md).

</td>
<td width="50%">

### 🧩 Diffusers-Native Integration
Model adapters preserve upstream pipeline lifecycles. AGKV, USP, CFP/CP, and
parallel VAE compose through a shared executor instead of a second inference
stack.

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

Support is explicit and model-specific. Unsupported cache/model combinations
fail early instead of silently falling back to uncached execution.

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
`uv sync`. Install the optional USP backend with `uv sync --extra usp`.

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
bash script/srun_direct.sh 1 4 chitu_diffusion/examples/wan_epac.py \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --output outputs/wan.mp4
```

Every distributed rank must enter `generate`; only the full-world leader
returns and writes the final output.

### 4. Start Persistent Serving

```bash
chitu serve --stage-config /path/to/stage.yaml
```

All ranks enter `serve`. The stage leader exposes HTTP while follower ranks run
the distributed worker loop. This path explicitly rejects non-`none`
FlexCache strategies.

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
| [`chitu_diffusion/epac/`](chitu_diffusion/epac/) | Model-independent executor, cost model, scheduler, pulse protocol, and workers |
| [`chitu_diffusion/models/`](chitu_diffusion/models/) | Model-family tensor glue and Diffusers adapters |
| [`chitu_diffusion/parallel/`](chitu_diffusion/parallel/) | Process groups, context-parallel attention, and parallel VAE communication |
| [`chitu_diffusion/flexcache/`](chitu_diffusion/flexcache/) | Request-local cache strategies without model-forward branches |
| [`chitu_diffusion/serve/`](chitu_diffusion/serve/) | Configuration, HTTP, and distributed service lifecycle |

See the [runtime guide](chitu_diffusion/README.md), [EPAC design
guide](chitu_diffusion/epac/README.md), and [FlexCache extension
guide](chitu_diffusion/flexcache/README.md).

---

## 🧠 FlexCache

FlexCache is available only for single-request `generate`:

```bash
chitu generate \
  --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --cache-strategy magcache \
  --output outputs/wan-magcache.mp4
```

Mutable state is request-local and serial CFG branches are isolated.
Fresh/reuse decisions use replicated inputs or shared schedules so CFP and CP
ranks enter identical collectives. Official MagCache and MeanCache profiles
fail early outside their calibrated models and step counts.

See the [FlexCache guide](chitu_diffusion/flexcache/README.md) for support
constraints, parallel-safety rules, and strategy integration.

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

1. Keep model-independent behavior in `epac/`, `parallel/`, or `flexcache/`.
2. Keep checkpoint-specific tensor adaptation in `models/<family>/`.
3. Fail explicitly for unsupported capabilities.
4. Add CPU contract tests and record GPU commands and results.
5. Never let rank-local cache signals change distributed control flow.

## ⚠️ Current Limitations

- The project is not production GA and does not guarantee recovery from an
  arbitrary rank failure.
- FlexCache is generate-only; persistent EPE serving rejects cache strategies.
- Cache profiles are calibrated for specific models, schedulers, and step
  counts. Changing these requires a new quality/performance evaluation.
- FLUX.2-klein currently provides a fixed static-CP baseline only.
- Historical ChituBench, DiTango, staged runtime, configurations, tests, and
  results are frozen under
  [`backup/chitu_diffusion_legacy/`](backup/chitu_diffusion_legacy/) and are
  excluded from packages and default tests.

## 📄 License

ChituDiffusion is released under the [MIT License](LICENSE). Model checkpoints
and upstream dependencies remain subject to their own licenses and acceptable
use policies.
