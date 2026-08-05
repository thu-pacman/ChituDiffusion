# ChituDiffusion

[English](README.md) | [简体中文](README.zh-CN.md)

ChituDiffusion 是一个基于 Diffusers 生命周期的高性能上下文并行扩散推理运行时。它保留
上游 Diffusers 的 tokenizer、文本编码器、scheduler、VAE 和输出处理，只通过统一执行
后端并行化 DiT 去噪计算。

> **项目状态：**开发者预览版。静态生成路径和 EPE 核心运行时已经可用并通过测试，但
> 服务端尚不具备跨 rank 容错能力。生产部署前请阅读[当前限制](#当前限制)。

## 核心特性

- **同一后端、两种生命周期：**`chitu generate` 在 full-world 静态 CP lane 中执行单
  请求；`chitu serve` 使用同一 executor，增加常驻队列和 EPE pulse 调度。
- **弹性并行执行（EPE）：**基于启动实测代价和端到端 SLO，在 pulse 边界动态重组 CP
  lane。
- **Diffusers-native：**通过模型适配器保留上游 pipeline 生命周期，而不是维护第二套
  推理栈。
- **并行 attention 与 decode：**默认使用 AGKV 上下文并行 attention；部分模型支持
  USP 和 tile-parallel VAE。
- **Generate-only FlexCache：**MagCache、MeanCache、TeaCache、TaylorSeer 和 PAB 均为
  请求内隔离策略，并保证各 rank 的缓存控制流一致。

## 模型支持

| 模型 | `generate` | EPE `serve` | FlexCache | 输出 |
| --- | --- | --- | --- | --- |
| Z-Image | 静态 CP / CFP+CP | 支持 | MeanCache、TaylorSeer、PAB | 图片 |
| FLUX.1-dev | 静态 CP | 支持 | MagCache、TeaCache、TaylorSeer、PAB | 图片 |
| Qwen-Image | 静态 CP / CFP+CP | 支持 | MagCache、MeanCache、TaylorSeer、PAB | 图片 |
| Wan 2.1 T2V | 静态 CP / CFP+CP | 支持 | MagCache、TeaCache、TaylorSeer、PAB | MP4 视频 |
| FLUX.2-klein | 静态 CP baseline | 不支持弹性切换 | 尚未验收 | 图片 |

支持范围是显式且与模型相关的。未支持的缓存/模型组合会提前报错，不会静默退化为无缓存
执行。

## 安装

环境要求：

- Linux、NVIDIA GPU，以及可用的 CUDA/PyTorch 环境
- Python 3.12 或更高版本
- 本地或可访问的 Diffusers 格式模型权重
- 多卡执行需要 `torchrun` 或 Slurm

仓库使用 `uv` 管理开发环境：

```bash
git clone <repository-url>
cd ChituDiffusion
uv sync
source .venv/bin/activate
chitu --help
```

运行 `uv sync` 前，请根据集群 CUDA 版本修改 `pyproject.toml` 中的 PyTorch index。可选
attention 后端通过 extra 安装，例如：

```bash
uv sync --extra usp
```

## 快速开始

执行单个请求：

```bash
chitu generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --prompt "a red cube on a white table" \
  --output outputs/zimage.png
```

使用四个本地进程执行 full-world 静态 CP：

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model flux1 \
  --model-path /path/to/FLUX.1-dev \
  --prompt "a lighthouse above a stormy sea" \
  --output outputs/flux1.png
```

在 Slurm 上可以使用仓库提供的 wrapper，模型参数保持不变：

```bash
bash script/srun_direct.sh 1 4 chitu_diffusion/examples/wan_epac.py \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --output outputs/wan.mp4
```

所有分布式 rank 都必须进入 `generate`；只有 full-world leader 返回并保存最终 Diffusers
输出。

从原生 stage 配置启动常驻服务：

```bash
chitu serve --stage-config /path/to/stage.yaml
```

所有 rank 都进入 `serve`；stage leader 提供 HTTP，其他 rank 运行分布式 worker loop。
该路径会显式拒绝非 `none` 的 FlexCache 策略。

## 架构

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

维护中的源码按职责拆分：

- [`chitu_diffusion/epac/`](chitu_diffusion/epac/)：模型无关的 executor、代价模型、
  调度策略、pulse 协议和 worker runtime。
- [`chitu_diffusion/models/`](chitu_diffusion/models/)：模型家族相关 tensor glue 和
  Diffusers adapter。
- [`chitu_diffusion/parallel/`](chitu_diffusion/parallel/)：进程组、上下文并行
  attention 和并行 VAE 通信。
- [`chitu_diffusion/flexcache/`](chitu_diffusion/flexcache/)：请求内缓存策略，不在模型
  forward 中添加策略分支。
- [`chitu_diffusion/serve/`](chitu_diffusion/serve/)：配置、HTTP 和分布式服务生命周期。

实现细节见[运行时说明](chitu_diffusion/README.md)、[EPAC 设计文档
](chitu_diffusion/epac/README.zh-CN.md)和[FlexCache 扩展指南
](chitu_diffusion/flexcache/README.zh-CN.md)。

## FlexCache

FlexCache 仅用于单请求生成：

```bash
chitu generate \
  --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --cache-strategy magcache \
  --output outputs/wan-magcache.mp4
```

策略的可变状态属于单个请求，串行 CFG 分支也会相互隔离。fresh/reuse 判定只使用各 rank
复制的输入或共享日程，保证 CFP/CP rank 进入相同 collective。官方 MagCache 和
MeanCache profile 根据模型与步数选择，超出标定范围会直接报错。完整支持约束和新策略
接入流程见 [FlexCache README](chitu_diffusion/flexcache/README.zh-CN.md)。

可复现的速度/质量文字记录：

- [MagCache 对比](outputs/flexcache/magcache_compare_20260805/result.md)
- [MeanCache 对比](outputs/flexcache/meancache_compare_20260804/result.md)

原始图片、视频和临时 benchmark 数据默认不纳入版本控制。

## 开发与验证

运行当前 CPU 测试和静态检查：

```bash
python -m pytest -q
python -m ruff check chitu_diffusion test
python -m build
```

GPU 正确性必须按模型使用相同 seed 与原生 Diffusers baseline 对比。示例 launcher 和
参数见 [`chitu_diffusion/examples/README.md`](chitu_diffusion/examples/README.md)。

贡献代码时请遵守以下边界：

1. 模型无关行为放在 `epac/`、`parallel/` 或 `flexcache/`。
2. checkpoint 相关 tensor 适配放在 `models/<family>/`。
3. 未支持能力必须显式报错。
4. 增加 CPU contract 测试，并记录 GPU 命令和结果。
5. 禁止用 rank-local 缓存信号改变分布式控制流。

## 当前限制

- 项目尚未达到 production GA，不保证任意 rank 故障后的恢复。
- FlexCache 仅支持 `generate`；常驻 EPE 服务会拒绝缓存策略。
- cache profile 针对特定模型、scheduler 和步数标定；改变这些条件后必须重新评估速度
  和质量。
- FLUX.2-klein 目前仅提供固定静态 CP baseline。
- 历史 ChituBench、DiTango、分阶段 runtime、配置、测试和结果冻结在
  [`backup/chitu_diffusion_legacy/`](backup/chitu_diffusion_legacy/)；该目录不会进入
  wheel，也不会被默认测试收集。

## 许可证

ChituDiffusion 使用 [MIT License](LICENSE)。模型权重和上游依赖仍受各自许可证与使用
政策约束。
