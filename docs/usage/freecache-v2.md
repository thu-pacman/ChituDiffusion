# FreeCache Preview

FreeCache 通过复用相邻去噪步骤的模型输出，减少生成耗时。Preview 提供 Z-Image、
Qwen-Image 和 FLUX.1-dev 的内置模型预设，无需自行下载或拟合 profile。

## 选择预算

| Fresh 数量 | 用途 |
| --- | --- |
| **25（默认）** | 优先保留画面，建议从这一档开始 |
| 17 | 更快生成 |
| 10 | 快速预览，画面变化更明显 |

`fresh_budget` 支持 **1–50 的任意整数**，不限于上面的三个参考档位。
10 / 17 / 25 保留原有预设；其他预算由内置离线 profile 自动生成 Fresh 调度，
无需重新标定，也不读取当前 prompt 或 seed。所有预算都完成完整的 50-step 采样区间，
只减少完整模型计算的次数，并非提前停止生成。预算 50 为全 Fresh；极低预算可能明显损坏画面。
当前预设适用于默认 sampler、1024²、50 steps。FreeCache 是有损加速，实际速度和画面
变化见 [FlexCache 评测](../features/flexcache.md#freecache-v2)。
评测展示 F10 / F13 / F17 / F25 / F31 五个实测预算；支持其他预算不等于保证其画质，
也不保证每张图的指标随预算严格单调改善。

## 命令行

```bash
chitu generate --model zimage --model-path /path/to/Z-Image \
  --prompt "A porcelain coffee cup on a cafe table, soft morning light." \
  --width 1024 --height 1024 --steps 50 --seed 53 \
  --cache-strategy freecache --freecache-profile zimage:25 \
  --output outputs/coffee-freecache.png
```

例如将 `zimage:25` 改为 `zimage:13` 或 `zimage:31`，即可使用非定档预算。
也可只传 `--cache-strategy freecache --freecache-fresh-budget 13`，自动选择当前模型的
profile；不传预算时为 25。这两个预算参数不能同时使用。

| 模型 | `--model` | `--freecache-profile` |
| --- | --- | --- |
| Z-Image | zimage | zimage:25 |
| Qwen-Image | qwen-image | qwen_image:25 |
| FLUX.1-dev | flux1 | flux1:25 |

## Python API

在已有 pipeline 的请求中传入缓存配置：

```python
from chitu_diffusion import CacheConfig, FreeCacheConfig, ZImageRequest

cache = CacheConfig(
    strategy="freecache",
    params=FreeCacheConfig.preview("zimage", fresh_budget=25),
)
result = pipeline.generate(ZImageRequest(
    prompt="A porcelain coffee cup on a cafe table, soft morning light.",
    width=1024, height=1024, num_steps=50, seed=53, cache=cache,
))
```

完整可执行示例是仓库中的 `examples/generate_zimage_freecache.py`，负责模型加载、
输出保存与资源释放。先检查配置，不加载模型：

```bash
.venv/bin/python examples/generate_zimage_freecache.py --budget 25 --dry-run
```

使用 Slurm 的环境可通过 debug 分区生成：

```bash
sbatch --partition=debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=8 --time=00:10:00 \
  --wrap='.venv/bin/python examples/generate_zimage_freecache.py --model-path /path/to/Z-Image --budget 25 --output outputs/freecache-preview.png'
```

## 使用范围

- 保留 `--steps 50`；通过 profile 的 Fresh 数量调整预算，不能据此外推 sampler 的步数或噪声网格。
- profile 必须与模型匹配；更换 checkpoint、分辨率、CFG 或 sampler 后需要检查画面。
- 当前仅支持单卡 `generate()` 和确定性 `FlowMatchEulerDiscreteScheduler`；不支持多卡并行、其他 solver、随机采样或 EPE `serve`。
- 关闭缓存使用 `--cache-strategy none`；Python API 使用没有默认缓存的 pipeline 与 `CacheConfig()`。
- 本功能为 preview；需要严格保持原始生成结果时，请关闭缓存。
