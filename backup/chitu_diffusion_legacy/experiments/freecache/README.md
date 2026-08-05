# FreeCache 探索实验

Train-free、step-level 的速度场缓存策略，目标是在 Qwen-Image 上超过官方 MeanCache（硬编码 fresh-step 表 + JVP）。

## 核心思路

| 组件 | MeanCache | FreeCache |
|------|-----------|-----------|
| 缓存粒度 | step（整模型 `noise_pred`） | 同左 |
| 外推变量 | 真实 `sigma` | 同左 |
| 外推方法 | 一阶 JVP（平均速度差） | **同 MeanCache JVP**（一步 Δσ 修正） |
| Fresh 调度 | `_FRESH_STEP_TABLE` 魔法表 | JVP 漂移累积 `<= tol`（train-free） |
| 可调参数 | `fresh_steps` + 表 | `tol` + `max_gap` |

实现代码：

- `chitu_diffusion/flexcache/freecache_core.py` — 纯数值控制器（可 CPU 单测）
- `chitu_diffusion/flexcache/strategy/freecache.py` — FlexCache 策略封装

## 实验矩阵（Qwen-Image, 50 step, 1328×1328, CFG parallel=2）

对照官方 MeanCache（`ChituBench/result_flexcache.md`）：

| case | 说明 |
|------|------|
| `torch_sdpa` | 无缓存 baseline（symlink 复用已有 run） |
| `qwen_meancache25_50_cfp2` | 官方 MeanCache fresh=25 |
| `qwen_meancache17_50_cfp2` | 官方 MeanCache fresh=17 |
| `qwen_freecache_tol010` | FreeCache JVP + 漂移调度：`tol=0.10` |
| `qwen_freecache_tol015` | FreeCache JVP + 漂移调度：`tol=0.15`（默认） |
| `qwen_freecache_tol020` | FreeCache JVP + 漂移调度：`tol=0.20` |
| `qwen_freecache_tol030` | FreeCache JVP + 漂移调度：`tol=0.30` |

### 待验证假设

1. **自适应 fresh**：`tol` 驱动的误差控制能否在相近 speedup 下获得更高 PSNR/1-LPIPS？
2. **自适应阶数**：二阶外推是否在轨迹高曲率区（首尾）减少误差，且开销可忽略？
3. **与 MeanCache10 对比**：官方 `fresh_steps=10` 质量崩溃（PSNR≈10）；FreeCache 应在相近加速比下保持可用质量。

## 运行方式

GPU 通过 `chitu run` → `script/srun_direct.sh` 申请（与现有 ChituBench 一致）。

### Smoke（4 step, 1 seed, 1 GPU）

```bash
./experiments/freecache/run_smoke.sh
```

### 完整 benchmark

```bash
# 复用已有 torch_sdpa baseline（推荐）
export CHITUBENCH_REFERENCE_DIR=ChituBench/results/qwen_image_attention/qwen_image_attn_50step_20260615_1550/chitubench-qwen-image-attn-torch-sdpa-20260615_154538-torch_sdpa

# 只跑部分 case
export CHITUBENCH_CASES="qwen_freecache_tol005_o2 qwen_meancache25_50_cfp2"

./experiments/freecache/run_bench.sh
```

环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `SRUN_PARTITION` | `debug` | Slurm 分区 |
| `CHITUBENCH_GPUS_PER_NODE` | `2` | GPU 数（与官方 cfp2 一致） |
| `CHITUBENCH_CFP` | `2` | CFG parallel |
| `CHITUBENCH_STEPS` | `50` | 采样步数 |
| `CHITUBENCH_REFERENCE_DIR` | 空 | baseline run 目录（symlink 复用） |

### CPU 单元测试（无需 GPU）

```bash
.venv/bin/python -m pytest test/test_freecache_core.py -q
```

## 产物

每次 run 在 `output.root_dir` 下生成：

- `flexcache_freecache_step_trace.json` — 逐步 fresh/reuse 决策、阶数、预测误差
- ChituBench 标准：`summary.json`、`quality/`、`plots/`

结果目录：`ChituBench/results/qwen_image_freecache/<run_id>/`

## 成功标准（相对官方 MeanCache）

在 **相近 DiT speedup**（±15%）下：

- PSNR ≥ MeanCache 对应点
- 1-LPIPS ≥ MeanCache 对应点
- HPSv3 不显著下降

若 speedup 更高且质量仍优于 MeanCache，记为超越。
