# FreeCache Worklog

更新时间：2026-06-26（归档整理）

> 本文已退化为简短索引。**完整结论看 [STATUS.md](STATUS.md)**。
> 历史日期戳长文档已移到 `archive/legacy_docs/`。

## 2026-06-26：specific→unified 多轮 driver + 归档整理

### 做了什么
- 实现 specific→unified 多轮收敛 driver（`ChituBench/scripts/freecache_dataset_converge.py` + `run_zimage_dataset_converge.sh`），给 `z_image_benchmark.py` 加 per-item `flexcache_params`。
- 跑了两次 Z-Image GPU smoke（默认 / 收紧 trust-region），**均 converged=False**。
- 结论：unified 聚合在当前目标函数（positive gap area）下质量劣于 base，根因是目标函数与图像质量错位（promote 末步等无效 fresh），**非过拟合、非"单 prompt 已够"**。主动暂停，转在线 planner。
- 详见 STATUS.md §3。

### 归档整理（炼丹中止）
- `experiments/freecache` 从 4.7G 清到 ~1.7M：删除所有 `.pt` 向量 / PNG / MP4 / memory snapshot / 运行 log（旧策略产物，重跑 chitu 即可再生）。
- `ChituBench/results/z_image_flexcache` 从 178M 清到 5.3M（同上，只删大文件保留小 CSV/JSON）。
- 关键小产物归档到 `archive/`：Z-Image base/trust B25 policy、Qwen datasetdp_p90 schedule、Flux B25/B17/B15 policy、Z-Image dataset gap-DP iter1、两次失败 smoke 留证。
- 历史日期戳文档移入 `archive/legacy_docs/`，traceplanner/ 只留 CONTEXT.md。

### 未清理（待确认）
- `ChituBench/results/flux1_dev_flexcache`（~386M）、`wan2_1_t2v_1_3b_flexcache`（~82M）的大文件未动——可能服务其它 benchmark，需确认归属后再决定。

## 重启指引
1. 读 `STATUS.md`。
2. 重跑入口在 STATUS §4（不依赖已删的 `.pt`）。
3. 若续 specific→unified 线，先验证 STATUS §3 的目标函数诊断（禁 promote tail，或 gap 按 step 衰减）。
