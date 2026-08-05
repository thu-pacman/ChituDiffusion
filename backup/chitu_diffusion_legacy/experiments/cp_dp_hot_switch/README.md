# CP/DP Hot Switch

本目录维护 diffusion serving 的动态并行实验：低负载时用 SP/CP 降低单请求延迟，突发和排队时在 denoise phase 边界切成更多 request-level DP lane。当前仍是内部优化，不作为公开稳定特性。

只维护三类内容：

- [`HOT_SWITCH_README.md`](HOT_SWITCH_README.md)：当前实现、模型适配状态和使用方法。
- [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md)：当前版本的关键实验结论、限制和复现命令。
- 可复现源码：`simulate.py`、`run_serve_policies.py`、`gen_client_trace.py`、`profile_worker.py`、`profile_stages.py` 和 `bench_3way_analyze.py`。

历史 worklog、里程碑报告、研究笔记、旧 sweep/绘图脚本不再维护。trace、cost model、图表、JSON 和运行输出均为本地生成物，由 `.gitignore` 排除。

## 快速验证

```bash
cd experiments/cp_dp_hot_switch
python3 gen_client_trace.py \
  --burst-sizes 1,4,1 --intra-burst-ms 70 --lull-ms 12000 \
  --sizes 512x512,1024x1024 --steps 20 --slo-factor 4 \
  --cost-model rtx4090 --out traces/serve/smoke.json
python3 run_serve_policies.py --cost-model rtx4090 \
  --serve-traces traces/serve/smoke.json --output-dir out/latest
python3 -m pytest \
  ../../test/test_slo_scheduler.py ../../test/test_serve_sim.py \
  ../../test/test_cp_dp_simulator.py ../../test/test_cost_model.py -q
```

当前下一步只有一个：在同一代码、trace、checkpoint 和采样配置下重跑 Z-Image 的 `pure_dp`、`pure_sp`、`slo_elastic` 三臂 runtime benchmark，并用结果覆盖现有报告中的跨版本读数。
