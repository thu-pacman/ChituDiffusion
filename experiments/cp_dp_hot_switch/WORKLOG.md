# CP/DP 热切换 Worklog

## 2026-06-23 请求异构性与 CP/DP 比例

新增观察：

- hot switch 的输入请求不一定同构。图像/视频分辨率不同，会直接反映为 DiT token/patch 序列长不同，进而改变每个请求的单 step 计算量、attention 通信量和显存压力。
- 因此问题不应只被定义为固定的“两卡 CP/SP -> 两路单卡 DP”切换，而应定义为动态资源重分配：不同请求可以拥有不同 CP/SP degree，同时系统整体保留若干 DP replica。
- 长序列请求可能更需要 CP/SP 来降低单 step latency 或满足显存约束；短序列请求更适合用 DP replica 并发处理，从而填补长请求并行执行时的资源碎片。

对问题定义的影响：

- 调度变量从二元模式 `CP/SP` vs `DP` 扩展为比例选择：`每个请求的 cp_degree`、`dp_replica_count`、`每个 replica 上的请求队列`。
- hot switch 的收益需要按请求 shape 分层统计，不能只看整体平均 latency。
- 模拟器需要显式建模分辨率或序列长，否则会低估 shape-aware 调度的价值。

下一步计划：

1. 在离线模拟器中加入请求 shape 字段，例如 `resolution`、`seq_len` 或 `compute_scale`。
2. 为不同 shape 准备逐 step latency 表：单卡、两卡 CP/SP、更多 CP/SP degree。
3. 增加 `shape_aware_ratio` 策略，与 `static_dp`、`static_sp`、`early_hot_switch` 和 `admission_control` 对比。
4. 指标除平均/P95/P99 外，增加不同 shape 请求的 slowdown 和 fairness，避免短请求吞吐提升掩盖长请求尾延迟恶化。

## 2026-07-04 首轮离线模拟跑通

- 延迟表 `traces/profiles.json` 锚定到真实实测（ChituBench `result_parallel_dit.md` +
  fast_cp worklog）：flux1-dev、flux2-klein、Qwen-Image 的 dp1/cp2/cp4 逐步延迟。
- 构造 4 个合成 trace（`single_long / short_burst / mixed_long_short / heavy_burst_4gpu`），
  到达序列合成、延迟锚真实。新增 `run_experiment.py` 遍历 trace × 6 策略并汇总。
- 修复模拟器 `_run_dp_replicas` 的完成态回收 bug（原本丢失 DP 完成请求；`shape_aware_ratio`
  在纯短请求 burst 下死循环）。现改为在函数内直接收集 `completed`。
- 结论（详见 `results.md`）：低负载 CP 单请求 1.71x；高负载 DP 吞吐 1.8x（4 卡）；cp2 在 4 卡
  上利用率仅 0.5。热切换的收益集中在排队 / 尾延迟（mixed 排队 -83%），原始吞吐上"按负载选静态
  layout + admission control"已吃掉大部分收益。

## 2026-07-04 上下文同步 / Handoff

路径提醒：用户口头提到 `experiments/hot_switch`，当前仓库实际目录是
`experiments/cp_dp_hot_switch`。

当前产物：

- 文档：`README.md` 定义 CP/SP -> DP hot switch 问题；`literature_notes.md` 映射
  Shift Parallelism、FLYING SERVING、LoongServe、Gyges、Seesaw、ReMP、AlpaServe 等动态并行
  工作；`simulator_plan.md` 记录离线模拟器输入、策略和指标；`results.md` 汇总首轮结果。
- 代码：`simulate.py` 是无依赖事件模拟器；`run_experiment.py` 负责加载
  `traces/profiles.json` 与 4 个场景 trace，跑 `static_dp/static_cp/admission_control/
  early_hot_switch/shape_aware_ratio/oracle`。
- runtime 脚手架：`serve_config.yaml` 新增 scheduling/hotswitch 配置；
  `scheduler.py` 新增 `ScheduleDecision`、execution group profile、shape metadata 和 hotswitch
  action；`generator.py` 新增 denoise step interleave 与 `prepare_hotswitch`；
  `hotswitch.py` 提供 CP latent materialize/shard primitives；`batching.py` 提供按 shape/step
  聚合的连续 batching metadata。

已验证：

- `python3 experiments/cp_dp_hot_switch/run_experiment.py --output-dir experiments/cp_dp_hot_switch/out`
  可复现 `out/summary.md` 与 `out/summary.json`；`out/` 是生成物，不进入提交。
- `python3 -m py_compile experiments/cp_dp_hot_switch/simulate.py experiments/cp_dp_hot_switch/run_experiment.py chitu_diffusion/runtime/batching.py chitu_diffusion/runtime/hotswitch.py chitu_diffusion/runtime/scheduler.py`
  通过。

关键结论记忆：

- 低负载单重请求：CP2 单请求延迟 80.935s，对比 DP1 138.820s，约 1.71x。
- 同构高负载 burst：4 卡 `static_dp` 吞吐 0.087 req/s，对比固定 `static_cp` 0.048 req/s，
  约 +81%；固定 cp2 在 4 卡上只吃到约 0.50 GPU busy。
- 中等并发 `short_burst` 是临界区：flux1 的 CP2 加速比接近 DP 并发度，DP 吞吐略好，CP 均值略好。
- 热切换目前主要改善排队/尾延迟，不明显超过最优静态吞吐；优先级应是 admission-time
  layout 选择，其次才是 step-boundary in-flight switch。

重要风险 / 下一步：

- `mixed_long_short` 中 `static_dp` 的 `gpu_busy_ratio=1.253` 超过 1，说明 DP 模拟分支在异构
  step latency 下会过度推进 GPU slot：`_run_dp_replicas` 返回最早结束时间，但没有跟踪仍在运行的
  replica。引用异构结果前应先把 DP 执行改成每个 GPU slot 独立事件队列；同构 burst 的数值相对不受影响。
- 模拟器还没建模 cp4 组、多 CP 组、step 方差、VAE/TextEncode、真实 QPS/泊松 sweep 和 batching。
- runtime 代码仍是接口/状态迁移原型，不等价于已支持在线 execution group reroute；下一步应先收敛
  scheduler decision 的可观测性与 simulator bug，再决定是否做真实多 worker 热切换。

## 2026-07-05 M1 解耦成本模型完成（worker roofline + 通信）

- 完成吞吐连续 batching 引擎计划的 M1：解耦成本模型（compute 项 + comm 项按 step 组合）。
  报告见 [`m1_cost_model_report.md`](m1_cost_model_report.md)，标定数据见
  [`cost_model.json`](cost_model.json)（15 点 compute 网格 + comm），单测
  `test/test_cost_model.py` 全通过。模拟器 `simulate.py` 的 `default_profiles()` 已改为消费该模型。
- 数据（H20, bf16, Z-Image，flash_attn）：B=2≈1.96×、B=4≈3.9× 近线性 → 该模型 **B=1 已 compute-bound**，
  `batch_saturation` 拐点 = 1，同形状 batching 在实测区间内无免费吞吐。held-out 插值误差
  max 4.82% / mean 2.56%（< 10% 门限，PASS）。
- comm 标定（4×H20 NVLink，Ulysses all_to_all_single）effective `nvlink_gbps = 123.4`，
  单节点 comm ≪ compute（1536² 时 <4ms/step，<2%）；SP 是有效延迟杠杆（1024² k=4 → 188ms，约 3.2×）。
- 对计划的启示：主吞吐杠杆是 **SP degree 右尺寸化**（避免过度切分）与 **FlexCache 减步数**，
  而非同形状 batching；据此收敛 M7 贪心策略与 M3 前提。
- 注意：SP>1 延迟由 SP=1 网格 + 标定 comm **组合而来**，尚未对 live cp2/cp4 Z-Image 交叉验证（下一步 spot-check）。

## 2026-07-06 再同步 / 脚手架审计

新增进展：

- 7 月 5 日已经不只是离线模拟：新增 `session_summary_2026-07-05.md`，并完成/记录 M1 成本模型、
  M2 work-item、M3/M4 continuous/mixed-step batching、M6 dynamic SP、Poisson serve trace、
  non-denoise stage profiling 等机制层工作。
- M4 结论：mixed-step batching 可行；per-row timestep 数学 bitwise 正确。batched vs solo 的差异来自
  Z-Image 既有的 batch-size fp reduction-order 非确定性，不是混步耦合 bug。收益主要是 step-boundary
  admission / queue delay，而非 compute-bound H20 上的吞吐。
- M6 结论：dynamic SP degree 切换可行；pre-warmed groups 后切换是 O(1) active CP group 指针切换，
  Z-Image replicated latent 下 0 bytes 迁移，报告测得约 17us，fixed degree / mid-run switch bitwise。
- stage profiling 结论：text encode / VAE decode 在 H20 上也偏 compute-bound；batch 非去噪阶段不是好吞吐杠杆。
  image save 是 CPU/disk 开销，适合 async overlap。

本次验证：

- 使用主仓库 venv：
  `/home/chenyy/WORK/cyy/ChituDiffusion/.venv/bin/python -m pytest test/test_cost_model.py test/test_sp_migration.py test/test_continuous_batch_sched.py test/test_work_item.py -q`
  通过，31 passed。
- `py_compile` 覆盖 `simulate.py/run_experiment.py/gen_client_trace.py/profile_worker.py/profile_stages.py`
  以及 `cost_model.py/data_parallel.py/hotswitch.py/sp_migration.py/scheduler.py/task.py` 通过。
- `python3 experiments/cp_dp_hot_switch/run_experiment.py --output-dir experiments/cp_dp_hot_switch/out`
  可复现旧 sweep；修复前 `mixed_long_short static_dp gpu_busy=1.253` 仍存在，说明旧 DP slot 模拟 bug
  需要处理（见下节已修复）。

脚手架是否足够完善：

- **机制层基本足够**：成本估计、work-item 抽象、mixed-step admission、dynamic SP actuator、serve trace
  harness、stage profiling 都已经各自能跑，并且有轻量单测或报告支撑。
- **闭环控制层还不够**：现在机制主要由 static config flag / env var / test driver 驱动；没有一个统一的
  scheduler action schema 把 `{work-items, batch group, SP degree, DP replica, cache policy, predicted cost, SLO}`
  串成一次可观测 decision。
- `ScheduleDecision` 仍偏 metadata，不表达 M7 真正需要的动作：目标 `sp_degree`、batch/work-item 集合、
  cache/FlexCache level、cost model 估计、deadline/SLO slack、执行后 metrics 回填。
- M4 continuous-batch engine 在 `generator._cb_round()` 内自己做 admission，绕开了 `schedule_decisions()`；
  M6 dynamic SP 由 `CHITU_SP_INITIAL/CHITU_SP_SWITCH_STEP/CHITU_SP_TARGET` 控制。两者尚未统一到 scheduler
  的 policy decision。
- runtime 的 DP helper 当前主要是 `n_sample` seed-slice DP，不等价于 serving 里“多个请求分配到多个 DP
  replica”的 request-level DP；CP->DP hot switch 的 replica routing 仍未实现。
- 配置面需要收敛：`serve_config.yaml` 与 scheduler 使用了 `scheduling_policy / step_interleave /
  execution_profiles / hotswitch_enabled / switch_allowed_until_step`，但 `ServeConfig.DiffusionConfig` 目前只
  显式增加了 `continuous_batch / max_batch_items / dynamic_sp`。loader 现在是非 structured YAML，不一定立刻报错，
  但正式脚手架应把这些字段补进 schema 和验证。
- 离线模拟器分叉：`simulate.py default_profiles()` 已能从 `cost_model.json` 派生 profile，但
  `run_experiment.py` 默认仍加载 `traces/profiles.json` 的旧 Flux/Qwen 实测 profile；且 `_run_dp_replicas`
  还没有 per-GPU slot 状态。因此旧 `results.md` 可作方向参考，不能作 M7 策略的精确闭环依据。

下一步计划：

1. **先修离线闭环**：把 `_run_dp_replicas` 改成 per-GPU slot 事件模型，保证 `gpu_busy<=1`；同时让
   `run_experiment.py` 可选择 `cost_model.json` 派生 Z-Image profiles，并增加 load-aware M7 策略的模拟版。
2. **定义统一 action schema**：新增/扩展 `ScheduleDecision` 为 `SchedulingPlan`，显式包含 work-item ids、
   target SP degree、batch group、replica/group、cache action、predicted step/request cost、deadline slack、policy reason。
3. **让 generator 只执行 plan**：把 M4 admission 和 M6 env-driven SP switch 收口到 scheduler plan；
   dynamic SP 应先作为 group-level action（同一 batched forward 只能有一个 active CP group），再考虑更细粒度。
4. **补 multi-GPU serve harness**：解决 world_size>1 下到达间 idle-sync / rank barrier 问题，打通
   Poisson trace -> scheduler plan -> batch/SP execution -> `serve_metrics.json` 的真实闭环。
5. **再做真实 GPU 对比**：先用 static baselines（SP1/SP2/SP4、continuous_batch on/off）校准 cost model；
   再跑 M7 策略，比较 SLO attainment、queue delay、mean/p95、GPU busy，而不是只看吞吐。
6. **FlexCache 先模拟再实现**：在 simulator 中加入质量/步数/SLO tradeoff，验证 “FlexCache 作为调度动作”
   是否真的 1+1>2；若只是 always-cache 最优，就不要把它做进 M7 runtime。
7. **落地顺序**：优先提交 M1-M6 机制层和报告，再单独提交 M7 scheduler/serve-harness 闭环，避免把
   已验证机制和策略实验揉成一个难 review 的大改。

## 2026-07-06 DP slot 事件模型修复

- 修复 `simulate.py` 中 DP 分支的资源模型：新增持久 `DpRunningStep` / `dp_slots`，每个 GPU slot 持有
  正在运行的 step，只有 step 完成后才推进该 request 的 `current_step` 并重新入队。这样同一 request
  不会在前一 step 尚未完成时被再次调度。
- `_run_dp_replicas` 现在先 harvest 已完成 slot，再填充空闲 slot，并返回下一次 completion event；
  `simulate()` 主循环在存在 running DP slot 时优先推进 DP event，避免同时启动 CP 与 DP。
- 新增 `test/test_cp_dp_simulator.py`，覆盖异构 DP trace 的两个不变量：`gpu_busy_ratio <= 1`，以及同一
  request 的 step trace 不重叠。
- 验证：
  `/home/chenyy/WORK/cyy/ChituDiffusion/.venv/bin/python -m pytest test/test_cp_dp_simulator.py test/test_cost_model.py test/test_sp_migration.py test/test_continuous_batch_sched.py test/test_work_item.py -q`
  通过，32 passed。
- 重跑 `run_experiment.py` 后，`mixed_long_short static_dp gpu_busy` 从 1.253 修正为 **0.939**；
  该场景的 static_dp 吞吐从旧的 0.043 修正为 **0.032 req/s**，vs static_cp 0.030 约 +7%。
  因此旧版 `results.md` 中 mixed 场景的 `+43%` 已废弃并更新。
- 语义变化：`early_hot_switch/oracle` 的 switch count 从多次重复切换收敛为 1 次。修复后，一旦进入
  DP slot 执行，会持续按 slot completion 事件推进到当前 DP 工作耗尽，更接近“一次 CP->DP layout
  transition 后运行 DP replicas”的语义。

## 2026-07-06 静态 DP/SP baseline

- 新增 `run_static_baselines.py`，对现有抽象 trace 与 `traces/serve/*.json` 跑固定部署 baseline：
  `dp`=每张 GPU 一个 1-GPU replica；`sp2`=每个请求占 2 GPU；`sp4`=每个请求占 4 GPU。
  每个 group FIFO 跑完整个 denoise phase，不做 hot switch / continuous batching / FlexCache。
- 输出写入 `out/static_baseline.md` 与 `out/static_baseline.json`（生成物，不进入提交）。当前 trace 都没有显式 deadline，
  所以报告用固定 latency SLO 阈值 `10/30/60/120/180s` 的 attainment，并同时给 throughput、mean/p95/p99、
  queue delay、GPU busy。
- 注意口径：这个 baseline 的 `sp2` 在 4-GPU trace 上会使用两个 2-GPU SP groups，是理想静态部署；
  `run_experiment.py` 里的 `static_cp` 仍是单个 2-GPU CP group（4 GPU 时有意留下半节点空闲），两者不能直接同名比较。
- 关键观察：
  - 抽象 `single_long`：SP2 明显更好，80.9s vs DP 138.8s，120s SLO 下 SP2=100%、DP=0%。
  - 抽象 burst：`short_burst` 中 DP/SP2 接近；`heavy_burst_4gpu` 若允许两个 SP2 groups，SP2 吞吐略高于 DP
    （0.096 vs 0.088 req/s），但 queue 更高。SP4 单 group 与 DP 吞吐接近但排队更重。
  - serve 1024 trace（4 GPU, Z-Image cost model）：DP 吞吐最高约 0.330 req/s；SP2/4 降低单请求 service
    time 但 group 数减少，整体吞吐略低。30s SLO 下 SP2 比 DP 高（83% vs 67%），60s 以上全满足。
  - serve 512 trace：SP 对 512 没有 latency 收益（cost model 在小 shape clamp），DP 明显最好；
    30s SLO 下 DP/SP2=100%，SP4=50%。
  - serve mixed trace：DP 在吞吐和 SLO 上最好（60s SLO 92%），SP2=75%，SP4=46%。这说明在当前
    Z-Image/H20 cost model 下，高负载 mixed serving 的静态默认应优先 DP，SP 作为低负载/紧 SLO latency lever。
- 为解释每个 case 的执行形态，`run_static_baselines.py` 现在还会输出 GPU-time timeline SVG：
  `out/static_baseline_timelines/*.svg`，索引见 `out/static_baseline_timelines.md`。纵轴是 GPU lane，
  横轴是 time since first arrival，填色块表示该 GPU 正在执行 denoise；空白表示 idle 或请求仍在排队。
- 2026-07-06 追加 request-level timeline：每张 SVG 下半部分按请求画 arrival → queue → compute → finish。
  arrival 是空心点/竖线，queue 是灰色线段，compute 是实色条，finish 是黑点；与上半部分共享同一时间轴，
  用来解释 SLO miss 是来自排队还是单请求 service time。

## 2026-07-06 Trace 到达率调整

- 现有 `serve/poisson_1024_r3n12`、`serve/poisson_512_r6n16`、`serve/poisson_mixed` 更像压力测试：
  到达率接近或超过当前 4-GPU cost model 的固定部署容量，timeline 中大部分请求会在前一两个请求执行期间
  全部到达并排队，GPU lane 很快被打满，难以观察 hot-switch / dynamic SP 的机会窗口。
- 新增低/中到达率 Poisson trace：
  `poisson_1024_r008n12`、`poisson_1024_r016n16`、`poisson_mixed_r018n20`、
  `poisson_mixed_r030n20`。这些 trace 覆盖 idle-rich、medium-load、near-capacity mixed 三种区间。
- 新增 `gen_structured_trace.py` 和 `traces/serve/staggered_mixed.json`。该 trace 使用固定 warmup +
  short burst + cooldown 到达序列，15 个请求跨 137s，包含 1024/512 与少量 `n_sample=2`，避免随机 Poisson
  把所有请求过早堆进队列。
- 追加 `realistic_sd3_mixed` preset 和 `traces/serve/realistic_sd3_mixed.json`：使用 SD3-like 尺寸组合
  128/256/512/1024/1536，26 请求跨 468s，间隔从 2s 到 39s 不等，包含两段局部 burst。它比
  `sd3_trace_stride300_n60` 更适合作为默认调度评估 trace，因为到达过程不是固定间隔。
- 重跑 `run_static_baselines.py` 后，`out/static_baseline.md` 与 `out/static_baseline_timelines.md` 已包含新 trace。
  关键读数：`serve/staggered_mixed` 在 30s SLO 下三种静态策略都 100%，但 10s SLO 下
  DP=47%、SP2=93%、SP4=93%；busy 分别为 0.249/0.322/0.474，说明它更适合观察低负载下 SP 降低 latency、
  中间 burst 下 DP replica count 避免排队的 tradeoff。
  `serve/realistic_sd3_mixed` 的关键读数：DP mean/p95=30.3/75.9s，60s SLO=77%，busy=0.351；
  SP2 mean/p95=19.9/38.7s，60s SLO=100%，busy=0.432；SP4 mean/p95=20.5/43.1s，60s SLO=100%，busy=0.604。

## 2026-07-06 SD3 trace 可用性检查

- 检查 `traces/sd3_trace.txt`：45,000 行，每行一个 `Request(...)`；字段包含 `request_id/timestamp/
  height/width/num_frames/prompt/negative_prompt/num_inference_steps`。解析无坏行，timestamp 单调递增。
- 原始 trace 到达率是固定约 30 req/s（间隔 0.033333s），持续 1499.95s；尺寸为
  128/256/512/1024/1536 各 9000 个请求，全部 `num_frames=1`、`num_inference_steps=50`。原样对当前
  4-GPU offline baseline 过载太严重，不适合直接用于看优化机会。
- 新增 `convert_sd3_trace.py`，把 `Request(...)` 文本 trace 转成本地 `traces/serve/*.json` 格式，并支持
  `--stride / --time-scale / --start-s / --duration-s / --max-requests` 控制负载。
- 生成可用样本：
  `python3 experiments/cp_dp_hot_switch/convert_sd3_trace.py --input experiments/cp_dp_hot_switch/traces/sd3_trace.txt --out experiments/cp_dp_hot_switch/traces/serve/sd3_trace_stride300_n60.json --stride 300 --max-requests 60`
  得到约 0.102 req/s、60 请求、589.99s span 的 mixed trace。注意：由于原始 trace 是固定 30 req/s，
  简单 stride 后的 inter-arrival 仍是完全固定的 9.9999s；它只适合作为外部 trace 格式接入验证，不应作为
  主要调度评估 trace。
- 用 `run_static_baselines.py --abstract-traces --serve-traces traces/serve/sd3_trace_stride300_n60.json`
  跑通，输出在 `out/sd3_check_only/`。关键结果：DP mean/p95=34.4/76.2s，60s SLO=73%，busy=0.704；
  SP2 mean/p95=22.9/44.5s，60s SLO=100%，busy=0.829；SP4 mean/p95=52.5/111.8s，60s SLO=62%，busy=0.992。
  该样本保留了真实 prompt/shape mix，但到达过程不真实。后续策略验证应优先使用
  `realistic_sd3_mixed` 或进一步从真实服务日志拟合 non-homogeneous Poisson / burst process。

## 2026-07-06/07 M7 N-GPU pool 引擎 + elastic_hot_switch（历史，报告已折叠）

- 新增 `simulate.simulate_pool()`：真正的 N-GPU 事件模拟器。每个 in-flight 请求持有 width
  `k∈{1,2,4}`（SP degree，映射 `dp1/cp2/cp4`），占用 k 个 GPU index，全程保持 `Σwidths ≤ N`；
  只在 denoise step 边界 / arrival 决策，只在 switch window 内换 width 并付 `SwitchCostModel.total_ms`；
  记录 GPU-index 级 `PoolSegment` trace 驱动 timeline。旧 2-GPU `simulate()` 保留作 regression。
- `elastic_hot_switch` 策略：keep-by-default + shape-aware initial degree（4090 上 512²→k=1、1024²→k=4）
  + 填充 idle GPU + 空闲 upshift + **pairwise 完成时间 benefit test 驱动的 burst downshift**。
- 结论：CP/DP 热切换收益 **真实但 regime-specific 且中等**——mixed/bursty 上 **+9~23% p95**、
  吞吐持平或更好；homogeneous / SP-negative 上正确 no-op。收益主要来自“不让大 SP 请求 head-of-line
  阻塞小 DP-friendly 请求”。代表数据（4090）：`poisson_mixed_r030` elastic p95 9.5s vs best-static 11.8s、
  吞吐 0.312；`realistic_sd3_mixed` elastic mean/p95 13.0/31.0s 优于所有 static extreme（仅 2 次 switch）。
- **唯一 Pareto loss（M8 已修复）**：`poisson_1024_r3`（饱和同构 1024²）elastic=shape_aware=SP4，
  但 best static 是 DP——4 路 DP 并发在吞吐（0.362 vs 0.286）和 p95（29.8 vs 36.5s）都更好，因为 SP4
  加速比 sub-linear（3.1×<4×）、队列延迟主导。M7 的 shape-aware degree 是 load-oblivious 的。

## 2026-07-07 M8 SLO-aware `slo_elastic` 调度器 + FlexCache（当前主线）

- 依据 [`slo_elastic_scheduler_strategy.md`](slo_elastic_scheduler_strategy.md) 实现，与 M7 并存不替换。
  在 `simulate_pool` 内新增 `slo_elastic` policy：**合法 GPU layout 枚举 → rolling-horizon 前向模拟打分
  → 字典序 SLO-first 目标择优**，外加 **FlexCache 紧急闸门**（simulator-only，硬门控）。
- 字典序目标（越前越优）：`slo_miss → max_tardiness → total_tardiness → max_slowdown → starvation →
  flexcache_used → flexcache_steps → switch_count → switch_cost → total_flow → mean_slowdown`。
  switch_count 之前的连续字段做量化（tardiness 500ms / max_slowdown 0.25），只有“有意义”的收益才值得 switch；
  末位用 total_flow（≈mean latency）而非 GPU-busy（后者会病态偏好宽 SP packing）。前向 fallback 是 load-aware：
  队列深就 DP 并发、空闲就 SP —— 这让目标函数“看见”饱和同构下 DP 更优。
- 结果（4-GPU, slo_factor=4；vs M7 elastic）：
  - **修复 M7 Pareto loss**：`poisson_1024_r3n12` p95 36.5→29.9s（4090）/ 39.6→33.1s（H20），吞吐 +25%/+23%，
    max_slowdown 3.52→2.75；timeline 里明确从“串行 SP4”变成“4 路 DP 并发”。
  - **公平性大幅改善**：`poisson_mixed` max_slowdown 10.74→6.00 且少 2 次 miss；`realistic_sd3` 2.50→1.89；
    H20 `poisson_mixed_r018` 1.54→1.01。SP-negative / 简单同构（512、r008、staggered）为正确 no-op / 平手。
  - **诚实代价**：`poisson_mixed_r030` p95 有小幅退让（9.5→12.5s），但 SLO 仍 0-miss、公平性持平——
    符合策略 SLO≫p95 的优先级，不是 SLO 退化。
- **FlexCache**（sensitivity study，非保质量生产策略）：硬门控，仅在正常 layout 仍存在极端 SLO 风险时开启，
  取最小 step reduction。impossible-SLO 压测（slo_factor=1.5, r3n12, min_steps=12）：p95 36.5→18.3s、
  max_tardiness 33→13s、miss 11→10；SLO 宽松时即便 enable 也 0 使用（有测试守护）。
- **Burst/idle trace**：`gen_client_trace.py` 新增 `--burst-sizes`/`--lull-ms` 分相到达模式，
  生成 `traces/serve/bursty_idle_mixed.json`（solo→burst→solo→burst）。它清晰展示“空闲→SP4、突发→DP 并发+公平、
  突发尾部 upshift 回 SP”：max_slowdown 1.52 vs elastic 5.62 / SP4 9.00，p95/tardiness 均最优。
- 测试：`test/test_slo_scheduler.py`（策略 §12 九个必测场景）+ `test/test_serve_sim.py`，全绿；
  `test/test_cost_model.py`、`test/test_cp_dp_simulator.py` 无回归。

## 2026-07-07 整理与清理（准备接入 runtime）

- **确定当前主线（canonical）**：设计 = [`slo_elastic_scheduler_strategy.md`](slo_elastic_scheduler_strategy.md)，
  结果 = [`m8_slo_elastic_report.md`](m8_slo_elastic_report.md)（图表作为本地生成物保留在 gitignored `figures/`，不入库）。
  核心代码收敛为 `simulate.py`（引擎 + `slo_elastic`）、`run_serve_policies.py`（自包含 runner+可视化）、
  `gen_client_trace.py`（trace 生成，含 bursty 模式）；成本模型 `cost_models/{rtx4090,h20}.json` + 传统默认 `cost_model.json` 均为本地生成/拷贝的数据文件，不入库。
  runtime 机制参考保留：`m1_cost_model_report.md`（成本模型）、`m6_dynamic_sp_report.md`（step 边界 SP 切换机制）、
  `profile_worker.py` / `profile_stages.py`（真机标定）。
- **删除的冗余文件**（结论已并入本 WORKLOG，故安全删除）：
  - 文档：`results.md`（首轮抽象模拟结果，已被 M7/M8 取代）、`simulator_plan.md`（早期计划）、
    `session_summary_2026-07-05.md`（handoff）、`m4_mixed_step_report.md`、`m7_elastic_hot_switch_report.md`、
    `stage_profile_report.md`。
  - 代码：`run_experiment.py`（旧抽象 sweep，被 `run_serve_policies` + `simulate_pool` 取代）、
    `run_static_baselines.py`（静态 baseline 已由 pool 引擎的 `static_*` policy 覆盖；其 3 个可视化 helper
    `_safe_name/_nice_tick_seconds/_shape_color` 已内联进 `run_serve_policies.py`）、
    `gen_structured_trace.py`、`convert_sd3_trace.py`（一次性 trace 转换器）。
  - 数据：`out/` 下体量大的 SVG dump、`figures/` PNG、trace JSON、cost model JSON 都可由 runner/profile/trace generator 重新生成，统一 gitignore。
- **下一步（runtime 接入）**：把 `slo_elastic` 的决策语义映射到 runtime 的统一 `SchedulingPlan`——
  每次决策输出 `{work-item ids, 每请求 target sp_degree/width, GPU 分配, 可选 switch, 可选 FlexCache 动作,
  predicted cost, deadline slack, deciding reason}`；generator 只执行 plan（收口 M4 admission 与 M6 env-driven SP 切换）；
  补齐 request-level DP replica routing 与 world_size>1 的 idle-sync；先用 static baselines 校准 cost model，
  再上线策略并回填 SLO/queue/p95/busy 指标。

## 2026-07-08 `slo_elastic` runtime pool engine + 三臂 benchmark

- **runtime 接入完成**：`slo_elastic` 不再只是 simulator policy。`DiffusionScheduler.plan_pool_round`
  把 live `DiffusionTask` 映射成 policy 层请求状态，调用共享的 `plan_pool_layout`，输出 `SchedulingPlan`
  / `LaneAssignment`；`Generator` 的 pool engine 在 rank 0 规划 lane layout 并广播，worker rank 在各自
  lane subgroup 上执行 denoise phase。
- **同 engine baseline**：新增 `pure_dp` / `pure_sp` 两个固定布局策略，复用完全相同的 pool engine
  admission、per-lane denoise、barrier、latent migration 和 retire 逻辑；差异只剩 layout decision，
  便于和 `slo_elastic` 做干净对比。
- **runtime overhead 收敛**：在最初 7/7 三臂结果里，`slo_elastic` 因为 planner 只建模 compute+comm，
  低估了同步 round 的固定开销，burst 阶段偏向 SP2，导致 aggregate 落后于 `pure_dp`。后续 runtime
  增加 multi-step phase、placement affinity、targeted latent migration、precise migration、async output，
  并给 planner 增加 residual overhead 入口（`CHITU_POOL_PER_STEP_COST_MS` /
  `CHITU_POOL_BOUNDARY_COST_MS`）。
- **当前读数口径**：`outputs/bench_3way_50step/comparison/report.md` 是 7/7 三臂同版本结果：
  `pure_sp` makespan/p95 = 155.6/115.9s，`pure_dp` = 144.9/109.5s，`slo_elastic` = 178.0/141.7s；
  该结果明确指出 per-round overhead 是主要 runtime gap。7/8 最新 `slo_elastic` 单臂结果
  `slo_elastic-20260708_170254-ad2a7d03` 已改善到 makespan/p95 = 138.3/102.1s、throughput 0.065 req/s，
  优于旧 pure baselines；但严格结论仍需用当前代码重跑 `pure_dp` / `pure_sp`，避免跨版本比较。
