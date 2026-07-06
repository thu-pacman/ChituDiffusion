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
