# 工作记录：PCIe 上的并行优化（Z-Image 序列并行）

> 第二篇知乎文章「在 PCIe 上的并行优化」的里程碑 + key insight 沉淀。
> 与第一篇（NVLink 上的 CUDA Graph + compile）不同，本篇主战场是 **PCIe 互联**：
> 卡间带宽远低于 NVLink、且本机无 GPUDirect P2P（全走 host 中转），通信极易成为瓶颈。
> 因此"通信 vs 计算"的精确拆分是一切优化的前提。
>
> 平台：4× RTX 4090，PCIe 互联（GPU0-1 / GPU2-3 各为一个 NUMA，跨 NUMA 走 SYS）。
> 负载：Z-Image 文生图，DiT 主干迭代去噪，single-stream 打包序列 `[image, text]`，1024²。
> 并行：序列并行（context parallel, CP），对标 qwen_image 的 split-Q / all-gather-KV，适配 Z-Image。

> Fast CP / Hybrid Context Parallel 总纲见 [README.md](README.md)。
> 2026-07-01 的 NVLink async overlap、Ulysses all-to-all、compile/graph、txt+img joint
> 阶段结论见 [milestones/2026-07-01.md](milestones/2026-07-01.md)。

---

## 1. 方法论：先把"通信 vs 计算"拆开

PCIe 上做并行优化，第一步不是写 kernel，而是**量化通信占比**。

- **细粒度 profiling 分桶**（`CHITU_Z_IMAGE_CP_PROFILE=1`，CUDA event 同步，关闭零开销）：
  `qkv`（投影 GEMM）/ `attn`（flash 注意力）为计算；`kv`（AGKV all-gather）/ `a2a`（Ulysses all-to-all）/
  `ring`（Ring P2P）/ `out`（输出 all-gather）为通信。
- **三种"纯策略"同布局可比**（`CHITU_Z_IMAGE_CP_MODE=agkv|ulysses|ring`）：文本 token 每卡复制、只对图像 token
  做序列切分，输出布局一致 → transformer-forward 包装与 output gather 与策略无关，可做严格 apples-to-apples。

> **教训（值得写进文章）**：框架在 `cp_size>1` 时会无条件把裸 attention kernel 包成通用 CP 包装器。
> 而"模型自管 CP"的 processor 再叠一层 → **双重 CP**（重复 all-gather、Ulysses 拼错 head、Ring 维度越界）。
> 凡是 processor 自己实现了 CP，必须解包到底层裸 kernel（`getattr(backend, "attn", backend)`）。

---

## 2. 核心事实：PCIe 上"通信决定一切"

cp2 干净 profile（修复双重 CP 后）：

- 真实 attention 计算只有 **~0.16s**，而 CP 通信高达 **0.32–0.42s**——单看 CP 注意力，**通信是计算的 2–2.6 倍**。
  这与 NVLink 场景正好相反（NVLink 上通信近乎免费、计算是瓶颈）。
- **三策略计算量基本相等**（qkv+attn ≈ 0.36s），区别全在通信模式，**没有哪种策略省计算**。
- **通信数据量几乎一样**（~4.4 GiB，对端图像 K/V 省不掉），差异在**搬运效率**（有效带宽）：

  | 原语 | 有效带宽 | 特征 |
  | --- | ---: | --- |
  | Ring P2P | **14.8 GB/s** | 成对收发、消息大、次数少 |
  | all-gather (AGKV) | 12.1 GB/s | 一次集合走完 |
  | all-to-all (Ulysses) | 11.3 GB/s | 消息碎、次数最多、启动开销摊薄带宽 |

**Key insight #1：PCIe 上 CP 是 comm-bound，优化的本质是"少搬字节 / 搬得更整"。**

---

## 3. 全卡横评：cp2 选 Ring，cp4 选 AGKV

非 profile 墙钟（10 步，CFG-off，速度口径见 §7 的重要修正）：

| Config | 加速比 vs 1GPU | 备注 |
| --- | ---: | --- |
| 1 GPU | 1.00× | 484ms/step 纯计算 |
| cp2 AGKV / Ulysses | 0.84× / 0.70× | **比单卡还慢**：通信吃掉计算节省 |
| **cp2 Ring** | **1.56×** | 只有 P2P 重叠能在 cp2 跑正 |
| cp4 AGKV / Ulysses | **2.02×** | 计算减到 1/4 压过通信 |
| cp4 Ring | 1.91× | 多轮 P2P 受拓扑拖累 |

**Key insight #2：策略最优解随卡数翻转**——cp2 通信占比高、Ring 的重叠救场；cp4 计算份额被摊薄、
AGKV 的"一次 all-gather"比 Ring 多轮 P2P 更省。

附带结论：
- **异构拓扑**：`nvidia-smi topo -m` 显示 GPU0-1 / GPU2-3 跨 NUMA 走 SYS。cp4 ring 环里一半链路走慢路径。
- **CUDA Graph 在 PCIe 不成立**：graph replay 把原本 overlap 的计算/通信也序列化了，全场景比 eager 慢。
  （第一篇 NVLink 上 graph 成立的前提是"通信短、发射抖动是瓶颈"，PCIe 上通信本身就长。）

---

## 4. 分层通信 unified + 最反直觉的发现

猜想：把通信分层——NUMA 内做一种、最少跳数过慢的 NUMA 边界。即 USP 2D 切分 `cp = up × ring`
（`up` 子组内 all-to-all，子组间 ring 流转 `cp/up` 轮）。落地为 core 的 `unified` 模式
（`up=2` → 子组 `{0,1}`/`{2,3}` 均 NUMA 内，ring 仅 2 轮跨 NUMA）。

cp4 实测：unified up2×ring2 把纯 Ring 的 254ms 拉回 243ms ≈ AGKV（241ms），**架构方向对，但没反超 AGKV**。

profile 分项揭示了**本篇最反直觉、也最重要的发现**：

| 分项 | 有效带宽 | |
| --- | ---: | --- |
| Ring P2P（**跨 NUMA**） | **13.9 GB/s** | 逼近 cp2 NUMA 内的 14.8——大块连续 P2P 对 SYS 不敏感 |
| Ulysses all-to-all（**NUMA 内**） | **7.9 GB/s** | 即便走最快链路也最慢——750 次碎消息，启动开销摊薄 |

**Key insight #3：cp4 ring 慢的真凶不是"SYS 带宽慢"，而是 all-to-all 的消息碎片化。**
这修正了 §3 把锅甩给拓扑的简单归因——**大块连续传输对跨 NUMA 并不敏感，碎消息才致命**。
所以 cp4 PCIe 上 AGKV 仍最优：一次 all-gather 既无 ring 多轮串行、也无 a2a 碎片化。

---

## 5. Ring 重叠天花板 & 无 GPUDirect P2P

- **重叠消融**（`CHITU_CP_RING_NO_OVERLAP=1` 强制串行）：eager ring 本就做了 compute/comm 重叠，但只藏掉
  **~4% 墙钟**。原因：每个 ring step 的 K/V 传输很大，能与之重叠的只有**一个块的 attention**（全程才 0.11s）——
  **计算太少，藏不住通信**。1024² 下 ring 的重叠天花板已摸到。
- **本机无 GPUDirect P2P**：`can_device_access_peer` 对所有配对（含同 NUMA）都 False → **所有跨卡通信经
  host 中转**（PCIe→pinned host→PCIe）。"copy-engine 直连、把 SM 让给 attention"这条路在本机不成立；
  也解释了为何 NUMA 内 ring 也只有 14.8 GB/s——上限就是 host-staged PCIe。

**Key insight #4：comm-bound 且无 P2P 时，抬 ceiling 的最大杠杆是"减少搬运字节"，而非"重叠"。**
重叠是分辨率相关的：分辨率↑ → attention 计算二次增长、K/V 通信只线性增长 → 同一份重叠代码才会"重新生效"。

---

## 6. 少搬字节：fp8 K/V 通信

把跨卡 K/V 用 **fp8(e4m3) per-head scaling** 传输，收完反量化回 bf16，**attention 仍 bf16**
（开关 `CHITU_CP_FP8_KV`，仅落在 core，adapter/processor 不动）。

- **AGKV-fp8**：KV all-gather 字节精确砍半（6.59→3.30 GiB），cp4 端到端 **−11.6%**（cp2 −5.4%，cp4 通信占比更大收益更高）。
- **local 分片保 bf16（免费优化）**：本 rank 的分片根本不出卡，没必要走有损 round-trip。让本地槽位直接用原始 bf16，
  **1/cp 的 K/V 无损、零额外通信**，张量级 KV 重建 MSE **−25%**（cp4，正好 1/cp）。已设为默认。
- **推广到 unified**（a2a 用 up 组内全局 per-head scale，ring 随数据带 scale）：再 −4.4%。

**Key insight #5：fp8 只压通信、不动计算**——arithmetic intensity 不变，所以它能省的上限就是"通信那一块的一半"。

---

## 7. 正确性体检 + CFG 是正交加速轴（关键修正）

复盘出图：prompt 要求一块写着「Z-Image × ChituDiffusion」的牌子，但早期所有图都杂乱无字。一度怀疑 CP 写错。结论相反：

- **"无字"是 `guidance_scale=1.0`（CFG 关）现象，不是 CP bug**。Z-Image 在 CFG=1 下 prompt 服从性极差。
  CFG=4 下 cp4 出图与单卡逐像素一致（PSNR 41 dB，牌子清晰）→ **CP 数值正确**。
- **CFG 下纯 cp4 一度加速消失（已修）**：`cfg_size=1` 把 cond+uncond `repeat(2)` 打包成 `len(x)=2`，命中
  `cp_forward` 的 `len(x)!=1` fallback → CP 没启用，4 卡冗余算 batch=2。修复 = CP 开启时拆成两次 len=1 前向再合 CFG。
  cp4@g4 **50.6s → 23.7s（2.13×）**。
- **CFG 并行（cfp）与 CP 正交可叠**：cfp2×cp2 在 g4 下 **3.09×**。8 卡 `cfp2×cp4` 应能复利——这正是"把 cp4 本身做强"的意义。

**真实设置下（guidance=4, 50 步）cp4 横评**（速度口径，与 §3 的 CFG-off 排序不同）：

| 模式 | vs 单卡 | +fp8 |
| --- | ---: | ---: |
| AGKV | 2.13× | 2.43× |
| Ring | 2.11× | — |
| **Unified up2×ring2** | **2.37×** | **2.48×** |

> §3 的 CFG-off 排序（AGKV 最优）在真实 guidance 下翻转为 unified 最快——加速比口径必须区分 CFG-off / CFG-on。

---

## 8. 画质方法论：像素漂移 ≠ 感知损失

CP / fp8 出图 vs 单卡 PSNR 常只有 ~20–24 dB，看着不小。五条证据表明这是**轨迹漂移、不是质量退化**：

1. 单卡逐位可复现（两次 run PSNR=inf）→ 差异 100% 来自 CP，不是 RNG。
2. CP 数学上与单卡等价（all-gather/a2a/ring online-softmax 都是精确重排），唯一变量是 **bf16 累加顺序**。
3. 单步扰动极小（1 步 35.4 dB），随去噪步数累积放大（50 步 24 dB）——混沌 ODE 的轨迹发散特征。
4. **CFG 是放大器**：同 cp4 AGKV，g1→g4 直接掉 13 dB（pos+4·(pos−neg) 每步把 bf16 差 ×4）。
5. 三种 CP 模式发散到同一水平（23–24 dB）→ 无某模式特有 bug。

**Key insight #6：质量 baseline 要用「无损 cp4（bf16 同布局）」而非单卡**——cp4 本身相对单卡就有漂移，
要隔离"近似策略（fp8/cache）的净损失"必须以同布局精确 CP 为参照。
且 **~20 dB 区间单图 PSNR 不是"谁更准"的判据**（硬证据看张量级 MSE 或感知指标）。

### 最终质量横评（CP4 各策略，guidance=4, 50 步, seed=42）

离线质量栈打通后（LPIPS AlexNet + HPSv3/Qwen2-VL 本地权重），对 contact sheet 8 张源图补测。
参考指标（PSNR/SSIM/LPIPS）以**无损 cp4 bf16 AGKV** 为 reference；HPSv3 为无参考 prompt-image reward（越高越好）。

![CP4 策略出图对照](assets/cp4_contact_sheet.png)

| 配置 | 加速 | PSNR↑ | SSIM↑ | LPIPS↓ | HPSv3↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| cp4 bf16 AGKV（ref） | 2.13× | ref | ref | 0.000 | 10.310 |
| cp4 fp8 all | 2.43× | 21.37 | 0.809 | 0.250 | 10.145 |
| cp4 fp8 local-bf16 | 2.43× | 19.36 | 0.730 | 0.372 | **10.822** |
| KV-cache w8/i4/t3 | 2.67× | 18.59 | 0.763 | 0.287 | 9.890 |
| unified up2×ring2 bf16 | 2.37× | **22.86** | **0.878** | **0.125** | 10.292 |
| unified up2×ring2 fp8 | 2.48× | 19.18 | 0.709 | 0.407 | 10.668 |
| single-GPU origin | 1.00× | 23.11 | 0.864 | 0.135 | 10.377 |

- **LPIPS 佐证主观判断**：unified-bf16 / 单卡最接近无损 cp4（0.12–0.13）；fp8 / cache 落在 0.25–0.41 的可接受漂移区，无伪影。
- **HPSv3 全部落在 9.9–10.8 窄区间**，无语义崩坏；最高的反而是 `fp8 local-bf16`（解释了它 PSNR 偏低但牌子最清晰）。
- 结果文件：`outputs/quality_cp4_assets/quality_summary.csv`。复现 contact sheet：`python3 outputs/make_contact_sheet.py`。

> 运维注记：HPSv3 是 16 GB safetensors，CFS 上 remap 会长时间 I/O wait，需先把模型拷到 `/dockerdata` 再转换；
> 离线依赖（含 `matplotlib`/`tensorboard`）已并入 `[eval]` extra 与 `wheelhouse/` 说明。

---

## 9. 跨步 KV-cache：更快但更不准

思路（仅 AGKV）：相邻步 K/V 变化慢 → 远端 KV 不必每步 all-gather。fresh 步全量 gather 并缓存，stale 步零通信
（复用缓存的远端分片 + 本步新算的 local 分片拼回）。约 29/50 步零 KV 通信。

| 配置 | 加速 | PSNR vs 无损 cp4 |
| --- | ---: | ---: |
| fp8 KV | 2.43× | 21.37 |
| **KV-cache（w8/i4/t3）** | **2.67×** | 18.59 |

**Key insight #7：cache 比 fp8 更快（stale 步完全不通信）但更不准**——fp8 是每步全量、只丢量化噪声；
cache 复用最旧差 3 步的远端 KV，在结构成形期是**系统性偏置**，比舍入级量化更伤。
- 反直觉但重要：调度 2（warmup 8→3、把 fresh 预算挪到 cooldown）**又慢又差**。
  **warmup（早期高噪声步）比 cooldown 重要得多**——stale KV 不能进结构成形的前几步。

---

## 10. 加速比上限分解：fp8 为什么只 +0.3×？

把 cp4 AGKV 的一步归一化到单卡=1.00（g4 50 步：cp4 bf16=1/2.13=0.470）。fp8 halve KV 字节后省了 0.058
（=½ KV 通信），反推：

| 组成 | 归一化时间 | 占比 |
| --- | ---: | ---: |
| 非通信地板（compute + 不可并行开销） | 0.354 | ~75% |
| 暴露的 KV 通信（bf16） | 0.116 | ~25% |

- **fp8 只动 25% 的通信、且只砍一半** → 移除 ~12% 步时间：`2.13× / (1−0.123) = 2.43×`。75% 的计算地板纹丝不动。
- **comm = 0 的理论上限 ≈ 2.8×**（`1/0.354`）；**comm=0 且 compute 完美 1/4 = 4.0×**。
- 4× → 2.13× 的差距几乎**平分**给两半：通信（0.116）与"计算不可完美 1/4"（0.354−0.25=0.104）。
  后者来自**文本 token 每卡复制**（per-rank ≈ 29% 而非 25%）+ 小 GEMM 利用率下降 + 固定开销。

**Key insight #8：comm-free 也只能到 ~2.8×**。要越过它必须攻**计算地板**（切分文本 token、融合 adaLN/norm 尾巴、
增大有效 GEMM），而非继续抠通信。fp8/cache 这类"省通信"手段的天花板就是那 25%。

---

## 11. 贯穿式序列切分：把"复制开销"压到底

§10 指出 comm=0 也只能到 ~2.8×，瓶颈转向**计算地板**。地板里有一块不是"算得慢"，而是
"**没切干净**"：序列切分原先只覆盖 30 层主干，而 2 层 `noise_refiner`、`final_layer`、patch
embed 仍在**全图**上每卡复制执行；且主层算完后要在**全 hidden 维（3840）**把图像 all-gather
回全长，才能过 final_layer。卡越多，这块固定开销被摊得越显眼。本节把切分**贯穿整条前向**。

- **贯穿前（main-only 切分）**：refiner / final 复制跑全图；主层后在全 hidden 维 gather 回全长。
- **贯穿后（end-to-end 切分）**：图像 token 在 `noise_refiner` 前切一次，一路分片穿过
  `noise_refiner → build_unified → 主层 → final_layer`，只在 `final_layer` 之后做**唯一一次** gather
  ——此时每 token 已降到 patch-output 维（~64，比 hidden 维小 ~60×），gather 近乎免费。
  `context_refiner` 是纯文本（几十个 token），保持复制、在切分前先跑完。
  数值上仍是精确重排：g1 下对全复制 **47.9 dB**（benign bf16 漂移），50 步 g4 出图清晰、牌子可读。

### 单卡 / 2 卡 / 4 卡 序列并行横评（guidance=4, 50 步, bf16, 同 prompt/seed）

| 配置 | 贯穿前 | 贯穿后 | Δ墙钟 | Δ加速 |
| --- | ---: | ---: | ---: | ---: |
| 1 GPU | 50.51s / 1.00× | — | — | — |
| cp2 AGKV | 32.37s / 1.56× | 31.40s / 1.61× | −3.0% | +0.05× |
| cp2 Ulysses | 32.16s / 1.57× | 31.21s / 1.62× | −3.0% | +0.05× |
| cp2 Ring | 30.05s / 1.68× | 28.75s / 1.76× | −4.3% | +0.08× |
| cp4 AGKV | 23.78s / 2.12× | 21.98s / 2.30× | −7.6% | +0.17× |
| cp4 Ring | 23.98s / 2.11× | 22.56s / 2.24× | −5.9% | +0.13× |
| **cp4 u2r2** | 21.30s / 2.37× | **19.46s / 2.60×** | **−8.6%** | **+0.22×** |

> 贯穿前 cp4 三项 **2.12 / 2.11 / 2.37×** 与 §7 历史值 **2.13 / 2.11 / 2.37×** 逐项吻合 → A/B 复刻可信。

**Key insight #9：复制开销随卡数放大，所以"贯穿切分"是卡数越多越划算的纯计算优化。**
- refiner + final 的**绝对**耗时固定；卡越多主层被摊得越薄，这块占比越高 → 贯穿收益随卡数上升
  （cp2 −3~4%，cp4 −6~9%）。这正面验证了 §10"计算地板"里"没切干净"的那部分是可回收的。
- gather 从"全 hidden 维 / 每步"挪到"final 之后 / 64 维 / 一次"，搬运字节 ~60× 缩水 → 对 gather 较重的
  模式增益最大（ring +0.13×、u2r2 +0.22×；AGKV 的一次 all-gather 本就最省，仍 +0.17×）。
- **真实 guidance 下新最优：cp4 u2r2 2.60×、cp4 AGKV 2.30×**。纯结构优化，与 fp8 / cfp 正交可叠，
  也不与 FlexCache 互斥（贯穿只改"哪里切/哪里 gather"，不改去噪步间的缓存语义）。

---

## 12. 结论速览

- **PCIe + 无 P2P 上 CP 是 comm-bound**；最优范式是**一次大块 all-gather（AGKV）**，而非 ring 多轮 P2P 或 a2a 碎消息。
- **拓扑不是主因，消息粒度才是**：大块连续传输对跨 NUMA 不敏感，all-to-all 碎片化才是 cp4 的真瓶颈。
- **少搬字节是第一杠杆**：fp8 K/V（local 保 bf16）cp4 −11.6% 且画质代价可控；KV-cache 更快但精度更低。
- **CFG 与 CP 正交**：真实 guidance 下 unified 2.37×、+fp8 2.48×；cfp×cp 复利（cfp2×cp2 3.09×）。
- **画质要看感知指标 + 同布局 baseline**：PSNR 在漂移区会误导；LPIPS/HPSv3 显示各策略均无质量崩坏。
- **加速天花板由计算地板决定**：comm=0 也只 ~2.8×，重心要转向计算侧可扩展性。
- **贯穿式序列切分回收"没切干净"的地板**：refiner/final 也分片、gather 移到 final 之后（64 维），
  卡数越多收益越大；真实 guidance 下把 cp4 u2r2 抬到 **2.60×**、cp4 AGKV 抬到 **2.30×**，且与 fp8/cfp/FlexCache 正交。

### 后续

- [x] 贯穿式序列切分：refiner/final 一并分片、gather 移到 final 之后（done，§11）。
- [ ] 继续攻**计算地板**：切分文本 token、融合 adaLN/norm、block-level `torch.compile`（与 CP 正交）。
- [ ] 8 卡 `cfp2×cp4` 复利实测。
- [ ] KV-cache 收紧（`INTERVAL=2` / 保 warmup / cache+fp8 混合），看能否在 fp8 速度下追平 fp8 精度。
- [ ] fp8 画质收紧：per-token scale / 只量化 K / 仅后段低噪声步启用。
- [ ] 高分辨率下重叠"重新生效"的划算分辨率拐点。
