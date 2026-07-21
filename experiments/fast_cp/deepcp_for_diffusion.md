# DeepCP for Diffusion

> 一套面向 DiT 推理的、可解释 / 可测量 / 可迁移的 Context Parallel 调度系统的设计总结。
> 本文综合 `fast_cp` 已有实验(harness microbench + 真实模型短步 + milestone 负结论)与
> DeepEP / NVSHMEM / triton-distributed 的技术调研,给出问题拆解、经验规律与 DeepCP 的技术路线。

配套结论:
- `results_zimage_cp.md`：Z-Image 在 H20 NVLink 上的纯 CP scaling。
- `results_wan_ring_graph.md`：Wan 1.3B CP8 扩展性与 Ring Graph 边界。

一次性 trace、probe、microbenchmark demo 和逐次实验日志已在阶段结束后清理；最小可复用
CP core harness 保留在 `harness/`，本文沉淀其稳定结论。

---

## 1. 背景与命题

- 当前主流 DiT 单卡基本放得下,所以 CP(序列并行)相比 TP / PP 通常有更高效率:
  它不切权重、不引入流水气泡,只沿序列维切分,通信模式相对规整。
- **但 CP 在 DiT 上的可扩展性不强**:卡数一多,加速比就掉。目标是**提升 CP 的可扩展效率**。
- 核心认知(来自 milestone):**CP 不是"换一个通信 primitive",而是一个调度问题**。
  是否该切、切多细、走哪种通信模式、用哪个后端、要不要 overlap / graph,都取决于
  **硬件互联 × 负载形状**。没有银弹,只有 policy。

---

## 2. 可扩展性差的两个根因

把所有观察归因到两点:

1. **通信太慢**:通信开销无法被计算掩盖。极端情形下(PCIe)通信 kernel 比计算 kernel 还长。
2. **过度切分**:序列被切得太碎 → 单卡 GEMM / SDPA 打不满 + 每次切分带来额外启动/同步开销。

DeepCP 的三条技术线正是分别针对这两点(见 §6)。

---

## 3. 硬件 × 负载:三种 regime

CP 的**主要瓶颈随硬件而变**,不能用同一套策略:

| 硬件 | 算力 | 互联 | 主瓶颈 | 优化重心 |
| --- | --- | --- | --- | --- |
| **4090** | 高 | PCIe(点对点、带宽低、**本机无 GPUDirect P2P**) | **通信 bound**:通信 kernel ≫ 计算 kernel | 减字节、减消息碎片、榨 overlap |
| **H20** | 低 | NVLink(全互联) | **切分 bound**:序列变短后 GPU 打不满 + 额外启动 | 保大 kernel 效率、少切、慎 overlap |
| **H100** | 高 | NVLink(全互联) | 计算密集,通信占比小;短序列时同样面临打不满 | 保效率为主,overlap 收益有限 |

**PCIe 是 comm-bound 的实测证据**(4×4090,Z-Image 1024²,cp2 干净 profile):
真实 attention 计算只有 **~0.16s**,而 CP 通信高达 **0.32–0.42s** —— 单看 CP 注意力,**通信是计算的 2–2.6 倍**。
三种策略计算量基本相等(qkv+attn≈0.36s)、通信数据量也几乎一样(~4.4 GiB,对端图像 K/V 省不掉),
**没有哪种策略省计算,差异全在搬运效率**。而且本机 `can_device_access_peer` 全 False → 所有跨卡通信经 host 中转
(PCIe→pinned host→PCIe),"copy-engine 直连让出 SM"这条路在 4090 上根本不成立 —— 这也是为什么
**同一套 SM-free/overlap 手段在 H20 NVLink 和 4090 PCIe 上结论会相反**。

关键补充:**Diffusion 绝大多数情况是计算密集**,但**序列越短,GPU 越容易打不满**。
例如 Z-Image / Flux 在 4096 序列长时,GEMM 和 SDPA 都不满 —— 这时再沿序列切 CP,
利用率进一步下降,过度切分的代价被放大。这解释了为什么 NVLink 上 overlap 往往不赚:
瓶颈根本不是通信,而是"切碎后算不满"。

> 实测印证(milestone「今日关键数据」):
> - AGKV cp4 / img=4096 / H=24,eager overlap = **-3.7%**(NVLink 上人为 overlap 反而亏)。
> - 只有人为制造 comm-bound(img=8192 / H=48)时 overlap 才 **+9.1%**。
> - 真实模型 Z-Image / Qwen-Image cp4 上 overlap 基本持平或略负。

---

## 4. CP 通信模式与选型经验

四类基本模式(及其混合)在 footprint / 消息模式 / 计算效率上的权衡:

| 模式 | 通信模式 | KV 显存 | SDPA 形状 | 特点 |
| --- | --- | --- | --- | --- |
| **Ring** | 邻居 P2P、多小步流水 | **最低**(只存一块 KV chunk) | online-softmax 分块 | 匹配点对点链路 / 跨节点;但 step 多→launch bound |
| **AGKV** | all-gather K/V | **最高**(每卡持全量 KV) | **一个大 SDPA、全部 head** | SDPA 效率最好;吃内存 |
| **Ulysses** | all-to-all(切 head) | 中低(全序列但 1/cp 个 head) | 全序列、head 变少 | 省内存;head 太少时 SDPA 掉效率 |
| **Ring+Ulysses** | 层次化:节点内 all-to-all,节点间 Ring | 中 | 组合 | 跨机时最省跨节点字节 |

### 你的经验(实测得出)

| 场景 | 最优模式 |
| --- | --- |
| PCIe 2 卡 | **Ring** |
| PCIe >2 卡 | **AGKV** |
| NVLink + 内存充足 | **AGKV** |
| NVLink + 内存不足 | **Ulysses** |
| 跨机 | **Ring + Ulysses** |

**PCIe 横评实测**(4×4090,Z-Image,CFG-off 口径):

| Config | 加速比 vs 1GPU | 关键读数 |
| --- | ---: | --- |
| cp2 AGKV / Ulysses | 0.84× / 0.70× | **比单卡还慢**:通信吃掉计算节省 |
| **cp2 Ring** | **1.56×** | 只有 P2P 重叠能在 cp2 跑正 |
| cp4 AGKV / Ulysses | **2.02×** | 计算减到 1/4 压过通信 |
| cp4 Ring | 1.91× | 多轮 P2P 受拖累 |

有效带宽(同~4.4 GiB 通信量,差异全在搬运效率):Ring P2P **14.8 GB/s** > all-gather 12.1 > all-to-all 11.3。

### 尝试提炼底层规律

这些选择其实由三个变量决定,可以合成一条 tentative rule:

1. **显存预算** → 决定能否 AGKV(复制全量 KV)。内存够就 AGKV(拿"最好的大 SDPA"),
   内存紧就退 Ulysses(切 head 降 footprint)或 Ring(footprint 最低)。
2. **消息粒度(大块连续 vs 碎片化)** → 这是比"拓扑"更本质的变量。
   **反直觉实测**:cp4 unified 里 Ring P2P **跨 NUMA** 仍有 **13.9 GB/s**
   (逼近 cp2 NUMA 内的 14.8),而 Ulysses all-to-all **走最快 NUMA 内链路却只有 7.9 GB/s**
   (750 次碎消息,启动开销摊薄带宽)。→ **大块连续传输对慢链路不敏感,碎消息才致命。**
   所以 AGKV(一次大块 all-gather)在 PCIe cp4 最优:既无 ring 多轮串行,也无 a2a 碎片化。
3. **互联拓扑 = mesh 还是 point-to-point** → 决定 all-pairs 是否划算(次于粒度)。
   NVLink/NVSwitch 全互联 → all-pairs 高效(AGKV/Ulysses);跨节点慢 P2P → 把 all-pairs 限制在
   节点内(Ulysses over NVLink)、节点间只走大块 Ring → Ring+Ulysses。
4. **计算余量** → 决定能否接受切 head/切块带来的 kernel 变小。
   算力高 + 序列短(易打不满)时,优先保大 SDPA(偏 AGKV),避免 Ulysses 把 head 切太碎。

> PCIe >2 卡为什么从 Ring 转向 AGKV:卡数一多,Ring 的串行跳数(latency)线性增长,
> 而 all-gather 一次大块集合走完,既避开长 ring 流水、也避开 a2a 碎片化 —— 实测 cp4 AGKV 2.02× > Ring 1.91×。

**一句话规律**:`模式 = f(显存预算, 消息粒度, 拓扑是否 mesh, 计算余量)`;
AGKV 是"内存换 SDPA 效率",Ring 是"内存/链路友好换流水开销",Ulysses 是"footprint 换 head 效率",
Ring+Ulysses 是"按拓扑分层,把慢链路的字节压到最少"。

---

## 5. 通信后端:NCCL / Symmetric Memory + Copy Engine / NVSHMEM / triton-distributed

选模式之外,**用什么后端搬字节**同样关键。核心结论先给:
**它们大多走同一条物理线(节点内 NVLink、节点间 RDMA 网卡),差别在"用什么方式驱动这条线",
不是"更快的线"。**(算子级横评显示:NCCL / copy-engine 在同一条 NVLink 上,
纯算子速度并无本质差距。)

| 后端 | 怎么驱动链路 | 占 SM? | 会规约? | 优点 | 代价 / 限制 |
| --- | --- | --- | --- | --- | --- |
| **NCCL** | SM-resident 集合 kernel | **占** | ✅ | 通用(任意拓扑/规约/跨节点回退)、一次 launch、成熟 | kernel 抢 SM→与计算争用;误用 `list+cat` 有额外 staging |
| **Symmetric Memory + Copy Engine** | 对称内存 P2P,由 **DMA/拷贝引擎**搬 | **不占** | ❌ | **SM-free**,能和计算真并行 | 需 P2P + 对称分配;不能规约;小消息延迟高;主要节点内;逐次发射有 host 开销 |
| **NVSHMEM (+IBGDA)** | PGAS 单边 put/get;跨节点 **GPU 直发 RDMA** | 可轻(IBGDA) | 取决用法 | 低延迟、细粒度、跨节点 GPU-initiated | 重依赖、需 IB(本节点起不来);DeepEP V2 已弃用改 NCCL Gin |
| **triton-distributed** | 编译期把通信原语降到 NVSHMEM,**融进 Triton kernel** | 占(但 tile 级调度重叠) | 可 | tile 级 comm/compute 融合、单源 Python | 依赖 NVSHMEM;要用 Triton 重写 kernel(注意力难追平 FA);收益主要在 comm-bound |

### 关键区分:Copy Engine 是老硬件,新的是"用法"

- **Copy Engine(DMA 引擎)**是 GPU 上一直就有的、**独立于 SM** 的搬运硬件
  (`cudaMemcpyAsync` 的 compute/copy overlap 就靠它)。它让"搬字节"**不占 SM**;
  但传统上**由 CPU 发起** —— 所以是"SM-free 执行、CPU 发射"。
- **新在思路**:把本来给 copy/compute overlap 的 CE,配合**对称内存(peer 寻址)**,
  拿来做**集合通信**(all-gather),从而绕开 SM-resident 的 NCCL kernel。这就是 DeepEP
  "0 SM CP with Copy Engine" 的核心。
- **跨节点**的"GPU 发起、不占 SM"靠的不是 CE,而是 **IBGDA**(GPU 直接向网卡投 RDMA)。
- 别混淆:Hopper 的 **TMA / `cp.async`** 是**芯片内** global↔shared 的异步拷贝,由 SM 发起,
  不是跨卡通信。

### NCCL 为什么不默认用 CE 做集合通信(代价)

1. **CE 不会规约**:all-reduce/reduce-scatter 是"搬+加",DMA 没有 ALU,做不了加法。
2. **做不了细粒度流水/flag 同步**:NCCL 用 SM 线程把 8B 数据+flag 打包、边收边发;CE 是粗粒度 memcpy。
3. **多跳/拓扑感知覆盖不了**:ring/tree/跨节点需要主动转发者(SM kernel 或 CPU proxy)。
4. **CE 数量少、小消息延迟高**:实测 0.5 MiB 时 NCCL 0.018ms 反而快过 CE 0.033ms。
5. **通用性/依赖**:CE-P2P 需对称内存 + 全 P2P 连通,且每次搬运 host 发起(需 CUDA graph 摊销)。

> 所以 CE 集合通信是**窄场景特化**:只有"all-gather 类 + 要和吃满 SM 的计算重叠"时才值。
> 业界正是这么用的(PyTorch `_low_contention_all_gather`、NCCL user-buffer registration)。

### 选型决策规则(deepCP runtime)

- 通信在关键路径、**无法 overlap** → **NCCL**(`all_gather_into_tensor` / `all_to_all_single`,
  永远别 `list+cat`),它同链路上一样快甚至更快,且最简单。
- 通信**能和计算 overlap** → **Copy Engine(对称内存)**,靠"不抢 SM"取胜,而非算子更快。
- **跨节点、长序列、comm-bound** → 才考虑 **NVSHMEM/IBGDA** 或 **triton-distributed** 的
  tile 级融合(前提:IB/NVSHMEM 可用,愿意付重写成本)。单节点 NVLink 上收益不足以抵复杂度。

---

## 6. DeepCP 的三条技术线

针对 §2 的两个根因(通信太慢 / 过度切分):

### 技术 1 —— 极致 overlap 隐藏通信(治"通信太慢")

用 **copy engine + 专用通信原语**,让通信最大程度和**无依赖计算**重叠,且**不抢 SM**。

- **已验证的硬件信号**(4×H20):把 all-gather 和一个独立 GEMM 并发,
  - NCCL(SM-resident):GEMM 膨胀 **+100%**(相当于慢一倍);
  - Copy-engine(SM-free):GEMM 膨胀 **+0.7%**(几乎免费)。
  → DeepEP 的 "0-SM CP" 思路在 CP all-gather 上成立。
- **但纯算子速度 CE 并不更快**:同一条 NVLink,小消息 NCCL 更快、
  大消息二者收敛(CE 仅 ~5–9% 领先)。**CE 的价值是 SM-free overlap,不是搬得更快。**
- **核心 tradeoff(挑战)**:要 overlap 就得把计算拆开暴露"无依赖片段",但**拆得越细,
  GEMM/SDPA 越难打满**。milestone 已实证:naive txt/image split 把一个大 SDPA 拆成两个小的,
  总 SDPA 时间上升,**吃掉了通信隐藏的收益**。所以 overlap 不能"到处拆",必须在保大 kernel 效率
  的前提下找无依赖计算(如 gather K/V 的同时做 Q/V projection)。
- **overlap 是"分辨率相关"的,不是恒定收益**:4090 上强制串行消融显示,
  eager Ring 本就做了 compute/comm 重叠,但 1024² 下**只藏掉 ~4% 墙钟** —— 每个 ring step 的 K/V
  传输很大,能与之重叠的只有一个块的 attention(全程才 0.11s),**计算太少藏不住通信**。
  分辨率↑ → attention 二次增长、K/V 通信只线性增长 → 同一份 overlap 代码才会"重新生效"。
  这与 NVLink 上 AGKV eager overlap **-3.7%**、人为 comm-bound 才 **+9.1%** 是同一条力学:
  **overlap 的收益 = 可隐藏的无依赖计算量 ÷ 暴露的通信量,两端都不满足时它就是噪声。**
- **CUDA graph 的边界**:
  - 串行 AGKV 是 **GPU-bound**,graph replay **中性(~1.00–1.04×)**,别指望它加速普通 AGKV。
  - lean copy-engine gather 是 **DMA-bound**(GPU 0.086ms > CPU 发射 0.031ms),graph 也无从优化。
  - graph 只对 **launch-bound** 结构有用:Ring 的多小步 P2P(NVLink 上实测 +8.5%),
    以及多流 overlap 调度的**跨 rank 对齐**。

> 结论:overlap 的正确落点是 **copy-engine SM-free gather + 粗粒度重叠(不拆碎大 kernel)**,
> 并只在 launch-bound 的调度上叠 CUDA graph。

### 技术 2 —— 通信实在盖不住时,用 diffusion 的容噪能力**减少通信量**(治"通信太慢")

Diffusion 天生对噪声/近似鲁棒,可用来压通信。**实测这条线是 PCIe 上的第一杠杆**
(4×4090,guidance=4,50 步):

| 手段 | cp4 加速 | PSNR↑(vs 无损 cp4) | LPIPS↓ | HPSv3↑ | 读数 |
| --- | ---: | ---: | ---: | ---: | --- |
| cp4 bf16 AGKV(ref) | 2.13× | ref | 0.000 | 10.31 | 基准 |
| cp4 fp8 all | 2.43× | 21.37 | 0.250 | 10.15 | KV 字节砍半 6.59→3.30 GiB |
| cp4 fp8 local-bf16 | 2.43× | 19.36 | 0.372 | **10.82** | 本地分片保 bf16,KV 重建 MSE **−25%** |
| KV-cache w8/i4/t3 | **2.67×** | 18.59 | 0.287 | 9.89 | 更快但更不准 |
| unified u2r2 bf16 | 2.37× | **22.86** | **0.125** | 10.29 | 最接近无损 |
| single-GPU | 1.00× | 23.11 | 0.135 | 10.38 | — |

三条被数据钉死的结论:
- **fp8 只压通信不动计算**:AGKV-fp8 cp4 端到端 **−11.6%**(cp2 只 −5.4%,通信占比小),
  因为它只动"暴露通信"那一块、且只砍一半(见 §6 加速上限分解)。
- **画质没崩,只是轨迹漂移**:HPSv3 全部落在 **9.9–10.8** 窄区间无语义崩坏;LPIPS 显示 fp8/cache 在
  0.25–0.41 的可接受漂移区,无伪影。**质量 baseline 必须用"无损 cp4 同布局"而非单卡**
  (cp4 相对单卡本就有 bf16 累加顺序漂移)。
- **cache 比 fp8 更快但更不准**:cache 复用最旧差 3 步的远端 KV,在结构成形期是**系统性偏置**,
  比 fp8 的舍入级量化更伤;且 warmup(早期高噪步)比 cooldown 重要得多,stale KV 不能进前几步。

具体量化手段:

1. **量化 K/V 通信**:用 FP8 搬 K/V,字节减半。
   - **教训**:naive **per-head** FP8 人眼画质掉得厉害(`image_cp_attention.py` 当前实现:
     `_quantize_fp8_per_head` 对 head_dim 方向共用一个 scale;Ulysses 路径更是 global per-head scale)。
   - **原因**:E4M3 只有 3 位尾数;per-head 在 head_dim(QK 收缩维)和 token 维都没保护,
     离群通道撑大 scale→小通道被压没;而 `softmax(QK)` 会指数放大 K 的误差。
   - **修法阶梯**:①per-head → **per-token**(amax 只在 D 上求,开销仅 `4/D`);
     ②**K 保 bf16 / 只 V 走 FP8**(softmax 对 K 更敏感);③**文本/条件 token 保 bf16**;
     ④**按 step 调度**(高噪前几步 bf16,低噪后段再 FP8);⑤必要时上 **Hadamard 旋转**摊平离群 + stochastic rounding。
2. **Feature cache 设计**:利用相邻去噪步 feature 的高相似度,跨步复用/跳过部分通信与计算
   (与 FlexCache 系列思路一致),把"每步都全量通信"降为"按需通信"。

### 技术 3 —— 治"过度切分":step-level scheduling

**为什么必须攻计算侧**(cp4 AGKV 加速上限分解,单卡=1.00):

| 组成 | 归一化时间 | 占比 |
| --- | ---: | ---: |
| 非通信地板(compute + 不可并行开销) | 0.354 | **~75%** |
| 暴露的 KV 通信(bf16) | 0.116 | ~25% |

→ **comm 完全为 0 的理论上限也只有 ~2.8×**(`1/0.354`);comm=0 且计算完美 1/4 才 4.0×。
fp8/cache 这类"省通信"手段的天花板就是那 25%(所以 fp8 只能 2.13×→2.43×)。**要越过 2.8× 必须攻计算地板。**

计算地板里有一块不是"算得慢",而是"**没切干净**":refiner/final/patch-embed 仍在全图每卡复制。
**贯穿式序列切分**(把切分穿过整条前向、gather 移到 final 之后的 ~64 维、只做一次)实测:
cp4 AGKV **2.30×**、cp4 u2r2 **2.60×**,且卡数越多收益越大(cp2 −3~4%,cp4 −6~9%),
与 fp8/cfp/FlexCache 正交。这验证了"计算地板可回收"。

在此之上,DeepCP 进一步在**去噪步之间**引入调度,用请求间的 batch/DP 把算力拉满:

- **Continuous batching**:去噪步之间允许不同请求拼批,填满 GEMM/SDPA。
- **CP–DP hot switch**:根据当前 batch 规模/序列长度,在去噪步之间动态切换 CP 与 DP —— 
  batch 大时用 DP(每卡算完整、无通信、利用率高),batch 小/序列长时切 CP(靠切分降单步延迟)。

> 直觉:CP 是"用通信换单请求延迟",DP 是"用显存/批量换吞吐"。二者的最优点随负载漂移,
> 在去噪步边界做 hot switch,就能始终待在利用率最高的工作点,从根上缓解"过度切分打不满"。

---

## 7. 已固化的负结论(重要,避免走回头路)

阶段实验已固化以下负结论:

1. NVLink 上"到处 async overlap"**不是主线收益**(真实模型多为噪声级甚至负)。
2. all-to-all API 已收敛到 `all_to_all_single`,继续换 API 空间很小。
3. `torch.compile` 无法直接吞掉 CP comm(stream/event/NCCL)路径,CP attention 只能是 eager island。
4. naive txt/image query split 会降低 SDPA 效率,不值得迁移。
5. CUDA graph 是**模式相关**优化(Ring/多小步值,AGKV 不值),不是全局银弹。

后续重点因此应是:**减字节、减 pack/unpack、保大 kernel 效率、按模式用 graph、按 topology+profile 做 runtime policy。**

---

## 8. 关键实测数据速查

**H20 / NVLink(harness microbench)**

| 实验 | 结论 | 出处 |
| --- | --- | --- |
| 同链路 gather:0.5 MiB | NCCL 0.018ms vs CE 0.033ms(**小消息 NCCL 更快**) | 阶段实验 |
| 同链路 gather:≥8 MiB | 收敛,CE 仅 ~5–9% 领先 | 阶段实验 |
| SM-free 证据 | 与独立 GEMM 并发:CE 膨胀 +0.7% vs NCCL +100% | 阶段实验 |
| CE 成本构成 | DMA-bound(GPU 0.086ms > CPU 0.031ms);barrier 仅 0.009ms | 阶段实验 |
| AGKV + CUDA graph | 中性 ~1.00–1.04×(GPU-bound) | 阶段实验 |
| AGKV CE vs NCCL(串行) | CE 快 ~4–11%(无 overlap 时仅算子小赢) | 阶段实验 |
| Ring + CUDA graph | +8.5%(launch-bound 才吃 graph) | milestone / results |
| AGKV cp4 img4096 eager overlap | -3.7%(NVLink 不赚) | milestone |
| AGKV cp4 img8192 H48 comm-bound | +9.1%(制造 comm-bound 才赚) | milestone |
| Wan2.1-1.3B cp4 Ulysses overlap | +1.8%(steady-state) | milestone |

**H20 / NVLink(Z-Image 真实模型纯 CP scaling，见 `results_zimage_cp.md`)**

| 实验 | 结论 |
| --- | --- |
| Z-Image cp4 最优 | AGKV/Ulysses **3.04×**(76% 效率);cp2 AGKV 1.75× |
| 模式排序 | `AGKV≈Ulysses > Ring > unified`(与 PCIe 相反:mesh 上一次大集合最省事) |
| CP 有效带宽 | ~126 GB/s(comm 仅占个位数%,瓶颈是计算地板非通信) |
| 修复:`do_cp` 门控 bug | 本 diffusers 版 `_prepare_sequence` 总返回全 1 mask→CP 从未生效;trivial mask 归一化为 None 才解锁 |
| CFG 与 CP 正交 | g4 与 g1 加速比逐项一致(cp4 3.04×);"CFG batching 税"假设已证伪 |

**H20 / NVLink(Wan 1.3B Ring Graph，见 `results_wan_ring_graph.md`)**

| 实验 | 结论 |
| --- | --- |
| Wan CP8 local workload | 约 4095 tokens/卡，高于 H20 上约 2K tokens 的 launch-bound 分界 |
| attention-only Ring Graph | GPU 时间仅约 1% 改善，端到端 steady state 持平或略慢 |
| 与短序列模型差异 | Wan 仍是 compute/data-movement-bound；Flux CP8 约 576 tokens/卡，graph 更有价值 |
| 输出缓冲区 | 静态 graph output 不能直接跨 replay 复用，模型路径必须 clone |

**4090 / PCIe(Z-Image 真实模型)**

| 实验 | 结论 |
| --- | --- |
| cp2 干净 profile | attention 计算 ~0.16s vs CP 通信 0.32–0.42s(**通信是计算 2–2.6×**) |
| 有效带宽(同 4.4 GiB) | Ring 14.8 > all-gather 12.1 > all-to-all 11.3 GB/s |
| 碎消息才是真凶 | Ring 跨 NUMA 13.9 GB/s vs Ulysses NUMA 内 all-to-all **7.9 GB/s** |
| cp2 最优 | Ring 1.56×(AGKV/Ulysses 0.84×/0.70×,**比单卡慢**) |
| cp4 最优 | AGKV 2.02×(Ring 1.91×) |
| Ring overlap 天花板(1024²) | 只藏 ~4% 墙钟(计算太少藏不住) |
| fp8 K/V(AGKV) | cp4 −11.6%(cp2 −5.4%);画质 HPSv3 9.9–10.8 无崩坏 |
| KV-cache w8/i4/t3 | 2.67×(更快)但 PSNR 18.59(更不准) |
| 加速上限分解 | 非通信地板 ~75% → **comm=0 也仅 ~2.8×** |
| 贯穿式序列切分 | cp4 AGKV 2.30× / u2r2 **2.60×**(纯计算优化,卡数越多越赚) |
| CUDA graph on PCIe | 全场景比 eager 慢(把 overlap 也序列化了) |

---

## 9. Roadmap

**近期**
- copy-engine SM-free gather 接入 AGKV(behind flag),NCCL 作为无 P2P/对称内存时的回退。
- FP8 K/V 从 per-head 改 **per-token** + K/V 分开开关 + 文本保 bf16;harness 上做画质 A/B。
- overlap 默认关、只留 flag;`all_to_all_single` 作 Ulysses 默认。

**中期**
- projection 直出 rank-major layout,削 Ulysses 的 pack/unpack。
- Ring CUDA graph 工程化;copy-engine overlap 调度 + graph 捕获(跨 rank 对齐)。
- Feature cache × CP 联合设计;step-level continuous batching 原型。

**长期**
- **topology + profile-aware runtime policy**:模型只暴露 Q/K/V producer、RoPE、mask、txt/img layout,
  由 DeepCP runtime 决定 CP mode / 是否 graph / 是否 overlap / 是否压缩 K/V / 是否 hybrid 切分,
  以及去噪步间的 **CP–DP hot switch**。
- 目标:不是某模型的特化 trick,而是一套**可解释、可测量、可迁移**的序列并行调度系统。
