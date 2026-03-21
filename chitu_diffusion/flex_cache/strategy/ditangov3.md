# DiTango 方法说明（补充数学公式）

## 背景与核心洞察

DiTango 针对扩散模型（DiT）在多节点分布式推理中的**可扩展性与生成质量权衡问题**提出解决方案。

核心洞察来自两个观察的自然对齐：

1. **计算的空间局部性**：在分布式 Context Parallelism（CP）中，将序列切分为若干 KV partition 后，attention 输出对各 partition 的依赖程度与空间距离负相关——距离越近的 partition 贡献越大，越远的贡献越小。这一规律在不同模型、不同输入下均一致。

2. **硬件拓扑的层次性**：分布式系统中通信带宽天然分层——本地显存访问开销极低，节点内 NVLink 带宽高，跨节点 InfiniBand 带宽低且延迟高。

这两者的对齐意味着：**贡献最大的 partition 恰好是通信代价最小的 partition，而贡献最小的远程 partition 恰好是通信代价最高的**。因此，对远程 partition 的 attention 计算进行历史结果复用，可以在最小化质量损失的前提下最大化通信节省。

---

## Attention State 的定义与可组合性

### 数学定义

对于序列 partition $i$，其 attention state $\text{AS}_t(i)$ 在时间步 $t$ 包含两部分：

$$\text{AS}_t(i) = \begin{bmatrix}
\mathbf{OUT}_t(i) \\
\mathbf{LSE}_t(i)
\end{bmatrix}$$

其中：

- $\mathbf{LSE}_t(i) = \log \sum_{j \in \mathcal{I}_i} \exp(\mathbf{q}_t \cdot \mathbf{k}_j)$
- $\mathbf{OUT}_t(i) = \sum_{j \in \mathcal{I}_i} \frac{\exp(\mathbf{q}_t \cdot \mathbf{k}_j)}{\exp(\mathbf{LSE}_t(i))} \mathbf{v}_j$

$\mathcal{I}_i$ 为 partition $i$ 包含的序列索引集合，$\mathbf{q}_t, \mathbf{k}_j, \mathbf{v}_j$ 分别为 query、key、value 向量。

### 组合操作公式

两个 attention state 的组合操作定义为：

$$\text{AS}_t(i) \oplus \text{AS}_t(j) = \begin{bmatrix}
\frac{e^{\mathbf{LSE}_t(i)} \mathbf{OUT}_t(i) + e^{\mathbf{LSE}_t(j)} \mathbf{OUT}_t(j)}{e^{\mathbf{LSE}_t(i)} + e^{\mathbf{LSE}_t(j)}} \\
\log(e^{\mathbf{LSE}_t(i)} + e^{\mathbf{LSE}_t(j)})
\end{bmatrix}$$

该操作满足结合律和交换律，可扩展到多个 partition：

$$\text{AS}_{combined} = \text{AS}_t(1) \oplus \text{AS}_t(2) \oplus ... \oplus \text{AS}_t(n)$$

---

## 误差传播分析

### 误差传播公式

设两个 attention state 分别有误差 $\delta\mathbf{OUT}_i, \delta\mathbf{LSE}_i$ 和 $\delta\mathbf{OUT}_j, \delta\mathbf{LSE}_j$。

定义归一化权重：
$$w_i = \frac{e^{\mathbf{LSE}_i}}{e^{\mathbf{LSE}_i} + e^{\mathbf{LSE}_j}}, \quad w_j = \frac{e^{\mathbf{LSE}_j}}{e^{\mathbf{LSE}_i} + e^{\mathbf{LSE}_j}}$$

组合后的误差为：
$$\delta\mathbf{OUT} = w_i\delta\mathbf{OUT}_i + w_j\delta\mathbf{OUT}_j + w_iw_j(\mathbf{OUT}_i - \mathbf{OUT}_j)(\delta\mathbf{LSE}_i - \delta\mathbf{LSE}_j)$$

$$\delta\mathbf{LSE} = w_i\delta\mathbf{LSE}_i + w_j\delta\mathbf{LSE}_j$$

### 简化误差界

忽略二阶项（$w_iw_j$ 项通常很小），对 $n$ 个 partition 的组合，输出误差的上界为：

$$\|\delta\mathbf{OUT}\|_2 \leq \sum_{i=1}^n w_i\|\delta\mathbf{OUT}_i\|_2$$

---

## 选择性计算与复用策略

### ASE 误差模型

每个 partition group $G_i$ 的 Attention State Error 定义为：

$$\text{ASE}(G_i, t) = \sum_{j \in G_i} w(j, t) \cdot \delta(j, t, t_c)$$

其中：

- $w(j, t) = \frac{e^{\mathbf{LSE}_t(j)}}{\sum_{k=1}^n e^{\mathbf{LSE}_t(k)}}$ 为 partition $j$ 的重要性权重
- $\delta(j, t, t_c) = \|\text{AS}_t(j) - \text{AS}_{t_c}(j)\|_2$ 为状态漂移量

### 权重和漂移量估计公式

由于直接计算 $w(j,t)$ 和 $\delta(j,t,t_c)$ 代价高昂，采用以下估计：

**权重估计（仅在 anchor step $t_a$ 更新）：**
$$w(j, t) = w(j, t_a), \quad \forall t \in (t_a, t_a + \tau_{\max}]$$

**漂移量估计（基于本地 partition 外推）：**
首先在 anchor step 计算比例因子：
$$\alpha(j) = \frac{\|\text{AS}_{t_a}(j)\|_2}{\|\text{AS}_{t_a}(i_{\text{loc}})\|_2}$$

然后外推远程 partition 误差：
$$\delta(j, t, t_a) = \alpha(j) \cdot \|\text{AS}_t(i_{\text{loc}}) - \text{AS}_{t_a}(i_{\text{loc}})\|_2$$

### 决策算法

给定误差阈值 $\epsilon$，每个 group 的决策为：

$$\text{Decision}(G_i, t) = \begin{cases}
\texttt{COMPUTE} & \text{if } \text{ASE}(G_i, t) > \epsilon \\
\texttt{REUSE}(t_{c,i}) & \text{otherwise}
\end{cases}$$

**Anchor step 触发条件：**
$$\text{IsAnchor}(t) = \begin{cases}
\text{True} & \text{if } \forall G_i: \text{ASE}(G_i, t) > \epsilon \\
\text{True} & \text{if } \exists G_i: t - t_{c,i} \geq \tau_{\max} \\
\text{False} & \text{otherwise}
\end{cases}$$

---

## Dynamic Group State Compose 公式

当内存使用率超过阈值 $M_{\max}$ 时，执行 group 合并操作。给定合并因子 $k$（通常为2），将 $k$ 个相邻 group 合并：

$$\text{GroupCompose}(g \rightarrow kg): AS_t(G_i) \oplus AS_t(G_{i+1}) \oplus ... \oplus AS_t(G_{i+k-1}) \rightarrow AS_t(G'_{\lfloor i/k \rfloor})$$

合并后 group 数量变为 $\lceil m/k \rceil$，内存占用约减少 $k$ 倍。

---

## 实现目标总结

| 目标                   | 实现手段                                                     |
| ---------------------- | ------------------------------------------------------------ |
| 减少跨节点通信量       | 对低重要性远程 partition 进行 attention state 复用，跳过 KV 传输 |
| 控制复用引入的质量损失 | 基于 ASE 误差模型的阈值决策 + anchor step 周期重置           |
| 降低缓存显存开销       | 以 attention state（半 KV 大小）代替 KV 缓存 + Dynamic Group Compose |
| 最大化 GPU 利用率      | 双流异步调度，复用操作填补通信气泡                           |
| 保证输出正确性         | 每步结束时所有 partition 均被 COMPUTE 或 REUSE 覆盖，确保完整性 |

|      |      |
| ---- | ---- |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |