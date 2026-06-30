# 在 PCIe 上做并行优化：当卡间带宽不再廉价（以 Z-Image 序列并行为例）

> 这是「ChituDiffusion 工程实录」系列的第二篇。第一篇我们在 NVLink 多卡上，用 torch.compile +
> CUDA Graph 把 Flux 的端到端推理压到 9.74×；那篇的隐含前提是——**卡间通信足够快**。
>
> 但现实里大量机器是 **PCIe 互联**：卡间带宽只有 NVLink 的一个零头，通信轻易就成为主导项。
> 这一篇换个战场：在 PCIe 上为 Z-Image 打通**序列并行（context parallel）**，并回答一个朴素的问题
> ——**带宽变贵之后，并行还划不划算，瓶颈到底在哪，又能怎么优化？**
>
> 老规矩：全文每个数字都可一键复现，配置、命令、trace、代码都附在文末归档目录里。

<!-- 本文为骨架/大纲，随 worklog.md 的实验推进逐节填充。 -->

---

## 0. 为什么 PCIe 值得单独写一篇

- NVLink vs PCIe 的带宽量级差异，对集合通信（all_gather / p2p）意味着什么。
- 序列并行的通信特征：split-Q 后每层都要 all-gather KV——通信量随层数线性累加。
- 命题：在 PCIe 上，**通信占比**可能从"边角料"变成"主角"，优化思路也随之不同。

## 1. 负载与并行范式：Z-Image 的序列并行

- Z-Image single-stream 架构：image + text 在 diffusers 内部打包成联合序列 `[image, text]`。
- 对标 qwen_image 的 split-Q / all-gather-KV：本地 Q × 全量 KV，RoPE 按 rank offset 切片。
- 非侵入式注入：wrap `transformer.forward` + CP 感知 attention processor（不改 diffusers 源码）。

## 2. First thing first：把通信和计算拆开看

- profiling 插桩：QKV proj GEMM / attention compute / KV all-gather / output gather 四段计时。
- 计时口径：CUDA event + synchronize，`CHITU_Z_IMAGE_CP_PROFILE=1` 开关，关闭零开销。
- 跑测配置：cp2、关 CFG、5 步纯测性能。
- **实测数据**：通信占比、KV all-gather 有效带宽 vs PCIe 理论带宽。（见 worklog 1.4）

## 3. 诊断：PCIe 上的通信瓶颈长什么样

- 有效带宽与理论带宽的差距来源。
- 通信占比随分辨率 / 序列长度的变化趋势。

## 4. 优化（待展开）

- 计算-通信重叠？减少 gather 数据量 / 次数？拓扑感知？
- 每一项的收益与代价。

## 5. 结语

- PCIe 场景下并行优化的一般性结论。

---

## 附录 A：可复现环境与归档

- 硬件：PCIe 互联多卡（型号 / 卡数待填）
- 软件：torch / ChituDiffusion 版本待填
- 模型：Z-Image
- 关键配置：`test/configs/z_image_cp2_profile.yaml`
- profile 开关：`CHITU_Z_IMAGE_CP_PROFILE=1`

```bash
CHITU_Z_IMAGE_CP_PROFILE=1 chitu run test/configs/z_image_cp2_profile.yaml
```
