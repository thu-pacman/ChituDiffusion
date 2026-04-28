# 为什么选择 ChituDiffusion？
[English Edition](./whySmart.md)

## Diffusion 推理的特点

Diffusion 推理是计算密集型任务，有以下几个特点：

1. **逐样本处理**：Batch 对 GPU 利用率提升不大，单样本流式处理就够用了。  
2. **长序列小模型**：激活序列很长但模型参数量相对较小，所以序列并行（Context Parallelism）是最经济的并行方式。  
3. **Attention 是瓶颈**：长序列场景下，Full Attention 占了 80% 以上的延迟，优化重点就是 Attention。  
4. **激活值变化小**：相邻去噪步之间的激活值变化不大，用 Feature Cache 这种简单方法就能明显加速。

## ChituDiffusion 的设计理念

### 三个优化方向：并行、算子、算法  
这三个方向可以单独优化，但配合起来效果最好。  
（具体技术细节会陆续更新，欢迎提 PR 一起完善。）

### 面向多用户、多任务的服务框架  
我们提供的是**常驻运行、支持热升级、可横向扩展**的 Diffusion 服务，不是每次都要冷启动的脚本。  
核心思路是把 Diffusion Pipeline 拆成多个可编排的阶段，用统一的调度器来管理：

- 让用户自己调节质量和效率的平衡：推理步数、CFG、Cache 比例都可以在运行时调整。  
- 让系统资源充分利用：不只是计算资源，显存、带宽、CPU 都要用起来。

## 开发指南

感谢你参与 ChituDiffusion 开源项目！为了让代码审查更顺利，请先了解一下参数分类：

| 参数类别 | 生命周期 | 配置位置 | 谁能改 | 最佳实践 |
|---|---|---|---|---|
| 模型参数 | 静态 | `chitu_core/config/models/<model>.yaml` | 不能改 | 跟权重绑定，改了就会出问题 |
| 用户参数 | 动态（每次请求） | `DiffusionUserParams` | 用户 | 只暴露必要的参数，别搞太复杂 |
| 系统参数 | 半动态（启动时） | `chitu launch args` | 运维/调度器 | 启动后不能改，避免分布式状态混乱 |

记住：  
每多一个参数就多一份文档、测试和使用负担。灵活性不等于参数越多越好。

### 目录结构

`/chitu_core` 是 Chitu 原生代码。`ServeConfig` 和 `ParallelState` 非必要不修改。
`/chitu_diffusion` 是我们基于 Chitu 逻辑搭建的 diffusion 框架，可以修改，但要保持基本结构。
* `chitu_diffusion_main.py`：系统初始化、启动、关闭等主要参数
* `backend.py`：基于系统参数搭建的后端，储存模型，调度任务。
