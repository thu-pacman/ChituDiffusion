# 更新日志

Smart-Diffusion 的所有重要更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)，
此项目遵循 [语义化版本](https://semver.org/spec/v2.0.0.html)。

## [0.1.2] - 2026-03-16

### 更改
- 启动入口统一为仅使用 `run.sh`
- 启动/系统参数统一由 `system_config.yaml` 管理
- 用显式的 `cfp -> infer.diffusion.cfg_size` 替代旧的 CFG 并行开关
- 文档启动示例统一为 `bash run.sh system_config.yaml ...`

## [0.1.1] - 2026-02-16 除夕

### 新增
- VBench 测评支持
- 为核心模块添加了全面的英文文档字符串
- 增强的 README.md，包括：
  - 改进的结构和格式
  - 全面的功能描述
  - 详细的安装说明，支持 uv
  - 使用示例和配置指南
  - 贡献指南和路线图
- 使用 MkDocs Material 的完整文档网站
  - 安装指南
  - 快速入门教程
  - 架构概览
  - 常见问题部分
  - 配置指南
- 用于自动文档部署的 GitHub Actions 工作流
- 支持文档搜索和代码高亮
- 中文文档和语言切换功能

### 更改
- 重组文档结构，划分清晰的部分
- 改进整个代码库的代码文档标准
- 修复文档中的图标渲染问题
- 将性能测试数据更新为"待测试"状态

## [0.1.0] - 2026-01-27

### 新增
- Smart-Diffusion 初始发布
- 支持 Wan-T2V 系列模型（1.3B、14B、A14B）
- 多种注意力后端支持：
  - FlashAttention（默认）
  - SageAttention（量化）
  - SpargeAttention（稀疏）
- 内存优化功能：
  - 带模型卸载的低内存模式
  - VAE 分块支持
  - 多级内存管理（0-3）
- 用于特征重用的 FlexCache 系统：
  - TeaCache 策略
  - 金字塔注意力广播（PAB）策略
- 并行支持：
  - 上下文并行（CP）
  - 分类器自由引导（CFG）并行
- 评估支持：
  - VBench 自定义模式评估
- 配置系统：
  - 基于 Hydra 的配置
  - 三层参数系统（模型/用户/系统）
- 任务管理：
  - 任务池和调度器
  - 用于分布式执行的请求序列化

### 已知问题
- 数据并行尚未实现
- 有限的模型支持（仅 Wan-T2V）
- 某些领域的文档不完整

## 未来路线图

### 计划功能
- [ ] Models
  - [ ] Flux-2
  - [ ] FireRed-Image-edit
  - [ ] Longcat
- [ ] AutoVideoParallel
  - [ ] DiTango
  - [ ] 混合并行组合
- [ ] FlexCache
  - [ ] 统一的缓存策略
  - [ ] 量化改进
- [ ] 生产功能
  - [ ] HTTP API 服务器
  - [ ] 批处理和请求排队
  - [ ] 监控和指标
- [ ] 更好的算子实现
  - [ ] 自定义 CUDA 内核
  - [ ] Triton 实现
- [ ] 全面的基准测试
  - [ ] 性能比较
  - [ ] 质量指标

### 文档改进
- [ ] 所有模块的完整 API 参考
- [ ] 更多使用示例
- [ ] 视频教程
- [ ] 社区贡献指南

## 贡献

有关如何为 Smart-Diffusion 做出贡献，请参阅 [贡献指南](contributing/developer-guide.md)。

---

有关详细的提交历史，请参阅 [GitHub 提交](https://github.com/chen-yy20/SmartDiffusion/commits/main)。
