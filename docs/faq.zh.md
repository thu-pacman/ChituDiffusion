# 常见问题

## 安装与设置

### 问：支持哪些 Python 版本？

**答**：Python 3.8 或更高版本。推荐使用 Python 3.10。

### 问：支持哪些 GPU？

**答**：需要具有 CUDA 计算能力 8.0+ 的 NVIDIA GPU：
- A100、A6000（推荐）
- H100、H800（最佳性能）
- RTX 3090、RTX 4090（消费级）

### 问：最低内存要求是什么？

**答**：取决于模型和配置：
- **1.3B 模型**：最低 16GB VRAM（使用低内存模式）
- **14B 模型**：最低 24GB VRAM（使用低内存模式）
- **完整性能**：建议 40GB+ VRAM

### 问：可以在 CPU 上运行吗？

**答**：不建议。Smart-Diffusion 针对 GPU 推理进行了优化。CPU 推理会非常慢（慢 100 倍以上）。

## 性能

### 问：如何减少内存使用？

**答**：使用低内存模式：

```bash
python test_generate.py infer.diffusion.low_mem_level=2
```

### 问：如何加速生成？

**答**：几种策略：

1. **使用 SageAttention**：`infer.attn_type=sage`（性能测试中）
2. **减少推理步数**：`num_inference_steps=30`（原为 50）
3. **启用 FlexCache**：`flexcache='teacache'`
4. **降低分辨率**：`height=480, width=848`
5. **使用多个 GPU**：启用上下文并行

### 问：不同注意力后端有什么区别？

**答**：
- **flash_attn**：默认，准确，性能良好
- **sage**：量化 (INT8)，性能测试中
- **sparge**：稀疏 + 量化，性能测试中
- **auto**：自动选择最佳可用后端

### 问：为什么第一次生成很慢？

**答**：第一次生成包括：
- 模型加载
- CUDA 内核编译
- 初始化开销

后续生成会快得多。

### 问：可以批量处理多个提示吗？

**答**：目前不支持批处理。每个请求按顺序处理。数据并行计划在未来版本中推出。

## 使用

### 问：如何指定输出目录？

**答**：在配置中设置：

```python
args.output_dir = "/path/to/output"
```

### 问：支持哪些分辨率？

**答**：常见分辨率：
- **480×848**：标准，平衡质量和速度
- **720×1280**：高清，需要更多 VRAM
- **自定义**：必须是 8 的倍数

### 问：可以控制随机种子吗？

**答**：是的，用于可重现性：

```python
params = DiffusionUserParams(
    prompt="A cat",
    seed=42  # 固定种子
)
```

### 问：生成时长是多少？

**答**：取决于模型配置：
- 默认：81 帧（约 3-5 秒，30fps）
- 最大：121 帧（约 4-6 秒，30fps）
- 自定义：通过 `num_frames` 参数设置

## 多 GPU

### 问：如何使用多个 GPU？

**答**：使用 torchrun：

```bash
torchrun --nproc_per_node=2 test_generate.py \
    infer.diffusion.cp_size=2
```

### 问：哪些并行策略可用？

**答**：
1. **上下文并行 (CP)**：在 GPU 之间拆分序列维度
2. **CFG 并行**：在 GPU 之间拆分正/负提示
3. **数据并行**：计划中

### 问：可以混合并行策略吗？

**答**：是的！例如 4 GPU 设置：

```bash
torchrun --nproc_per_node=4 test_generate.py \
    infer.diffusion.cfg_size=2 \
    infer.diffusion.cp_size=2
```

## 故障排除

### 问：遇到"CUDA 内存不足"错误

**答**：
1. 增加 `low_mem_level`（最高为 3）
2. 降低分辨率或帧数
3. 使用 SageAttention (`infer.attn_type=sage`)
4. 启用上下文并行
5. 关闭其他 GPU 进程

### 问：生成挂起或冻结

**答**：
1. 检查 GPU 利用率：`nvidia-smi`
2. 验证所有 GPU 可见：`echo $CUDA_VISIBLE_DEVICES`
3. 检查日志中的错误
4. 尝试使用单个 GPU 隔离问题

### 问：输出质量较差

**答**：
1. 增加 `num_inference_steps`（例如 50）
2. 调整 `guidance_scale`（尝试 7.0-15.0）
3. 禁用 FlexCache 进行测试
4. 使用 FlashAttention 而不是量化后端

### 问：导入错误

**答**：
1. 验证安装：`pip list | grep -E "torch|diffusers"`
2. 重新安装依赖：`pip install -r requirements.txt`
3. 检查 CUDA 版本：`nvcc --version`
4. 重建扩展（如果使用）

## 配置

### 问：如何更改默认配置？

**答**：编辑配置文件：

```yaml
# chitu_core/config/infer.yaml
diffusion:
  low_mem_level: 2
  attn_type: sage
```

### 问：可以覆盖每个请求的配置吗？

**答**：是的，通过 `DiffusionUserParams`：

```python
params = DiffusionUserParams(
    prompt="A cat",
    num_inference_steps=30,
    guidance_scale=7.5,
    flexcache="teacache"
)
```

### 问：如何启用调试日志？

**答**：设置环境变量：

```bash
export CHITU_DEBUG=1
python test_generate.py
```

## 高级

### 问：可以添加自定义模型吗？

**答**：是的！请参阅 [自定义模型指南](../advanced/custom-models.md) 获取详细说明。

### 问：如何进行性能分析？

**答**：使用 PyTorch 分析器：

```python
with torch.profiler.profile() as prof:
    chitu_generate()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 问：支持量化吗？

**答**：是的，通过 SageAttention 和 SpargeAttention 支持 INT8 量化。INT4 量化计划中。

## 贡献

### 问：如何贡献？

**答**：
1. Fork 仓库
2. 创建功能分支
3. 进行更改
4. 添加测试
5. 提交拉取请求

请参阅 [开发者指南](../contributing/developer-guide.md) 了解详情。

### 问：如何报告错误？

**答**：在 GitHub 上提交问题：
- 包含完整的错误消息
- 提供重现步骤
- 指定环境详细信息（GPU、CUDA、Python 版本）

## 另请参阅

- [安装指南](../getting-started/installation.zh.md)
- [快速入门](../getting-started/quick-start.zh.md)
- [性能调优](../user-guide/performance-tuning.md)
- [多 GPU 设置](../user-guide/multi-gpu.md)
