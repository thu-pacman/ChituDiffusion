# 快速入门

本指南将帮助您在几分钟内使用 ChituDiffusion 生成您的第一个视频。

## 前置条件

开始之前，请确保您已经：

1. [安装了 ChituDiffusion](installation.md)
2. 下载了模型检查点（参见[模型下载](#模型下载)）

## 模型下载 {#模型下载}

ChituDiffusion 目前支持 Wan-T2V 系列模型：

| 模型 | 大小 | 下载链接 |
|-------|------|----------|
| Wan2.1-T2V-1.3B | 13亿参数 | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) \| [ModelScope](https://www.modelscope.cn/Wan-AI/Wan2.1-T2V-1.3B) |
| Wan2.1-T2V-14B | 140亿参数 | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) \| [ModelScope](https://www.modelscope.cn/Wan-AI/Wan2.1-T2V-14B) |
| Wan2.2-T2V-A14B | 140亿参数 | [Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) \| [ModelScope](https://www.modelscope.cn/Wan-AI/Wan2.2-T2V-A14B) |

将模型检查点下载到本地目录，例如 `/path/to/Wan2.1-T2V-1.3B`。

## 基本生成

### 步骤 1：创建测试脚本

创建名为 `test_generate.py` 的文件：

```python
from chitu_diffusion import chitu_init, chitu_generate, chitu_start, chitu_terminate
from chitu_diffusion.task import DiffusionUserParams, DiffusionTask, DiffusionTaskPool
from hydra import compose, initialize

# 初始化配置
initialize(config_path="config", version_base=None)
args = compose(config_name="wan")

# 设置模型检查点路径
args.models.ckpt_dir = "/path/to/Wan2.1-T2V-1.3B"

# 初始化后端
chitu_init(args)
chitu_start()

# 创建生成任务
user_params = DiffusionUserParams(
    role="user1",
    prompt="一只猫在草地上行走。",
    num_inference_steps=50,
    height=480,
    width=848,
    num_frames=81,
    guidance_scale=7.0,
)

# 将任务添加到池中
task = DiffusionTask.from_user_request(user_params)
DiffusionTaskPool.add(task)

# 生成直到完成
while not DiffusionTaskPool.all_finished():
    chitu_generate()

# 终止后端
chitu_terminate()

print(f"✅ 视频已保存至: {task.buffer.save_path}")
```

### 步骤 2：运行脚本

**单 GPU：**

```bash
bash run.sh system_config.yaml --num-nodes 1 --gpus-per-node 1 --cfp 1
```

**单机多 GPU：**

```bash
bash run.sh system_config.yaml --num-nodes 1 --gpus-per-node 2 --cfp 2
```

**多机 SLURM：**

```bash
bash run.sh system_config.yaml --num-nodes 2 --gpus-per-node 2 --cfp 2  # 4 个 GPU
```

### 步骤 3：查看输出

生成的视频将保存到：
```
./outputs/<时间戳>_<任务ID>.mp4
```

使用任何视频播放器打开文件以查看您的第一个生成！

## 自定义生成

### 更改提示词

修改 `prompt` 参数以生成不同的内容：

```python
user_params = DiffusionUserParams(
    prompt="日落时分的山脉，色彩鲜艳",  # 您的自定义提示
    num_inference_steps=50,
)
```

### 调整分辨率

支持的分辨率（宽度和高度必须是 8 的倍数）：

```python
# 标清
user_params = DiffusionUserParams(
    prompt="...",
    height=480,
    width=848,
)

# 高清
user_params = DiffusionUserParams(
    prompt="...",
    height=720,
    width=1280,
)
```

### 控制视频长度

调整 `num_frames` 以改变视频持续时间：

```python
# 短视频（约 2.7 秒，30fps）
user_params = DiffusionUserParams(
    prompt="...",
    num_frames=81,  # 默认
)

# 长视频（约 4 秒，30fps）
user_params = DiffusionUserParams(
    prompt="...",
    num_frames=121,
)
```

### 设置随机种子

对于可重现的生成：

```python
user_params = DiffusionUserParams(
    prompt="...",
    seed=42,  # 固定种子获得一致的结果
)
```

## 性能优化

### 使用 SageAttention

通过量化注意力加速生成：

```bash
python test_generate.py \
    models.ckpt_dir=/path/to/checkpoint \
    infer.attn_type=sage
```

### 启用低内存模式

如果 GPU 内存不足：

```bash
python test_generate.py \
    models.ckpt_dir=/path/to/checkpoint \
    infer.diffusion.low_mem_level=2
```

内存级别：
- **0**：无优化（最快，内存最多）
- **1**：VAE 分块
- **2**：CPU 卸载文本编码器
- **3**：积极卸载（最慢，内存最少）

### 启用 FlexCache

使用特征缓存加速：

```python
user_params = DiffusionUserParams(
    prompt="...",
    flexcache="teacache",  # 或 "PAB"
)
```

## 多 GPU 生成

### 上下文并行

在多个 GPU 之间拆分序列：

```bash
torchrun --nproc_per_node=2 test_generate.py \
    models.ckpt_dir=/path/to/checkpoint \
    infer.diffusion.cp_size=2
```

### CFG 并行

自动启用（当使用 2+ GPU 且 CFG 开启时）：

```bash
torchrun --nproc_per_node=2 test_generate.py \
    models.ckpt_dir=/path/to/checkpoint \
    infer.diffusion.cfg_size=2
```

### 混合并行

组合 CP 和 CFG 以实现 4 GPU 设置：

```bash
torchrun --nproc_per_node=4 test_generate.py \
    models.ckpt_dir=/path/to/checkpoint \
    infer.diffusion.cp_size=2 \
    infer.diffusion.cfg_size=2
```

## 故障排除

### 内存不足 (OOM)

**症状**：`RuntimeError: CUDA out of memory`

**解决方案**：
1. 增加 `low_mem_level`（最高为 3）
2. 降低分辨率或帧数
3. 使用 `infer.attn_type=sage`
4. 启用上下文并行

### 生成缓慢

**症状**：生成花费很长时间

**解决方案**：
1. 使用 `infer.attn_type=sage`
2. 启用 `flexcache='teacache'`
3. 减少 `num_inference_steps`
4. 使用多个 GPU

### 质量较差

**症状**：生成的视频质量不佳

**解决方案**：
1. 增加 `num_inference_steps`（尝试 50-100）
2. 调整 `guidance_scale`（尝试 7.0-15.0）
3. 使用更大的模型（14B 而非 1.3B）
4. 禁用缓存进行测试

## 下一步

现在您已经运行了第一个生成，接下来可以：

1. [探索配置选项](configuration.md)
2. [学习性能调优](../user-guide/performance-tuning.md)
3. [了解多 GPU 设置](../user-guide/multi-gpu.md)
4. [阅读架构概览](../architecture/overview.md)

## 示例提示

以下是一些示例提示，可帮助您开始：

```python
# 自然场景
"日落时分的山脉，色彩鲜艳，电影般的照明"

# 动物
"一只可爱的熊猫在竹林中吃竹子"

# 城市
"繁忙的城市街道，夜晚，霓虹灯，未来主义"

# 抽象
"五彩缤纷的烟雾漩涡，流动运动，艺术"
```

## 获取帮助

如果您遇到问题：

1. 查看 [常见问题](../faq.zh.md)
2. 在 [GitHub Issues](https://github.com/chen-yy20/SmartDiffusion/issues) 上搜索
3. 加入 [讨论](https://github.com/chen-yy20/SmartDiffusion/discussions)
