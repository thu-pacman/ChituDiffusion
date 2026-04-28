# 配置指南

本页说明 ChituDiffusion 的主要配置项。

## 三层配置体系

ChituDiffusion 使用三层配置：

## 1. 模型参数（静态）

位置：`chitu_core/config/models/<model>.yaml`

用途：定义模型结构（层数、维度、注意力头等）。

## 2. 用户参数（动态）

位置：`DiffusionUserParams`

用途：控制单次生成请求。

### FlexCache 统一参数

推荐使用 `flexcache_params` 进行配置：

```python
from chitu_diffusion.task import DiffusionUserParams, FlexCacheParams

DiffusionUserParams(
    prompt="A cat on grass",
    num_inference_steps=50,
    flexcache_params=FlexCacheParams(
        strategy="teacache",
        cache_ratio=0.4,
        warmup=5,
        cooldown=5,
    ),
)
```

语义约定：

- `warmup`: 前 N 步完整计算
- `cooldown`: 后 N 步完整计算
- `cache_ratio`: 0 表示质量优先，1 表示速度优先

兼容旧写法：

```python
DiffusionUserParams(
    prompt="A cat on grass",
    flexcache="teacache",
)
```

## 3. 系统参数（半静态）

位置：启动配置（命令行或配置文件）

用途：并行、算子、内存、评测等系统行为。

### 推荐的 `system_config.yaml` 模板

```yaml
launch:
    tag: my-exp
    num_nodes: 1
    gpus_per_node: 4
    python_script: test/test_generate.py
    enable_launch_log: false

parallel:
    cfp: 1  # 仅支持 1 或 2

infer:
    attn_type: flash_attn
    low_mem_level: 0
    enable_flexcache: true
    up_limit: 81

output:
    root_dir: outputs
    enable_run_log: true
    enable_timer_dump: true
    hydra_dump_mode: video_dir  # default/video_dir/off
```

与启动脚本 `run.sh` 对应关系：
- `launch.tag` 会导出为 `CHITU_RUN_TAG`，并作为输出目录前缀。
- `parallel.cfp` 会映射为 `infer.diffusion.cfg_size`。
- `infer.diffusion.cp_size` 会按 `(num_nodes * gpus_per_node) / cfp` 自动推导。

## 常用系统参数

### 注意力后端

```bash
infer.attn_type=flash_attn   # 或 sage / sparge / auto
```

### 低内存模式

```bash
infer.diffusion.low_mem_level=2
```

### FlexCache 全局开关

```bash
infer.enable_flexcache=true
```

仅当全局开关开启时，请求侧 FlexCache 才会生效。

启动后会通过 Hydra 覆盖为：

```bash
infer.diffusion.enable_flexcache=true
```

### 输出与运行元数据

- `output.hydra_dump_mode`:
    - `default`: 保留 Hydra 运行目录中的 `.hydra`
    - `video_dir`: 将 `.hydra` 移动到视频输出目录
    - `off`: 运行后清理 `.hydra`
- `output.enable_timer_dump=true` 时，会在每次输出目录中写入 `time_stats.csv`。
- `launch.enable_launch_log=true` 时，启动日志会写入 `output.root_dir/launch_<timestamp>.log`。

## Hydra 覆盖示例

```bash
python test_generate.py \
    models.name=Wan2.1-T2V-14B \
    models.ckpt_dir=/path/to/checkpoint \
    infer.attn_type=sage \
    infer.diffusion.low_mem_level=2
```
