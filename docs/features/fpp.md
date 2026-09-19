# Wan FPP：跨步细粒度流水线

本实现将 **SmartDiffusion `exp/fpp@b64ddaa268516901a19372e421af56eaaf6f43c9`**
的最终 FPP 算法移植到现在的 Diffusers Wan 2.1 模型与请求接口。Transformer 分层加载、
`WanDenoiseState`、标准 Diffusers 数值求解器和现有生成/解码入口仍是基础。

`pipeline_parallel_degree > 1` 或显式 `fpp_config` 启用 FPP。默认
`schedule="stream"` 在连续分块步骤间保持流水线，不在每个去噪步收齐全 world。
普通生成未启用 FPP 时仍走原有 EPE 路径。

这是有损的实验性执行路径。真实 BF16 对照已观察到条纹和主体形状偏移，尚不能承诺
画质或加速收益；参见[质量记录](../validation/wan-fpp.md)。

## 算法与来源

- 首级在最初 P 个 patch 使用完整预热后的 latents；随后每处理一个 patch，消费一个
  末级回传的候选 latents，再重新计算 patch embedding。P 是流水线级数。
- 各层按请求与 CFG 分支保存滚动 KV，当前 patch 原位更新；后续 patch 可见此前已更新的 KV。
- 末级另存完整的最后一层 token 特征。每完成一个 patch，先更新特征缓存，再以**当前时间步**
  的调制参数计算完整输出头。缓存的不是上一步已经投影的噪声预测。
- 同一逻辑步的每个候选都使用相同的起始 latents 和正式采样历史。前 M−1 个 patch
  只预览，最后一个 patch 提交一次；预览不会改变 UniPC 的输出历史、步号或阶数。
- 首步遍历 `0..M−1`，每个分块步之后令
  `order = order[M-P:] + reversed(order[:M-P])`。刷新不会重置该顺序。
- 预热、周期刷新和收尾使用完整序列；在分块段末尾排空剩余 P 个反馈，再同步 latents。
  修正了旧分支长请求刷新时 `save_cache` / `init_cache` 参数不一致的问题。

CP 使用等长条带，每条带再分成 M 个等长 patch。全序列补齐到 `CP × M` 的倍数，
embedding 补零，RoPE 补 `cos=1, sin=0`；与旧分支一致，补齐位置参与注意力，输出头前移除。
例如 7800 tokens、CP=2、M=21：补齐为 7812，每个 CP 条带 3906，每个 patch 186。
这与无补齐的 dense baseline 可能产生差异，属于需要测量的近似误差。

## 运行

8 卡 CFG=2、PP=2、CP=2 示例（4 卡 PP=4 可改为 PP=4、CP=1、关闭 CFG 并行、M=7）：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m chitu_diffusion.commands.generate.wan \
  --model-path /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --pipeline-parallel-degree 2 --context-parallel-degree 2 --cfg-parallel \
  --fpp-patches 3 --fpp-warmup-steps 2 --fpp-cooldown-steps 2 \
  --sample-solver unipc --flow-shift 3 --guidance-scale 7.5 \
  --height 480 --width 832 --frames 81 --steps 50 \
  --prompt 'A cat walking on grass.' \
  --save-frames --output outputs/wan-fpp.mp4
```

所有 rank 执行相同的 Python 代码，输出由 rank 0 保存：

```python
import torch
from chitu_diffusion import FppConfig, WanPipeline, WanRequest

pipeline = WanPipeline.from_pretrained(
    "/path/to/Wan2.1-T2V-1.3B-Diffusers",
    torch_dtype=torch.bfloat16,
    pipeline_parallel_degree=2,
    context_parallel_degree=2,
    cfg_parallel=True,
    fpp_config=FppConfig(patches=3, warmup_steps=2, cooldown_steps=2),
    sample_solver="unipc",
    flow_shift=3,
)
try:
    result = pipeline.generate(WanRequest(
        prompt="A cat walking on grass.", num_frames=81, num_steps=50,
        guidance_scale=7.5,
    ))
    if pipeline.parallel_context.rank == 0:
        print(pipeline.last_fpp_stats)
finally:
    pipeline.close()
```

rank 布局为 `rank = (cfg_rank × PP + pp_rank) × CP + cp_rank`。
必须满足 `world_size = CFG × PP × CP`，CFG 为 1 或 2；PP 不超过 Transformer 层数。
CP 只在本级、本 CFG 分支内收集 KV；CFG 合并在末级对应 CP rank 之间进行。
反馈使用独立通信组，保留发送张量直至通信完成。

| 设置 | 默认 | 含义 |
| --- | --- | --- |
| `pipeline_parallel_degree` | 1 | 实际模型分层数 |
| `context_parallel_degree` | 1 | FPP 每级的 CP 度数；普通 EPE CP 使用 lane 配置 |
| `cfg_parallel` | FPP 时 false | true 时两组流水线分别计算正/负分支 |
| `FppConfig.schedule` / `--fpp-schedule` | stream | 跨步反馈；step 为先前同步实现的显式对照 |
| `patches` / `--fpp-patches` | 4 | M≥P；M=1 为完整序列同步 PP |
| `warmup_steps` / `--fpp-warmup-steps` | 1 | 至少一个完整步以建立缓存与 UniPC 历史 |
| `cooldown_steps` / `--fpp-cooldown-steps` | 1 | 末尾完整步数，可为 0 |
| `refresh_interval` / `--fpp-refresh-interval` | 50 | 预热后每第 N 个位置改做完整刷新；0 关闭 |
| `rotate_patches` / `--fpp-rotate-patches` | true | 使用来源分支的级数相关遍历规则 |
| `sample_solver` / `--sample-solver` | stream 时 unipc，否则 euler | 支持确定性的 flow Euler 与 UniPC |
| `flow_shift` / `--flow-shift` | 8 | 新框架原默认；复现来源实验时显式使用 3 |

短请求的预热/收尾可重叠，不会改变总步数。`schedule="step"` 保留此前无补齐的
均衡 patch、每步同步和简单轮转语义，仅支持本地 CFG、CP=1，便于比较调度影响。

## 单卡参考与实验

单进程可用 `FppConfig(reference_stages=4, reference_context_degree=2, patches=7)`
模拟 PP=4、CP=2 的相同反馈延迟、遍历顺序及 token 布局。CLI 对应
`--fpp-reference-stages 4 --fpp-reference-context-degree 2`。
参考模式加载完整模型，仅用于数值/质量对照，不能反映多卡速度或显存节省。
旧 debug 路径根据 patch 数猜级数且提前轮转，这里显式指定实际级数。

以下脚本保留原 `experiments/exp02.sh` 的 11 组几何配置，以及最终分支的 20 组
warmup/cooldown 组合，使用现有 torchrun CLI 替代旧 Slurm/Hydra 和硬编码集群路径：

```bash
# 先输出计划；加 --execute 才实际生成。可用 --case 2cfg2cp2pp 选择一组。
python tools/benchmarks/wan_fpp.py --mode matrix \
  --model-path /path/to/model --output-dir outputs/fpp-matrix

# dense baseline + 单卡参考的 20 组 warmup/cooldown，种子及提示词保持一致。
python tools/benchmarks/wan_fpp.py --mode warmup \
  --model-path /path/to/model --output-dir outputs/fpp-quality \
  --reference-stages 4 --patches 7 --seeds 42 43

python tools/benchmarks/wan_fpp_quality.py \
  --reference outputs/fpp-quality/baseline-seed42.npz \
  --candidate outputs/fpp-quality/reference-w2-c1-seed42.npz \
  --output outputs/fpp-quality/metrics.json
```

CLI JSON 记录参数、实际调度和生成耗时，`--save-frames` 保存压缩编码前的浮点 NPZ。
PSNR 按 [0,1] 的完整视频计算，形状不一致会报错；相同视频用 `identical=true` 表示无限 PSNR。
可选 LPIPS 需要安装 `lpips` 并通过 `--lpips-checkpoint` 提供**包含 AlexNet backbone 的完整**
LPIPS state_dict；脚本不下载权重。比较时必须配对同 checkpoint、提示词、种子、步数、solver、shift。
计划中的耗时是未预热的端到端 generate，排除模型加载和文件编码，不能直接当作稳态去噪加速比。

## 实现范围与验证

- 支持 Wan 2.1 T2V、静态 `generate()`、本地/并行 CFG、AGKV CP、FP32/BF16。
  普通未启用 FPP 的 EPE、CP、TP 和 FlexCache 路径继续使用原接口。
- Diffusers checkpoint 在 meta 上建立模型，只加载本级 block；保留原生 FP32 conditioning
  与 RoPE 精度。原始 Wan checkpoint 仍先在 CPU 转换再分层，加载峰值未降低。
  文本编码器、VAE、conditioning 和输出头仍复制到每个 rank。
- FPP 的正式 scheduler 历史由末级持有；每段结束时全 rank 拿到一致 latents。
  不通过 EPE 的 `denoise_step`、state migration 或动态 lane 扩缩容执行。
- TP、Fast CP transport、FlexCache、弹性 serving、Wan 2.2 双模型、I2V、自定义
  attention kwargs 仍不支持；相关入口明确拒绝。DPM 不在已验证的预览求解器范围内。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/unit/parallel/test_fpp_stream.py \
  tests/unit/models/test_wan_fpp_config.py tests/distributed/test_wan_fpp_stream.py

# 八张 GPU；只创建微型随机权重，不下载 checkpoint。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CHITU_FPP_TEST_CUDA=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/distributed/test_wan_fpp_stream.py -k matches_serial
```

测试涵盖 2/3 级流水线、CFG×PP×CP=2×2×2、非整除 token/层数、FP32/BF16、重复请求、
真实 P2P、公开 generate 入口、精确的 P 个 patch 反馈延迟和 54 步周期刷新。
UniPC golden fixture 来自指定提交的原始求解器，覆盖不同 shift 和 6/50 步的候选输出与历史隔离。
这些验证证明微型模型的算法/接口一致性。已有 1.3B/14B 真实权重的历史质量记录，
但不等价于证明近似无损；移植到新主分支后的大模型质量和稳态速度仍需复测。
