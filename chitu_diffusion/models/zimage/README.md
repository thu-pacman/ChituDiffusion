# Z-Image

该包保存 `ZImagePipeline`、`ZImageRequest`、executor、transformer tensor 布局和
attention glue。EPE 只通过公共 executor contract 使用该包。

```bash
chitu generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --output outputs/zimage.png
```

服务使用统一入口：

```bash
chitu serve --stage-config examples/stage-zimage.yaml
```

## Tensor parallelism

`tensor_parallel.py` 声明了一张 `TensorParallelPlan`：attention 的 Q/K/V 与
SwiGLU 的 `w1`/`w3` 按输出行切，`to_out.0` 与 `w2` 按输入列切并 all-reduce，
modulation、embedding 和输出头保持完整——因此 block 之间传递的始终是完整激活，
不需要额外 gather。它的 qk-norm 在展开到 head 之后才归一，所以 norm 无需跨 shard
归约。加载时先在 meta 设备建图、套用这张表，再由公共 loader 只读本 rank 的分片，
峰值显存不会经过完整权重。

degree 必须整除 30 个 attention head，所以 2、3、5、6 可用，4 和 8 不可用。

```bash
torchrun --nproc_per_node=2 -m chitu_diffusion.commands.generate.zimage \
  --model-path /path/to/Z-Image --tensor-parallel-degree 2 --no-cfg-parallel \
  --output outputs/zimage-tp2.png
```

正确性以 fp32 为准：`tests/distributed/test_zimage_tensor_parallel_cpu.py` 用小
模型比对切分与稠密前向（1e-4 以内），8×Pro5000 上 fp32 的 TP2 与单卡出图逐像素
平均差 0.006/255。bf16 下 TP2 与单卡的差异（1.73/255）小于 bf16 本身相对 fp32 的
偏移（2.61/255）。

不存在模型专用服务入口。EPE、parallel 和 FlexCache 的边界见 `docs/features/`。
