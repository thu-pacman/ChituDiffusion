# Wan 2.1 T2V

该包保存 Wan 文本编码、video latent、scheduler、transformer、VAE finalize 和 Request
差异。共享 EPE、CP transport 和 FlexCache 算法位于各自公共包。

```bash
chitu generate \
  --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --height 480 \
  --width 832 \
  --frames 17 \
  --output outputs/wan.mp4
```

缓存 profile 与模型和步数绑定。使用前查看 `docs/features/flexcache.md`。

## Tensor parallel

`tensor_parallel.py` 里的一张 `TensorParallelPlan` 覆盖本包加载的 Wan 2.1
checkpoint——T2V 和 I2V、两种尺寸都在内。它们只在宽度和层数上不同，而这些值是从
模型上读出来的，不写在表里。

切法是常规的那一套：attention 的 q/k/v 和 FFN 的第一层按输出行切，attention 的
输出投影和 FFN 的第二层按输入列切并 all-reduce，conditioning 和 `proj_out` 保持
完整。I2V 的 `add_k_proj`/`add_v_proj` 跟着 attn2 的 head 一起切。

Wan 是公共 plan 里 `norm` 这一档存在的原因。它的 qk-norm 是
`rms_norm_across_heads`——`RMSNorm(heads × head_dim)` 作用在整个投影上，在拆成
head 之前。切了投影之后没有哪个 rank 还持有算 scale 所需要的完整向量，所以这些
norm 必须跨 rank 归约，而不是各自 replicate。归约的量是每个 token 一个标量，比
紧随其后的 row-parallel all-reduce 小一个 hidden size。

这一档也是有安全网的：`tp/plan.py` 会拒绝一个宽度正好等于同级 column linear 输出
宽度、却没有被列进 `norm` 的 per-feature scale。忘了声明 `norm_q` 不会静默算错，
会直接报错。

degree 需要同时整除 head 数和 FFN 宽度。后者才是实际的约束——14B 的 head 数 40
允许 TP5，但 13824 不允许——所以可用的是 14B 的 2/4/8 和 1.3B 的 2/4。

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.commands.generate.wan \
  --model-path /path/to/Wan2.1-T2V-14B-Diffusers \
  --tensor-parallel-degree 2 \
  --output outputs/wan-tp2.mp4
```

按 rank 读分片需要 diffusers 布局的 checkpoint；原始 Wan 发布布局是转换出来的，
不能直接切。
