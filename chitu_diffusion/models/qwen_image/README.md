# Qwen-Image

该包保存 Qwen-Image 的 Request、Pipeline、executor、transformer tensor 布局和
attention glue。EPE 调度和 CP transport 位于公共运行时。

```bash
chitu generate \
  --model qwen-image \
  --model-path /path/to/Qwen-Image \
  --output outputs/qwen-image.png
```

模型包不创建 process group，也不实现 FlexCache 策略。缓存支持范围见
`docs/features/flexcache.md`。

context parallelism 默认同时分片 image 和 text sequence。text 长度不能被当前
lane width 整除时回退为 replicated text；`shard_text=False` 可显式关闭分片。

图文同时分片时，processor 使用 packed QKV projection，并将配对 QK RMSNorm、
RoPE 和图文 joint QKV 布局转换融合，直接生成连续的 Q/K/V 供 CP attention 使用。

模型 block benchmark 还会显式安装 `prepare_qwen_image_sharded_agkv`：准备阶段实测
packed projection 和 CE/Q projection overlap 两种完整 block 调度，按所有 rank 的
最大耗时选择并缓存。首次调优会创建两种候选的 symmetric buffer，直到 runtime
关闭才回收；可显式指定 `packed` 或 `overlap` 跳过调优。该 prepared processor
要求固定输入存储，目前不是完整生成 pipeline 自动安装的默认 processor。
