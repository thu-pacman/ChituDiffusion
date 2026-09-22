# CP Core Benchmark Harness

该 harness 用合成 Q/K/V 驱动真实 `ImageContextParallelAttention`，用于在不加载模型权重和
去噪 pipeline 的情况下验证 CP correctness、通信/计算 overlap 与 CUDA Graph 调度。

保留它是因为它仍是可复用的 core-level regression/benchmark 入口；一次性 probe、trace 和
microbenchmark demo 已在阶段结束后清理。

## 文件

- `cp_async_harness.py`：correctness、benchmark、component profile 和 trace 入口。
- `overlap_cp.py`：harness-only overlap/graph 调度，被主 harness 直接依赖。
- `run_harness.sh`：单节点 `torchrun` 启动器。

## 使用

```bash
# correctness
bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode agkv --check
bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode ulysses --check

# benchmark
bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode agkv --bench
bash experiments/fast_cp/harness/run_harness.sh 8 -- --mode ulysses --bench
```

### fast-ulysses

`CHITU_HARNESS_A2A=fast_ulysses` uses the optional
[`triple-mu/fast-ulysses`](https://github.com/triple-mu/fast-ulysses) backend.
It requires a single NVLink node, NVSHMEM 3.7+, a built `fast_ulysses`
extension, and fp16/bf16 4D tensors. The CP groups may be the contiguous,
disjoint WORLD partition produced by CFP2.

```bash
CHITU_HARNESS_A2A=fast_ulysses \
bash experiments/fast_cp/harness/run_harness.sh 8 -- \
  --production-group --mode ulysses --check --bench
```

The generic parallel Ulysses routing check can be launched with:

```bash
PYTHONPATH="$PWD" CHITU_CP_ULYSSES_BACKEND=fast_ulysses \
script/srun_direct.sh 1 2 test/test_fast_ulysses_parallel.py

# CFP2: two independent CP subgroups on four GPUs
PYTHONPATH="$PWD" CHITU_CP_ULYSSES_BACKEND=fast_ulysses TEST_CFG_SIZE=2 \
script/srun_direct.sh 1 4 test/test_fast_ulysses_parallel.py
```

The production Ulysses transport defaults to capability-gated auto selection:

```bash
export CHITU_CP_ULYSSES_BACKEND=auto  # Hopper + P2P + extension, else NCCL
export CHITU_FAST_ULYSSES_USE_TMA=auto
```

Use `fast_ulysses` instead of `auto` to require the extension and fail on an
unsupported host, or `nccl` to disable it. Install the optional package with
`NVSHMEM_HOME=/path/to/nvshmem uv sync --extra fast-ulysses`; detailed hardware
and version requirements are in `docs/install_and_launch.md`.

`CHITU_FAST_ULYSSES_USE_TMA` accepts `auto`, `tma`/`1`, or `non_tma`/`0`.
For an end-to-end transport A/B, pin `infer.cp_backend=ucp`; the default
`cp_backend=auto` may select AGCP at small CP sizes and bypass Ulysses entirely.
`CHITU_FAST_ULYSSES_POOL_BYTES` controls the WORLD-wide symmetric arena and
defaults to 2 GiB per PE. Unsupported non-contiguous/overlapping subgroup,
multi-node, or import configurations fail explicitly. 3D varlen, FP8, and CUDA
Graph-captured all-to-all calls remain on NCCL because the fast kernel supports
only eager fp16/bf16 4D swaps. The model's attention head count must also be
divisible by the CP size for its
Ulysses mode to be selected. Generic pure-Ulysses attention reuses the
fast-enabled CP group instead of allocating a second NVSHMEM heap for the
equivalent full-size UP group. Unified Ulysses+Ring keeps its smaller UP
subgroup on the existing NCCL path.

默认形状近似 Z-Image 1024×1024。结果只表示单次 CP attention 的机制与理论上限，最终性能
仍须在真实模型的端到端 denoise 墙钟上确认。

## 实验纪律

1. 先用 `--check` 确认 serial 与候选调度数值一致。
2. 再做 component profile，判断通信是否值得 overlap。
3. 只在需要确认调度关系时生成短 trace；trace 写入 `outputs/`，不提交。
4. 最后在同一节点、同一模型配置下做真实模型 A/B。
