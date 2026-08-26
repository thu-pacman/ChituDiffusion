# MiniMax-H3 VAE Parallel on RTX PRO 5000

## 实现

H3 video VAE 按 release 配方切成 256 px、至少 64 px overlap 的二维空间
tile。VAEP 使用独立于 DiT TP/CP 的 process group；tile 以 row-major 编号并
round-robin 分配，各 rank 顺序 decode 本地 tile，经一次等形状 all-gather 后，
仅 leader 按 release 的先纵向、后横向顺序 blend。Audio VAE 保持 leader 单卡。

`vae_parallel_degree` 独立配置并行度。TP4×CP2 配置使用 VAEP8；768×768 输入
产生 4×4 共 16 个 tile，每个 VAEP rank 处理 2 个。

## VAE-only 真权重结果

输入为 `[1, 24, 37, 48, 48]`，输出为 `[1, 3, 124, 768, 768]`：

- release 串行 tiled decode：8139.29 ms
- VAEP2：4190.51 ms，**1.94×**
- VAEP8：1109.18 ms，**7.34×**
- 最大绝对误差：`2.3841858e-7`
- MSE：`3.4929561e-17`
- PSNR：164.57 dB
- VAEP8 rank 0 peak allocated 16.57 GB，其余 rank 约 15.69 GB；串行
  rank 0 为 17.24 GB

原始 benchmark 数据写入 `outputs/`，不纳入版本库。

## TP4×CP2 端到端验收

Fast Ulysses、FA4、768×768、5 秒、24 FPS、20 steps：

- 请求总延迟：49.51 秒
- denoise：约 2.030 秒/transition
- GPU finalize：1.810 秒，其中 VAE 1.578 秒
- 输出：H.264 768×768 24 FPS，AAC 32 kHz stereo
- 本地验收产物写入 `outputs/h3-vae-parallel/`，不纳入版本库

此前同步 Fast Ulysses 同规格预热结果为 55.63 秒，VAEP8 缩短 6.12 秒
（11.0%）；相对 VAEP2 的 52.90 秒再缩短 3.39 秒（6.4%）。VAEP2 和 VAEP8
产出的 MP4 SHA256 完全一致。全模型启动后每卡驻留约 46.1 GB；请求后约
51.7 GB（rank 0 为 54.0 GB），在 72 GB 显存内仍有余量。

## 复现 VAE-only

```bash
export NCCL_IB_DISABLE=1

/dockerdata/chitudiffusion-venv-sm120/bin/torchrun \
  --standalone --nproc-per-node=8 \
  experiments/vae_parallel_pro5000/benchmark_h3.py \
  --degree 8 \
  --output outputs/vae-parallel-pro5000/results-vaep8.json
```
