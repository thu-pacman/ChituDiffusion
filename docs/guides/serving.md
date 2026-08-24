# 服务

`chitu serve` 启动 EPE 常驻服务：

```bash
chitu serve --stage-config examples/stage-zimage.yaml
```

stage 配置定义 GPU、模型工厂、EPE lane、SLO 和 HTTP 地址。最小结构如下：

```yaml
name: image
process: image
factory: zimage
gpu: [0, 1, 2, 3]
parallelism:
  sp: 4
  chitu_pool:
    policy: elastic
    allowed_lane_widths: [1, 2, 4]
    warmup_resolutions: [512]
    warmup_steps: 3
factory_args:
  model_path: /path/to/Z-Image
service:
  host: 0.0.0.0
  port: 18200
```

所有 rank 进入服务生命周期。leader 提供 HTTP 并管理队列，其他 rank 执行 worker
loop。`policy` 可选 `elastic`、`static_cp` 或 `static_dp`。

EPE 服务不接受 FlexCache 配置。该限制避免缓存状态跨排队请求或动态 lane 共享。
服务目前不保证任意 rank 故障后的恢复。

宿主已管理进程时，可参考根目录 `examples/epe_embedded.py` 使用
`EmbeddedDiffusionRuntime`。
