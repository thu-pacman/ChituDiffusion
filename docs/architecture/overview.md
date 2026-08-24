# 架构总览

```text
Diffusers pipeline lifecycle
          |
models: request、tensor layout、executor
          |
   +------+------------------+
   |                         |
generate                  serve
静态 NCCL/Fast CP          EPE 动态 lane
可选 FlexCache             队列、SLO、worker
```

目录职责：

- `chitu_diffusion/models/`：模型差异。
- `chitu_diffusion/epe/`：动态 lane、SLO 调度和 worker runtime。
- `chitu_diffusion/parallel/`：静态 NCCL CP 与 Fast CP 的通信和 attention。
- `chitu_diffusion/flexcache/`：request-local 缓存策略。
- `chitu_diffusion/serve/`：配置、HTTP 和分布式服务生命周期。
- `chitu_diffusion/commands/`：正式 CLI 实现。
- `examples/`：用户示例。

这些边界避免模型实现复制调度和通信代码，也避免缓存策略改变分布式控制流。
