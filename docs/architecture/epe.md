# EPE

Elastic Parallel Engine 管理常驻服务的动态执行：

- 接收请求并记录端到端 deadline；
- 用启动 warmup 和在线测量建立 step、terminal 和 state-transfer 代价；
- 在 pulse 边界为请求分配 lane 和步数；
- 驱动 worker、状态迁移、完成和取消；
- 在 lane 变化时复用预先创建的 process group。

EPE 调度输入是 request profile、SLO 和测量代价。模型 tensor 布局属于
`models/`，collective 与 attention 属于 `parallel/`。

`generate` 使用固定 full-world lane，不经过队列和动态 lane planner。`serve` 才使用
EPE 的队列、SLO 和 worker 生命周期。

FlexCache 当前不接入 EPE 服务。服务对非 `none` 的 cache 配置直接报错。

EPE 是开发者预览能力。当前实现不保证任意 rank 故障后的恢复。
