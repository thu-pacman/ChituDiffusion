# `chitu_diffusion` 包结构

| 目录 | 职责 |
| --- | --- |
| `commands/` | `chitu generate` 与 `chitu serve` 的正式 CLI |
| `models/` | 模型 Request、Pipeline、tensor 布局和 executor 差异 |
| `epe/` | 动态 lane、SLO 调度、worker 和状态迁移 |
| `parallel/` | 静态 NCCL CP 与 Fast CP |
| `flexcache/` | request-local 缓存配置、session 和策略 |
| `serve/` | stage 配置、HTTP、torchrun 和 embedded 生命周期 |

## 公共 API

根包只导出模型 Pipeline/Request、FlexCache 配置和必要服务 API：

```python
from chitu_diffusion import (
    CacheConfig,
    EPEServeConfig,
    ZImagePipeline,
    ZImageRequest,
)
```

调度器、worker、lane lease 和 transport 是内部 API，应从其所属子包导入。模型包继续
导出 executor wrapper，供服务集成使用。

## 边界

- EPE 只用于常驻服务的动态 lane、SLO 和 worker 管理。
- `parallel/` 只实现 NCCL CP 与 Fast CP，不处理请求。
- FlexCache 只用于 `generate`，可叠加单卡或静态 CP，不接入 EPE 服务。
- `models/` 不实现调度、通信 transport 或缓存算法。
- 用户示例位于仓库根目录 `examples/`。

完整用法见仓库根 README 和 `docs/`。
