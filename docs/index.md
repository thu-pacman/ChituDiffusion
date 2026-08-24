# ChituDiffusion

ChituDiffusion 在 Diffusers pipeline 生命周期内提供扩散模型生成、上下文并行和服务运行时。

三个功能边界如下：

- **EPE** 管理常驻服务中的请求队列、端到端 SLO、动态 lane 和 worker。
- **并行层** 提供静态 NCCL CP 和单机 Fast CP。它不管理请求调度。
- **FlexCache** 加速单卡或静态 CP 的 `generate`。它不创建序列并行，也不接入 EPE 服务。

用户入口：

- `chitu` 是正式命令行入口，实现位于 `chitu_diffusion.commands`。
- `chitu_diffusion` 根包导出模型 Pipeline、Request、FlexCache 配置和必要的服务 API。
- 根目录 `examples/` 保存可直接阅读和修改的用户示例。

从[快速开始](quickstart.md)安装并运行第一个请求。
