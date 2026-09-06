# Parallel

该包按执行域分为 `cp`、`tp` 和 `vae`。模型必须从对应子包导入，不再从
`chitu_diffusion.parallel` 根包导入符号。根包只放跨域共用的主机事实：
`interconnect.py` 从驱动读出本机 GPU 之间的互连形态（有无 NVLink、每张卡的 NUMA
节点），供需要在"铺开传输"和"串行传输"之间取舍的调度使用——PCIe 主机一张卡的所有
peer 共用一个 egress 端口，NVLink 主机每个 peer 有独立链路，两者的最优调度相反。

## CP

`cp/` 负责 lane/process-group 生命周期、context-parallel attention 和 transport：

- `cp/nccl/`：Torch/NCCL K/V gather 与 Ulysses all-to-all，作为默认路径。
- `cp/fast/`：单机 NVSHMEM Fast AGKV/Fast Ulysses；Fast Ring 等代码仅供实验。
  AGKV 走 SM 还是 copy engine 由 `interconnect.py` 探测的互连决定。CUDA/C++
  扩展源码在 `cp/fast/csrc/`。
- `cp/context.py`：创建候选 lane group 并跟踪 active lane。
- `cp/topology.py`：Ulysses/USP topology。
- `cp/agkv_transport.py` 与 `cp/ulysses_transport.py`：transport 协议和 factory。

Fast CP 构建条件和 benchmark 见 [`cp/fast/README.md`](cp/fast/README.md)。

## TP

`tp/` 负责 tensor-parallel topology、Column/Row/MergedColumn/Replicated linear
以及按 rank 加载 checkpoint。TP 不依赖 CP 或 VAE。

TP 的机制是公共的，策略按模型声明。`tp/plan.py` 接受一张
`TensorParallelPlan` 表——哪些 linear 按输出行切（column）、哪些按输入列切并
all-reduce（row）、哪些保持完整（replicated）、哪些 norm 的归一维正好是被切的那
一维（norm），以及哪些属性存着需要同步缩小的 head 数——然后完成模块替换；
`tp/loader.py` 按模块类型只从 safetensors 读本 rank 的分片，不先加载整份再切；
`tp/build.py` 把"meta 上建图、按表切、只读本 rank 分片"这套 diffusers 加载流程收
在一处。

表里没提到的 linear 会直接报错而不是默默 replicate。这一条是有意的：column 和
row 配错不会抛异常，只会算错，而上游新增一个投影时静默漏掉它正是最容易发生的
情况。模式按路径段匹配，`*` 不跨 `.`、`**` 跨任意段，所以 `**.attention` 选中
attention 模块而不会连它下面的投影一起选中。

`norm` 这一档针对的是切开之后分母就不对的归一，例如 Wan 的 qk-norm 在展开 head
之前就对整个 `heads × head_dim` 做 RMS。`TensorParallelRMSNorm` 用一次跨 rank 的
平方和归约补回来，代价是每个 token 一个标量，比紧随其后的 row-parallel
all-reduce 小一个 hidden size。同样有安全网：宽度正好等于同级 column linear 输出
宽度、却没有被列进 `norm` 的 per-feature scale 会直接报错，因为它几乎必然是漏
声明。

给一个新模型加 TP 因此是"写一张表 + 一个数值等价性测试"，而不是重写一遍模块图。
`validate_tensor_parallel_plan` 不需要 process group，可以在 meta 模型上把每个
degree 的覆盖和整除都过一遍；等价性测试则要跑真的分片。两者都不能省，但要注意各
自能抓什么：整模型的 fp32 等价性对"漏了跨 shard 归约"其实不敏感——错掉的 Q/K
scale 经过 attention 再进残差流，在小模型上只挪动输出 1e-4 量级——所以归约本身另
有单测直接对着 `nn.RMSNorm` 比。参考实现见
[`models/zimage/tensor_parallel.py`](../models/zimage/tensor_parallel.py) 和
[`models/wan/tensor_parallel.py`](../models/wan/tensor_parallel.py)。

## VAE

`vae/` 负责 decode lane 的最小 topology contract、空间 tile 规划、通信与 leader
重组。`VaeParallelPlacement` 统一两种归属：不指定 degree 时解码跟随当前 denoise
lane，指定 degree 时创建一个独立于 DiT TP/CP 的固定 decode group，例如 TP4×CP2
搭配 VAEP8。固定 group 的生命周期由 stage runtime 持有。具体 VAE 的 latent
归一化和单 tile decode 仍由模型包实现。

依赖方向固定为 `cp -> tp`；`vae` 持有独立 decode group；`tp` 不反向依赖
其他并行域。EPE 决定 lane，parallel 层不管理请求、SLO 或 worker。
