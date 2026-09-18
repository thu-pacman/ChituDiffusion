# FreeCache 离线 preprocess

内置 Preview 可以直接使用，无需重新标定。这里提供两条独立路径：
**复现已发布配置**，以及**采集数据、拟合并验证新的候选配置**。
工具位于源码仓库 `tools/freecache/`，不依赖本机研究目录，不随推理 wheel 安装。
所有命令从仓库根目录执行，使用已安装项目依赖的 Python 环境。

## 复现现有三模型配置

仓库包含约 4 MiB 的紧凑标定输入：FLUX.1-dev、Qwen-Image、Z-Image 每模型
40 条完整轨迹的速度 Gram 矩阵、真实 sigma 网格、传播响应 CSV、历史选定预设和来源哈希。
不包含模型权重、生成图片或完整高维速度张量。

```bash
python -m tools.freecache.preprocess reproduce --check-runtime \
  --output outputs/freecache-reproduction/profiles.json
```

此命令无需 GPU：校验输入 SHA256，重新计算任意预算的节点排序，应用历史预设覆盖，
逐项核对运行时的 **3 模型 × 50 个预算**。输出已有文件会被拒绝覆盖。
`profiles.json` 中的 `compiled` 保存排序和系数索引，`profiles` 保存可读的具体调度。

F10/F17/F25 是历史选定结果；其他预算使用传播加权目标的贪心删点排序。
复现命令不重新进行那九个预设的历史质量筛选，也不声称它们来自同一条优化路径。
详细来源见仓库 [`tools/freecache/data/README.md`](https://github.com/thu-pacman/ChituDiffusion/blob/main/tools/freecache/data/README.md)。

## 实际拟合方法

在完整计算轨迹上记录 scheduler 消耗的 CFG 后速度和 sigma，以
`G[i,j] = <v_i,v_j>` 压缩。候选预测只使用此前真实 Fresh 的两个速度：

```text
v_hat_i = v_a + lambda * (sigma_i - sigma_a)/(sigma_a - sigma_p) * (v_a - v_p)
lambda ∈ {0, .25, .5, .75, 1}
```

传播测量沿归一化的 `v[i-2] - v[i]` 方向，在第 i 次 Euler 更新后注入 latent 扰动，
再完整计算到终点。注入幅度由更新前 latent 范数乘 `relative_epsilon` 决定。
CSV 同时记录注入幅度、实际低精度加法后的幅度、终点偏差和偏差/幅度比值。

为忠实复现 v2，当前拟合取 **`final_deviation`，不是 `gain`**，按位置平均、归一化，
对数插值并在两端外推为 `w_i`。目标为：

```text
d_i = v_hat_i - v_i; h_i = sigma_(i+1) - sigma_i
J² = mean_trajectory [ sum_(i,j) rho^|i-j| h_i h_j w_i w_j <d_i,d_j>
                      / ||sum_i h_i v_i||² ]
```

`rho` 是保留跨步误差相关性的经验参数。历史值为 Z-Image 0、Qwen 1、FLUX .98；
它们参考过历史候选的质量排序，不能视为任意新模型的默认值。
这个目标使用完整计算轨迹上的缺陷，不是实际 cache 轨迹的完整误差传播，
也不是终点质量保证。新 collector 用 CPU float64 累积 Gram；随仓库发布的历史矩阵
来自原有 float32 累积，CPU 复现始终使用原始数据。

拟合从全 Fresh 开始逐个删除节点，最小化代理目标；连续 warmup 前缀不可删除，
且计入总预算。每个 lambda 得到一条排序，再为每个预算选择代理目标最小的 lambda。
没有全局最优或质量单调保证。历史任意预算编译只固定前两步；**新候选必须显式设置
`--warmup`，不能把两步默认值当成已验证的通用先验。**

## 新候选的标准流程

采集前固定 checkpoint/revision、sampler、实际 sigma 网格、分辨率、CFG、精度、
后端、开发/留出 prompt 和 seed、总成本预算及验收条件。
目前 GPU 前端接通 FLUX.1-dev / Qwen-Image / Z-Image 的单卡 50-step 确定性
FlowMatch Euler。数学拟合不读取 DiT 内部特征；增加其他模型仍需实现并验证采集适配，
不表示 Wan/LLaDA 或任意 flow-matching 模型已经完成同等标定。

以下命令展示工具链。`warmup=10`、`coherence=0`、`epsilon=.1` 是显式示例，
不是重新搜索或推荐出的最优参数。不要将一次 smoke test 当成质量确认。

### 1. 准备开发和留出数据

创建 `calibration.json`，格式如下；实际标定应包含多种场景：

```json
[
  {"id": "simple", "prompt": "A red cube on a white table."},
  {"id": "detail", "prompt": "A crowded outdoor market with people and handwritten signs."}
]
```

另建 `holdout.json`，使用未参与选择的新 prompt；示例采用不同的 seed。
历史 `tools/freecache/data/prompts.json` 已是开发数据，不能再称为独立留出。

### 2. 采集完整轨迹

```bash
sbatch --partition=debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=8 --mem=96G --time=01:00:00 \
  --wrap='python -m tools.freecache.collect trace \
    --model zimage --model-path /path/to/Z-Image \
    --prompts calibration.json --seeds 42 44 \
    --width 1024 --height 1024 --guidance 5 \
    --output outputs/zimage-calibration/trace'
```

模型权重须预先准备在本地。输出 `traces.json` 和 `protocol.json`，记录版本、请求、
实际网格、配置哈希、作业号、GPU 和包括加载的耗时。采集器检查观测 hook 前后结果相同，
并在失败时恢复 hook。配置哈希不覆盖权重内容；请另行保留 checkpoint revision 或权重清单。

### 3. 测量传播响应

等待 trace 完成后运行；示例默认测量第 4/6/8/10/13/16/20/24/28/32/36/40/43/45/47/48 步：

```bash
sbatch --partition=debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=8 --mem=96G --time=01:00:00 \
  --wrap='python -m tools.freecache.collect propagation \
    --model zimage --model-path /path/to/Z-Image \
    --prompts calibration.json --seeds 42 \
    --width 1024 --height 1024 --guidance 5 \
    --relative-epsilon .1 --output outputs/zimage-calibration/propagation'
```

输出 `propagation.csv` 和协议。有限幅度、方向、早期位置及外推区间都影响解释；
不能将旧模型的权重直接套到新模型，也不能只用几个中后期点声称覆盖早期传播。

### 4. CPU 拟合候选

```bash
python -m tools.freecache.preprocess fit \
  --traces outputs/zimage-calibration/trace/traces.json \
  --propagation outputs/zimage-calibration/propagation \
  --coherence 0 --warmup 10 --budgets 17 25 31 \
  --output outputs/zimage-calibration/candidates-w10.json
```

输入必须完成，且模型、网格和采样配置相符；budget 必须不小于 warmup。
输出标记为 `candidate_requires_heldout_validation`，不会写入运行时。
warmup 和 coherence 的选择应在开发数据上进行，先与同预算 warmup＋周期 ZOH 对照，
不能仅凭更低的代理目标晋级。若比较多个 warmup，提前固定候选和总预算。

### 5. 配对质量验收

```bash
sbatch --partition=debug --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=8 --mem=96G --time=01:00:00 \
  --wrap='python -m tools.freecache.evaluate \
    --model zimage --model-path /path/to/Z-Image \
    --prompts holdout.json --seeds 53 54 \
    --width 1024 --height 1024 --guidance 5 \
    --candidates outputs/zimage-calibration/candidates-w10.json \
    --lpips --meancache --output outputs/zimage-calibration/validation'
```

LPIPS 需安装项目 `eval` extra 并准备 AlexNet 权重。评测包含 Origin、F50 精确一致检查、
候选，以及相同总预算/相同 warmup 长度的周期 ZOH。`--meancache` 增加官方支持预算；
Qwen/FLUX 可用 `--magcache` 增加默认 MagCache。不支持的配置不会伪造测量。
若两种方法的预测系数也不同，这个对照比较整体策略，不单独识别调度贡献。

保存原始 float32 RGB `.npy`、配置、实际缓存统计、逐样本 PSNR 和可选的原尺寸 Alex LPIPS，
不缩放、不裁剪。Origin 的无限 PSNR 用 `null`＋`exact=true` 表示。
计时在完整 generate 前后同步 CUDA，排除加载、保存、评分；每个单元一次计时，
不能据此声称稳定性能或置信区间。异常需保留并重复检查。该工具不自动绘图或发布前沿。

候选只有通过独立配对评测才具备发布依据。固定预算的代理改善、更多 Fresh 或更长 warmup
都不能替代终点验证；保留失败场景。使用过的留出集随后应视为开发证据。

## 成本和发布边界

每个 GPU 作业只使用 `debug` 单卡；作业超时会留下 `complete=false` 的部分产物，拟合拒绝使用。
各阶段的加载、采集、开发集筛选、重试都应计入 preprocess 总成本，不能把多个一小时作业
说成总共一小时。提交前按一次请求实测耗时规划样本数；一小时限制不会自动证明任务能完成。
历史发布输入来自多轮研究，尚无从零开始总成本不超过一张 H20 一小时的完整核算。
纯 CPU 复现不需要重新支付采集成本，但也不能抹除历史成本。

运行时接入显式 profile：

```python
import json
from chitu_diffusion import CacheConfig, FreeCacheConfig, FreeCacheProfile

data = json.load(open("outputs/zimage-calibration/candidates-w10.json"))
selected = next(p for p in data["profiles"] if len(p["fresh_steps"]) == 25)
cache = CacheConfig(
    strategy="freecache",
    params=FreeCacheConfig(profile=FreeCacheProfile(**selected)),
)
# 将 cache 传入对应模型的 Request。仅在完成上述验收后用于目标配置。
```

工具输出永远不会自动替换 `presets.py` / `budget_profiles.py`。本次公开的是复现与标定工具链，
没有重新标定现有发布配置，也没有新增质量提升结论。
