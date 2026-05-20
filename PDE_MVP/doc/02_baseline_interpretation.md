# 三个 Baseline 解读：FNO、DeepONet、PI-DeepONet

## 1. FNO Baseline

### 核心思想

Fourier Neural Operator 将积分算子核参数化到 Fourier 空间。对 1D Burgers 这类周期或近周期空间场，FNO 的优势是：

- 低频模式能高效表示整体波形传播。
- 频域卷积对分辨率具有较好泛化潜力。
- 推理速度快，适合作为比赛主模型。

### 本地代码现状

当前仓库的 `FNO/` 是 `neuraloperator` 风格代码库，包含：

- `FNO/neuralop/models/fno.py`：FNO 模型定义。
- `FNO/neuralop/layers/spectral_convolution.py`：频谱卷积核心。
- `FNO/scripts/train_burgers.py`：Burgers 训练入口。
- `FNO/config/burgers_config.py`：Burgers 配置。
- `FNO/neuralop/losses/equation_losses.py`：包含 PDE 方程损失。

`train_burgers.py` 中已经支持：

- FNO 模型构建。
- L2/H1/初值/方程残差损失组合。
- AdamW、调度器、Trainer。
- 可选 multi-grid patching。

但它默认加载的是 mini Burgers 数据，配置里写着 `16x17` 小数据示例，不是本赛题的 PDEBench 200x256/1000x256 长窗格式。要用于比赛，需要重写数据层和输出层。

### 优势

- 最适合 Task 1 的固定物理环境预测。
- 推理速度容易满足 2 分钟限制。
- 可加入 PINO 风格的 Burgers 方程残差。
- 可做自回归一步模型，也可做直接多步模型。

### 风险

- 纯 FNO 长时 rollout 容易误差累积。
- 若只优化 MSE，第三段统计分数可能差。
- Task 2 测试缺失 `Nu`，标准 FNO 需要额外参数识别机制。

### 比赛改造建议

1. 数据适配：统一读取 HDF5，输出 `(N, T, X)`。
2. 输入窗口：使用前 10 步作为通道输入，输出未来 190 步。
3. 结构：先做 1D 空间 FNO + 时间解码头，后续再做时空 FNO。
4. 损失：分段 Rel-MSE + rollout MSE + 频谱损失 + 方程残差。
5. Task 2：增加隐式参数 encoder，从前 10 步估计 latent physics embedding。

## 2. DeepONet Baseline

### 核心思想

DeepONet 用 branch/trunk 结构学习算子：

- Branch net 编码输入函数，例如初始条件或短窗观测。
- Trunk net 编码查询点，例如 `(t, x)`。
- 两者内积输出查询点处解值。

它天然适合“给定函数，查询任意时空点”的算子学习。

### 本地代码现状

当前 `deeponet/` 是 Lu 等人的论文复现代码，包含：

- `deeponet/src/deeponet_pde.py`
- `deeponet/src/deeponet_dataset.py`
- `deeponet/seq2seq/`
- 多个传统 PDE/ODE 数据生成与 DeepONet 示例。

该代码依赖较老版本 DeepXDE，并且部分案例依赖 Matlab/TensorFlow 1。它更像论文复现仓库，不是开箱即用的 PyTorch 比赛工程。

### 优势

- 理论表达清晰，适合写 methodology。
- 可查询任意时空点，适合跨分辨率。
- 对 Task 3 可用前 20 步作为 branch 输入，`(t, x)` 作为 trunk 输入。

### 风险

- 对规则要求的完整网格预测，逐点 trunk 查询可能推理慢。
- 原仓库工程栈较旧，直接迁移成本高。
- 长时混沌预测中，点查询模型若没有时序稳定机制，容易统计崩溃。

### 比赛改造建议

1. 不直接依赖老 DeepXDE 工程，建议在 MVP 中实现轻量 PyTorch DeepONet。
2. Branch 使用 CNN/Transformer 编码初始窗口。
3. Trunk 使用 Fourier features 编码 `(t, x)`。
4. 输出阶段批量查询全部 `(T, X)` 网格，避免 Python 循环。
5. 作为 FNO 的对照实验，或作为 Task 2/3 的参数识别辅助分支。

## 3. PI-DeepONet Baseline

### 核心思想

PI-DeepONet 在 DeepONet 基础上加入物理约束，通过自动微分或离散差分将 PDE 残差作为软约束：

- Burgers：`u_t + u u_x - nu u_xx = 0`
- KS：`u_t + u u_x + lambda2 u_xx + u_xxxx = 0`

目标是减少纯数据驱动模型的物理漂移。

### 本地代码现状

当前 `PI-DeepONet/` 以 notebook 为主：

- `PI-DeepONet/Burger/PI_DeepONet_Burger.ipynb`
- `PI-DeepONet/Advection/PI_DeepONet_adv.ipynb`
- `PI-DeepONet/Diffusion-reaction/`
- `PI-DeepONet/Antiderivative/`

README 说明该项目已更新到 JAX 0.3.25，但 notebook 形式不适合直接作为比赛工程入口。

### 优势

- 非常契合赛题“物理信息约束”的叙事。
- 对长时稳定性有帮助。
- 可减少模型在未见参数下的物理不一致。

### 风险

- 自动微分 PDE 残差在完整 `(T, X)` 网格上代价高。
- Notebook 工程化困难。
- 若物理残差权重过大，会损害短期数据拟合。

### 比赛改造建议

1. 对规则网格使用离散差分残差，避免对所有点做高阶自动微分。
2. 残差只在随机采样的时间/空间点上计算，控制训练成本。
3. 对 Task 1 固定 `nu=0.001`。
4. 对 Task 2 用隐式 `nu_hat` 或 latent embedding 参与残差。
5. 对 Task 3 用 KS 频谱残差、能谱损失和统计损失，而不是全量四阶自动微分。

## 4. 三者在比赛中的推荐定位

| 模型 | 推荐角色 | 适用任务 | 原因 |
|---|---|---|---|
| FNO | 主力模型 | Task 1/2/3 | 快、稳定、容易输出完整网格 |
| DeepONet | 对照与参数化分支 | Task 2/3 | branch/trunk 对未知物理参数有解释性 |
| PI-DeepONet | 物理约束思想来源 | Task 1/2/3 | 提供 PDE 残差、边界/初值一致性损失 |

最稳妥的 MVP 路线：

1. 先实现一个轻量 FNO/Conv 预测器，跑通 HDF5 输入输出。
2. 加入自回归 rollout 和分段评分指标。
3. 加入 Burgers/KS 物理残差与频谱损失。
4. 再引入 DeepONet 风格 latent parameter encoder。
5. 最后让 Agent 自动搜索配置、比较实验、生成日志。

