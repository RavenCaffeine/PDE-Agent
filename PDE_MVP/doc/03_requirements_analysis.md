# 详细需求分析文档

## 1. 项目目标

构建一个面向 PDE 神经算子竞赛的科研 Agent MVP，使其能够：

- 读取 PDEBench 风格 HDF5 数据。
- 训练神经算子模型。
- 对 Task 1/2/3 生成合规预测文件。
- 记录训练、推理、实验结论与 Agent 科研日志。
- 将代码、预测、时间文件、日志和方法论打包为 submission。
- 后续扩展为可自动诊断、改代码、跑实验、反思迭代的闭环 Agent。

## 2. 功能需求

### 2.1 数据读取

必须支持：

- Task 1/2 Burgers：输入前 10 步，输出 200 步，空间 256。
- Task 3 KS：输入前 20 步，输出 400 步，空间 256。
- HDF5 常见 key：`tensor`、`t-coordinate`、`x-coordinate`、`nu`、`lambda2`。
- 自动识别 tensor shape，并做必要降采样或裁剪。

建议支持：

- train/val/test 三类 split。
- 内存映射或延迟读取，避免一次性读爆内存。
- 数据标准化器保存与加载。

### 2.2 模型训练

MVP 必须支持：

- 从头训练一个轻量 1D 神经算子模型。
- 保存 checkpoint。
- 记录 epoch、loss、验证指标。
- 使用 GPU 或 CPU 自动选择。

进阶支持：

- Task 1 加载官方 checkpoint 微调。
- 混合精度训练。
- 梯度裁剪。
- 多段 rollout 训练。
- 配置驱动实验。

### 2.3 模型结构

MVP 建议模型：

- 输入：`(B, input_steps, X)`。
- 主干：1D Conv/FNO block。
- 输出：`(B, output_steps, X)`。
- 约束：前 10/20 步在最终预测中强制复制输入。

后续结构：

- FNO1D/Temporal-FNO。
- U-Net 1D 多尺度模型。
- DeepONet branch/trunk 辅助模型。
- 隐式物理参数 encoder。
- 多模型 ensemble。

### 2.4 损失函数

MVP：

- MSE。
- 分段加权 MSE。
- Rel-MSE metric。

进阶：

- Burgers 离散 PDE 残差。
- KS 离散 PDE 残差。
- 频谱能量损失。
- 统计损失：均值、方差、自相关、能谱。
- 初值一致性损失。
- rollout consistency loss。

### 2.5 推理

必须支持：

- 加载 checkpoint。
- 读取 test HDF5。
- 输出 `task{N}_pred.hdf5`。
- 前 10/20 步严格复制输入。
- 记录 `task{N}_time.csv`，包含 `train_time,inference_time`。

性能要求：

- 推理必须控制在 2 分钟内。
- 避免逐点 Python 循环。
- 批量推理并一次性写 HDF5。

### 2.6 日志

比赛日志要求：

- 每行合法 JSON。
- 必须包含 `timestamp`、`elapsed_seconds`。
- 必须包含 `response` 或 `tool_calls`。

MVP 日志建议分两层：

- `runs/.../train.log`：普通训练日志。
- `submission/task{N}_logs.log`：比赛要求的 Agent JSONL 科研日志。

Agent 日志内容应覆盖：

- 文献/代码理解。
- 假设提出。
- 代码修改说明。
- 实验配置。
- 指标结果。
- 失败分析。
- 下一步决策。

### 2.7 打包提交

必须生成：

- `submission.json`
- `task{N}_pred.hdf5`
- `task{N}_time.csv`
- `task{N}_logs.log`
- `methodology.pdf`
- `code/`
- `submission.zip`

MVP 可先生成 Markdown 版 methodology，再通过工具转换为 PDF。

## 3. 非功能需求

### 3.1 合规性

- 不得调用数值求解器生成测试预测。
- Task 2/3 不得加载公开预训练权重。
- 代码必须由 Agent 生成，并能在日志中追溯。
- 提交日志时间跨度不得超过 12 小时。

### 3.2 可复现性

- 固定随机种子。
- 保存配置文件。
- 保存模型参数、训练时间、推理时间。
- 每个实验有唯一 run id。

### 3.3 可扩展性

- 模型、数据集、损失函数、指标、Agent 策略分模块。
- 配置文件与代码解耦。
- 支持新任务注册。

### 3.4 工程可靠性

- 输入 shape 校验。
- 输出 shape 校验。
- HDF5 key 校验。
- 前缀观测窗口一致性校验。
- 推理时间统计。

## 4. Agent 闭环需求

Agent 应具备以下状态机：

1. `READ`：读取赛题、数据 schema、baseline 代码。
2. `DIAGNOSE`：分析当前实验日志和指标。
3. `HYPOTHESIZE`：提出一个可验证改进假设。
4. `PATCH`：修改模型/损失/配置代码。
5. `RUN`：执行训练或小样本干实验。
6. `EVALUATE`：计算分段指标、耗时、稳定性。
7. `REFLECT`：决定保留、回滚或继续改进。
8. `SUBMIT`：生成合规提交物。

每轮迭代必须写入 JSONL 日志，保证“代码演进”和“Agent 思考”可追踪。

## 5. MVP 验收标准

第一阶段 MVP 达标条件：

- `mvp/train.py` 能在小样本数据上启动训练。
- `mvp/infer.py` 能生成正确 shape 的 HDF5。
- `mvp/scripts/package_submission.py` 能生成提交目录。
- README 清楚说明如何放置数据、训练、推理、打包。
- 文档说明后续如何接入 Agent、物理损失和沙箱执行。

第二阶段达标条件：

- Task 1 跑通真实数据。
- 输出文件前 10 步一致性通过。
- 推理小于 2 分钟。
- JSONL Agent 日志通过格式校验。

第三阶段达标条件：

- Task 1 有可提交结果。
- Task 3 有可提交结果。
- 至少三轮实验迭代可在日志中追溯。

