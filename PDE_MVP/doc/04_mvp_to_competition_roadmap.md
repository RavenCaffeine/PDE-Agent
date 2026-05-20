# 从最小 MVP 到完整比赛系统的迭代路线

## 阶段 0：仓库与文档初始化

目标：

- 建立 `doc/` 与 `mvp/`。
- 固化赛题理解、baseline 解读、需求分析、迭代路线。

交付：

- `doc/01_competition_interpretation.md`
- `doc/02_baseline_interpretation.md`
- `doc/03_requirements_analysis.md`
- `doc/04_mvp_to_competition_roadmap.md`
- `mvp/README.md`

## 阶段 1：最小可运行训练与推理

目标：

- 读取 HDF5。
- 训练轻量模型。
- 生成预测 HDF5。
- 生成 time.csv。

模型：

- 1D Conv/FNO-like baseline。
- 输入前 10/20 步作为通道。
- 直接输出完整预测窗口。

验收：

- 小样本能完成 train/infer。
- 输出 shape 正确。
- 前 10/20 步严格复制输入。

## 阶段 2：比赛指标与验证闭环

目标：

- 实现 Task 1/2 分段 Rel-MSE、RMSE、统计指标。
- 实现 Task 3 分段指标近似。
- 每次训练后自动评估验证集。

策略：

- 用验证集模拟评分公式。
- 指标不仅看全局 MSE，还看三段得分。
- 保存 best checkpoint。

验收：

- 每次实验输出 `metrics.json`。
- 能比较不同 run 的分段表现。

## 阶段 3：Task 1 强 baseline

目标：

- 在固定 `nu=0.001` 上获得稳定可提交结果。

改进方向：

- 使用官方 checkpoint 微调。
- 使用 FNO/UNet 主干。
- 分段加权损失：前段 MSE 权重大，后段统计损失权重大。
- 加入 Burgers 残差：
  `u_t + u u_x - nu u_xx = 0`
- rollout consistency：预测片段再作为输入继续预测。

验收：

- Task 1 真实测试集推理小于 2 分钟。
- `task1_pred.hdf5` 前 10 步一致性通过。
- `task1_logs.log` 合法 JSONL。

## 阶段 4：Agent 日志与科研闭环

目标：

- 让 Agent 的每次关键动作都进入比赛日志。
- 形成“假设-实验-结果-反思”的记录。

设计：

- `agent_loop.py` 管理实验状态。
- `agent_logger.py` 写 JSONL。
- 每次代码生成、配置修改、实验结论写入 response。
- 工具调用写入 tool_calls。

验收：

- 日志逐行 JSON。
- 每行有 `timestamp`、`elapsed_seconds`。
- code 目录中的关键文件能在日志里找到对应生成说明。

## 阶段 5：Task 3 KS 附加题

目标：

- 构建混沌系统长时预测模型，优先争取方案 B 高分。

模型策略：

- 前 20 步 encoder 推断状态和隐式 `lambda2`。
- 短期使用 MSE 保点态精度。
- 中长期加入能谱、均值方差、自相关损失。
- 用随机长窗 rollout 训练稳定性。

物理策略：

- KS 残差：
  `u_t + u u_x + lambda2 u_xx + u_xxxx = 0`
- 若测试没有 `lambda2`，使用 latent 参数或辅助头估计 `lambda2_hat`。
- 训练阶段可监督 `lambda2_hat`，推理阶段只用观测窗口。

验收：

- `task3_pred.hdf5` shape `(100, 400, 256)`。
- 前 20 步一致性通过。
- 推理小于 2 分钟。

## 阶段 6：Task 2 多物理 Burgers

目标：

- 从头训练多 `Nu` 泛化模型。

模型策略：

- 输入前 10 步。
- Physics encoder 从短窗识别隐式 `Nu`。
- 训练时使用 `Nu` 作为辅助监督，不在推理时读取测试 `Nu`。
- 支持 conditional normalization 或 FiLM。

验收：

- 不加载公开 checkpoint。
- 推理小于 2 分钟。
- 多 `Nu` 验证集分段指标稳定。

## 阶段 7：实验自动化与沙箱执行

目标：

- Agent 能在受控沙箱里改代码、运行、解析结果。
- 避免一次实验破坏主线代码。

沙箱建议：

- 每个实验创建 `runs/{run_id}`。
- 配置、日志、checkpoint、metrics 独立保存。
- 代码修改先生成 patch，再执行 quick test。
- 失败实验保留错误日志和反思。

可加入的进阶能力：

- 自动小样本 smoke test。
- 自动 shape 校验。
- 自动比较 best checkpoint。
- 自动生成 methodology 草稿。
- 自动打包 submission。

## 阶段 8：最终提交

目标：

- 生成完整 `submission.zip`。

最终检查清单：

- 至少一个任务三件套完整。
- HDF5 shape 正确。
- 初始窗口复制误差小于 `1e-3`。
- `time.csv` 格式正确。
- `logs.log` 每行合法 JSON。
- `submission.json` 正确。
- `methodology.pdf` 存在。
- `code/` 非空且可运行。

推荐最终组合：

- 保底：Task 1。
- 冲分：Task 1 + Task 3。
- 完整：Task 1 + Task 2 + Task 3。

