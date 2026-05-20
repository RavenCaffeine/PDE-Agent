# PDE Agent MVP

这是一个面向“AI4S 智能体 CNS 挑战赛：神经算子 PDE 智能体”的初始项目骨架。目标是先跑通比赛工程闭环，再逐步加入 FNO、DeepONet、物理约束损失、Agent 自动实验与代码沙箱。

## 目录结构

```text
mvp/
├── README.md
├── requirements.txt
├── configs/
│   ├── task1.yaml
│   ├── task2.yaml
│   └── task3.yaml
├── pde_agent/
│   ├── __init__.py
│   ├── agent_logger.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   └── utils.py
├── train.py
├── infer.py
├── agent_loop.py
└── scripts/
    ├── package_submission.py
    └── validate_submission.py
```

## 安装

```bash
cd mvp
pip install -r requirements.txt
```

## 数据放置

建议在仓库根目录创建 `data/`：

```text
data/
├── task1_train.hdf5
├── task1_val.hdf5
├── task1_test.hdf5
├── task2_train.hdf5
├── task2_val.hdf5
├── task2_test.hdf5
└── task3_data_sample_submission/
    └── train_val_test_init/
        ├── KS_train.hdf5
        ├── KS_val.hdf5
        └── KS_test.hdf5
```

配置文件中的路径可以按实际文件名修改。

## 训练

```bash
python train.py --config configs/task1.yaml
python train.py --config configs/task3.yaml
```

输出默认保存在：

```text
runs/{task_name}_{timestamp}/
├── config.json
├── checkpoint.pt
├── metrics.json
├── train.log
└── agent_logs.jsonl
```

## 推理

```bash
python infer.py --config configs/task1.yaml --checkpoint runs/task1_xxx/checkpoint.pt --out submission/task1_pred.hdf5
```

推理脚本会：

- 读取 test HDF5。
- 生成比赛要求 shape 的预测。
- 强制复制前 10/20 个输入时间步。
- 记录推理耗时。

## 打包提交

```bash
python scripts/package_submission.py \
  --submission-id your_team_name \
  --tasks 1 \
  --source-dir . \
  --work-dir submission \
  --zip-path submission.zip
```

## 进阶改进路线

1. 用 `pde_agent/model.py` 中的 `SpectralConv1d` 替换更强的 FNO block。
2. 在 `losses.py` 中启用 Burgers/KS 离散残差。
3. 在 Task 2/3 中加入参数推断头，训练时监督 `nu` 或 `lambda2`，推理时只用观测窗口。
4. 用 `agent_loop.py` 管理实验：读取指标、提出下一轮配置、写 Agent JSONL 日志。
5. 引入代码沙箱：每个实验复制到独立 run 目录，只允许 Agent 修改白名单文件，跑 smoke test 后再进入长训练。

## 合规提醒

- Task 2/3 禁止加载公开预训练权重。
- 测试推理不得调用数值求解器。
- `task{N}_logs.log` 必须逐行 JSON，且每行包含 `timestamp`、`elapsed_seconds`、`response` 或 `tool_calls`。
- 预测文件前 10/20 步必须与测试输入一致，容差 `1e-3`。

