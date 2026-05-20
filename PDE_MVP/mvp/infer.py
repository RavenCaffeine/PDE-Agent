from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pde_agent.agent_logger import AgentLogger
from pde_agent.data import TASK_SPECS, load_test_input, write_prediction_hdf5
from pde_agent.model import build_model
from pde_agent.utils import Timer, get_device, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--time-csv", default=None)
    parser.add_argument("--log-jsonl", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    task_id = int(config["task"]["id"])
    spec = TASK_SPECS[task_id]
    device = get_device(str(config.get("train", {}).get("device", "auto")))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "config" in checkpoint:
        config = checkpoint["config"]

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    test_x = load_test_input(config["data"]["test_path"], spec)
    loader = DataLoader(
        TensorDataset(test_x),
        batch_size=args.batch_size or int(config.get("data", {}).get("batch_size", 8)),
        shuffle=False,
    )

    log_path = args.log_jsonl or str(Path(args.out).with_suffix(".log"))
    agent_log = AgentLogger(log_path)
    agent_log.record(response=f"Start inference for task {task_id} using checkpoint {args.checkpoint}.")

    preds = []
    with Timer() as timer:
        with torch.no_grad():
            for (x,) in loader:
                x = x.to(device)
                pred = model(x)
                pred[:, : spec.input_steps] = x
                preds.append(pred.detach().cpu().numpy())
    pred_np = np.concatenate(preds, axis=0).astype(np.float32)
    write_prediction_hdf5(args.out, pred_np)

    time_csv = args.time_csv or str(Path(args.out).with_name(f"task{task_id}_time.csv"))
    with open(time_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["train_time", "inference_time"])
        writer.writeheader()
        writer.writerow(
            {
                "train_time": float(checkpoint.get("train_time", 0.0)),
                "inference_time": float(timer.elapsed),
            }
        )

    agent_log.record(
        response=(
            f"Inference completed in {timer.elapsed:.4f} seconds. "
            f"Prediction shape={tuple(pred_np.shape)}. The observed prefix was copied into the output."
        ),
        extra={"inference_time": timer.elapsed, "prediction_shape": list(pred_np.shape)},
    )
    print(f"Wrote prediction: {args.out}")
    print(f"Wrote time CSV: {time_csv}")


if __name__ == "__main__":
    main()

