from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pde_agent.agent_logger import AgentLogger
from pde_agent.data import TASK_SPECS, TrajectoryDataset
from pde_agent.losses import build_training_loss
from pde_agent.metrics import segment_metrics
from pde_agent.model import build_model
from pde_agent.utils import count_parameters, get_device, load_config, now_run_id, save_json, set_seed


def evaluate(model: torch.nn.Module, loader: DataLoader, config: dict, device: torch.device) -> dict:
    model.eval()
    input_steps = int(config["task"]["input_steps"])
    all_metrics = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            all_metrics.append(segment_metrics(pred, y, input_steps=input_steps))
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    return {key: float(sum(item[key] for item in all_metrics) / len(all_metrics)) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    task_id = int(config["task"]["id"])
    spec = TASK_SPECS[task_id]
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})

    set_seed(int(train_cfg.get("seed", 42)))
    device = get_device(str(train_cfg.get("device", "auto")))
    run_root = Path(args.run_dir or train_cfg.get("run_dir", "runs"))
    run_dir = run_root / now_run_id(config["task"]["name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "source_config.yaml")
    save_json(run_dir / "config.json", config)

    agent_log = AgentLogger(run_dir / "agent_logs.jsonl")
    agent_log.record(
        response=(
            f"Initialize training for task {task_id}. "
            "MVP hypothesis: a compact Fourier-style 1D operator can map the observed window "
            "to the full trajectory while preserving the observed prefix exactly."
        )
    )

    train_ds = TrajectoryDataset(data_cfg["train_path"], spec, max_samples=data_cfg.get("max_train_samples"))
    val_ds = TrajectoryDataset(data_cfg["val_path"], spec, max_samples=data_cfg.get("max_val_samples"))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(data_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(train_cfg.get("epochs", 1))))
    agent_log.record(
        response=f"Model created with {count_parameters(model)} trainable parameters. Device={device}."
    )

    best = float("inf")
    best_metrics = {}
    train_start = time.perf_counter()
    epochs = int(train_cfg.get("epochs", 20))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    with open(run_dir / "train.log", "w", encoding="utf-8") as text_log:
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            bar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
            for x, y in bar:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(x)
                loss = build_training_loss(config, pred, y)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                bar.set_postfix(loss=sum(losses) / len(losses))
            scheduler.step()

            metrics = evaluate(model, val_loader, config, device)
            val_score = metrics.get("rel_mse_all", float("inf"))
            line = {"epoch": epoch, "train_loss": sum(losses) / len(losses), **metrics}
            text_log.write(str(line) + "\n")
            text_log.flush()

            agent_log.record(
                response=(
                    f"Epoch {epoch} finished. train_loss={line['train_loss']:.6g}, "
                    f"val_rel_mse={val_score:.6g}. "
                    "Decision rule: keep checkpoint when validation relative MSE improves."
                )
            )

            if val_score < best:
                best = val_score
                best_metrics = metrics
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": config,
                        "metrics": metrics,
                        "train_time": time.perf_counter() - train_start,
                    },
                    run_dir / "checkpoint.pt",
                )

    train_time = time.perf_counter() - train_start
    save_json(run_dir / "metrics.json", {"best_rel_mse": best, "best_metrics": best_metrics, "train_time": train_time})
    agent_log.record(
        response=(
            f"Training completed in {train_time:.2f} seconds. "
            f"Best validation relative MSE={best:.6g}. Checkpoint saved to checkpoint.pt."
        ),
        extra={"train_time": train_time, "best_metrics": best_metrics},
    )
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()

