from __future__ import annotations

from typing import Dict

import torch


def rel_mse_value(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    num = torch.sum((pred - target) ** 2, dim=-1)
    den = torch.sum(target**2, dim=-1).clamp_min(eps)
    rel = torch.clamp(num / den, max=5.0)
    return float(rel.mean().detach().cpu())


def rmse_value(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).detach().cpu())


def segment_metrics(pred: torch.Tensor, target: torch.Tensor, input_steps: int) -> Dict[str, float]:
    horizon = pred.shape[1] - input_steps
    first = input_steps + max(1, int(horizon * 0.25))
    second = input_steps + max(2, int(horizon * 0.50))
    result = {
        "rel_mse_all": rel_mse_value(pred[:, input_steps:], target[:, input_steps:]),
        "rmse_all": rmse_value(pred[:, input_steps:], target[:, input_steps:]),
    }
    result["rel_mse_seg1"] = rel_mse_value(pred[:, input_steps:first], target[:, input_steps:first])
    result["rel_mse_seg2"] = rel_mse_value(pred[:, first:second], target[:, first:second])
    result["rel_mse_seg3"] = rel_mse_value(pred[:, second:], target[:, second:])
    result["rmse_seg3"] = rmse_value(pred[:, second:], target[:, second:])
    return result

