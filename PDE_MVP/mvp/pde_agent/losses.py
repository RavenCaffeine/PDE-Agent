from __future__ import annotations

import torch
import torch.nn.functional as F


def relative_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = torch.sum((pred - target) ** 2, dim=-1)
    den = torch.sum(target**2, dim=-1).clamp_min(eps)
    return (num / den).mean()


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_ft = torch.fft.rfft(pred, dim=-1)
    target_ft = torch.fft.rfft(target, dim=-1)
    return F.mse_loss(torch.abs(pred_ft), torch.abs(target_ft))


def segmented_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    input_steps: int,
    segment_weights: tuple[float, float, float] = (0.35, 0.35, 0.30),
) -> torch.Tensor:
    horizon = pred.shape[1] - input_steps
    if horizon <= 0:
        return F.mse_loss(pred, target)
    first = input_steps + max(1, int(horizon * 0.25))
    second = input_steps + max(2, int(horizon * 0.50))
    parts = [
        (pred[:, input_steps:first], target[:, input_steps:first]),
        (pred[:, first:second], target[:, first:second]),
        (pred[:, second:], target[:, second:]),
    ]
    loss = pred.new_tensor(0.0)
    for weight, (p, y) in zip(segment_weights, parts):
        if p.numel() > 0:
            loss = loss + weight * F.mse_loss(p, y)
    return loss


def burgers_residual_loss(pred: torch.Tensor, nu: float = 0.001, dx: float = 1.0, dt: float = 1.0) -> torch.Tensor:
    if pred.shape[1] < 3 or pred.shape[2] < 3:
        return pred.new_tensor(0.0)
    u = pred
    u_t = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2.0 * dt)
    u_mid = u[:, 1:-1, 1:-1]
    u_x = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2.0 * dx)
    u_xx = (u[:, 1:-1, 2:] - 2.0 * u_mid + u[:, 1:-1, :-2]) / (dx * dx)
    residual = u_t + u_mid * u_x - nu * u_xx
    return torch.mean(residual**2)


def ks_residual_loss(pred: torch.Tensor, lambda2: float = 1.25, dx: float = 1.0, dt: float = 1.0) -> torch.Tensor:
    if pred.shape[1] < 3 or pred.shape[2] < 5:
        return pred.new_tensor(0.0)
    u = pred
    c = u[:, 1:-1, 2:-2]
    u_t = (u[:, 2:, 2:-2] - u[:, :-2, 2:-2]) / (2.0 * dt)
    u_x = (u[:, 1:-1, 3:-1] - u[:, 1:-1, 1:-3]) / (2.0 * dx)
    u_xx = (u[:, 1:-1, 3:-1] - 2.0 * c + u[:, 1:-1, 1:-3]) / (dx * dx)
    u_xxxx = (
        u[:, 1:-1, 4:]
        - 4.0 * u[:, 1:-1, 3:-1]
        + 6.0 * c
        - 4.0 * u[:, 1:-1, 1:-3]
        + u[:, 1:-1, :-4]
    ) / (dx**4)
    residual = u_t + c * u_x + lambda2 * u_xx + u_xxxx
    return torch.mean(residual**2)


def build_training_loss(config: dict, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    task = config["task"]
    loss_cfg = config.get("loss", {})
    input_steps = int(task["input_steps"])
    loss = segmented_loss(pred, target, input_steps=input_steps)
    loss = loss + float(loss_cfg.get("spectral_weight", 0.0)) * spectral_loss(pred, target)
    physics_weight = float(loss_cfg.get("physics_weight", 0.0))
    if physics_weight > 0:
        if int(task["id"]) in (1, 2):
            loss = loss + physics_weight * burgers_residual_loss(pred, nu=float(loss_cfg.get("nu", 0.001)))
        elif int(task["id"]) == 3:
            loss = loss + physics_weight * ks_residual_loss(pred, lambda2=float(loss_cfg.get("lambda2", 1.25)))
    return loss

