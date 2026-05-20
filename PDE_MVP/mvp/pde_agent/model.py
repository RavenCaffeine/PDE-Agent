from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """Small Fourier layer for 1D spatial fields."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        weight = scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, width = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        modes = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :modes], self.weight[:, :, :modes])
        return torch.fft.irfft(out_ft, n=width, dim=-1)


class FNOBlock1d(nn.Module):
    def __init__(self, channels: int, modes: int, dropout: float = 0.0):
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.pointwise(x)
        y = F.gelu(self.norm(y))
        return x + self.dropout(y)


class WindowToTrajectoryFNO(nn.Module):
    """Map an observed time window to a full PDE trajectory on a fixed grid."""

    def __init__(
        self,
        input_steps: int,
        total_steps: int,
        spatial_size: int = 256,
        hidden_channels: int = 64,
        modes: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_steps = input_steps
        self.total_steps = total_steps
        self.spatial_size = spatial_size
        self.lift = nn.Conv1d(input_steps + 1, hidden_channels, 1)
        self.blocks = nn.Sequential(
            *[FNOBlock1d(hidden_channels, modes=modes, dropout=dropout) for _ in range(depth)]
        )
        self.proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels * 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_channels * 2, total_steps, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,X), got {tuple(x.shape)}")
        grid = torch.linspace(0.0, 1.0, x.shape[-1], device=x.device)
        grid = grid[None, None, :].expand(x.shape[0], 1, -1)
        z = torch.cat([x, grid], dim=1)
        z = self.lift(z)
        z = self.blocks(z)
        pred = self.proj(z)
        pred = pred.clone()
        pred[:, : self.input_steps] = x
        return pred


def build_model(config: dict) -> WindowToTrajectoryFNO:
    task = config["task"]
    model_cfg = config.get("model", {})
    return WindowToTrajectoryFNO(
        input_steps=int(task["input_steps"]),
        total_steps=int(task["total_steps"]),
        spatial_size=int(task.get("spatial_size", 256)),
        hidden_channels=int(model_cfg.get("hidden_channels", 64)),
        modes=int(model_cfg.get("modes", 32)),
        depth=int(model_cfg.get("depth", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )

