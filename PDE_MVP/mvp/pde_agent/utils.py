from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def now_run_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *_args: object) -> None:
        self.elapsed = time.perf_counter() - self.start

