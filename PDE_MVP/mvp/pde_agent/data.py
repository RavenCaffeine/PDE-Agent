from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TaskSpec:
    task_id: int
    input_steps: int
    total_steps: int
    spatial_size: int = 256
    tensor_key: str = "tensor"
    param_key: Optional[str] = None


TASK_SPECS: Dict[int, TaskSpec] = {
    1: TaskSpec(task_id=1, input_steps=10, total_steps=200, param_key="nu"),
    2: TaskSpec(task_id=2, input_steps=10, total_steps=200, param_key="nu"),
    3: TaskSpec(task_id=3, input_steps=20, total_steps=400, param_key="lambda2"),
}


def _pick_key(h5: h5py.File, preferred: str) -> str:
    if preferred in h5:
        return preferred
    candidates = ["tensor", "u", "data", "x"]
    for key in candidates:
        if key in h5:
            return key
    raise KeyError(f"Cannot find tensor key. Available keys: {list(h5.keys())}")


def load_hdf5_tensor(path: str | Path, tensor_key: str = "tensor") -> np.ndarray:
    with h5py.File(path, "r") as h5:
        key = _pick_key(h5, tensor_key)
        arr = h5[key][:]
    if arr.ndim != 3:
        raise ValueError(f"Expected tensor with shape (N,T,X), got {arr.shape} from {path}")
    return arr.astype(np.float32)


def resize_or_crop_space(arr: np.ndarray, spatial_size: int) -> np.ndarray:
    if arr.shape[-1] == spatial_size:
        return arr
    if arr.shape[-1] % spatial_size == 0:
        stride = arr.shape[-1] // spatial_size
        return arr[..., ::stride]
    return arr[..., :spatial_size]


def resize_or_crop_time(arr: np.ndarray, total_steps: int) -> np.ndarray:
    if arr.shape[1] == total_steps:
        return arr
    if arr.shape[1] % total_steps == 0:
        stride = arr.shape[1] // total_steps
        return arr[:, ::stride]
    return arr[:, :total_steps]


def adapt_tensor(arr: np.ndarray, spec: TaskSpec) -> np.ndarray:
    arr = resize_or_crop_space(arr, spec.spatial_size)
    arr = resize_or_crop_time(arr, spec.total_steps)
    if arr.shape[1] < spec.input_steps:
        raise ValueError(f"Need at least {spec.input_steps} time steps, got {arr.shape}")
    return arr.astype(np.float32)


class TrajectoryDataset(Dataset):
    def __init__(self, path: str | Path, spec: TaskSpec, max_samples: Optional[int] = None):
        self.path = Path(path)
        self.spec = spec
        arr = adapt_tensor(load_hdf5_tensor(self.path, spec.tensor_key), spec)
        if max_samples:
            arr = arr[:max_samples]
        self.tensor = torch.from_numpy(arr)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.tensor[idx]
        x = y[: self.spec.input_steps]
        return x, y


def load_test_input(path: str | Path, spec: TaskSpec) -> torch.Tensor:
    arr = load_hdf5_tensor(path, spec.tensor_key)
    arr = resize_or_crop_space(arr, spec.spatial_size)
    if arr.shape[1] >= spec.total_steps:
        arr = resize_or_crop_time(arr, spec.total_steps)
        arr = arr[:, : spec.input_steps]
    elif arr.shape[1] != spec.input_steps:
        arr = arr[:, : spec.input_steps]
    return torch.from_numpy(arr.astype(np.float32))


def write_prediction_hdf5(path: str | Path, pred: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("tensor", data=pred.astype(np.float32), compression="gzip")

